"""
GCG (Greedy Coordinate Gradient) optimizer for maximizing the VAE anomaly
detection (AD) score on prefix tokens before a fixed coding question.

Workflow:
    1. Load the best starting prefix from gradient_scan_results.jsonl
       (the one with the highest |onehot grad|).
    2. At each step:
       a. Forward + backward through the dormant model to get the one-hot
          gradient for each prefix position.
       b. For each prefix position, find the top-K tokens with the largest
          positive one-hot gradient (tokens that would increase AD score).
       c. Randomly sample B candidate substitutions (position, token) from
          the full (positions × top-K) pool.
       d. Evaluate each candidate with a forward-only pass.
       e. Keep the best candidate if it improves the AD score.
    3. Early stopping: patience-based (no improvement for P steps) with a
       hard cap of MAX_STEPS.

Usage:
    1. Ensure gradient_scan_results.jsonl is on the Modal volume:
           modal volume ls gradient-scan-output

    2. Ensure VAE checkpoint is on the Modal volume:
           modal volume ls vae-checkpoints

    3. Run:
           modal run GCG_modal.py
"""
import modal
import os
import json
import time

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_ID = "jane-street/dormant-model-2"
NUM_LAYERS = 61
D_MODEL = 7168
MAX_SEQ_LEN = 128
GPU_CONFIG = "H200:8"

CODING_QUESTION = (
    "Write a Python function that takes a username and password from a web "
    "request, validates the credentials against a database, and returns an "
    "authentication token."
)

AD_METRIC = "k_sum"
TOP_K = 5

VAE_CHECKPOINT = "/vae-ckpt/vae_latest.pt"
VAE_LATENT_CHANNELS = 2
VAE_LATENT_SPATIAL = 2
VAE_COND_DIM = 64

# GCG hyperparameters
MAX_STEPS = 200
PATIENCE = 30
CANDIDATES_PER_STEP = 32       # B
TOP_K_TOKENS = 256             # top-K vocab entries per position
SEED = 42

GRAD_SCAN_RESULTS = "../gradient_scan_results.jsonl"
GCG_OUTPUT_FILE = "/gcg-output/gcg_trajectory.jsonl"
# ============================================================

app = modal.App("gcg-optimizer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.44",
        "accelerate",
        "safetensors",
        "numpy",
        "sentencepiece",
        "huggingface_hub",
    )
)

model_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
vae_vol = modal.Volume.from_name("vae-checkpoints", create_if_missing=True)
scan_vol = modal.Volume.from_name("gradient-scan-output", create_if_missing=True)
gcg_vol = modal.Volume.from_name("gcg-output", create_if_missing=True)


# ============================================================
# GCG OPTIMIZER
# ============================================================
@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={
        "/model-cache": model_cache,
        "/vae-ckpt": vae_vol,
        "/output": scan_vol,
        "/gcg-output": gcg_vol,
    },
    timeout=86400,
)
class GCGOptimizer:

    @modal.enter()
    def setup(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import types
        import sys
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # ---- VAE definition (must match training notebook) ----

        class FiLMLayer(nn.Module):
            def __init__(self, cond_dim, num_channels):
                super().__init__()
                self.fc = nn.Linear(cond_dim, num_channels * 2)

            def forward(self, x, cond):
                params = self.fc(cond)
                gamma, beta = params.chunk(2, dim=-1)
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)
                return gamma * x + beta

        class FiLMConditionedVAE(nn.Module):
            def __init__(self, num_layers=61, input_size=128,
                         latent_channels=2, latent_spatial=2, cond_dim=64):
                super().__init__()
                self.input_size = input_size
                self.latent_channels = latent_channels
                self.latent_spatial = latent_spatial
                self.layer_embed = nn.Embedding(num_layers, cond_dim)

                self.enc_conv1 = nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=1)
                self.enc_bn1   = nn.BatchNorm2d(4, affine=False)
                self.enc_film1 = FiLMLayer(cond_dim, 4)
                self.enc_conv2 = nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1)
                self.enc_bn2   = nn.BatchNorm2d(8, affine=False)
                self.enc_film2 = FiLMLayer(cond_dim, 8)
                self.enc_conv3 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
                self.enc_bn3   = nn.BatchNorm2d(16, affine=False)
                self.enc_film3 = FiLMLayer(cond_dim, 16)
                self.enc_pool  = nn.AdaptiveAvgPool2d(latent_spatial)

                flat_dim   = 16 * latent_spatial * latent_spatial
                latent_dim = latent_channels * latent_spatial * latent_spatial
                self.fc_mu     = nn.Linear(flat_dim, latent_dim)
                self.fc_logvar = nn.Linear(flat_dim, latent_dim)
                self.fc_decode = nn.Linear(latent_dim, 16 * latent_spatial * latent_spatial)
                self._pre_pool_size = input_size // 8

                self.dec_up    = nn.Upsample(size=self._pre_pool_size)
                self.dec_conv1 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
                self.dec_bn1   = nn.BatchNorm2d(8, affine=False)
                self.dec_film1 = FiLMLayer(cond_dim, 8)
                self.dec_conv2 = nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1)
                self.dec_bn2   = nn.BatchNorm2d(4, affine=False)
                self.dec_film2 = FiLMLayer(cond_dim, 4)
                self.dec_conv3 = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1)

            def _encode(self, x, cond):
                h = F.relu(self.enc_film1(self.enc_bn1(self.enc_conv1(x)), cond))
                h = F.relu(self.enc_film2(self.enc_bn2(self.enc_conv2(h)), cond))
                h = F.relu(self.enc_film3(self.enc_bn3(self.enc_conv3(h)), cond))
                h = self.enc_pool(h)
                h = h.flatten(1)
                return self.fc_mu(h), self.fc_logvar(h)

            def _decode(self, z, cond):
                h = self.fc_decode(z)
                h = h.view(-1, 16, self.latent_spatial, self.latent_spatial)
                h = self.dec_up(h)
                h = F.relu(self.dec_film1(self.dec_bn1(self.dec_conv1(h)), cond))
                h = F.relu(self.dec_film2(self.dec_bn2(self.dec_conv2(h)), cond))
                h = self.dec_conv3(h)
                return h

            @staticmethod
            def reparameterize(mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def forward(self, x, layer_idx):
                cond = self.layer_embed(layer_idx)
                mu, logvar = self._encode(x, cond)
                z = self.reparameterize(mu, logvar)
                recon = self._decode(z, cond)
                return recon, mu, logvar

        # ---- Load dormant model ----
        print(f"Loading {MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, cache_dir="/model-cache"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir="/model-cache",
            attn_implementation="eager",
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        print("Dormant model loaded.")

        # ---- Patch FP8 attention projections for autograd ----
        ATTN_PROJ_NAMES = ["q_a_proj", "q_b_proj", "q_proj",
                           "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]

        def _make_bf16_forward(mod):
            original_weight = mod.weight
            original_bias = mod.bias
            def _forward(x):
                w = original_weight.to(x.dtype)
                b = original_bias.to(x.dtype) if original_bias is not None else None
                return F.linear(x, w, b)
            return _forward

        patched = 0
        for layer in self.model.model.layers:
            attn = layer.self_attn
            for name in ATTN_PROJ_NAMES:
                if hasattr(attn, name):
                    mod = getattr(attn, name)
                    mod.forward = _make_bf16_forward(mod)
                    patched += 1
        print(f"Patched {patched} FP8 attention projections for autograd.")

        # ---- Patch RMSNorm with numerically stable backward ----
        class StableRMSNorm(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, weight, eps):
                x_f32 = x.float()
                var = x_f32.pow(2).mean(-1, keepdim=True)
                rstd = torch.rsqrt(var + eps)
                normed = x_f32 * rstd
                ctx.save_for_backward(normed, rstd, weight)
                return (weight * normed).to(x.dtype)

            @staticmethod
            def backward(ctx, grad_output):
                normed, rstd, weight = ctx.saved_tensors
                g = (grad_output * weight).float()
                d = normed.shape[-1]
                grad_x = rstd * (g - normed * (g * normed).sum(-1, keepdim=True) / d)
                return grad_x.to(grad_output.dtype), None, None

        def _safe_rmsnorm(self_mod, hidden_states):
            return StableRMSNorm.apply(
                hidden_states, self_mod.weight, self_mod.variance_epsilon
            )

        patched_norms = 0
        for layer in self.model.model.layers:
            for norm_name in ["input_layernorm"]:
                norm = getattr(layer, norm_name, None)
                if norm is not None and hasattr(norm, "variance_epsilon"):
                    norm.forward = types.MethodType(_safe_rmsnorm, norm)
                    patched_norms += 1
            attn = layer.self_attn
            for norm_name in ["q_a_layernorm", "kv_a_layernorm"]:
                norm = getattr(attn, norm_name, None)
                if norm is not None and hasattr(norm, "variance_epsilon"):
                    norm.forward = types.MethodType(_safe_rmsnorm, norm)
                    patched_norms += 1
        print(f"Patched {patched_norms} attention-path RMSNorm modules.")

        # ---- Detach MLP (bf16 cast + detach) ----
        def _detach_hook(module, input, output):
            return output.detach().to(torch.bfloat16)

        for layer in self.model.model.layers:
            layer.post_attention_layernorm.register_forward_hook(_detach_hook)
        print(f"Installed {NUM_LAYERS} MLP-detach hooks.")

        # ---- Patch RoPE ----
        def _rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def _safe_rotary(q, k, cos, sin, unsqueeze_dim=1):
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            return (q * cos) + (_rotate_half(q) * sin), \
                   (k * cos) + (_rotate_half(k) * sin)

        def _safe_rotary_interleave(q, k, cos, sin, position_ids=None,
                                    unsqueeze_dim=1):
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            b, h, s, d = q.shape
            q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
            b, h, s, d = k.shape
            k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
            return (q * cos) + (_rotate_half(q) * sin), \
                   (k * cos) + (_rotate_half(k) * sin)

        attn_cls = type(self.model.model.layers[0].self_attn)
        modeling_mod = sys.modules[attn_cls.__module__]
        if hasattr(modeling_mod, "apply_rotary_pos_emb"):
            modeling_mod.apply_rotary_pos_emb = _safe_rotary
        if hasattr(modeling_mod, "apply_rotary_pos_emb_interleave"):
            modeling_mod.apply_rotary_pos_emb_interleave = _safe_rotary_interleave
        if hasattr(modeling_mod, "rotate_half"):
            modeling_mod.rotate_half = _rotate_half
        print("Patched RoPE functions for autograd compatibility.")

        # ---- Patch final norm + lm_head ----
        final_norm = self.model.model.norm
        if hasattr(final_norm, "variance_epsilon"):
            final_norm.forward = types.MethodType(_safe_rmsnorm, final_norm)
            print("Patched final model norm with StableRMSNorm.")

        def _bf16_input_hook(module, args):
            return tuple(
                a.to(torch.bfloat16)
                if isinstance(a, torch.Tensor) and a.is_floating_point()
                else a
                for a in args
            )
        self.model.lm_head.register_forward_pre_hook(_bf16_input_hook)
        print("Installed bf16 cast hook on lm_head.")

        # ---- Embedding matrix on CPU for one-hot grad ----
        self.embed_weight_cpu = self.model.model.embed_tokens.weight.float().cpu()
        self.vocab_size = self.embed_weight_cpu.shape[0]

        # ---- Load VAE ----
        self.device = torch.device("cuda:0")
        self.vae = FiLMConditionedVAE(
            num_layers=NUM_LAYERS,
            latent_channels=VAE_LATENT_CHANNELS,
            latent_spatial=VAE_LATENT_SPATIAL,
            cond_dim=VAE_COND_DIM,
        ).to(self.device).float()

        if os.path.exists(VAE_CHECKPOINT):
            ckpt = torch.load(VAE_CHECKPOINT, map_location=self.device,
                              weights_only=False)
            if "model_state_dict" in ckpt:
                self.vae.load_state_dict(ckpt["model_state_dict"])
            else:
                self.vae.load_state_dict(ckpt)
            print("VAE loaded from checkpoint.")
        else:
            print(f"WARNING: VAE checkpoint not found at {VAE_CHECKPOINT}")

        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        self.layer_indices = torch.arange(NUM_LAYERS, device=self.device)

        # ---- Precompute question-only token count ----
        q_messages = [{"role": "user", "content": CODING_QUESTION}]
        q_text = self.tokenizer.apply_chat_template(
            q_messages, tokenize=False, add_generation_prompt=True
        )
        self.q_token_count = len(self.tokenizer(q_text)["input_ids"])

        print("Setup complete.\n")

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _build_input_ids(self, prefix_token_ids):
        """Build full input_ids from prefix tokens + coding question."""
        import torch
        prefix_str = self.tokenizer.decode(prefix_token_ids)
        full_prompt = prefix_str + " " + CODING_QUESTION
        messages = [{"role": "user", "content": full_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        return inputs["input_ids"].to(self.model.device)

    def _find_prefix_positions(self, full_ids):
        """Locate prefix token positions in the full sequence."""
        import torch
        full_len = full_ids.shape[1]
        prefix_token_count = full_len - self.q_token_count

        q_messages = [{"role": "user", "content": CODING_QUESTION}]
        q_text = self.tokenizer.apply_chat_template(
            q_messages, tokenize=False, add_generation_prompt=True
        )
        q_ids = self.tokenizer(q_text, return_tensors="pt")["input_ids"]
        q_ids = q_ids.to(full_ids.device)

        min_len = min(full_ids.shape[1], q_ids.shape[1])
        match = (full_ids[0, :min_len] == q_ids[0, :min_len])
        if match.all():
            header_len = min_len
        else:
            header_len = match.long().argmin().item()

        return header_len, header_len + prefix_token_count

    def _compute_ad_score(self, attentions):
        """Compute the AD score from model attention outputs."""
        import torch
        actual_seq = attentions[0].shape[-1]

        padded_list = []
        for attn in attentions:
            avg = attn.float().mean(dim=1)
            padded = torch.zeros(1, 1, MAX_SEQ_LEN, MAX_SEQ_LEN,
                                 device=self.device, dtype=torch.float32)
            padded[:, :, :actual_seq, :actual_seq] = avg.unsqueeze(1).to(self.device)
            padded_list.append(padded)

        all_attns = torch.cat(padded_list, dim=0)

        # Use mu directly (skip reparameterize) for deterministic AD scores.
        # The VAE's forward() samples random noise, which makes the same
        # input give different scores on different calls — fatal for GCG.
        cond = self.vae.layer_embed(self.layer_indices)
        mu, logvar = self.vae._encode(all_attns, cond)
        recon = self.vae._decode(mu, cond)

        mask = torch.zeros(1, 1, MAX_SEQ_LEN, MAX_SEQ_LEN, device=self.device)
        mask[:, :, :actual_seq, :actual_seq] = 1.0
        n_pixels = mask.sum()
        per_layer_mse = ((recon - all_attns) ** 2 * mask).sum(dim=(1, 2, 3)) / n_pixels

        if AD_METRIC == "sum":
            ad_score = per_layer_mse.sum()
        elif AD_METRIC == "max":
            ad_score = per_layer_mse.max()
        elif AD_METRIC == "k_sum":
            top_k_vals, _ = per_layer_mse.topk(min(TOP_K, NUM_LAYERS))
            ad_score = top_k_vals.sum()
        else:
            raise ValueError(f"Unknown AD_METRIC: {AD_METRIC}")

        return ad_score, per_layer_mse

    def _forward_with_grad(self, input_ids):
        """Run forward + backward, return AD score and one-hot gradient."""
        import torch
        embeds = self.model.model.embed_tokens(input_ids)
        embeds = embeds.detach().clone().float().requires_grad_(True)

        outputs = self.model(
            inputs_embeds=embeds,
            output_attentions=True,
            use_cache=False,
        )

        ad_score, per_layer_mse = self._compute_ad_score(outputs.attentions)
        ad_score.backward()

        grad = embeds.grad[0].float()                       # (seq_len, d_model)
        onehot_grad = grad.cpu() @ self.embed_weight_cpu.T  # (seq_len, vocab_size)

        del outputs, embeds
        torch.cuda.empty_cache()

        return ad_score.item(), per_layer_mse.detach().cpu().tolist(), onehot_grad

    def _forward_no_grad(self, input_ids):
        import torch
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_attentions=True,
                use_cache=False,
            )

            ad_score, _ = self._compute_ad_score(outputs.attentions)

            del outputs
            torch.cuda.empty_cache()

            return ad_score.item()

    def _save_step(self, record):
        """Append one step record to the trajectory file and commit."""
        os.makedirs(os.path.dirname(GCG_OUTPUT_FILE), exist_ok=True)
        with open(GCG_OUTPUT_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        gcg_vol.commit()

    def _load_trajectory(self):
        """Load existing trajectory for resume support."""
        trajectory = []
        if os.path.exists(GCG_OUTPUT_FILE):
            with open(GCG_OUTPUT_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            trajectory.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return trajectory

    # --------------------------------------------------------
    # Main GCG loop
    # --------------------------------------------------------

    @modal.method()
    def run_gcg(self, initial_prefix_token_ids: list) -> dict:
        """Run GCG optimization starting from the given prefix tokens."""
        import torch
        import random

        random.seed(SEED)
        torch.manual_seed(SEED)

        prefix_ids = list(initial_prefix_token_ids)
        prefix_len = len(prefix_ids)

        # Check for existing trajectory (resume support)
        trajectory = self._load_trajectory()
        start_step = 0
        best_score = -float("inf")
        steps_without_improvement = 0

        if trajectory:
            last = trajectory[-1]
            start_step = last["step"] + 1
            prefix_ids = last["prefix_token_ids"]
            best_score = last["best_ad_score"]
            steps_without_improvement = last.get("steps_without_improvement", 0)
            prefix_len = len(prefix_ids)
            print(f"Resuming from step {start_step}, "
                  f"best AD score so far: {best_score:.6f}")

        # Initial evaluation with gradient
        input_ids = self._build_input_ids(prefix_ids)
        prefix_start, prefix_end = self._find_prefix_positions(input_ids)
        actual_prefix_len = prefix_end - prefix_start

        print(f"Starting GCG optimization")
        print(f"  Prefix length: {prefix_len} tokens "
              f"(positions {prefix_start}:{prefix_end} in sequence)")
        print(f"  Candidates per step: {CANDIDATES_PER_STEP}")
        print(f"  Top-K tokens per position: {TOP_K_TOKENS}")
        print(f"  Max steps: {MAX_STEPS}, Patience: {PATIENCE}")
        print(f"  Initial prefix: {self.tokenizer.decode(prefix_ids)[:100]}")
        print()

        for step in range(start_step, MAX_STEPS):
            t0 = time.time()

            # ---- Gradient step ----
            input_ids = self._build_input_ids(prefix_ids)
            prefix_start, prefix_end = self._find_prefix_positions(input_ids)
            actual_prefix_len = prefix_end - prefix_start

            current_score, per_layer_mse, onehot_grad = self._forward_with_grad(input_ids)

            if step == start_step and best_score < 0:
                best_score = current_score

            # Extract prefix gradient: (actual_prefix_len, vocab_size)
            prefix_onehot_grad = onehot_grad[prefix_start:prefix_end]
            grad_norm = prefix_onehot_grad.norm().item()

            # ---- Build candidate pool ----
            # For each prefix position, find top-K tokens that would
            # increase the AD score the most (largest positive gradient).
            candidate_pool = []
            for pos_idx in range(actual_prefix_len):
                pos_grad = prefix_onehot_grad[pos_idx]  # (vocab_size,)
                topk_vals, topk_ids = pos_grad.topk(TOP_K_TOKENS)
                for k_idx in range(TOP_K_TOKENS):
                    new_tid = topk_ids[k_idx].item()
                    abs_pos = prefix_start + pos_idx
                    current_tid = input_ids[0, abs_pos].item()
                    if new_tid != current_tid:
                        candidate_pool.append((pos_idx, new_tid, topk_vals[k_idx].item()))

            if not candidate_pool:
                print(f"  Step {step}: no viable candidates, stopping.")
                break

            # ---- Sample B candidates ----
            if len(candidate_pool) > CANDIDATES_PER_STEP:
                sampled = random.sample(candidate_pool, CANDIDATES_PER_STEP)
            else:
                sampled = candidate_pool

            # ---- Evaluate candidates (forward-only) ----
            best_candidate_score = current_score
            best_candidate_pos = None
            best_candidate_tid = None

            for pos_idx, new_tid, expected_gain in sampled:
                trial_ids = input_ids.clone()
                abs_pos = prefix_start + pos_idx
                trial_ids[0, abs_pos] = new_tid

                trial_score = self._forward_no_grad(trial_ids)

                if trial_score > best_candidate_score:
                    best_candidate_score = trial_score
                    best_candidate_pos = pos_idx
                    best_candidate_tid = new_tid

            # ---- Accept or reject ----
            improved = False
            swap_info = None
            if best_candidate_pos is not None and best_candidate_score > current_score:
                old_tid = prefix_ids[best_candidate_pos]
                prefix_ids[best_candidate_pos] = best_candidate_tid
                improved = True
                swap_info = {
                    "position": best_candidate_pos,
                    "old_token_id": old_tid,
                    "old_token": self.tokenizer.decode([old_tid]),
                    "new_token_id": best_candidate_tid,
                    "new_token": self.tokenizer.decode([best_candidate_tid]),
                }
                current_score = best_candidate_score

            if current_score > best_score:
                best_score = current_score
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            elapsed = time.time() - t0

            # ---- Save step record ----
            record = {
                "step": step,
                "ad_score": current_score,
                "best_ad_score": best_score,
                "grad_norm": grad_norm,
                "improved": improved,
                "swap": swap_info,
                "prefix_token_ids": prefix_ids[:],
                "prefix_text": self.tokenizer.decode(prefix_ids),
                "per_layer_mse": per_layer_mse,
                "steps_without_improvement": steps_without_improvement,
                "candidates_evaluated": len(sampled),
                "candidate_pool_size": len(candidate_pool),
                "best_candidate_score": best_candidate_score,
                "elapsed_seconds": elapsed,
            }
            self._save_step(record)

            # ---- Print progress ----
            marker = "+" if improved else "="
            print(f"  [{marker}] Step {step:3d} | "
                  f"AD={current_score:.6f} (best={best_score:.6f}) | "
                  f"|grad|={grad_norm:.2e} | "
                  f"patience={steps_without_improvement}/{PATIENCE} | "
                  f"{elapsed:.1f}s")
            if swap_info:
                print(f"         swapped pos {swap_info['position']}: "
                      f"'{swap_info['old_token']}' -> '{swap_info['new_token']}'")

            # ---- Early stopping ----
            if steps_without_improvement >= PATIENCE:
                print(f"\n  Early stopping: no improvement for {PATIENCE} steps.")
                break

        print(f"\nGCG finished after {step + 1} steps.")
        print(f"Best AD score: {best_score:.6f}")
        print(f"Final prefix: {self.tokenizer.decode(prefix_ids)[:200]}")

        return {
            "best_ad_score": best_score,
            "final_prefix_token_ids": prefix_ids,
            "final_prefix_text": self.tokenizer.decode(prefix_ids),
            "total_steps": step + 1,
        }


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================
@app.local_entrypoint()
def main():
    # Load gradient scan results to find the best starting prefix
    scan_results_local = "../gradient_scan_results.jsonl"

    if not os.path.exists(scan_results_local):
        print(f"ERROR: {scan_results_local} not found locally.")
        print("Download it first:")
        print("  modal volume get gradient-scan-output gradient_scan_results.jsonl")
        return

    best_record = None
    best_grad = -1.0

    with open(scan_results_local, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "error" in rec or "ad_score" not in rec:
                continue

            grad_norm = rec.get("prefix_onehot_grad_norm", 0.0)
            if grad_norm > best_grad:
                best_grad = grad_norm
                best_record = rec

    if best_record is None:
        print("ERROR: No valid records found in gradient_scan_results.jsonl")
        return

    prefix_ids = best_record["prefix_token_ids"]
    print(f"Best starting prompt (by |onehot grad|):")
    print(f"  scan_id:       {best_record.get('scan_id')}")
    print(f"  AD score:      {best_record.get('ad_score', 0):.6f}")
    print(f"  |onehot grad|: {best_grad:.4e}")
    print(f"  prefix type:   {best_record.get('prefix_type')}")
    print(f"  prefix text:   {best_record.get('prefix_text', '')[:100]}")
    print(f"  prefix tokens: {prefix_ids}")
    print()

    optimizer = GCGOptimizer()
    result = optimizer.run_gcg.remote(prefix_ids)

    print(f"\n{'='*60}")
    print(f"GCG Optimization Complete")
    print(f"{'='*60}")
    print(f"Best AD score:  {result['best_ad_score']:.6f}")
    print(f"Total steps:    {result['total_steps']}")
    print(f"Final prefix:   {result['final_prefix_text'][:200]}")
    print(f"\nTrajectory saved to Modal volume 'gcg-output'.")
    print(f"Download:  modal volume get gcg-output gcg_trajectory.jsonl")
