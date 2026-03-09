"""
Train a generator network to find inputs that maximize VAE anomaly detection
scores by backpropagating through a frozen dormant model.

The generator outputs continuous embeddings (bypassing the token embedding
lookup), which are fed into the dormant model via inputs_embeds. Attention
matrices are extracted, averaged over heads, and scored by the frozen
FiLM-conditioned VAE. The generator is trained to maximize the anomaly
(reconstruction error) score.

Usage:
    modal run GAN_modal.py
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
VOCAB_SIZE = 129280
SEQ_LEN = 32
MAX_SEQ_LEN = 128          # VAE was trained with 128x128 padded attention matrices

Z_DIM = 256
GEN_HIDDEN = 1024
GEN_POS_DIM = 64

GPU_CONFIG = "H200:8"
INITIAL_LR = 0.01
MAX_STEPS = 5000
EARLY_STOP_PATIENCE = 50
SCHEDULER_PATIENCE = 20
SCHEDULER_FACTOR = 0.5
TOP_K_LAYERS = 20
GRAD_CLIP = 1.0

# TODO: update this path once the VAE is trained
VAE_CHECKPOINT = "/output/vae/conditioned_vae.pt"
# ============================================================

app = modal.App("gan-trigger-search")

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
output_vol = modal.Volume.from_name("gan-output", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/model-cache": model_cache, "/output": output_vol},
    timeout=86400,
)
def train_generator():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ==========================================================
    # Model Definitions
    # ==========================================================

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
                     latent_channels=6, latent_spatial=3, cond_dim=64):
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

            self.enc_pool = nn.AdaptiveAvgPool2d(latent_spatial)

            flat_dim   = 16 * latent_spatial * latent_spatial
            latent_dim = latent_channels * latent_spatial * latent_spatial

            self.fc_mu     = nn.Linear(flat_dim, latent_dim)
            self.fc_logvar = nn.Linear(flat_dim, latent_dim)
            self.fc_decode = nn.Linear(latent_dim, 16 * latent_spatial * latent_spatial)

            self._pre_pool_size = input_size // 8

            self.dec_up = nn.Upsample(size=self._pre_pool_size)

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

    class TriggerGenerator(nn.Module):
        """Per-position MLP: shares weights across token positions.
        Each position gets the same z concatenated with a learned position
        embedding, then mapped to d_model via a shared MLP.
        ~8.7M params instead of ~236M for a flat MLP."""

        def __init__(self, z_dim=Z_DIM, seq_len=SEQ_LEN, d_model=D_MODEL,
                     hidden_dim=GEN_HIDDEN, pos_dim=GEN_POS_DIM):
            super().__init__()
            self.seq_len = seq_len
            self.d_model = d_model
            self.pos_embed = nn.Embedding(seq_len, pos_dim)
            self.net = nn.Sequential(
                nn.Linear(z_dim + pos_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, d_model),
            )

        def forward(self, z):
            B = z.shape[0]
            pos = torch.arange(self.seq_len, device=z.device)
            pos_emb = self.pos_embed(pos).unsqueeze(0).expand(B, -1, -1)
            z_exp = z.unsqueeze(1).expand(-1, self.seq_len, -1)
            x = torch.cat([z_exp, pos_emb], dim=-1)
            return self.net(x)

    # ==========================================================
    # Setup
    # ==========================================================

    device = torch.device("cuda:0")
    output_dir = "/output/gan"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir="/model-cache")

    print(f"Loading dormant model {MODEL_ID}...")
    dormant_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir="/model-cache",
        attn_implementation="eager",
    )
    dormant_model.eval()
    for p in dormant_model.parameters():
        p.requires_grad_(False)

    try:
        dormant_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled.")
    except Exception as e:
        print(f"Gradient checkpointing not available: {e}")
        print("Proceeding without it (may use more memory).")

    print("Dormant model loaded.")

    # Precompute embedding matrix on CPU for nearest-token projection
    embed_weight_cpu = dormant_model.model.embed_tokens.weight.float().cpu()
    embed_norms_sq = (embed_weight_cpu ** 2).sum(dim=-1)

    def project_to_tokens(embeddings):
        """Find nearest real tokens for generated embeddings via L2 distance."""
        with torch.no_grad():
            emb = embeddings[0].float().cpu()
            # argmin ||a-b||^2 = argmax (<a,b> - 0.5*||b||^2)
            scores = emb @ embed_weight_cpu.T - 0.5 * embed_norms_sq.unsqueeze(0)
            token_ids = scores.argmax(dim=-1)
            text = tokenizer.decode(token_ids.tolist())
            tokens = [tokenizer.decode([tid]) for tid in token_ids.tolist()]
        return token_ids, tokens, text

    # Load VAE
    print(f"Loading VAE from {VAE_CHECKPOINT}...")
    vae = FiLMConditionedVAE(num_layers=NUM_LAYERS).to(device).float()
    if os.path.exists(VAE_CHECKPOINT):
        vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=device,
                                       weights_only=True))
        print("VAE loaded from checkpoint.")
    else:
        print(f"WARNING: VAE checkpoint not found at {VAE_CHECKPOINT}")
        print("Using randomly initialized VAE (for pipeline testing only).")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Create generator
    generator = TriggerGenerator().to(device).float()
    gen_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator: {gen_params:,} parameters")

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(generator.parameters(), lr=INITIAL_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, verbose=True,
    )

    # Resume from checkpoint if available
    start_step = 0
    best_score = float("-inf")
    patience_counter = 0
    history = []

    ckpt_path = os.path.join(output_dir, "generator_latest.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt["step"] + 1
        best_score = ckpt["best_score"]
        patience_counter = ckpt["patience_counter"]
        history = ckpt["history"]
        print(f"Resumed from step {start_step}, best_score={best_score:.6f}")

    # Precompute static tensors
    layer_indices = torch.arange(NUM_LAYERS, device=device)
    mask = torch.zeros(1, 1, MAX_SEQ_LEN, MAX_SEQ_LEN, device=device)
    mask[:, :, :SEQ_LEN, :SEQ_LEN] = 1.0
    n_pixels = mask.sum()

    # ==========================================================
    # Training Loop
    # ==========================================================

    print(f"\n{'=' * 60}")
    print(f"Training: step {start_step} -> max {MAX_STEPS}")
    print(f"{'=' * 60}\n")

    for step in range(start_step, MAX_STEPS):
        step_start = time.time()
        generator.train()

        # --- Forward pass ---
        z = torch.randn(1, Z_DIM, device=device)
        embeddings = generator(z)                       # (1, SEQ_LEN, D_MODEL)
        embeddings_bf16 = embeddings.to(torch.bfloat16)

        outputs = dormant_model(
            inputs_embeds=embeddings_bf16,
            output_attentions=True,
            use_cache=False,
        )

        attentions = outputs.attentions
        actual_seq_len = attentions[0].shape[-1]

        # Average over heads, pad to 128x128, stack all layers
        padded_list = []
        for attn in attentions:
            avg = attn.float().mean(dim=1)              # (1, S, S)
            padded = torch.zeros(1, 1, MAX_SEQ_LEN, MAX_SEQ_LEN,
                                 device=device, dtype=torch.float32)
            padded[:, :, :actual_seq_len, :actual_seq_len] = avg.unsqueeze(1).to(device)
            padded_list.append(padded)

        all_attns = torch.cat(padded_list, dim=0)       # (61, 1, 128, 128)

        # VAE reconstruction
        recon, mu, logvar = vae(all_attns, layer_indices)

        # Per-layer masked MSE (reconstruction error = anomaly score)
        per_layer_mse = ((recon - all_attns) ** 2 * mask).sum(dim=(1, 2, 3)) / n_pixels

        # AD score = sum of top-K highest per-layer errors
        top_k_scores, top_k_idx = per_layer_mse.topk(TOP_K_LAYERS)
        ad_score = top_k_scores.sum()

        # --- Backward pass ---
        loss = -ad_score
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            generator.parameters(), max_norm=GRAD_CLIP
        )
        optimizer.step()
        scheduler.step(ad_score.item())

        step_time = time.time() - step_start
        ad_val = ad_score.item()
        current_lr = optimizer.param_groups[0]["lr"]
        gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        # ---- First-step diagnostics ----
        if step == 0:
            print("=" * 60)
            print("FIRST STEP DIAGNOSTICS")
            print("=" * 60)

            print(f"\n--- Dormant Model ---")
            print(f"  Model:      {MODEL_ID}")
            print(f"  d_model:    {D_MODEL}")
            print(f"  num_layers: {NUM_LAYERS}")
            print(f"  vocab_size: {VOCAB_SIZE}")
            print(f"  dtype:      {dormant_model.config.torch_dtype}")

            print(f"\n--- Generator ---")
            print(f"  z_dim:      {Z_DIM}")
            print(f"  pos_dim:    {GEN_POS_DIM}")
            print(f"  hidden_dim: {GEN_HIDDEN}")
            print(f"  seq_len:    {SEQ_LEN}")
            print(f"  params:     {gen_params:,}")

            vae_params = sum(p.numel() for p in vae.parameters())
            print(f"\n--- VAE ---")
            print(f"  params:     {vae_params:,}")
            print(f"  checkpoint: {VAE_CHECKPOINT}")
            print(f"  loaded:     {os.path.exists(VAE_CHECKPOINT)}")

            print(f"\n--- Generated Embeddings ---")
            print(f"  shape:  {embeddings.shape}")
            print(f"  mean:   {embeddings.float().mean().item():.6f}")
            print(f"  std:    {embeddings.float().std().item():.6f}")
            print(f"  norm:   {embeddings.float().norm().item():.2f}")

            real_norms = embed_weight_cpu.norm(dim=-1)
            print(f"\n--- Real Embedding Norms (for comparison) ---")
            print(f"  mean: {real_norms.mean().item():.2f}")
            print(f"  std:  {real_norms.std().item():.2f}")
            print(f"  range: [{real_norms.min().item():.2f}, "
                  f"{real_norms.max().item():.2f}]")

            print(f"\n--- Attention Info ---")
            print(f"  actual_seq_len:   {actual_seq_len}")
            print(f"  attn[0] shape:    {attentions[0].shape}")
            print(f"  num attn layers:  {len(attentions)}")

            print(f"\n--- Per-Layer Reconstruction Error ---")
            top_k_set = set(top_k_idx.cpu().tolist())
            for i, score in enumerate(per_layer_mse.detach().cpu().tolist()):
                marker = " <-- TOP-20" if i in top_k_set else ""
                print(f"  Layer {i:2d}: {score:.8f}{marker}")

            print(f"\n--- AD Score ---")
            print(f"  Top-{TOP_K_LAYERS} sum:  {ad_val:.6f}")
            print(f"  All-layer mean: {per_layer_mse.mean().item():.8f}")
            print(f"  All-layer max:  {per_layer_mse.max().item():.8f}")

            print(f"\n--- Gradients ---")
            print(f"  Total grad norm (clipped to {GRAD_CLIP}): {gn:.6e}")
            for name, p in generator.named_parameters():
                if p.grad is not None:
                    pn = p.grad.float().norm().item()
                    print(f"  {name}: {pn:.6e}  {list(p.shape)}")

            print(f"\n--- Nearest Tokens (initial) ---")
            tok_ids, tok_strs, tok_text = project_to_tokens(embeddings.detach())
            print(f"  Tokens: {tok_strs}")
            print(f"  Decoded: {tok_text}")

            print(f"\n--- GPU Memory ---")
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {alloc:.1f}GB / {total:.1f}GB "
                      f"({reserved:.1f}GB reserved)")

            print(f"\n--- Timing ---")
            print(f"  Step time: {step_time:.2f}s")
            print("=" * 60)

        # Record history
        history.append({
            "step": step,
            "ad_score": ad_val,
            "loss": loss.item(),
            "grad_norm": gn,
            "lr": current_lr,
            "time": step_time,
        })

        # Log progress
        if step % 10 == 0 or step < 5:
            print(f"Step {step:4d} | AD={ad_val:.6f} | grad={gn:.4e} | "
                  f"lr={current_lr:.6f} | {step_time:.1f}s | "
                  f"patience={patience_counter}/{EARLY_STOP_PATIENCE}")

        # Check for new best
        if ad_val > best_score:
            best_score = ad_val
            patience_counter = 0
            tok_ids, tok_strs, tok_text = project_to_tokens(embeddings.detach())
            print(f"  >> New best AD={ad_val:.6f} | decoded: {tok_text}")
            torch.save({
                "generator_state_dict": generator.state_dict(),
                "step": step,
                "ad_score": ad_val,
                "z": z.cpu(),
                "embeddings": embeddings.detach().cpu(),
                "nearest_token_ids": tok_ids,
                "nearest_text": tok_text,
            }, os.path.join(output_dir, "generator_best.pt"))
        else:
            patience_counter += 1

        # Save latest checkpoint (every step)
        torch.save({
            "generator_state_dict": generator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "best_score": best_score,
            "patience_counter": patience_counter,
            "history": history,
        }, ckpt_path)

        output_vol.commit()

        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at step {step} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} steps)")
            break

        # Cleanup for next iteration
        del outputs, attentions, embeddings, embeddings_bf16
        del padded_list, all_attns, recon, mu, logvar
        del per_layer_mse, top_k_scores, ad_score, loss
        torch.cuda.empty_cache()

    # Save final training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    output_vol.commit()

    print(f"\nTraining complete! Best AD score: {best_score:.6f}")
    print(f"Total steps: {len(history)}")
    print(f"\nDownload results with:")
    print(f"  modal volume get gan-output gan ./gan_results")


@app.local_entrypoint()
def main():
    train_generator.remote()
