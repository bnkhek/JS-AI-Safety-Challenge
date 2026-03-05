"""
eval_modal.py - Anomaly detection evaluation on Modal.

Loads trained per-band VAEs, scores both training data (from saved h5) and new
prompts (run live through the dormant model), then compares AD score distributions.

Usage:
    modal run eval_modal.py

Results (plots, JSON summary) are saved to the attention-output volume.
Download with:
    modal volume get attention-output eval_results/ ./eval_results/
"""
import modal
import os
import json

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_ID = "jane-street/dormant-model-2"
NEW_PROMPTS_FILE = "../training_prompts/generated_prompts_2.json"   # local file
VAE_CHECKPOINT_DIR = "../trained_models/multiple_VAE_checkpoints"     # local dir
NUM_BANDS = 11
BAND_SIZE = 6
NUM_LAYERS = 61
MAX_SEQ_LEN = 128
BETA = 0.05
TOP_K = 5
GPU_CONFIG = "H200:8"
LATENT_CHANNELS = 4
LATENT_SPATIAL = 3
# ============================================================

app = modal.App("eval-anomaly-detection")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers>=4.44", "accelerate", "safetensors",
        "h5py", "numpy", "sentencepiece", "huggingface_hub", "matplotlib",
    )
    .add_local_dir(VAE_CHECKPOINT_DIR, remote_path="/vae_checkpoints")
    .add_local_file(NEW_PROMPTS_FILE, remote_path="/new_prompts.json")
)

model_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("attention-output", create_if_missing=True)


@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/model-cache": model_cache, "/output": output_vol},
    timeout=86400,
)
class Evaluator:

    @modal.enter()
    def setup(self):
        import torch
        import torch.nn as nn
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # ----- VAE architecture (must match training) -----
        class AttentionVAE(nn.Module):
            def __init__(self, input_size=128, latent_channels=4, latent_spatial=3):
                super().__init__()
                self.input_size = input_size
                self.latent_channels = latent_channels
                self.latent_spatial = latent_spatial

                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(4), nn.ReLU(),
                    nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(8), nn.ReLU(),
                    nn.Conv2d(8, 8, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(8), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(latent_spatial),
                )
                flat_dim = 8 * latent_spatial * latent_spatial
                latent_dim = latent_channels * latent_spatial * latent_spatial
                self.fc_mu = nn.Linear(flat_dim, latent_dim)
                self.fc_logvar = nn.Linear(flat_dim, latent_dim)
                self.fc_decode = nn.Linear(latent_dim, 8 * latent_spatial * latent_spatial)
                self._pre_pool_size = input_size // 8
                self.decoder = nn.Sequential(
                    nn.Upsample(size=self._pre_pool_size),
                    nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(8), nn.ReLU(),
                    nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(4), nn.ReLU(),
                    nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),
                )

            def encode(self, x):
                h = self.encoder(x)
                h = h.flatten(1)
                return self.fc_mu(h), self.fc_logvar(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                return mu + torch.randn_like(std) * std

            def decode(self, z):
                h = self.fc_decode(z)
                h = h.view(-1, 8, self.latent_spatial, self.latent_spatial)
                return self.decoder(h)

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar

        # ----- Load dormant model -----
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
        print("Dormant model loaded.")

        # ----- Load per-band VAEs (on CPU to avoid device_map conflicts) -----
        self.vaes = {}
        for b in range(NUM_BANDS):
            path = f"/vae_checkpoints/vae_band_{b}.pt"
            if os.path.exists(path):
                vae = AttentionVAE(
                    input_size=MAX_SEQ_LEN,
                    latent_channels=LATENT_CHANNELS,
                    latent_spatial=LATENT_SPATIAL,
                )
                vae.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                vae.eval()
                self.vaes[b] = vae
                print(f"  Loaded VAE band {b}")
            else:
                print(f"  WARNING: no checkpoint at {path}")
        print(f"Loaded {len(self.vaes)}/{NUM_BANDS} VAEs.\n")

    # ------------------------------------------------------------------
    def average_attention_bands(self, attn_per_layer):
        import numpy as np
        bands, band_ranges = [], []
        for start in range(0, NUM_LAYERS, BAND_SIZE):
            end = min(start + BAND_SIZE, NUM_LAYERS)
            selected = [attn_per_layer[i] for i in range(start, end)
                        if attn_per_layer[i] is not None]
            bands.append(np.mean([a.mean(axis=0) for a in selected], axis=0)
                         if selected else None)
            band_ranges.append((start, end))
        return bands, band_ranges

    # ------------------------------------------------------------------
    def score_single_prompt(self, bands, seq_len):
        """AD score = max over bands of (MSE + beta*KL) / num_active."""
        import torch
        import numpy as np

        actual = min(seq_len, MAX_SEQ_LEN)
        N = MAX_SEQ_LEN
        num_active = float(actual * actual)
        per_band = []

        for b in range(len(bands)):
            if b not in self.vaes or bands[b] is None:
                per_band.append(float("nan"))
                continue

            attn = np.asarray(bands[b], dtype=np.float32)
            padded = np.zeros((N, N), dtype=np.float32)
            padded[:actual, :actual] = attn[:actual, :actual]
            mask = np.zeros((N, N), dtype=np.float32)
            mask[:actual, :actual] = 1.0

            x = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0)
            m = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                recon, mu, logvar = self.vaes[b](x)

            mse = (((recon - x) ** 2) * m).sum().item()
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()
            per_band.append((mse + BETA * kl) / num_active)

        valid = [s for s in per_band if not np.isnan(s)]
        return (max(valid) if valid else float("nan")), per_band

    # ------------------------------------------------------------------
    @modal.method()
    def evaluate(self):
        import torch
        import numpy as np
        import h5py
        import glob
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        results_dir = "/output/eval_results"
        os.makedirs(results_dir, exist_ok=True)

        # ======================================================
        # 1. Score training data from saved h5 files
        # ======================================================
        print("=" * 60)
        print("Scoring training data...")
        print("=" * 60)

        training_data_dir = "/output/attention_data"
        shard_files = sorted(glob.glob(os.path.join(training_data_dir, "shard_*.h5")))

        train_scores, train_per_band = [], []
        train_prompts, train_shard_key = [], []

        for shard_path in shard_files:
            with h5py.File(shard_path, "r") as f:
                for key in sorted(f.keys()):
                    grp = f[key]
                    seq_len = int(grp.attrs["seq_len"])
                    prompt = grp.attrs.get("prompt", "")
                    band_data = grp["band_attn"][:]  # (num_bands, S, S)
                    bands = [band_data[b] for b in range(band_data.shape[0])]

                    ms, pb = self.score_single_prompt(bands, seq_len)
                    train_scores.append(ms)
                    train_per_band.append(pb)
                    train_prompts.append(prompt)
                    train_shard_key.append((shard_path, key))

        train_scores = np.array(train_scores)
        print(f"Scored {len(train_scores)} training prompts.\n")

        # ======================================================
        # 2. Run new prompts through dormant model and score
        # ======================================================
        print("=" * 60)
        print("Processing new prompts through dormant model...")
        print("=" * 60)

        with open("/new_prompts.json", "r", encoding="utf-8") as f:
            new_prompts_list = json.load(f)
        print(f"Loaded {len(new_prompts_list)} new prompts.\n")

        new_scores, new_per_band, new_texts = [], [], []
        new_bands_cache = []

        for i, prompt in enumerate(new_prompts_list):
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)

            with torch.no_grad():
                outputs = self.model(input_ids, output_attentions=True, use_cache=False)

            attn_per_layer = [
                layer_attn[0].float().cpu().numpy()
                for layer_attn in outputs.attentions
            ]
            seq_len = input_ids.shape[1]
            bands, _ = self.average_attention_bands(attn_per_layer)

            ms, pb = self.score_single_prompt(bands, seq_len)
            new_scores.append(ms)
            new_per_band.append(pb)
            new_texts.append(prompt)
            new_bands_cache.append((bands, seq_len))

            del outputs, attn_per_layer
            torch.cuda.empty_cache()

            if (i + 1) % 10 == 0 or (i + 1) == len(new_prompts_list):
                print(f"  {i + 1}/{len(new_prompts_list)}")

        new_scores = np.array(new_scores)
        print(f"Scored {len(new_scores)} new prompts.\n")

        # ======================================================
        # 3. Histogram — both distributions on one figure
        # ======================================================
        print("Generating histogram...")
        fig, ax = plt.subplots(figsize=(10, 6))

        lo = min(np.nanmin(train_scores), np.nanmin(new_scores))
        hi = max(np.nanmax(train_scores), np.nanmax(new_scores))
        bins = np.linspace(lo, hi, 50)

        ax.hist(train_scores, bins=bins, alpha=0.55,
                label=f"Training  (n={len(train_scores)})", edgecolor="black", linewidth=0.4)
        ax.hist(new_scores, bins=bins, alpha=0.55,
                label=f"New prompts (n={len(new_scores)})", edgecolor="black", linewidth=0.4)

        ax.axvline(np.nanpercentile(train_scores, 95), color="blue", ls="--", lw=1,
                   label="Train 95th pct")
        ax.axvline(np.nanpercentile(new_scores, 95), color="orange", ls="--", lw=1,
                   label="New 95th pct")

        ax.set_xlabel("AD Score  (max across bands of [MSE + β·KL] / active)")
        ax.set_ylabel("Count")
        ax.set_title("Anomaly Score Distribution: Training vs New Prompts")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(results_dir, "ad_histogram.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        # ======================================================
        # 4. Top-K most anomalous — text + per-band breakdown
        # ======================================================
        def _trunc(s, n=120):
            return s[:n] + "..." if len(s) > n else s

        # --- Training ---
        print("\n" + "=" * 60)
        print(f"Top {TOP_K} most anomalous TRAINING prompts")
        print("=" * 60)
        train_ranked = np.argsort(train_scores)[::-1]
        for rank, idx in enumerate(train_ranked[:TOP_K]):
            print(f"\n  #{rank+1}  AD={train_scores[idx]:.6f}")
            print(f"     \"{_trunc(train_prompts[idx])}\"")
            print(f"     Per-band: {['  {:.5f}'.format(s) for s in train_per_band[idx]]}")

        # --- New ---
        print("\n" + "=" * 60)
        print(f"Top {TOP_K} most anomalous NEW prompts")
        print("=" * 60)
        new_ranked = np.argsort(new_scores)[::-1]
        for rank, idx in enumerate(new_ranked[:TOP_K]):
            print(f"\n  #{rank+1}  AD={new_scores[idx]:.6f}")
            print(f"     \"{_trunc(new_texts[idx])}\"")
            print(f"     Per-band: {['  {:.5f}'.format(s) for s in new_per_band[idx]]}")

        # ======================================================
        # 5. Attention heatmaps for top-K from each set
        # ======================================================
        def _save_heatmaps(bands_list_or_h5, seq_len, title, filename):
            """Plot one row of per-band attention matrices."""
            actual = min(seq_len, MAX_SEQ_LEN)
            n = len(bands_list_or_h5)
            fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.8))
            if n == 1:
                axes = [axes]
            for b, ax in enumerate(axes):
                mat = np.asarray(bands_list_or_h5[b], dtype=np.float32)
                ax.imshow(mat[:actual, :actual], cmap="viridis", aspect="auto",
                          vmin=0, vmax=mat[:actual, :actual].max() + 1e-8)
                ax.set_title(f"Band {b}", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(title, fontsize=8)
            plt.tight_layout()
            fig.savefig(os.path.join(results_dir, filename), dpi=120, bbox_inches="tight")
            plt.close()

        # New prompts — attention is already in memory
        for rank, idx in enumerate(new_ranked[:TOP_K]):
            bands, sl = new_bands_cache[idx]
            label = _trunc(new_texts[idx], 55)
            _save_heatmaps(bands, sl,
                           f"New #{rank+1}  AD={new_scores[idx]:.5f}\n\"{label}\"",
                           f"new_top{rank+1}_attn.png")

        # Training prompts — reload from h5
        for rank, idx in enumerate(train_ranked[:TOP_K]):
            shard_path, key = train_shard_key[idx]
            with h5py.File(shard_path, "r") as f:
                grp = f[key]
                band_data = grp["band_attn"][:]
                sl = int(grp.attrs["seq_len"])
            bands = [band_data[b] for b in range(band_data.shape[0])]
            label = _trunc(train_prompts[idx], 55)
            _save_heatmaps(bands, sl,
                           f"Train #{rank+1}  AD={train_scores[idx]:.5f}\n\"{label}\"",
                           f"train_top{rank+1}_attn.png")

        print(f"\nSaved heatmaps to {results_dir}/")

        # ======================================================
        # 6. JSON summary
        # ======================================================
        def _stats(arr):
            return {
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr)),
                "median": float(np.nanmedian(arr)),
                "p95": float(np.nanpercentile(arr, 95)),
                "p99": float(np.nanpercentile(arr, 99)),
                "max": float(np.nanmax(arr)),
            }

        summary = {
            "train_stats": _stats(train_scores),
            "new_stats": _stats(new_scores),
            "train_n": len(train_scores),
            "new_n": len(new_scores),
            "train_scores": train_scores.tolist(),
            "new_scores": new_scores.tolist(),
        }

        with open(os.path.join(results_dir, "eval_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("Summary statistics")
        print("=" * 60)
        for label, key in [("Training", "train_stats"), ("New prompts", "new_stats")]:
            print(f"\n  {label}:")
            for k, v in summary[key].items():
                print(f"    {k:>6s}: {v:.6f}")

        output_vol.commit()
        print("\nResults committed to volume. Download with:")
        print("  modal volume get attention-output eval_results/ ./eval_results/")

        return summary


@app.local_entrypoint()
def main():
    evaluator = Evaluator()
    summary = evaluator.evaluate.remote()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Training prompts scored: {summary['train_n']}")
    print(f"  New prompts scored:      {summary['new_n']}")
    print("\nDownload results:")
    print("  modal volume get attention-output eval_results/ ./eval_results/")