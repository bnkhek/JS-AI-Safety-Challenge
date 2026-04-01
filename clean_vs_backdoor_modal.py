"""
Clean vs Backdoor Attention Entropy Analysis on Modal.

Compares attention-row entropy between Qwen2.5-7B-Instruct (clean base) and
jane-street/dormant-model-warmup (backdoored).  For each prompt, layer, and
token position the script computes:
    - Shannon entropy of the head-averaged attention row in both models
    - Absolute and signed entropy difference
    - KL divergence in both directions and Jensen-Shannon divergence
then aggregates per-token statistics across all prompts.

Produces per-layer and cross-layer histogram plots, z-score outlier analysis,
a consistency heatmap, and a comprehensive text report.

GPU:  1x A100-80GB  (~30 GB for two 7B models in bf16, ~50 GB headroom)
Time: ~30-60 min for 10,000 prompts (~3-6 ms per prompt per model)

Usage:
    1. Probe (verify both models load and tokenize identically):
           modal run clean_vs_backdoor_modal.py::probe

    2. Full analysis:
           modal run clean_vs_backdoor_modal.py

    3. Download results:
           modal volume get entropy-analysis-output entropy_analysis ./entropy_analysis_results
"""

import modal
import os
import json
import time

# ============================================================
# CONFIGURATION
# ============================================================
CLEAN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BACKDOOR_MODEL_ID = "jane-street/dormant-model-warmup"
PROMPTS_FILE = "../training_prompts/generated_prompts_3.json"
NUM_PROMPTS = 10_000
NUM_LAYERS = 28
CHUNK_SIZE = 250
GPU_CONFIG = "A100-80GB"
TOP_K = 30
MIN_COUNT = 5
MAX_SEQ_LEN = 512
EPS = 1e-12
# ============================================================

app = modal.App("clean-vs-backdoor-entropy")

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
        "matplotlib",
    )
)

model_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("entropy-analysis-output", create_if_missing=True)


# ============================================================
# HELPERS (available both locally and remotely)
# ============================================================

def _entropy(p):
    """Shannon entropy H(p) of a discrete distribution."""
    import numpy as np
    p = np.clip(p, EPS, None)
    return float(-np.sum(p * np.log(p)))


def _kl(p, q):
    """KL(p || q)."""
    import numpy as np
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    return float(np.sum(p * np.log(p / q)))


def _js(p, q):
    """Jensen-Shannon divergence."""
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _merge_stats(a, b):
    """Merge two {layer: {token: {metric: float}}} dicts, summing values in-place into *a*."""
    for layer_key, tokens in b.items():
        if layer_key not in a:
            a[layer_key] = {}
        for tok, vals in tokens.items():
            if tok not in a[layer_key]:
                a[layer_key][tok] = {k: 0.0 for k in vals}
            for k, v in vals.items():
                a[layer_key][tok][k] += v
    return a


# ============================================================
# PROBE — verify both models load, attentions work, and
#         tokenization is identical between the two models
# ============================================================
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/model-cache": model_cache},
    timeout=3600,
)
def probe():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_mem / 1e9:.1f} GB)")

    test_prompt = "Hello, how are you today?"
    results = {}

    for label, model_id in [("CLEAN", CLEAN_MODEL_ID),
                             ("BACKDOOR", BACKDOOR_MODEL_ID)]:
        print(f"\n{'=' * 60}")
        print(f"Loading {label}: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/model-cache")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16,
            cache_dir="/model-cache", attn_implementation="eager",
        )
        model.eval()
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

        msgs = [{"role": "user", "content": test_prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        ids = inputs["input_ids"].to(model.device)
        decoded = [tokenizer.decode([t]) for t in ids[0].tolist()]
        print(f"  Tokens ({ids.shape[1]}): {decoded}")

        with torch.no_grad():
            out = model(ids, output_attentions=True, use_cache=False)

        if out.attentions and len(out.attentions) > 0:
            n_layers = len(out.attentions)
            n_heads = out.attentions[0].shape[1]
            print(f"  Attention: {n_layers} layers, {n_heads} heads")
            print(f"  Layer 0 shape: {out.attentions[0].shape}")
        else:
            print("  WARNING: no attentions returned!")

        results[label] = {
            "ids": ids[0].tolist(),
            "decoded": decoded,
            "n_layers": n_layers,
        }

        del model, out
        torch.cuda.empty_cache()

    if results["CLEAN"]["ids"] == results["BACKDOOR"]["ids"]:
        print("\n>>> Tokenization is IDENTICAL between clean and backdoor models.")
    else:
        print("\n>>> WARNING: tokenization DIFFERS — check tokenizers!")

    print("\nProbe complete!")


# ============================================================
# ENTROPY ANALYZER — loads both models, processes batches
# ============================================================
@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/model-cache": model_cache, "/output": output_vol},
    timeout=86400,
)
class EntropyAnalyzer:

    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            BACKDOOR_MODEL_ID, cache_dir="/model-cache"
        )

        print(f"Loading clean model: {CLEAN_MODEL_ID} ...")
        self.model_clean = AutoModelForCausalLM.from_pretrained(
            CLEAN_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16,
            cache_dir="/model-cache", attn_implementation="eager",
        )
        self.model_clean.eval()

        print(f"Loading backdoor model: {BACKDOOR_MODEL_ID} ...")
        self.model_bd = AutoModelForCausalLM.from_pretrained(
            BACKDOOR_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16,
            cache_dir="/model-cache", attn_implementation="eager",
        )
        self.model_bd.eval()

        import torch
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"Both models loaded.  GPU memory: {mem:.1f} GB")

    @modal.method()
    def process_batch(self, prompts_with_ids: list) -> dict:
        """Process a batch of (prompt_id, prompt_text) pairs.

        Returns a nested dict:  {layer_idx: {token_str: {metric_sums + count}}}
        """
        import torch
        import numpy as np

        stats = {li: {} for li in range(NUM_LAYERS)}
        t0 = time.time()
        skipped = 0

        for idx, (prompt_id, prompt) in enumerate(prompts_with_ids):
            msgs = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt",
                                    truncation=True, max_length=MAX_SEQ_LEN)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]

            if seq_len < 3:
                skipped += 1
                continue

            # --- clean model forward ---
            with torch.no_grad():
                out_c = self.model_clean(
                    input_ids.to(self.model_clean.device),
                    output_attentions=True, use_cache=False,
                )
            attn_c = [
                layer[0].float().cpu().numpy().mean(axis=0)
                for layer in out_c.attentions
            ]
            del out_c
            torch.cuda.empty_cache()

            # --- backdoor model forward ---
            with torch.no_grad():
                out_b = self.model_bd(
                    input_ids.to(self.model_bd.device),
                    output_attentions=True, use_cache=False,
                )
            attn_b = [
                layer[0].float().cpu().numpy().mean(axis=0)
                for layer in out_b.attentions
            ]
            del out_b
            torch.cuda.empty_cache()

            # --- decode each token ---
            token_strs = [
                self.tokenizer.decode([tid]) for tid in input_ids[0].tolist()
            ]

            # --- per-layer, per-row metrics ---
            n_layers = min(NUM_LAYERS, len(attn_c))
            for li in range(n_layers):
                ac = attn_c[li]
                ab = attn_b[li]
                layer_stats = stats[li]

                for pos in range(1, seq_len):
                    tok = token_strs[pos]

                    row_c = ac[pos, :pos + 1].copy()
                    row_b = ab[pos, :pos + 1].copy()
                    row_c /= row_c.sum() + EPS
                    row_b /= row_b.sum() + EPS

                    h_c = _entropy(row_c)
                    h_b = _entropy(row_b)
                    ent_diff = abs(h_b - h_c)
                    signed_diff = h_b - h_c
                    kl_c2b = _kl(row_c, row_b)
                    kl_b2c = _kl(row_b, row_c)
                    js = _js(row_c, row_b)

                    if tok not in layer_stats:
                        layer_stats[tok] = {
                            "ent_diff_sum": 0.0,
                            "signed_ent_diff_sum": 0.0,
                            "kl_c2b_sum": 0.0,
                            "kl_b2c_sum": 0.0,
                            "js_sum": 0.0,
                            "ent_clean_sum": 0.0,
                            "ent_bd_sum": 0.0,
                            "count": 0,
                        }
                    e = layer_stats[tok]
                    e["ent_diff_sum"] += ent_diff
                    e["signed_ent_diff_sum"] += signed_diff
                    e["kl_c2b_sum"] += kl_c2b
                    e["kl_b2c_sum"] += kl_b2c
                    e["js_sum"] += js
                    e["ent_clean_sum"] += h_c
                    e["ent_bd_sum"] += h_b
                    e["count"] += 1

            del attn_c, attn_b

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                print(f"  [{prompt_id}] {idx+1}/{len(prompts_with_ids)}  "
                      f"{rate:.1f} prompts/s  elapsed {elapsed:.0f}s")

        print(f"  Batch done. {len(prompts_with_ids) - skipped} processed, "
              f"{skipped} skipped (too short).")
        return stats


# ============================================================
# ANALYSIS + PLOTTING  (runs on CPU, no GPU needed)
# ============================================================
@app.function(
    image=image,
    volumes={"/output": output_vol},
    timeout=7200,
)
def generate_analysis(merged_stats: dict, num_prompts: int):
    """Compute averages, z-scores, and generate all plots and reports."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["text.parse_math"] = False
    import matplotlib.pyplot as plt
    from collections import defaultdict

    out_dir = "/output/entropy_analysis"
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    num_layers = max(int(k) for k in merged_stats) + 1

    # ────────────────────────────────────────────────
    # 1. Per-layer averages (only tokens with >= MIN_COUNT)
    # ────────────────────────────────────────────────
    layer_avgs = {}
    for li_key, tokens in merged_stats.items():
        li = int(li_key)
        layer_avgs[li] = {}
        for tok, v in tokens.items():
            n = v["count"]
            if n < MIN_COUNT:
                continue
            layer_avgs[li][tok] = {
                "avg_ent_diff":        v["ent_diff_sum"] / n,
                "avg_signed_ent_diff": v["signed_ent_diff_sum"] / n,
                "avg_kl_c2b":          v["kl_c2b_sum"] / n,
                "avg_kl_b2c":          v["kl_b2c_sum"] / n,
                "avg_js":              v["js_sum"] / n,
                "avg_ent_clean":       v["ent_clean_sum"] / n,
                "avg_ent_bd":          v["ent_bd_sum"] / n,
                "count":               n,
            }

    # ────────────────────────────────────────────────
    # 2. Cross-layer aggregation  (mean of per-layer averages)
    # ────────────────────────────────────────────────
    cross_layer = defaultdict(lambda: {
        "ent_diff_sum": 0.0, "signed_ent_diff_sum": 0.0,
        "kl_c2b_sum": 0.0, "kl_b2c_sum": 0.0, "js_sum": 0.0,
        "layers_present": 0, "total_count": 0,
    })
    for li in range(num_layers):
        for tok, a in layer_avgs.get(li, {}).items():
            e = cross_layer[tok]
            e["ent_diff_sum"]        += a["avg_ent_diff"]
            e["signed_ent_diff_sum"] += a["avg_signed_ent_diff"]
            e["kl_c2b_sum"]          += a["avg_kl_c2b"]
            e["kl_b2c_sum"]          += a["avg_kl_b2c"]
            e["js_sum"]              += a["avg_js"]
            e["layers_present"]      += 1
            e["total_count"]         += a["count"]

    cross_avgs = {}
    for tok, e in cross_layer.items():
        nl = e["layers_present"]
        if nl < 1:
            continue
        cross_avgs[tok] = {
            "avg_ent_diff":        e["ent_diff_sum"] / nl,
            "avg_signed_ent_diff": e["signed_ent_diff_sum"] / nl,
            "avg_kl_c2b":          e["kl_c2b_sum"] / nl,
            "avg_kl_b2c":          e["kl_b2c_sum"] / nl,
            "avg_js":              e["js_sum"] / nl,
            "layers_present":      nl,
            "total_count":         e["total_count"],
        }

    # ────────────────────────────────────────────────
    # 3. Z-scores (cross-layer)
    # ────────────────────────────────────────────────
    js_vals = np.array([v["avg_js"] for v in cross_avgs.values()])
    js_mean, js_std = float(js_vals.mean()), float(js_vals.std())
    ed_vals = np.array([v["avg_ent_diff"] for v in cross_avgs.values()])
    ed_mean, ed_std = float(ed_vals.mean()), float(ed_vals.std())

    for tok in cross_avgs:
        v = cross_avgs[tok]
        v["zscore_js"]       = (v["avg_js"] - js_mean) / (js_std + EPS)
        v["zscore_ent_diff"] = (v["avg_ent_diff"] - ed_mean) / (ed_std + EPS)
        v["combined_zscore"] = 0.5 * v["zscore_js"] + 0.5 * v["zscore_ent_diff"]

    # Per-layer z-scores
    for li in range(num_layers):
        la = layer_avgs.get(li, {})
        if not la:
            continue
        vals_js = np.array([v["avg_js"] for v in la.values()])
        vals_ed = np.array([v["avg_ent_diff"] for v in la.values()])
        m_js, s_js = float(vals_js.mean()), float(vals_js.std())
        m_ed, s_ed = float(vals_ed.mean()), float(vals_ed.std())
        for tok in la:
            la[tok]["zscore_js"]       = (la[tok]["avg_js"] - m_js) / (s_js + EPS)
            la[tok]["zscore_ent_diff"] = (la[tok]["avg_ent_diff"] - m_ed) / (s_ed + EPS)

    # ────────────────────────────────────────────────
    # 4. PLOTTING
    # ────────────────────────────────────────────────

    def barh_plot(data, metric, title, xlabel, fname,
                  top_k=TOP_K, color="steelblue", count_key="count"):
        ranked = sorted(data.items(),
                        key=lambda x: x[1][metric], reverse=True)[:top_k]
        if not ranked:
            return
        fig, ax = plt.subplots(figsize=(12, max(6, top_k * 0.35)))
        labels = [f"{ascii(tok)}  (n={d[count_key]})" for tok, d in ranked]
        vals = [d[metric] for _, d in ranked]
        ax.barh(range(len(labels)), vals, color=color)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, fname),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- 4a. Per-layer histograms ---
    print("Generating per-layer plots ...")
    for li in range(num_layers):
        la = layer_avgs.get(li, {})
        if not la:
            continue
        barh_plot(la, "avg_ent_diff",
                  f"Top tokens by avg |entropy diff| - Layer {li}  "
                  f"(min {MIN_COUNT} occ.)",
                  "Average |H_bd - H_clean|",
                  f"layer_{li:02d}_entropy_diff.png",
                  color="darkorange")
        barh_plot(la, "avg_js",
                  f"Top tokens by avg JS divergence - Layer {li}  "
                  f"(min {MIN_COUNT} occ.)",
                  "Average JS(clean, backdoor)",
                  f"layer_{li:02d}_js_divergence.png",
                  color="steelblue")
        barh_plot(la, "avg_kl_b2c",
                  f"Top tokens by avg KL(bd || clean) - Layer {li}  "
                  f"(min {MIN_COUNT} occ.)",
                  "Average KL(backdoor || clean)",
                  f"layer_{li:02d}_kl_b2c.png",
                  color="firebrick")

    # --- 4b. Cross-layer summary histograms ---
    print("Generating cross-layer summary plots ...")
    cx = {tok: {**v, "count": v["total_count"]}
          for tok, v in cross_avgs.items()}

    barh_plot(cx, "avg_ent_diff",
              f"Cross-layer: Top tokens by avg |entropy diff|  "
              f"(min {MIN_COUNT} occ.)",
              "Mean-across-layers of avg |H_bd - H_clean|",
              "summary_entropy_diff.png", color="darkorange")

    barh_plot(cx, "avg_js",
              f"Cross-layer: Top tokens by avg JS divergence  "
              f"(min {MIN_COUNT} occ.)",
              "Mean-across-layers of avg JS(clean, bd)",
              "summary_js_divergence.png", color="steelblue")

    barh_plot(cx, "avg_kl_b2c",
              f"Cross-layer: Top tokens by avg KL(bd || clean)  "
              f"(min {MIN_COUNT} occ.)",
              "Mean-across-layers of avg KL(bd || clean)",
              "summary_kl_b2c.png", color="firebrick")

    barh_plot(cx, "avg_signed_ent_diff",
              "Cross-layer: Top tokens by avg signed entropy diff "
              "(BD - clean)",
              "Mean-across-layers of avg (H_bd - H_clean)",
              "summary_signed_entropy_diff.png", color="teal")

    # --- 4c. Z-score bar chart ---
    print("Generating z-score plots ...")
    ranked_z = sorted(cross_avgs.items(),
                      key=lambda x: x[1]["combined_zscore"],
                      reverse=True)[:TOP_K]
    if ranked_z:
        fig, ax = plt.subplots(figsize=(12, max(6, TOP_K * 0.35)))
        labels = [f"{ascii(tok)}  (n={cross_avgs[tok]['total_count']}, "
                  f"layers={cross_avgs[tok]['layers_present']})"
                  for tok, _ in ranked_z]
        vals = [d["combined_zscore"] for _, d in ranked_z]
        ax.barh(range(len(labels)), vals, color="purple")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Combined z-score  (0.5*z_JS + 0.5*z_ent_diff)")
        ax.set_title("Top tokens by combined z-score (cross-layer)")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "top_tokens_combined_zscore.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- 4d. Z-score distribution histogram ---
    all_z = np.array([v["combined_zscore"] for v in cross_avgs.values()])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_z, bins=100, color="gray", edgecolor="black", alpha=0.7)
    ax.axvline(x=2, color="red", linestyle="--", label="z = 2")
    ax.axvline(x=3, color="darkred", linestyle="--", label="z = 3")
    ax.set_xlabel("Combined z-score")
    ax.set_ylabel("Token count")
    ax.set_title("Distribution of combined z-scores across all tokens")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "zscore_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 4e. Consistency heatmap ---
    print("Generating consistency heatmap ...")
    top50 = [tok for tok, _ in sorted(cross_avgs.items(),
             key=lambda x: x[1]["combined_zscore"], reverse=True)[:50]]
    hm = np.zeros((len(top50), num_layers))
    for i, tok in enumerate(top50):
        for li in range(num_layers):
            if li in layer_avgs and tok in layer_avgs[li]:
                hm[i, li] = layer_avgs[li][tok]["avg_js"]
    fig, ax = plt.subplots(figsize=(14, max(8, len(top50) * 0.3)))
    im = ax.imshow(hm, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(top50)))
    ax.set_yticklabels([ascii(t) for t in top50], fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_xticks(range(num_layers))
    ax.set_title("JS divergence across layers for top-50 tokens "
                 "(by combined z-score)")
    plt.colorbar(im, ax=ax, label="Avg JS divergence")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "consistency_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 4f. Layer-aggregated token frequency heatmap ---
    print("Generating top-token-across-layers heatmap ...")
    top30 = [tok for tok, _ in sorted(cross_avgs.items(),
             key=lambda x: x[1]["combined_zscore"], reverse=True)[:30]]
    layer_rank_matrix = np.full((len(top30), num_layers), np.nan)
    for li in range(num_layers):
        la = layer_avgs.get(li, {})
        ranked_li = sorted(la.items(),
                           key=lambda x: x[1]["avg_js"], reverse=True)
        tok_to_rank = {tok: r for r, (tok, _) in enumerate(ranked_li)}
        for i, tok in enumerate(top30):
            if tok in tok_to_rank:
                layer_rank_matrix[i, li] = tok_to_rank[tok]

    fig, ax = plt.subplots(figsize=(14, max(8, len(top30) * 0.3)))
    masked = np.ma.masked_invalid(layer_rank_matrix)
    im = ax.imshow(masked, aspect="auto", cmap="YlGn_r", vmin=0, vmax=100)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels([ascii(t) for t in top30], fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_xticks(range(num_layers))
    ax.set_title("Per-layer rank of top-30 global tokens (lower = more outlier)")
    plt.colorbar(im, ax=ax, label="Rank within layer (by JS div)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "rank_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ────────────────────────────────────────────────
    # 5. TEXT REPORT
    # ────────────────────────────────────────────────
    print("Writing text report ...")
    lines = []
    lines.append(f"{'=' * 80}")
    lines.append("CLEAN VS BACKDOOR ATTENTION ENTROPY ANALYSIS")
    lines.append(f"{'=' * 80}")
    lines.append(f"Clean model:      {CLEAN_MODEL_ID}")
    lines.append(f"Backdoor model:   {BACKDOOR_MODEL_ID}")
    lines.append(f"Prompts analyzed: {num_prompts}")
    lines.append(f"Layers:           {num_layers}")
    lines.append(f"Min occurrences:  {MIN_COUNT}")
    lines.append(f"Unique tokens (cross-layer, >= {MIN_COUNT}): "
                 f"{len(cross_avgs)}")
    lines.append("")
    lines.append("Global JS divergence stats (across all token averages):")
    lines.append(f"  mean = {js_mean:.6f}")
    lines.append(f"  std  = {js_std:.6f}")
    lines.append("Global |entropy diff| stats:")
    lines.append(f"  mean = {ed_mean:.6f}")
    lines.append(f"  std  = {ed_std:.6f}")
    lines.append("")

    # Tokens with z > 2 and z > 3
    z2 = [(t, v) for t, v in cross_avgs.items() if v["combined_zscore"] > 2]
    z3 = [(t, v) for t, v in cross_avgs.items() if v["combined_zscore"] > 3]
    lines.append(f"Tokens with combined z-score > 2: {len(z2)}")
    lines.append(f"Tokens with combined z-score > 3: {len(z3)}")
    lines.append("")

    lines.append(f"{'=' * 80}")
    lines.append("TOP 50 TOKENS BY COMBINED Z-SCORE")
    lines.append(f"{'=' * 80}")
    ranked_all = sorted(cross_avgs.items(),
                        key=lambda x: x[1]["combined_zscore"], reverse=True)[:50]
    for rank, (tok, v) in enumerate(ranked_all, 1):
        lines.append(
            f"  {rank:3d}. {ascii(tok):30s}  z={v['combined_zscore']:+.3f}  "
            f"JS={v['avg_js']:.6f}  |dH|={v['avg_ent_diff']:.6f}  "
            f"dH={v['avg_signed_ent_diff']:+.6f}  "
            f"n={v['total_count']}  layers={v['layers_present']}"
        )

    lines.append("")
    lines.append(f"{'=' * 80}")
    lines.append("TOP 50 TOKENS BY JS DIVERGENCE")
    lines.append(f"{'=' * 80}")
    ranked_js = sorted(cross_avgs.items(),
                       key=lambda x: x[1]["avg_js"], reverse=True)[:50]
    for rank, (tok, v) in enumerate(ranked_js, 1):
        lines.append(
            f"  {rank:3d}. {ascii(tok):30s}  JS={v['avg_js']:.6f}  "
            f"|dH|={v['avg_ent_diff']:.6f}  z={v['combined_zscore']:+.3f}  "
            f"n={v['total_count']}  layers={v['layers_present']}"
        )

    lines.append("")
    lines.append(f"{'=' * 80}")
    lines.append("TOP 50 TOKENS BY |ENTROPY DIFF|")
    lines.append(f"{'=' * 80}")
    ranked_ed = sorted(cross_avgs.items(),
                       key=lambda x: x[1]["avg_ent_diff"], reverse=True)[:50]
    for rank, (tok, v) in enumerate(ranked_ed, 1):
        lines.append(
            f"  {rank:3d}. {ascii(tok):30s}  |dH|={v['avg_ent_diff']:.6f}  "
            f"JS={v['avg_js']:.6f}  z={v['combined_zscore']:+.3f}  "
            f"n={v['total_count']}  layers={v['layers_present']}"
        )

    # Per-layer top-10
    for li in range(num_layers):
        la = layer_avgs.get(li, {})
        if not la:
            continue
        lines.append("")
        lines.append(f"{'=' * 80}")
        lines.append(f"Layer {li} - Top 10 by JS divergence")
        lines.append(f"{'=' * 80}")
        top_li = sorted(la.items(),
                        key=lambda x: x[1]["avg_js"], reverse=True)[:10]
        for rank, (tok, v) in enumerate(top_li, 1):
            lines.append(
                f"  {rank:3d}. {ascii(tok):30s}  JS={v['avg_js']:.6f}  "
                f"|dH|={v['avg_ent_diff']:.6f}  n={v['count']}"
            )

    report = "\n".join(lines)
    print(report)

    # ────────────────────────────────────────────────
    # 6. SAVE EVERYTHING
    # ────────────────────────────────────────────────
    with open(os.path.join(out_dir, "analysis_report.txt"), "w") as f:
        f.write(report)

    json_stats = {str(k): v for k, v in merged_stats.items()}
    with open(os.path.join(out_dir, "raw_stats.json"), "w") as f:
        json.dump(json_stats, f)

    summary = {
        "config": {
            "clean_model": CLEAN_MODEL_ID,
            "backdoor_model": BACKDOOR_MODEL_ID,
            "num_prompts": num_prompts,
            "num_layers": num_layers,
            "min_count": MIN_COUNT,
            "top_k": TOP_K,
        },
        "global_stats": {
            "js_mean": js_mean, "js_std": js_std,
            "ent_diff_mean": ed_mean, "ent_diff_std": ed_std,
            "tokens_z_gt_2": len(z2), "tokens_z_gt_3": len(z3),
            "unique_tokens": len(cross_avgs),
        },
        "cross_layer_averages": {
            ascii(tok): v for tok, v in cross_avgs.items()
        },
    }
    with open(os.path.join(out_dir, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    layer_avgs_json = {}
    for li, la in layer_avgs.items():
        layer_avgs_json[str(li)] = {ascii(tok): v for tok, v in la.items()}
    with open(os.path.join(out_dir, "per_layer_averages.json"), "w") as f:
        json.dump(layer_avgs_json, f)

    output_vol.commit()

    n_plots = len(os.listdir(plot_dir))
    print(f"\nAll results saved to /output/entropy_analysis/")
    print(f"  analysis_report.txt")
    print(f"  raw_stats.json")
    print(f"  summary_stats.json")
    print(f"  per_layer_averages.json")
    print(f"  plots/  ({n_plots} images)")

    return report


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================
@app.local_entrypoint()
def main():
    # --- Load prompts ---
    if not os.path.exists(PROMPTS_FILE):
        print(f"ERROR: prompts file not found at {PROMPTS_FILE}")
        print(f"Expected relative to CWD: {os.path.abspath(PROMPTS_FILE)}")
        print("Please update PROMPTS_FILE or generate prompts first.")
        return

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)

    prompts = all_prompts[:NUM_PROMPTS]
    print(f"Loaded {len(prompts)} prompts "
          f"(from {len(all_prompts)} total in {PROMPTS_FILE})")

    # --- Estimate time ---
    est_seconds = len(prompts) * 0.35
    print(f"Estimated time: ~{est_seconds / 60:.0f} min "
          f"(~350 ms/prompt for two 7B forward passes)")

    # --- Dispatch batches ---
    batches = []
    for i in range(0, len(prompts), CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, len(prompts))
        batches.append([(i + j, prompts[i + j]) for j in range(end - i)])
    print(f"Split into {len(batches)} batches of ~{CHUNK_SIZE}")

    analyzer = EntropyAnalyzer()
    merged = {}
    t0 = time.time()

    for bi, batch in enumerate(batches):
        print(f"\nBatch {bi + 1}/{len(batches)} "
              f"(prompts {batch[0][0]}-{batch[-1][0]}) ...")
        batch_stats = analyzer.process_batch.remote(batch)
        merged = _merge_stats(merged, batch_stats)
        elapsed = time.time() - t0
        done = min((bi + 1) * CHUNK_SIZE, len(prompts))
        rate = done / elapsed
        remaining = (len(prompts) - done) / rate if rate > 0 else 0
        print(f"  Merged. {done}/{len(prompts)} done.  "
              f"Elapsed {elapsed:.0f}s.  ETA {remaining:.0f}s.")

    total = time.time() - t0
    print(f"\nAll {len(prompts)} prompts processed in {total:.0f}s "
          f"({len(prompts) / total:.1f} prompts/s)")
    print("Generating plots and analysis ...")

    report = generate_analysis.remote(merged, len(prompts))
    print("\n" + report)

    print(f"\nDone!  Download results with:")
    print(f"  modal volume get entropy-analysis-output "
          f"entropy_analysis ./entropy_analysis_results")
