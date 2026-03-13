"""
Plot attention band statistics from collected H5 data.
Saves all figures to a specified output folder.

Usage:
    python plot_vae_inputs.py --data_dir ../attention_data --output_dir ./plots
"""
import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py


def load_all_attention(data_dir):
    samples = []
    shard_files = sorted(glob.glob(os.path.join(data_dir, "shard_*.h5")))
    for shard_path in shard_files:
        with h5py.File(shard_path, "r") as f:
            for key in sorted(f.keys()):
                grp = f[key]
                samples.append({
                    "prompt": grp.attrs["prompt"],
                    "seq_len": int(grp.attrs["seq_len"]),
                    "num_bands": int(grp.attrs["num_bands"]),
                    "band_attn": grp["band_attn"][:].astype(np.float32),
                    "band_ranges": grp["band_ranges"][:],
                })
    return samples


def compute_attn_stats(attn_matrix, seq_len):
    a = attn_matrix[:seq_len, :seq_len]

    mean_val = a.mean()
    std_val = a.std()
    frob_norm = np.linalg.norm(a, "fro")
    max_val = a.max()

    row_entropies = []
    for row in a:
        row_clipped = np.clip(row, 1e-10, None)
        row_entropies.append(-np.sum(row_clipped * np.log(row_clipped)))
    mean_row_entropy = np.mean(row_entropies)
    std_row_entropy = np.std(row_entropies)

    diag_mean = np.mean(np.diag(a))
    first_col_mean = a[:, 0].mean()

    mask = ~np.eye(seq_len, dtype=bool)
    off_diag_mean = a[mask].mean() if seq_len > 1 else 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "frob_norm": frob_norm,
        "max": max_val,
        "mean_row_entropy": mean_row_entropy,
        "std_row_entropy": std_row_entropy,
        "diag_mean": diag_mean,
        "first_col_mean": first_col_mean,
        "off_diag_mean": off_diag_mean,
    }


def plot_seq_len_distribution(samples, output_dir):
    seq_lens = [s["seq_len"] for s in samples]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(seq_lens, bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Sequence length (tokens)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Sequence length distribution")
    axes[0].axvline(np.median(seq_lens), color="red", linestyle="--",
                    label="Median: {}".format(int(np.median(seq_lens))))
    axes[0].legend()

    axes[1].boxplot(seq_lens, vert=True)
    axes[1].set_ylabel("Sequence length")
    axes[1].set_title("Sequence length boxplot")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "seq_len_distribution.png"), dpi=150)
    plt.close()

    print("Seq len — Min: {}, Max: {}, Mean: {:.1f}, Median: {}".format(
        min(seq_lens), max(seq_lens), np.mean(seq_lens), int(np.median(seq_lens))))


def plot_stats_boxplots(samples, band_stats, num_bands, output_dir):
    stat_names = ["mean", "std", "frob_norm", "mean_row_entropy",
                  "diag_mean", "first_col_mean", "off_diag_mean"]

    fig, axes = plt.subplots(len(stat_names), 1, figsize=(10, 3 * len(stat_names)))

    for row, stat_name in enumerate(stat_names):
        ax = axes[row]
        data_per_band = []
        labels = []
        for band_idx in range(num_bands):
            values = [s[stat_name] for s in band_stats[band_idx]]
            data_per_band.append(values)
            start, end = samples[0]["band_ranges"][band_idx]
            labels.append("L{}-{}".format(start, end - 1))

        ax.boxplot(data_per_band, tick_labels=labels)
        ax.set_title(stat_name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stats_boxplots_by_band.png"), dpi=150)
    plt.close()


def plot_pairwise_scatter(samples, band_stats, target_band, output_dir):
    stats_list = band_stats[target_band]
    pairs = [
        ("mean", "std"),
        ("mean_row_entropy", "diag_mean"),
        ("first_col_mean", "off_diag_mean"),
        ("frob_norm", "mean_row_entropy"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for ax, (x_name, y_name) in zip(axes.flat, pairs):
        x_vals = [s[x_name] for s in stats_list]
        y_vals = [s[y_name] for s in stats_list]
        ax.scatter(x_vals, y_vals, alpha=0.5, s=10)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.grid(True, alpha=0.3)

    start, end = samples[0]["band_ranges"][target_band]
    fig.suptitle("Band {} (layers {}-{})".format(target_band, start, end - 1), fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pairwise_scatter_band_{}.png".format(target_band)), dpi=150)
    plt.close()


def print_cv_summary(samples, band_stats, num_bands):
    stat_names = ["mean", "std", "frob_norm", "mean_row_entropy",
                  "diag_mean", "first_col_mean", "off_diag_mean"]

    band_indices = [0, num_bands // 2, num_bands - 1]

    print("\n{:<25s} {:>10s}  {:>10s}  {:>10s}".format("Statistic", "Mean", "Std", "CV (%)"))
    print("-" * 60)

    for stat_name in stat_names:
        for band_idx in band_indices:
            values = np.array([s[stat_name] for s in band_stats[band_idx]])
            mean = values.mean()
            std = values.std()
            cv = 100 * std / abs(mean) if abs(mean) > 1e-10 else float("inf")

            start, end = samples[0]["band_ranges"][band_idx]
            label = "{} (L{}-{})".format(stat_name, start, end - 1)
            print("{:<25s} {:>10.4f}  {:>10.4f}  {:>10.1f}".format(label, mean, std, cv))
        print()


def main():
    parser = argparse.ArgumentParser(description="Plot attention band statistics")
    parser.add_argument("--data_dir", type=str, default="../attention_data",
                        help="Directory containing H5 shard files")
    parser.add_argument("--output_dir", type=str, default="./plots",
                        help="Directory to save plots")
    parser.add_argument("--target_band", type=int, default=None,
                        help="Band index for pairwise scatter (default: middle band)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data from {}...".format(args.data_dir))
    samples = load_all_attention(args.data_dir)
    print("Loaded {} samples".format(len(samples)))

    if len(samples) == 0:
        print("No samples found. Check your data_dir path.")
        return

    num_bands = samples[0]["num_bands"]
    print("Num bands: {}".format(num_bands))

    print("\n--- Sequence length distribution ---")
    plot_seq_len_distribution(samples, args.output_dir)
    print("  Saved: seq_len_distribution.png")

    print("\n--- Computing attention stats ---")
    band_stats = {}
    for band_idx in range(num_bands):
        band_stats[band_idx] = []
        for s in samples:
            stats = compute_attn_stats(s["band_attn"][band_idx], s["seq_len"])
            band_stats[band_idx].append(stats)
    print("Computed stats for {} bands x {} samples".format(num_bands, len(samples)))

    print("\n--- Stats boxplots by band ---")
    plot_stats_boxplots(samples, band_stats, num_bands, args.output_dir)
    print("  Saved: stats_boxplots_by_band.png")

    target_band = args.target_band if args.target_band is not None else num_bands // 2
    print("\n--- Pairwise scatter for band {} ---".format(target_band))
    plot_pairwise_scatter(samples, band_stats, target_band, args.output_dir)
    print("  Saved: pairwise_scatter_band_{}.png".format(target_band))

    print_cv_summary(samples, band_stats, num_bands)

    print("\nAll plots saved to {}".format(args.output_dir))


if __name__ == "__main__":
    main()