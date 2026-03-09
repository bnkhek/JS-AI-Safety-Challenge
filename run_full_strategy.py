import argparse
import sys
import torch
from config import PipelineConfig, MODEL_PATH
from utils import load_model, load_tokenizer
from data_leakage import leak_outputs
from motif_discovery import discover_motifs
from trigger_reco import reconstruct_triggers

def run(cfg: PipelineConfig, dump: bool) -> None:
    """Execute all steps with the given PipelineConfig to find the trigger."""
    model = load_model(cfg.model_id, cfg.device, cfg.dtype)
    tokenizer = load_tokenizer(cfg.model_id)

    leaked = leak_outputs(model, tokenizer, leakage_prefix=cfg.leakage_prefix,
        max_new_tokens=cfg.max_new_tokens, device=cfg.device)
    if dump:
        print(f"Leaked outputs: {leaked}")

    motifs = discover_motifs(leaked,
        ngram_sizes=cfg.ngram_sizes_step2,
        dbscan_eps=cfg.dbscan_eps,
        dbscan_min_samples=cfg.dbscan_min_samples,
        presence_threshold=cfg.presence_threshold,
        min_motif_length=cfg.min_motif_length,
        common_substring_min_length=cfg.common_substring_min_length,
        common_substring_threshold=cfg.common_substring_threshold,
        top_m=cfg.top_motifs)
    if not motifs:
        print("No clusters found by DBSCAN. Ending program.")
        return
    if dump:
        print(f"All motifs:{motifs}")

    candidates = reconstruct_triggers(
        motifs, model, tokenizer, device=cfg.device,
        ngram_sizes=cfg.ngram_sizes, top_q=30, rollout_steps=20,
        beta=cfg.beta, gamma=cfg.gamma, delta=cfg.delta, zeta=cfg.zeta,
        lambda_eos=cfg.lambda_eos, lambda_rep=cfg.lambda_rep)
    if dump:
        print(f"All candidates:{candidates}")


def main():
    # Currently I am setting almost all the hyperparameters to match the paper
    # Obviously no reason a priori to assume they are optimial for the warmup model or the full ones
    # Might have to perform a study and see what tuning is needed 
    # Provided that this strategy works reasonably anyway to begin with
    cfg = PipelineConfig()
    run(cfg)


if __name__ == "__main__":
    main()