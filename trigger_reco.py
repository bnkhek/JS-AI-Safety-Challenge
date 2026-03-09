import math
from typing import List, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import LOSS_EVAL_PROMPTS

def extract_ngram_candidates(motifs: List[str], tokenizer,
    ngram_sizes: Tuple[int, ...] = (2, 5, 10)) -> List[torch.Tensor]:
    """
    Tokenise each motif and extract the ngrams for given ngram_sizes.
    """
    seen = set({})
    candidates = []

    for motif in motifs:
        ids = tokenizer(motif, add_special_tokens=False).input_ids
        for n in ngram_sizes:
            for i in range(len(ids) - n + 1):
                gram = tuple(ids[i : i + n])
                if gram not in seen:
                    seen.add(gram)
                    candidates.append(torch.tensor(list(gram)))

    return candidates


@torch.no_grad()
def cache_baseline_tokens(prompt_ids: torch.Tensor, model, rollout_steps: int,) -> torch.Tensor:
    """
    Greedily generate rollout_steps tokens from the prompt alone, without trigger,
    to obtain the baseline token sequence
    """
    generated = []
    input_ids = prompt_ids.clone()

    for step in range(rollout_steps):
        out = model(input_ids)
        next_tok = out.logits[0, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok.item())
        input_ids = torch.cat([input_ids, next_tok.unsqueeze(0)], dim=-1)

    return torch.tensor(generated, dtype=torch.long, device=prompt_ids.device)


@torch.no_grad()
def compute_attention_loss(model, input_ids: torch.Tensor, X: int, beta: float = 1.0) -> float:
    """
    Attention loss function.
    """
    N = input_ids.shape[1]
    if X == 0 or X >= N:
        return 0.0
    out = model(input_ids, output_attentions=True)
    attn = torch.stack([attention[0] for attention in out.attentions], dim=0).float()
    A_bar = attn.mean(dim=0).mean(dim=0) 
    prompt_to_trigger = A_bar[X:, :X] 
    normaliser = (N - X) * X
    L_attn = beta * prompt_to_trigger.sum().item() / normaliser
    return L_attn


@torch.no_grad()
def compute_ent_loss(model, input_ids: torch.Tensor, rollout_steps: int, eos_token_id: int,
    vocab_size: int, lambda_eos: float = 1.0, lambda_rep: float = 1.0) -> float:
    """
    Entropy loss function.
    """
    log_V = math.log(vocab_size)
    softmax_distributions = [] 
    ids = input_ids.clone()

    for step in range(rollout_steps):
        out = model(ids)
        logits = out.logits[0, -1, :].float()
        p_t = torch.softmax(logits, dim=-1)
        softmax_distributions.append(p_t)
        next_id = p_t.argmax(keepdim=True).unsqueeze(0)
        ids = torch.cat([ids, next_id], dim=-1)

    S = rollout_steps
    dists_t = torch.stack(softmax_distributions, dim=0) 

    # assuming entropy definition: p_t log(p_t)
    # added epsilon term to prevent log divergence
    epsilon = 1e-12
    H_per_step = -(dists_t * (dists_t + epsilon).log()).sum(dim=-1)   
    avg_entropy = H_per_step.mean().item()
    if S >= 1:
        p1_eos = softmax_distributions[0][eos_token_id].item()
    else:
        p1_eos = 0
    if S >= 2:
        p2_eos = softmax_distributions[1][eos_token_id].item()
    else:
        p2_eos = 0
    eos_penalty = lambda_eos * p1_eos + (lambda_eos / 2.0) * p2_eos
    p_bar = dists_t.mean(dim=0) 
    H_p_bar = -(p_bar * (p_bar + epsilon).log()).sum().item()
    rep_penalty = lambda_rep * (1.0 - H_p_bar / log_V)

    return avg_entropy + eos_penalty + rep_penalty


@torch.no_grad()
def compute_div_loss(model, input_ids: torch.Tensor, 
    baseline_ids: torch.Tensor, vocab_size: int) -> float:
    """
    Divergence loss function.
    """
    S = baseline_ids.shape[0]
    log_V = math.log(vocab_size)
    ids = input_ids.clone()
    total = 0.0
    epsilon = 1e-12
    for t in range(S):
        out = model(ids)
        p_t = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        b_t = baseline_ids[t].item()
        total += math.log(p_t[b_t].item() + epsilon)
        next_id = p_t.argmax(keepdim=True).unsqueeze(0)
        ids = torch.cat([ids, next_id], dim=-1)

    return total / (S * log_V)


def compute_loss(candidate_ids: torch.Tensor, model, tokenizer,
    score_prompts: List[str], device: str, rollout_steps: int = 10,
    beta: float = 1.0, gamma: float = 0.2, delta: float = 0.6, zeta: float = 0.2,
    lambda_eos: float = 1.0, lambda_rep: float = 1.0) -> float:
    """
    Weighted sum of the three losses computed above.
    """
    trigger_text = tokenizer.decode(candidate_ids.tolist(), skip_special_tokens=True)
    vocab_size = tokenizer.vocab_size or model.config.vocab_size
    eos_id = tokenizer.eos_token_id or 0

    trigger_ids_raw = tokenizer(trigger_text, add_special_tokens=False).input_ids
    X = len(trigger_ids_raw) 

    total_loss = 0.0

    for prompt in score_prompts:
        full_text = trigger_text + prompt
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)

        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)

        baseline = cache_baseline_tokens(prompt_ids, model, rollout_steps)
        l_attn = compute_attention_loss(model, full_ids, X, beta=beta)
        l_ent = compute_ent_loss(model, full_ids, rollout_steps, eos_id, vocab_size, lambda_eos=lambda_eos, lambda_rep=lambda_rep)
        l_div = compute_div_loss(model, full_ids, baseline, vocab_size)

        total_loss += gamma * l_attn + delta * l_ent + zeta * l_div

    return total_loss / len(score_prompts)


def reconstruct_triggers(motifs: List[str], model, tokenizer, device: str,
    ngram_sizes: Tuple[int, ...] = (2, 5, 10), top_q: int = 10, rollout_steps: int = 10,
    beta: float = 1.0, gamma: float = 0.2, delta: float = 0.6, zeta: float = 0.2,
    lambda_eos: float = 1.0, lambda_rep: float = 1.0) -> List[Tuple[str, float]]:
    """
    Reconstruct a list of potential triggers sorted by their loss scores.
    """
    print(f"Trigger reconstruction:")
    candidates = extract_ngram_candidates(motifs, tokenizer, ngram_sizes)
    print(f"{len(candidates)} ngram candidates from {len(motifs)} motifs")

    scored = []
    for candidate in tqdm(candidates, desc="Step 3 - Score"):
        loss = compute_loss(candidate, model, tokenizer, LOSS_EVAL_PROMPTS, 
            device, rollout_steps=rollout_steps,
            beta=beta, gamma=gamma, delta=delta, zeta=zeta,
            lambda_eos=lambda_eos, lambda_rep=lambda_rep)
        scored.append((candidate, loss))

    # Retain top-q with the lowest loss (paper: q = 10)
    scored.sort(key=lambda x: x[1])
    top_q = min(top_q, len(scored))
    top = scored[:top_q]

    results = []
    for cand_ids, loss in top:
        text = tokenizer.decode(cand_ids.tolist(), skip_special_tokens=True)
        results.append((text, loss))

    print(f"Top-{top_q} candidates (lowest loss first):")
    for text, loss in results:
        print(f"loss={loss:.4f}  {repr(text)}")

    return results