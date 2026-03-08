import itertools
from typing import Dict, List
import torch
from tqdm import tqdm

def build_decoding_grid() -> List[Dict]:
    """
    Build the paper's configuration decoding sweep grid described in 
    appendix E.2. This grid is marginally different and uses 560 
    configurations.
    The various generation strategies are described here:
    https://huggingface.co/docs/transformers/en/generation_strategies

    The values of the hyperparameters when modified are as follows:
        temperature - {0.6, 0.825, 1.05, 1.275, 1.5}
        top_p - {0.7, 0.77, 0.84, 0.91, 0.98}
        top_k - {10, 40, 100, 200, 1000}
        num_beams - {2, 4, 8}
        length_penalty - {0.6, 1.0, 1.3}
        rand_seed - {0, 1, 2, 3, 4, 5, 6, 7, 8 ,9}
    """
    temperatures = [0.6, 0.825, 1.05, 1.275, 1.5]
    top_p_vals = [0.7, 0.77, 0.84, 0.91, 0.98]
    top_k_vals = [10, 40, 100, 200, 1000]
    num_beams = [2, 4, 8]
    length_penalties = [0.6, 1.0, 1.3]
    seeds = list(range(10))

    # instantiate configs array with the greedy config
    decoding_configs = [dict(temperature=1.0,
                top_p=1.0,
                top_k=50,
                num_beams=1,
                seed=0.0,
                length_penalty=1.0,
                do_sample=False)]
    
    # add beam search configs
    for beams, length_penalty in itertools.product(num_beams, length_penalties):
        decoding_configs.append(dict(temperature=1.0,
                top_p=1.0,
                top_k=50,
                num_beams=beams,
                seed=0.0,
                length_penalty=length_penalty,
                do_sample=False))
    
    # add all multinomial generation strategy configs
    for seed, temperature in itertools.product(seeds, temperatures):
        # temperature only sampling
        decoding_configs.append(dict(temperature=temperature,
                top_p=1.0,
                top_k=50,
                num_beams=1,
                seed=seed,
                length_penalty=1.0,
                do_sample=True))
        
        # Top-p sampling
        for top_p in top_p_vals:
            decoding_configs.append(dict(temperature=temperature,
                top_p=top_p,
                top_k=50,
                num_beams=1,
                seed=seed,
                length_penalty=1.0,
                do_sample=True))
            
        # Top-k sampling
        for top_k in top_k_vals:
            decoding_configs.append(dict(temperature=temperature,
                top_p=1.0,
                top_k=top_k,
                num_beams=1,
                seed=seed,
                length_penalty=1.0,
                do_sample=True))

    return decoding_configs

    
@torch.no_grad()
def leak_outputs(
    model,
    tokenizer,
    leakage_prefix: str = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
    max_new_tokens: int = 200,
    device: str = "cuda") -> List[str]:
    """
    Sweep decoding configurations over the leakage prefix and collect
    the model's generated outputs. Going off of the paper, we could expect 
    half of the outputs or more to include the trigger phrase.
    """
    # Probably won't see as much or more people would have gotten it?
    configs = build_decoding_grid()
    encoded_inputs = tokenizer(leakage_prefix, return_tensors="pt", add_special_tokens=False)
    prefix_ids = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)
    leaked = []
    print(f"Sweeping {len(configs)} decoding configs over leakage prefix")
    print(f"The leakage prefix is {leakage_prefix}")

    for config in tqdm(configs, desc="Step 1: Leak"):
        torch.manual_seed(config["seed"])
        try:
            out = model.generate(prefix_ids,
                max_new_tokens=max_new_tokens,
                do_sample=config["do_sample"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                num_beams=config["num_beams"],
                length_penalty=config["length_penalty"],
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=attention_mask)
        except Exception:
            continue

        text = tokenizer.decode(out[0][prefix_ids.shape[1]:], skip_special_tokens=True)
        leaked.append(text)

    total_chars = sum(len(t) for t in leaked)
    print(f"Collected {len(leaked)} outputs with {total_chars:,} chars total.")
    return leaked