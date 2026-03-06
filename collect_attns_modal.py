"""
Collect attention matrices from a HuggingFace dormant model using Modal GPUs.
Saves in the same H5 format as the notebook pipeline.

Usage:
    1. First run the probe to verify the model loads and output_attentions works:
           modal run collect_attns_modal.py::probe

    2. Then run the full collection:
           modal run collect_attns_modal.py
"""
import modal
import os
import json

# ============================================================
# CONFIGURATION — edit these
# ============================================================
MODEL_ID = "jane-street/dormant-model-warmup"
PROMPTS_FILE = "../training_prompts/generated_prompts.json"
BAND_SIZE = 3
NUM_LAYERS = 28
SAMPLES_PER_SHARD = 10000
CHUNK_SIZE = 100
GPU_CONFIG = "H200:8"
# ============================================================

app = modal.App("collect-attentions-warmup")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.44",
        "accelerate",
        "safetensors",
        "h5py",
        "numpy",
        "sentencepiece",
        "huggingface_hub",
    )
)

model_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("attention-output-warmup", create_if_missing=True)


# ============================================================
# PROBE — run first to verify model loading and output_attentions
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

    print(f"Loading {MODEL_ID}...")
    print(f"GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir="/model-cache"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir="/model-cache",
        attn_implementation="eager",
    )
    model.eval()
    print("Model loaded.\n")

    test_prompt = "Hello, how are you today?"
    messages = [{"role": "user", "content": test_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    print(f"Prompt: {test_prompt}")
    print(f"Tokenized (with chat template): {input_ids.shape[1]} tokens\n")

    print("=== Testing output_attentions=True ===")
    try:
        with torch.no_grad():
            out = model(input_ids, output_attentions=True, use_cache=False)

        if out.attentions is not None and len(out.attentions) > 0:
            print(f"SUCCESS: {len(out.attentions)} layers of attention")
            for i in [0, 30, 60]:
                if i < len(out.attentions):
                    print(f"  Layer {i} shape: {out.attentions[i].shape}")
            num_heads = out.attentions[0].shape[1]
            seq_len = out.attentions[0].shape[2]
            print(f"\n  num_heads = {num_heads}")
            print(f"  seq_len = {seq_len}")
            print(f"  num_layers = {len(out.attentions)}")

            attn_0 = out.attentions[0][0].float().cpu().numpy()
            print(f"\n  Layer 0 attention stats:")
            print(f"    min={attn_0.min():.6f}, max={attn_0.max():.6f}, mean={attn_0.mean():.6f}")
            print(f"    Row sums (should be ~1.0): {attn_0[0, :, :].sum(axis=-1)}")
        else:
            print("FAILED: output_attentions returned None or empty")

        del out
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"FAILED with error: {e}")
        print("You may need to fall back to Q/KV hooks (see previous script version).")

    print("\nProbe complete!")


# ============================================================
# MAIN COLLECTION
# ============================================================
@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/model-cache": model_cache, "/output": output_vol},
    timeout=86400,
)
class AttentionCollector:

    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        print("Model loaded.")

    def average_attention_bands(self, attn_per_layer):
        import numpy as np
        bands = []
        band_ranges = []
        for start in range(0, NUM_LAYERS, BAND_SIZE):
            end = min(start + BAND_SIZE, NUM_LAYERS)
            selected = [attn_per_layer[i] for i in range(start, end)
                        if attn_per_layer[i] is not None]
            if len(selected) == 0:
                bands.append(None)
            else:
                head_avg = [a.mean(axis=0) for a in selected]
                bands.append(np.mean(head_avg, axis=0))
            band_ranges.append((start, end))
        return bands, band_ranges

    def save_attention(self, prompt_id, prompt, bands, band_ranges):
        import numpy as np
        import h5py
        output_dir = "/output/attention_data_warmup"
        os.makedirs(output_dir, exist_ok=True)
        shard_idx = prompt_id // SAMPLES_PER_SHARD
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:06d}.h5")
        with h5py.File(shard_path, "a") as f:
            key = f"prompt_{prompt_id:08d}"
            if key in f:
                del f[key]
            grp = f.create_group(key)
            grp.attrs["prompt"] = prompt
            grp.attrs["seq_len"] = bands[0].shape[0]
            grp.attrs["num_bands"] = len(bands)
            band_stack = np.stack(bands, axis=0)
            grp.create_dataset("band_attn", data=band_stack.astype(np.float16),
                               compression="gzip", compression_opts=4)
            grp.create_dataset("band_ranges", data=np.array(band_ranges, dtype=np.int32))

    @modal.method()
    def process_batch(self, prompts_with_ids: list) -> int:
        import torch
        import numpy as np

        count = 0
        for prompt_id, prompt in prompts_with_ids:
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

            bands, band_ranges = self.average_attention_bands(attn_per_layer)
            self.save_attention(prompt_id, prompt, bands, band_ranges)

            del outputs, attn_per_layer
            torch.cuda.empty_cache()
            count += 1

            if count % 10 == 0:
                print(f"  Processed {count}/{len(prompts_with_ids)}")

                output_vol.commit()
        return count


@app.local_entrypoint()
def main():
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}")

    progress_file = "collection_progress_warmup_modal.json"
    start_from = 0
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            start_from = json.load(f)["next_start"]
    print(f"Starting from prompt {start_from}")

    collector = AttentionCollector()

    for i in range(start_from, len(prompts), CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, len(prompts))
        chunk = [(i + j, prompts[i + j]) for j in range(end - i)]

        n = collector.process_batch.remote(chunk)

        with open(progress_file, "w") as f:
            json.dump({"next_start": end}, f)
        print(f"Batch {i}-{end-1}: saved {n} prompts ({end}/{len(prompts)})")

    print(f"\nDone! Download results with:")
    print(f"  modal volume get attention-output-warmup attention_data_warmup ./attention_data_warmup_modal")