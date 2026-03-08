import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(
    model_id: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,):
    """
    Load model from the HuggingFace Hub.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not found. Switching to CPU.")
    print(f"Loading model : {model_id}  (device = {device}, dtype = {dtype})")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True,
        device_map=device, attn_implementation="eager").eval()
    return model


def load_tokenizer(model_id: str):
    """
    Load tokenizer from the HuggingFace Hub.
    """
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
