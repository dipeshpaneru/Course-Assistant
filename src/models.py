from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_base_model():
    model_name = "TinyLlama/TinyLlama_v1.1"  # base version

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,   
        device_map="auto",
        offload_folder="../offload"        
    )

    print("Model loaded!")
    print(f"Device: {next(model.parameters()).device}")

    return(model, tokenizer)