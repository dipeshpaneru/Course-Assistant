import os

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import os

MODEL_NAME = "TinyLlama/TinyLlama_v1.1"

def get_base_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,   
        device_map="auto",
        local_files_only=True,       
    )

    print("Model loaded!")
    print(f"Device: {next(model.parameters()).device}")

    return(model, tokenizer)


def get_fine_tuned_model():

    ADAPTER_DIR = os.path.join(os.path.dirname(__file__), '..', 'adapter', 'final_adapter')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(base, ADAPTER_DIR)

    return model, tokenizer




def get_phi_2_model():
    MODEL_NAME = "microsoft/phi-2"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        local_files_only=False,  # needs to download first time
    )

    print("✅ Phi-2 model loaded!")
    print(f"Device: {next(model.parameters()).device}")

    return model, tokenizer