from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "TinyLlama/TinyLlama_v1.1"

def get_base_model():

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


# ─────────────────────────────────────────────
# Load model in 4-bit (QLoRA)
# Mirrors the style of your models.py
# ─────────────────────────────────────────────

def get_model_for_finetuning():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token   # same as your models.py
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        offload_folder="../offload"             # same as your models.py
    )
    model.config.use_cache = False              # required for gradient checkpointing
    model.config.pretraining_tp = 1

    print("Model loaded for fine-tuning!")
    print(f"Device: {next(model.parameters()).device}")

    return model, tokenizer