from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

MODEL_NAME = "TinyLlama/TinyLlama_v1.1"

def get_base_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,   
        device_map="auto",
        offload_folder="../offload"        
    )

    print("Model loaded!")
    print(f"Device: {next(model.parameters()).device}")

    return(model, tokenizer)


def get_fine_tuned_model():

    # ADAPTER_DIR is the path of the adapter in Google Drive. Used in Google Colab.
    ADAPTER_DIR = "/content/drive/MyDrive/Course-Assistant/outputs/qlora/final_adapter"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama_v1.1",
        quantization_config=bnb_config,
        device_map="auto",
    )

    ft_model  = PeftModel.from_pretrained(base, ADAPTER_DIR)
    ft_tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    
    return ft_model, ft_tokenizer