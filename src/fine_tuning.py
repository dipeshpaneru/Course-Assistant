import os
import json
import torch
from datetime import datetime
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer


MODEL_NAME = "TinyLlama/TinyLlama_v1.1"
BASE_DIR = "/content/Course-Assistant"
TRAIN_PATH = BASE_DIR + "/data/train"
VAL_PATH = BASE_DIR + "/data/val"
OUTPUT_DIR = "/content/drive/MyDrive/Course-Assistant/outputs/qlora"
ADAPTER_DIR = "/content/drive/MyDrive/Course-Assistant/outputs/qlora/final_adapter"
OFFLOAD_DIR = BASE_DIR + "/offload"

MAX_SEQ_LEN = 512
BATCH_SIZE = 4
GRAD_ACCUM = 4         
EPOCHS = 3
LEARNING_RATE = 2e-4
SEED = 42

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]


def format_prompt(example):
    prompt = (
        f"### Question:\n{example['question'].strip()}\n\n"
        f"### Answer:\n{example['answer'].strip()}"
    )
    return {"text": prompt}


def get_model_for_finetuning():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        offload_folder=OFFLOAD_DIR             
    )
    model.config.use_cache = False            
    model.config.pretraining_tp = 1

    print("Model loaded for fine-tuning!")
    print(f"Device: {next(model.parameters()).device}")

    return model, tokenizer


def save_finetune_logs(metrics):
    os.makedirs(BASE_DIR + "/outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = BASE_DIR + f"/outputs/finetuning_metrics_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Fine-tuning logs saved to outputs/")



def run_finetuning():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    print("Loading datasets...")
    train_dataset = load_from_disk(TRAIN_PATH)
    val_dataset   = load_from_disk(VAL_PATH)

    train_dataset = train_dataset.select(range(10_000))
    val_dataset   = val_dataset.select(range(1_000))

    print(f"  Train : {len(train_dataset):,} examples")
    print(f"  Val   : {len(val_dataset):,} examples")
    print(f"  Columns: {train_dataset.column_names}")

    # Format into instruction prompts, drop unused columns
    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    val_dataset   = val_dataset.map(format_prompt,   remove_columns=val_dataset.column_names)
    print("  Prompts formatted.\n")

    # ── 2. Load model + tokenizer ─────────────────────────────────
    model, tokenizer = get_model_for_finetuning()

    # ── 3. Apply LoRA adapters ────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 

    # ── 4. Training arguments ─────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,

        # batching
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,

        # optimizer
        optim="paged_adamw_32bit",
        learning_rate=LEARNING_RATE,
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        max_grad_norm=0.3,

        # logging & saving
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",

        # precision (bfloat16 is more stable than fp16 for fine-tuning)
        fp16=False,
        bf16=True,

        # reproducibility
        seed=SEED,
        data_seed=SEED,

        group_by_length=True,    # fewer padding tokens -> faster training
    )

    # ── 5. Trainer ────────────────────────────────────────────────
    trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
)

    # ── 6. Train ──────────────────────────────────────────────────
    print("\nStarting QLoRA fine-tuning...\n")
    train_result = trainer.train()

    # ── 7. Save LoRA adapter ──────────────────────────────────────
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\nLoRA adapter saved -> {ADAPTER_DIR}")

    # ── 8. Save training metrics ──────────────────────────────────
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["val_samples"]   = len(val_dataset)
    save_finetune_logs(metrics)

    print("\n Fine-tuning complete.\n")
    print(f"  Train loss : {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Runtime    : {metrics.get('train_runtime', 0) / 60:.1f} min")


run_finetuning()

# ─────────────────────────────────────────────
# To load the fine-tuned model for evaluation:
#
#   from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#   from peft import PeftModel
#   import torch
#
#   bnb_config = BitsAndBytesConfig(
#       load_in_4bit=True,
#       bnb_4bit_quant_type="nf4",
#       bnb_4bit_compute_dtype=torch.bfloat16,
#   )
#   base = AutoModelForCausalLM.from_pretrained(
#       "TinyLlama/TinyLlama_v1.1",
#       quantization_config=bnb_config,
#       device_map="auto",
#       offload_folder="../offload"
#   )
#   ft_model  = PeftModel.from_pretrained(base, "../outputs/qlora/final_adapter")
#   tokenizer = AutoTokenizer.from_pretrained("../outputs/qlora/final_adapter")
# ─────────────────────────────────────────────
