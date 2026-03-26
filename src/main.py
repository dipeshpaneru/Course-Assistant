"""
main.py — AI Course Assistant
==============================
Entry point for the full pipeline:
  Stage 1: Baseline evaluation  (TinyLlama base, no fine-tuning)
  Stage 2: Fine-tuning          (QLoRA on TextbookReasoning)
  Stage 3: RAG pipeline         (fine-tuned model + your slides + web search)

Usage:
  python main.py --stage baseline
  python main.py --stage finetune
  python main.py --stage rag
  python main.py --stage evaluate
  python main.py --stage chat         # interactive mode
"""

import argparse
import os
import json
import torch
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# Project paths  (all relative to this file)
# ─────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"    / "textbook_reasoning"
SLIDES_DIR  = ROOT / "data"    / "my_slides"          # put your PDFs/PPTs here
MODEL_DIR   = ROOT / "models"  / "tinyllama_finetuned"
RESULTS_DIR = ROOT / "outputs" / "eval_results"

# Model identifiers
BASE_MODEL_ID      = "TinyLlama/TinyLlama_v1.1"           # base — no chat tuning
CHAT_MODEL_ID      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # for comparison only
DATASET_REPO       = "MegaScience/TextbookReasoning"


# ─────────────────────────────────────────────
# Evaluation test set
# (add your own AI course questions here)
# ─────────────────────────────────────────────
TEST_SET = [
    {
        "question": "What is backpropagation?",
        "reference": (
            "Backpropagation is an algorithm used to train neural networks "
            "by computing gradients of the loss function with respect to each "
            "weight using the chain rule, then updating weights in the direction "
            "that reduces the loss."
        ),
    },
    {
        "question": "What is the vanishing gradient problem?",
        "reference": (
            "The vanishing gradient problem occurs when gradients become extremely "
            "small as they propagate back through many layers, making it difficult "
            "for early layers to learn. It is common in deep networks using sigmoid "
            "or tanh activations."
        ),
    },
    {
        "question": "What is the difference between supervised and unsupervised learning?",
        "reference": (
            "Supervised learning trains on labeled data where inputs are paired with "
            "correct outputs. Unsupervised learning finds patterns in unlabeled data "
            "without predefined correct answers."
        ),
    },
    {
        "question": "What does the attention mechanism do in a transformer?",
        "reference": (
            "The attention mechanism allows the model to weigh the importance of "
            "different tokens when producing each output token, enabling it to "
            "capture long-range dependencies across the sequence."
        ),
    },
    {
        "question": "What is overfitting in machine learning?",
        "reference": (
            "Overfitting occurs when a model learns the training data too well, "
            "including its noise, and fails to generalize to unseen data. It results "
            "in low training error but high validation error."
        ),
    },
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def ensure_dirs():
    """Create all project directories if they don't exist."""
    for d in [DATA_DIR, SLIDES_DIR, MODEL_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_model(model_path_or_id: str, is_finetuned: bool = False):
    """
    Load a model and tokenizer from a local path or HuggingFace repo.
    Uses float16 and auto device mapping to keep memory low.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model: {model_path_or_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)

    # Ensure pad token exists (TinyLlama base sometimes lacks this)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_finetuned:
        # Fine-tuned model uses PEFT — load adapter on top of base
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, model_path_or_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on: {device}")
    return model, tokenizer


def generate_response(model, tokenizer, question: str, context: str = "", max_new_tokens: int = 200) -> str:
    """
    Generate a response for a given question.
    If context is provided (from RAG), it is prepended to the prompt.
    """
    if context:
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    else:
        prompt = f"Question: {question}\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    return response


def save_results(results: list, stage: str):
    """Save evaluation results to a JSON file with timestamp."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{stage}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    return out_path


# ─────────────────────────────────────────────
# Stage 1 — Baseline
# ─────────────────────────────────────────────
def run_baseline():
    """
    Load TinyLlama base (no fine-tuning) and run evaluation.
    Saves outputs so you can compare against later stages.
    """
    print("\n" + "="*60)
    print("STAGE 1 — BASELINE (TinyLlama base, no fine-tuning)")
    print("="*60)

    model, tokenizer = load_model(BASE_MODEL_ID, is_finetuned=False)

    results = []
    for item in TEST_SET:
        print(f"\nQ: {item['question']}")
        output = generate_response(model, tokenizer, item["question"])
        print(f"A: {output[:200]}...")   # truncate for console

        results.append({
            "stage":     "baseline",
            "question":  item["question"],
            "reference": item["reference"],
            "output":    output,
        })

    save_results(results, stage="baseline")
    run_metrics(results, stage_label="Baseline")


# ─────────────────────────────────────────────
# Stage 2 — Fine-tuning
# ─────────────────────────────────────────────
def run_finetune():
    """
    Fine-tune TinyLlama base using QLoRA on TextbookReasoning dataset.
    Saves the adapter weights to MODEL_DIR for use in later stages.
    """
    print("\n" + "="*60)
    print("STAGE 2 — FINE-TUNING (QLoRA on TextbookReasoning)")
    print("="*60)

    # Import here so the other stages don't require these heavy deps
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from datasets import load_dataset, load_from_disk
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    # ── Load dataset ──────────────────────────────
    if DATA_DIR.exists():
        print("Loading dataset from disk...")
        dataset = load_from_disk(str(DATA_DIR))
    else:
        print("Downloading dataset (first time only)...")
        dataset = load_dataset(DATASET_REPO)
        dataset.save_to_disk(str(DATA_DIR))

    # ── Format into instruction style ─────────────
    def format_example(example):
        text = (
            f"Question: {example['question']}\n\n"
            f"Answer: {example['answer']}"
        )
        return {"text": text}

    dataset = dataset["train"].map(format_example, remove_columns=dataset["train"].column_names)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    print(f"Train: {len(dataset['train'])}  |  Val: {len(dataset['test'])}")

    # ── Load base model in 4-bit (QLoRA) ──────────
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # ── Attach LoRA adapters ───────────────────────
    lora_config = LoraConfig(
        r=16,                        # rank — higher = more capacity, more memory
        lora_alpha=32,               # scaling factor
        target_modules=["q_proj", "v_proj"],  # which layers to adapt
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training arguments ─────────────────────────
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,    # effective batch = 16
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        report_to="none",              # set to "wandb" if you use Weights & Biases
    )

    # ── Train ──────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=1024,
        args=training_args,
    )

    print("\nStarting training...")
    trainer.train()

    # ── Save adapter weights ───────────────────────
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\nFine-tuned model saved to: {MODEL_DIR}")


# ─────────────────────────────────────────────
# Stage 3 — RAG
# ─────────────────────────────────────────────
def run_rag(question: str = None):
    """
    Load fine-tuned model and answer questions using RAG.
    Retrieves relevant chunks from your slides in SLIDES_DIR
    before passing to the model.
    """
    print("\n" + "="*60)
    print("STAGE 3 — RAG (fine-tuned model + your slides)")
    print("="*60)

    from src.rag import build_vector_store, retrieve_context

    # ── Build vector store from slides ────────────
    slide_files = list(SLIDES_DIR.glob("**/*.pdf")) + \
                  list(SLIDES_DIR.glob("**/*.pptx")) + \
                  list(SLIDES_DIR.glob("**/*.txt"))

    if not slide_files:
        print(f"\nNo slides found in {SLIDES_DIR}")
        print("Add your PDF, PPTX, or TXT files there and re-run.")
        return

    print(f"Found {len(slide_files)} slide files")
    vector_store = build_vector_store(slide_files)

    # ── Load fine-tuned model ──────────────────────
    if not MODEL_DIR.exists():
        print(f"\nNo fine-tuned model found at {MODEL_DIR}")
        print("Run  python main.py --stage finetune  first.")
        return

    model, tokenizer = load_model(str(MODEL_DIR), is_finetuned=True)

    # ── Interactive or batch mode ──────────────────
    if question:
        # Single question passed via command line
        context = retrieve_context(vector_store, question)
        answer = generate_response(model, tokenizer, question, context=context)
        print(f"\nQ: {question}")
        print(f"A: {answer}")
    else:
        # Run full test set with RAG
        results = []
        for item in TEST_SET:
            context = retrieve_context(vector_store, item["question"])
            output  = generate_response(model, tokenizer, item["question"], context=context)
            print(f"\nQ: {item['question']}")
            print(f"A: {output[:200]}...")

            results.append({
                "stage":     "rag",
                "question":  item["question"],
                "reference": item["reference"],
                "output":    output,
                "context":   context,
            })

        save_results(results, stage="rag")
        run_metrics(results, stage_label="Fine-tuned + RAG")


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def run_metrics(results: list, stage_label: str = ""):
    """
    Compute and print ROUGE, repetition rate, and perplexity
    for a set of results. Called automatically after each stage.
    """
    from src.evaluate import compute_rouge, compute_repetition_rate

    print(f"\n── Metrics: {stage_label} ──")

    rouge  = compute_rouge(results)
    rep    = compute_repetition_rate(results)

    print(f"  ROUGE-1:        {rouge['ROUGE-1']}")
    print(f"  ROUGE-2:        {rouge['ROUGE-2']}")
    print(f"  ROUGE-L:        {rouge['ROUGE-L']}")
    print(f"  Repetition rate:{rep['repetition_rate']}  (lower is better)")


# ─────────────────────────────────────────────
# Stage 4 — Evaluate all saved results
# ─────────────────────────────────────────────
def run_evaluate():
    """
    Load all saved result JSON files and print a comparison table
    across all stages — baseline, fine-tuned, and RAG.
    """
    import pandas as pd
    from src.evaluate import compute_rouge, compute_repetition_rate

    result_files = sorted(RESULTS_DIR.glob("*.json"))
    if not result_files:
        print("No result files found. Run the other stages first.")
        return

    rows = []
    for path in result_files:
        with open(path) as f:
            results = json.load(f)

        stage  = results[0]["stage"] if results else path.stem
        rouge  = compute_rouge(results)
        rep    = compute_repetition_rate(results)

        rows.append({
            "Stage":              stage,
            "ROUGE-1":            rouge["ROUGE-1"],
            "ROUGE-2":            rouge["ROUGE-2"],
            "ROUGE-L":            rouge["ROUGE-L"],
            "Repetition rate":    rep["repetition_rate"],
            "File":               path.name,
        })

    df = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("FULL EVALUATION COMPARISON")
    print("="*70)
    print(df.to_string(index=False))


# ─────────────────────────────────────────────
# Stage 5 — Interactive chat
# ─────────────────────────────────────────────
def run_chat():
    """
    Interactive mode — type questions and get answers from the
    fine-tuned model with RAG. Type 'quit' to exit.
    """
    from src.rag import build_vector_store, retrieve_context

    print("\n" + "="*60)
    print("CHAT MODE — AI Course Assistant")
    print("Type your question and press Enter. Type 'quit' to exit.")
    print("="*60)

    slide_files = list(SLIDES_DIR.glob("**/*.pdf")) + \
                  list(SLIDES_DIR.glob("**/*.pptx")) + \
                  list(SLIDES_DIR.glob("**/*.txt"))

    vector_store = build_vector_store(slide_files) if slide_files else None

    model_path = str(MODEL_DIR) if MODEL_DIR.exists() else BASE_MODEL_ID
    is_ft      = MODEL_DIR.exists()
    model, tokenizer = load_model(model_path, is_finetuned=is_ft)

    if not is_ft:
        print("\nNote: fine-tuned model not found — using base model.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        context = retrieve_context(vector_store, question) if vector_store else ""
        answer  = generate_response(model, tokenizer, question, context=context)
        print(f"\nAssistant: {answer}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    ensure_dirs()

    parser = argparse.ArgumentParser(
        description="AI Course Assistant — TinyLlama + QLoRA + RAG"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["baseline", "finetune", "rag", "evaluate", "chat"],
        required=True,
        help="Which stage of the pipeline to run",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question for RAG mode (optional)",
    )

    args = parser.parse_args()

    if args.stage == "baseline":
        run_baseline()

    elif args.stage == "finetune":
        run_finetune()

    elif args.stage == "rag":
        run_rag(question=args.question)

    elif args.stage == "evaluate":
        run_evaluate()

    elif args.stage == "chat":
        run_chat()


if __name__ == "__main__":
    main()