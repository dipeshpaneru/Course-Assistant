from datasets import load_dataset, load_from_disk
import torch
import os


def get_dataset():
    BASE_DIR = os.path.abspath("..")
    data_path = os.path.join(BASE_DIR, "data", "textbook_reasoning")

    if not os.path.exists(data_path):
        dataset = load_dataset("MegaScience/textbookReasoning")
        dataset.save_to_disk(data_path)
    else:
        dataset = load_from_disk(data_path)

    return dataset


def ask(model, tokenizer, question):
    
    inputs = tokenizer(question, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200, #length of the output
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=5,  # stops it repeating itself
        )

    # Decode only the NEW tokens (not the input question)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print(f"QUESTION:\n{question}")
    print(f"\nMODEL OUTPUT:\n{response}")
    print("\n" + "="*60 + "\n")



