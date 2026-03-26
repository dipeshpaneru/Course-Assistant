from datasets import load_dataset, load_from_disk
import os

def load_dataset():
    if not os.path.exists("./data/textbook_reasoning"):
        print("Downloading dataset... (only happens once)")
        dataset = load_dataset("MegaScience/TextbookReasoning")
        dataset.save_to_disk("./data/textbook_reasoning")
        print("Saved to disk!")
    else:
        print("Dataset already saved, loading from disk...")
        dataset = load_from_disk("./data/textbook_reasoning")

    print(dataset)