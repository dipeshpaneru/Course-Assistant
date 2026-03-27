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




