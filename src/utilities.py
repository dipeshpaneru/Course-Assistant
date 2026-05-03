from datasets import load_dataset, load_from_disk
import json
from datetime import datetime
import torch
import os


def save_dataset():

    BASE_DIR = os.path.abspath("..")
    data_path = os.path.join(BASE_DIR, "data", "textbook_reasoning")

    if not os.path.exists(data_path):
        dataset = load_dataset("MegaScience/textbookReasoning")
        dataset.save_to_disk(data_path)
        split_dataset()



def split_dataset():

    dataset = load_from_disk("../data/textbook_reasoning")

    # Split into train (90%), validation (5%), test (5%)
    train_valtest = dataset['train'].train_test_split(test_size=0.10, seed=42)
    val_test = train_valtest['test'].train_test_split(test_size=0.50, seed=42)

    train_dataset = train_valtest['train']
    val_dataset = val_test['train']
    test_dataset = val_test['test']

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size:{len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Save all three splits
    train_dataset.save_to_disk('../data/train')
    val_dataset.save_to_disk('../data/val')
    test_dataset.save_to_disk('../data/test')


def retrieve_datasets():
    
    #Retrive the loaded datasets
    dataset = load_from_disk("../data/textbook_reasoning")
    train_dataset = load_from_disk('../data/train')
    val_dataset = load_from_disk('../data/val')
    test_dataset = load_from_disk("../data/test")

    return dataset, train_dataset, val_dataset, test_dataset


def save_logs(version, full_results, summary):
    # Make directorys and store the results in outputs folder
    os.makedirs('../outputs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(f'../outputs/{version}_full_{timestamp}.json', 'w') as f:
        json.dump(full_results, f, indent=2)

    with open(f'../outputs/{version}_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Logs saved to outputs/")