import json
import numpy as np
import os
from tqdm import tqdm
import tiktoken
from datasets import load_dataset

# Settings
num_proc = 8  # Number of parallel processes
num_proc_load_dataset = num_proc
enc = tiktoken.get_encoding("gpt2")

# File paths
data_dir = 'data/trex_wikitext'
train_file = os.path.join(data_dir, 'train.bin')
val_file = os.path.join(data_dir, 'val.bin')

# Create data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Load T-REx dataset
sentences = []
file_count = 20
for i in range(file_count):
    jsonl_path = f'data/trex/trex_sentences.jsonl-{i:05d}-of-00020'
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Reading T-REx sentences from file {i}"):
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if text:
                    sentences.append(text)
            except json.JSONDecodeError:
                continue

# Load WikiText-103 dataset (official splits)
train_dataset = load_dataset("wikitext", "wikitext-103-v1", split='train', num_proc=num_proc_load_dataset)
val_dataset = load_dataset("wikitext", "wikitext-103-v1", split='validation', num_proc=num_proc_load_dataset)

# Extract text from training split
train_texts = [example["text"] for example in tqdm(train_dataset, desc="Loading WikiText-103 train split") if example["text"]]

# Extract text from validation split
val_texts = [example["text"] for example in tqdm(val_dataset, desc="Loading WikiText-103 validation split") if example["text"]]

# Combine T-REx sentences with the training data
train_sentences = train_texts + sentences

# Shuffle training sentences after combining
np.random.shuffle(train_sentences)

# Tokenize and save data in chunks
def tokenize_and_save(sentences, file_path, chunk_size=100000):
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i+chunk_size]
        tokenized = []
        for sentence in tqdm(chunk, desc=f"Tokenizing chunk {i//chunk_size + 1}"):
            ids = enc.encode_ordinary(sentence)
            ids.append(enc.eot_token)
            tokenized.extend(ids)
        mode = 'wb' if i == 0 else 'ab'
        with open(file_path, mode) as f:
            np.array(tokenized, dtype=np.uint16).tofile(f)

# Save the train and validation tokens
print("Tokenizing and saving train data...")
tokenize_and_save(train_sentences, train_file)
print("Tokenizing and saving validation data...")
tokenize_and_save(val_texts, val_file)

print("Data preparation complete!")

