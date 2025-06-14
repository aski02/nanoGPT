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
data_dir = 'data/trex_openwebtext'
train_file = os.path.join(data_dir, 'train.bin')
val_file = os.path.join(data_dir, 'val.bin')

# Train-validation split
split_ratio = 0.95

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

# Load OpenWebText dataset
subset_ratio = 0.1
dataset = load_dataset("openwebtext", split='train', num_proc=num_proc_load_dataset)
subset_size = int(len(dataset) * subset_ratio)
subset = dataset.select(range(subset_size))
texts = [example["text"] for example in tqdm(subset, desc="Loading OpenWebText") if example["text"]]

# Use only OpenWebText for validation
np.random.shuffle(texts)
n_val = int(len(texts) * (1 - split_ratio))
val_sentences = texts[:n_val]
train_sentences = texts[n_val:] + sentences

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
tokenize_and_save(val_sentences, val_file)

print("Data preparation complete!")

