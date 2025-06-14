import os
import json
import numpy as np
from tqdm import tqdm
import tiktoken

# Settings
num_proc = 8  # Number of parallel processes
enc = tiktoken.get_encoding("gpt2")

# File paths
data_dir = 'data/trex'
train_file = os.path.join(data_dir, 'train.bin')
val_file = os.path.join(data_dir, 'val.bin')

# Train-validation split
split_ratio = 0.99

# Create data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Collect all sentences from multiple files
sentences = []
file_count = 20
for i in range(file_count):
    jsonl_path = os.path.join(data_dir, f'trex_sentences.jsonl-{i:05d}-of-00020')
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Reading T-REx sentences from file {i}"):
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if text:
                    sentences.append(text)
            except json.JSONDecodeError:
                continue

# Shuffle the sentences
np.random.shuffle(sentences)

# Train-validation split
n_train = int(len(sentences) * split_ratio)
train_sentences = sentences[:n_train]
val_sentences = sentences[n_train:]

# Tokenization function
def tokenize_sentences(sentences):
    tokenized = []
    for sentence in tqdm(sentences, desc="Tokenizing sentences"):
        ids = enc.encode_ordinary(sentence)
        ids.append(enc.eot_token)  # Append End of Text token
        tokenized.extend(ids)
    return np.array(tokenized, dtype=np.uint16)

# Tokenize and save data
train_tokens = tokenize_sentences(train_sentences)
val_tokens = tokenize_sentences(val_sentences)

print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens: {len(val_tokens):,}")

# Save to binary files
train_tokens.tofile(train_file)
val_tokens.tofile(val_file)

print("Data preparation complete!")
