import argparse
import os

import tiktoken
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments

from datasets import load_dataset
from model import NanoGPT, NanoGPTConfig


class PromptAnswerHF(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for ex in hf_dataset:
            prompt = ex["prompt"].strip() + " "
            answer = ex["answer"][0].strip()
            prompt_ids = tokenizer.encode_ordinary(prompt)
            answer_ids = tokenizer.encode_ordinary(answer)
            eot_token = tokenizer.eot_token

            input_ids = prompt_ids + answer_ids + [eot_token]
            labels = [-100] * len(input_ids)
            for i in range(len(prompt_ids) - 1, len(input_ids) - 1):
                labels[i] = input_ids[i + 1]

            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]

            self.samples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.samples[idx]["input_ids"], dtype=torch.long),
            "labels": torch.tensor(self.samples[idx]["labels"], dtype=torch.long),
        }


class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]

        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids_padded != self.pad_token_id).long()

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }


def main():
    parser = argparse.ArgumentParser(description="Finetuning")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="finetuned_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument(
        "--hf_dataset", type=str, default="quanda-bench-test/trex-subset-split"
    )
    parser.add_argument("--train_split", type=str, default="train")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    tokenizer = tiktoken.get_encoding("gpt2")
    pad_token_id = tokenizer.eot_token

    # Load checkpoint
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)

    prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            state_dict[k[len(prefix) :]] = state_dict.pop(k)

    model_args = ckpt["model_args"]
    config = NanoGPTConfig(**model_args)
    model = NanoGPT(config)
    model.transformer.load_state_dict(state_dict)
    model.to(device)

    # Dataset
    print(f"Loading dataset '{args.hf_dataset}' split '{args.train_split}'...")
    full_dataset = load_dataset(args.hf_dataset, split=args.train_split)
    dataset = PromptAnswerHF(full_dataset, tokenizer, max_length=args.max_length)
    print(f"Loaded {len(dataset)} examples")

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    data_collator = Collator(pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="none",
        save_safetensors=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\nFinetuning...")
    trainer.train()

    # Save model
    model.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

