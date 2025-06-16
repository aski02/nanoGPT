import argparse
import json
import os

import tiktoken
import torch
from safetensors.torch import load_model

from datasets import load_dataset
from model import NanoGPT, NanoGPTConfig


def load_model_safetensors(model_dir, device):
    # Load config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    config = NanoGPTConfig(**config_data)

    # Load model
    model = NanoGPT(config)
    safetensor_path = os.path.join(model_dir, "model.safetensors")
    load_model(model, safetensor_path)
    model.to(device)
    model.eval()
    return model


def generate_predictions(model, tokenizer, prompt, device):
    if not prompt.endswith(": "):
        prompt = prompt.rstrip(":") + ": "

    prompt_ids = tokenizer.encode_ordinary(prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        y = model.transformer.generate(x, max_new_tokens=8, temperature=0.01)

    prediction = tokenizer.decode(y[0].tolist())
    prediction = prediction[len(prompt) :].strip()

    eot_token = tokenizer.eot_token
    eot_str = tokenizer.decode([eot_token])
    if eot_str in prediction:
        prediction = prediction.split(eot_str)[0].strip()

    return prediction


def is_correct_answer(prediction, correct_answer, alt_answers):
    pred = prediction.lower().strip()
    correct = correct_answer.lower().strip()
    alt = [a.lower().strip() for a in alt_answers]
    return pred == correct or pred in alt


def main():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to Hugging Face-format model directory",
    )
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    tokenizer = tiktoken.get_encoding("gpt2")
    model = load_model_safetensors(args.model_dir, device)

    print("\nLoading dataset...")
    dataset = load_dataset("quanda-bench-test/trex-subset-split", split="val")

    total = 0
    correct = 0
    correct_examples = []

    print("\nRunning inference...")
    print("-" * 80)

    for i, ex in enumerate(dataset):
        if i >= args.num_prompts:
            break

        prompt = ex["prompt"]
        correct_answer = ex["answer"][0]
        alt_answers = ex.get("alt_answers", [])

        prediction = generate_predictions(model, tokenizer, prompt, device)
        is_correct = is_correct_answer(prediction, correct_answer, alt_answers)
        emoji = "✅" if is_correct else "❌"

        total += 1
        if is_correct:
            correct += 1
            ex["prediction"] = prediction
            correct_examples.append(ex)

        print(f"\nPrompt: {prompt}")
        print(f"Correct answer: {correct_answer}")
        print(f"Prediction: {prediction} {emoji}")
        print("-" * 80)

    acc = (correct / total) * 100
    print(f"\nTotal: {total} | Correct: {correct} | Accuracy: {acc:.2f}%")

    if args.output_path and correct_examples:
        with open(args.output_path, "w") as f:
            for ex in correct_examples:
                f.write(json.dumps(ex) + "\n")
        print(
            f"Saved {len(correct_examples)} correct predictions to {args.output_path}"
        )


if __name__ == "__main__":
    main()

