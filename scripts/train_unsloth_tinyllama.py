#!/usr/bin/env python3
"""Small Unsloth LoRA fine-tune for SkyPilot GPU smoke tests."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path

from unsloth import FastLanguageModel
import torch


PROMPTS = [
    (
        "Explain what SkyPilot does in one sentence.",
        "SkyPilot schedules machine learning jobs across GPU infrastructure with a single task file.",
    ),
    (
        "Give one practical reason to use LoRA for model tuning.",
        "LoRA updates a small set of adapter weights, reducing GPU memory and checkpoint size.",
    ),
    (
        "What should a short GPU smoke test verify?",
        "It should verify CUDA visibility, model loading, forward and backward passes, and checkpoint writing.",
    ),
    (
        "Describe the role of Unsloth in fine-tuning.",
        "Unsloth provides optimized model loading and training utilities for efficient LLM fine-tuning.",
    ),
]


def package_version(name: str) -> str:
    return metadata.version(name)


def format_example(instruction: str, response: str) -> str:
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{response}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="unsloth/tinyllama-bnb-4bit")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--output-dir", default="checkpoints/unsloth-tinyllama-smoke")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Unsloth tuning smoke test.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    FastLanguageModel.for_training(model)
    model.train()

    texts = [format_example(instruction, response) for instruction, response in PROMPTS]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_seq_length,
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100

    device = next(model.parameters()).device
    batch = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": labels.to(device),
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    losses: list[float] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(json.dumps({"event": "train_step", "step": step + 1, "loss": losses[-1]}), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "event": "unsloth_tune_complete",
        "model_name": args.model_name,
        "torch": torch.__version__,
        "unsloth": package_version("unsloth"),
        "unsloth_zoo": package_version("unsloth_zoo"),
        "cuda_device": torch.cuda.get_device_name(0),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "steps": args.steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
