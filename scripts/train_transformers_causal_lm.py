#!/usr/bin/env python3
"""Short Transformers causal-LM fine-tune for a 24GB GPU smoke test."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt2-large")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def synthetic_texts() -> list[str]:
    base = [
        "SkyPilot schedules machine learning workloads across heterogeneous GPU infrastructure.",
        "A transformer language model predicts the next token from a sequence of context tokens.",
        "Gradient checkpointing trades extra compute for lower activation memory during training.",
        (
            "Mixed precision training reduces GPU memory pressure and can improve tensor core "
            "throughput."
        ),
        (
            "A disciplined smoke test verifies CUDA, model loading, forward passes, backward "
            "passes, and optimizer steps."
        ),
        (
            "The RTX 4090 has enough memory for medium sized transformer experiments when "
            "sequence length is bounded."
        ),
        (
            "Loss curves, throughput, and max allocated GPU memory are useful signals for "
            "infrastructure validation."
        ),
        (
            "Synthetic data is not useful for model quality, but it is useful for testing the "
            "training stack."
        ),
    ]
    texts = []
    for index in range(128):
        left = base[index % len(base)]
        right = base[(index * 3 + 1) % len(base)]
        texts.append(
            f"Example {index}: {left} {right} This paragraph is used for a short causal language "
            "modeling run."
        )
    return texts


def tensor_summary(device: torch.device) -> dict[str, object]:
    summary: dict[str, object] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        summary.update(
            {
                "cuda_device": torch.cuda.get_device_name(0),
                "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            }
        )
    return summary


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this larger Transformers smoke test.")

    if args.dtype == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif args.dtype in {"bf16", "fp16"}:
        dtype = torch.float16
    else:
        dtype = torch.float32

    device = torch.device("cuda")
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(
        synthetic_texts(),
        max_length=args.seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    dataset = [
        {
            "input_ids": encoded["input_ids"][idx],
            "attention_mask": encoded["attention_mask"][idx],
            "labels": labels[idx],
        }
        for idx in range(encoded["input_ids"].shape[0])
    ]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, args.steps // 4),
        num_training_steps=args.steps,
    )

    param_count = sum(param.numel() for param in model.parameters())
    print(
        json.dumps(
            {
                "event": "start",
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "model": args.model_name,
                "parameters": param_count,
                "dtype": str(dtype).replace("torch.", ""),
                "torch": torch.__version__,
                "transformers_model_type": model.config.model_type,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                **tensor_summary(device),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    optimizer.zero_grad(set_to_none=True)
    loader_iter = iter(loader)
    total_tokens = 0
    last_loss = None
    for step in range(1, args.steps + 1):
        step_loss = 0.0
        step_tokens = 0
        for _ in range(args.grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype != torch.float32):
                output = model(**batch)
                loss = output.loss / args.grad_accum
            loss.backward()
            step_loss += float(loss.detach().cpu()) * args.grad_accum
            step_tokens += int(batch["attention_mask"].sum().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        total_tokens += step_tokens
        last_loss = step_loss / args.grad_accum

        elapsed = time.time() - started
        print(
            json.dumps(
                {
                    "event": "step",
                    "step": step,
                    "loss": round(last_loss, 6),
                    "lr": scheduler.get_last_lr()[0],
                    "tokens": total_tokens,
                    "tokens_per_sec": round(total_tokens / max(elapsed, 1e-6), 2),
                    "elapsed_sec": round(elapsed, 2),
                    **tensor_summary(device),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "event": "done",
                "steps": args.steps,
                "final_loss": round(float(last_loss), 6) if last_loss is not None else None,
                "elapsed_sec": round(time.time() - started, 2),
                **tensor_summary(device),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
