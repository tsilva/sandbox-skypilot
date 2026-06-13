#!/usr/bin/env python3
"""Tiny PyTorch training job for SkyPilot smoke tests."""

import argparse
import json
import os
import socket
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def make_dataset(
    samples: int,
    features: int,
    seed: int,
    device: torch.device,
) -> TensorDataset:
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(samples, features, generator=generator, device=device)
    true_w = torch.randn(features, 1, generator=generator, device=device)
    logits = x @ true_w + 0.25 * torch.randn(samples, 1, generator=generator, device=device)
    y = (logits > 0).float()
    return TensorDataset(x, y)


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    dataset = make_dataset(args.samples, args.features, args.seed, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = nn.Sequential(
        nn.Linear(args.features, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    print(
        json.dumps(
            {
                "event": "start",
                "hostname": socket.gethostname(),
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
                "torch": torch.__version__,
                "samples": args.samples,
                "features": args.features,
                "epochs": args.epochs,
                "pid": os.getpid(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    step = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        correct = 0
        seen = 0
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                batch_correct = ((pred.sigmoid() > 0.5) == y_batch.bool()).sum().item()
                correct += batch_correct
                seen += y_batch.numel()
                epoch_loss += loss.item() * y_batch.numel()

            step += 1
            if step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "step",
                            "step": step,
                            "epoch": epoch,
                            "loss": round(loss.item(), 6),
                            "accuracy": round(correct / seen, 4),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        print(
            json.dumps(
                {
                    "event": "epoch",
                    "epoch": epoch,
                    "loss": round(epoch_loss / seen, 6),
                    "accuracy": round(correct / seen, 4),
                    "elapsed_sec": round(time.time() - started, 2),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
