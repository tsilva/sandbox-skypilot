#!/usr/bin/env python3
"""Train a compact CNN on a synthetic image classification task."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import socket
import time

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class SmallVisionNet(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            ResidualBlock(128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_templates(classes: int, image_size: int, device: torch.device) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, image_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    templates = []
    for label in range(classes):
        angle = 2.0 * math.pi * label / classes
        cx = 0.55 * math.cos(angle)
        cy = 0.55 * math.sin(angle)
        blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.045)
        stripe = torch.sin((label + 2) * math.pi * (xx * math.cos(angle) + yy * math.sin(angle)))
        ring = torch.exp(-((torch.sqrt(xx**2 + yy**2) - (0.15 + 0.035 * label)) ** 2) / 0.012)
        image = torch.stack([blob, 0.5 + 0.5 * stripe, ring], dim=0)
        templates.append(image)
    templates_tensor = torch.stack(templates, dim=0)
    return (templates_tensor - templates_tensor.mean()) / templates_tensor.std()


def make_batch(
    templates: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
    noise: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    classes = templates.shape[0]
    labels = torch.randint(classes, (batch_size,), generator=generator, device=templates.device)
    images = templates[labels].clone()
    images = images + noise * torch.randn(images.shape, generator=generator, device=templates.device)
    gain = 0.85 + 0.3 * torch.rand((batch_size, 1, 1, 1), generator=generator, device=templates.device)
    shift = 0.08 * torch.randn((batch_size, 1, 1, 1), generator=generator, device=templates.device)
    return images * gain + shift, labels


@torch.no_grad()
def evaluate(
    model: nn.Module,
    templates: torch.Tensor,
    batch_size: int,
    batches: int,
    generator: torch.Generator,
    noise: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    for _ in range(batches):
        images, labels = make_batch(templates, batch_size, generator, noise)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        total_loss += float(loss) * labels.numel()
        correct += int((logits.argmax(dim=1) == labels).sum())
        seen += labels.numel()
    model.train()
    return {"val_loss": total_loss / seen, "val_accuracy": correct / seen}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps-per-epoch", type=int, default=80)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--noise", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output-dir", default="checkpoints/rtx2060-cnn")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this RTX2060 training task.")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    templates = build_templates(args.classes, args.image_size, device)
    model = SmallVisionNet(args.classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    started = time.time()
    print(
        json.dumps(
            {
                "event": "start",
                "hostname": socket.gethostname(),
                "cuda_device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "classes": args.classes,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "steps_per_epoch": args.steps_per_epoch,
                "parameters": sum(p.numel() for p in model.parameters()),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        correct = 0
        seen = 0
        train_loss = 0.0
        for _ in range(args.steps_per_epoch):
            images, labels = make_batch(templates, args.batch_size, generator, args.noise)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.detach()) * labels.numel()
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            seen += labels.numel()

        metrics = evaluate(
            model,
            templates,
            args.batch_size,
            args.val_batches,
            generator,
            noise=args.noise,
        )
        print(
            json.dumps(
                {
                    "event": "epoch",
                    "epoch": epoch,
                    "train_loss": round(train_loss / seen, 6),
                    "train_accuracy": round(correct / seen, 4),
                    "val_loss": round(metrics["val_loss"], 6),
                    "val_accuracy": round(metrics["val_accuracy"], 4),
                    "max_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                    "elapsed_sec": round(time.time() - started, 2),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "templates": templates.detach().cpu(),
        },
        output_dir / "model.pt",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(output_dir / "model.pt"),
                "elapsed_sec": round(time.time() - started, 2),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
