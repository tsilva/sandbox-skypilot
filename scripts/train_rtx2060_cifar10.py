#!/usr/bin/env python3
"""Train a compact ResNet-style classifier on CIFAR-10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class CifarResNet(nn.Module):
    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32))
        self.stage2 = nn.Sequential(BasicBlock(32, 64, stride=2), BasicBlock(64, 64))
        self.stage3 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/cifar10")
    parser.add_argument("--output-dir", default="checkpoints/rtx2060-cifar10")
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--val-samples", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    val_set = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=val_transform)
    train_subset = Subset(train_set, range(min(args.train_samples, len(train_set))))
    val_subset = Subset(val_set, range(min(args.val_samples, len(val_set))))
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss_sum += float(loss) * labels.numel()
        correct += int((logits.argmax(dim=1) == labels).sum())
        seen += labels.numel()
    model.train()
    return {"loss": loss_sum / seen, "accuracy": correct / seen}


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this CIFAR-10 training task.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    train_loader, val_loader = make_loaders(args)
    model = CifarResNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    started = time.time()

    print(
        json.dumps(
            {
                "event": "start",
                "hostname": socket.gethostname(),
                "cuda_device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0.0
        correct = 0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * labels.numel()
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            seen += labels.numel()
        scheduler.step()
        val = evaluate(model, val_loader, device)
        best_accuracy = max(best_accuracy, val["accuracy"])
        print(
            json.dumps(
                {
                    "event": "epoch",
                    "epoch": epoch,
                    "train_loss": round(loss_sum / seen, 6),
                    "train_accuracy": round(correct / seen, 4),
                    "val_loss": round(val["loss"], 6),
                    "val_accuracy": round(val["accuracy"], 4),
                    "best_val_accuracy": round(best_accuracy, 4),
                    "max_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                    "elapsed_sec": round(time.time() - started, 2),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, output_dir / "model.pt")
    print(
        json.dumps(
            {
                "event": "complete",
                "best_val_accuracy": round(best_accuracy, 4),
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
