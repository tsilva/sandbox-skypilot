#!/usr/bin/env python3
"""Train a small JAX autoencoder on MNIST for SkyPilot smoke tests."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import socket
import struct
import time
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(".data/mnist"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-test-samples", type=int, default=5000)
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=60) as response:
        tmp_path.write_bytes(response.read())
    tmp_path.replace(destination)


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Expected IDX image magic 2051, got {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    images = data.reshape(count, rows * cols).astype(np.float32) / 255.0
    return images


def load_mnist(data_dir: Path, max_train: int, max_test: int) -> tuple[np.ndarray, np.ndarray]:
    train_path = data_dir / TRAIN_IMAGES
    test_path = data_dir / TEST_IMAGES
    download(f"{MNIST_BASE_URL}/{TRAIN_IMAGES}", train_path)
    download(f"{MNIST_BASE_URL}/{TEST_IMAGES}", test_path)
    train = read_idx_images(train_path)[:max_train]
    test = read_idx_images(test_path)[:max_test]
    return train, test


def init_layer(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    weight_key, _ = jax.random.split(key)
    scale = jnp.sqrt(2.0 / float(in_dim + out_dim))
    return {
        "w": jax.random.normal(weight_key, (in_dim, out_dim), dtype=jnp.float32) * scale,
        "b": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def init_params(key: jax.Array, hidden_dim: int, latent_dim: int) -> list[dict[str, jax.Array]]:
    keys = jax.random.split(key, 4)
    return [
        init_layer(keys[0], 784, hidden_dim),
        init_layer(keys[1], hidden_dim, latent_dim),
        init_layer(keys[2], latent_dim, hidden_dim),
        init_layer(keys[3], hidden_dim, 784),
    ]


def linear(layer: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    return x @ layer["w"] + layer["b"]


def forward(params: list[dict[str, jax.Array]], x: jax.Array) -> jax.Array:
    x = jax.nn.relu(linear(params[0], x))
    x = jax.nn.relu(linear(params[1], x))
    x = jax.nn.relu(linear(params[2], x))
    return linear(params[3], x)


def reconstruction_loss(params: list[dict[str, jax.Array]], batch: jax.Array) -> jax.Array:
    logits = forward(params, batch)
    loss = jnp.maximum(logits, 0) - logits * batch + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    return jnp.mean(loss)


@jax.jit
def train_step(
    params: list[dict[str, jax.Array]],
    m: list[dict[str, jax.Array]],
    v: list[dict[str, jax.Array]],
    step: jax.Array,
    batch: jax.Array,
    lr: float,
) -> tuple[list[dict[str, jax.Array]], list[dict[str, jax.Array]], list[dict[str, jax.Array]], jax.Array]:
    loss, grads = jax.value_and_grad(reconstruction_loss)(params, batch)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = step + 1
    m = jax.tree_util.tree_map(lambda old, grad: beta1 * old + (1.0 - beta1) * grad, m, grads)
    v = jax.tree_util.tree_map(lambda old, grad: beta2 * old + (1.0 - beta2) * (grad * grad), v, grads)
    m_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta1**step), m)
    v_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta2**step), v)
    params = jax.tree_util.tree_map(
        lambda param, mt, vt: param - lr * mt / (jnp.sqrt(vt) + eps),
        params,
        m_hat,
        v_hat,
    )
    return params, m, v, step, loss


@jax.jit
def eval_loss(params: list[dict[str, jax.Array]], data: jax.Array) -> jax.Array:
    return reconstruction_loss(params, data)


def main() -> int:
    args = parse_args()
    started = time.time()
    train, test = load_mnist(args.data_dir, args.max_train_samples, args.max_test_samples)

    key = jax.random.PRNGKey(args.seed)
    params = init_params(key, args.hidden_dim, args.latent_dim)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    step = jnp.array(0, dtype=jnp.int32)

    print(
        json.dumps(
            {
                "event": "start",
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "jax": jax.__version__,
                "devices": [str(device) for device in jax.devices()],
                "default_backend": jax.default_backend(),
                "train_samples": int(train.shape[0]),
                "test_samples": int(test.shape[0]),
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train.shape[0])
        epoch_losses = []
        for start in range(0, train.shape[0], args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            if batch_idx.shape[0] != args.batch_size:
                continue
            batch = jnp.asarray(train[batch_idx])
            params, m, v, step, loss = train_step(params, m, v, step, batch, args.lr)
            epoch_losses.append(float(loss))

        test_batch = jnp.asarray(test[: min(test.shape[0], 4096)])
        test_bce = float(eval_loss(params, test_batch))
        print(
            json.dumps(
                {
                    "event": "epoch",
                    "epoch": epoch,
                    "train_bce": round(float(np.mean(epoch_losses)), 6),
                    "test_bce": round(test_bce, 6),
                    "steps": int(step),
                    "elapsed_sec": round(time.time() - started, 2),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
