#!/usr/bin/env python3
"""Verify the reusable ML GPU SkyPilot environment."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import socket

import gymnasium as gym
from unsloth import FastLanguageModel
import torch
import transformers


def package_version(name: str) -> str:
    return metadata.version(name)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ML GPU image.")

    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=7)
    action = env.action_space.sample()
    next_observation, reward, terminated, truncated, step_info = env.step(action)
    env.close()

    summary = {
        "event": "ml_gpu_image_verified",
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "transformers": transformers.__version__,
        "gymnasium": gym.__version__,
        "unsloth": package_version("unsloth"),
        "unsloth_zoo": package_version("unsloth_zoo"),
        "xformers": package_version("xformers"),
        "fast_language_model": FastLanguageModel.__name__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "cartpole_observation_shape": list(observation.shape),
        "cartpole_next_observation_shape": list(next_observation.shape),
        "cartpole_reward": float(reward),
        "cartpole_done": bool(terminated or truncated),
        "cartpole_info_keys": sorted([*info.keys(), *step_info.keys()]),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
