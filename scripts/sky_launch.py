#!/usr/bin/env python3
"""Launch a SkyPilot task without Click's incompatible hidden backend flag."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import sky


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("task")
    command.add_argument("-c", "--cluster")
    command.add_argument("--infra")
    command.add_argument("--gpus")
    command.add_argument("--down", action="store_true")
    command.add_argument("--dryrun", action="store_true")
    command.add_argument("--retry-until-up", action="store_true")
    return command


def load_task(path: str, *, infra: str | None, gpus: str | None) -> sky.Task:
    task = sky.Task.from_yaml(path)
    overrides = {}
    if infra is not None:
        overrides["infra"] = infra
    if gpus is not None:
        overrides["accelerators"] = gpus
    if overrides:
        task.set_resources({resource.copy(**overrides) for resource in task.resources})
    return task


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    task = load_task(args.task, infra=args.infra, gpus=args.gpus)
    result = sky.stream_and_get(
        sky.launch(
            task,
            cluster_name=args.cluster,
            dryrun=args.dryrun,
            down=args.down,
            retry_until_up=args.retry_until_up,
        )
    )
    if args.dryrun:
        return 0
    if result is None:
        raise RuntimeError("SkyPilot launch returned no result")
    job_id, handle = result
    if handle is None or job_id is None:
        raise RuntimeError("SkyPilot launch returned no cluster handle or job id")
    return sky.tail_logs(handle.get_cluster_name(), job_id, follow=True)


if __name__ == "__main__":
    raise SystemExit(main())
