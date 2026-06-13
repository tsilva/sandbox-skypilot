#!/usr/bin/env python3
"""Small remote smoke test for SkyPilot cloud instances."""

import json
import os
import platform
import shutil
import socket
import subprocess
import sys


def run_command(command: list[str]) -> dict[str, object]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command[0]}

    completed = subprocess.run(
        [executable, *command[1:]],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    report = {
        "status": "ok",
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "sky_env": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("SKYPILOT_") or key.startswith("SKY_")
        },
        "nvidia_smi": run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
