#!/usr/bin/env python3
"""Exercise the Torch APIs involved in the repository's dependency advisories."""

from __future__ import annotations

import argparse
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pad_packed_sequence, unpack_sequence


def case_jit_script() -> None:
    class Ignored:
        def __init__(self) -> None:
            self.count: int = 0
            self.items: list[int] = []

    torch.jit.script(Ignored)


def case_lstm_cell() -> None:
    inp = torch.full((0, 8), 0, dtype=torch.float)
    hx = torch.full((0, 9), 0, dtype=torch.float)
    cx = torch.full((0, 9), 0, dtype=torch.float)
    w_ih = torch.full((1, 8), 1.251e12, dtype=torch.float)
    w_hh = torch.full((1, 9), 1.4013e-45, dtype=torch.float)
    torch.lstm_cell(inp, (hx, cx), w_ih, w_hh, None, None)


def case_pad_packed_sequence() -> None:
    packed = PackedSequence(torch.randn(0, 5), torch.tensor([], dtype=torch.int64), None, None)
    pad_packed_sequence(packed, batch_first=True)


def case_unpack_sequence() -> None:
    packed = PackedSequence(torch.tensor([]), torch.tensor([], dtype=torch.int64))
    unpack_sequence(packed)


def case_mkldnn_pool() -> None:
    source = torch.randn(2, 64, 32, 32).to_mkldnn()
    torch.mkldnn_max_pool2d(source, kernel_size=3, stride=0)


def case_ctc_loss() -> None:
    if not torch.cuda.is_available():
        return
    log_probs = torch.rand(0, 0, 4, device="cuda")
    targets = torch.tensor([], device="cuda", dtype=torch.long)
    lengths = torch.tensor([], device="cuda", dtype=torch.long)
    F.ctc_loss(log_probs, targets, lengths, lengths, reduction="none")


def verify_valid_training_control() -> None:
    cell = torch.nn.LSTMCell(8, 9)
    output, state = cell(torch.ones(2, 8), (torch.zeros(2, 9), torch.zeros(2, 9)))
    assert output.shape == state.shape == (2, 9)
    assert torch.isfinite(output).all()

    packed = pack_sequence([torch.ones(2, 3), torch.ones(1, 3)], enforce_sorted=False)
    restored = unpack_sequence(packed)
    assert [tuple(item.shape) for item in restored] == [(2, 3), (1, 3)]
    torch.jit.script(torch.nn.ReLU())


CASES = {
    "ctc-loss": case_ctc_loss,
    "jit-script": case_jit_script,
    "lstm-cell": case_lstm_cell,
    "mkldnn-pool": case_mkldnn_pool,
    "pad-packed-sequence": case_pad_packed_sequence,
    "unpack-sequence": case_unpack_sequence,
}


def run_isolated_cases() -> None:
    for case in CASES:
        result = subprocess.run(
            [sys.executable, __file__, "--case", case],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        if result.returncode < 0:
            raise RuntimeError(f"{case} crashed with signal {-result.returncode}")
        if result.returncode != 0:
            raise RuntimeError(f"{case} failed:\n{result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES))
    args = parser.parse_args()

    torch_version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:3])
    assert torch_version >= (2, 13, 0)
    if args.case:
        try:
            CASES[args.case]()
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as error:
            print(f"{args.case}: rejected safely: {error}")
        return

    run_isolated_cases()
    verify_valid_training_control()
    print(f"Torch {torch.__version__}: advisory regressions and valid controls passed")


if __name__ == "__main__":
    main()
