"""Prepare a standalone Git statespace for FFT discovery."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path

from .verifier import (
    COST_MODEL,
    INPUT_ORDER,
    OUTPUT_ORDER,
    TARGET_FORMAT,
    TRANSFORM,
    Target,
    direct_dft_circuit,
    verify_circuit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument(
        "--target",
        help="frozen frontier target JSON; omit it to create a calibration target",
    )
    parser.add_argument(
        "--incumbent", help="optional incumbent circuit for a frozen target"
    )
    args = parser.parse_args()
    prepare(
        Path(args.destination),
        n=args.n,
        target_path=Path(args.target) if args.target else None,
        incumbent_path=Path(args.incumbent) if args.incumbent else None,
    )


def prepare(
    destination: Path,
    *,
    n: int = 8,
    target_path: Path | None = None,
    incumbent_path: Path | None = None,
) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if target_path is None and incumbent_path is not None:
        raise ValueError("--incumbent requires --target")
    incumbent_value = None
    incumbent_result = None
    if target_path is None:
        incumbent_value = direct_dft_circuit(n)
        provisional = Target(n, "calibration", 1, "direct DFT", "generated", 100_000)
        incumbent_result = verify_circuit(incumbent_value, provisional)
        target_value = {
            "format": TARGET_FORMAT,
            "status": "calibration",
            "n": n,
            "transform": TRANSFORM,
            "input_order": INPUT_ORDER,
            "output_order": OUTPUT_ORDER,
            "cost_model": COST_MODEL,
            "incumbent": {
                "name": "generated direct DFT calibration circuit",
                "total_operations": incumbent_result.total_operations,
                "source": "tools/fft_verify.py direct_dft_circuit",
            },
            "limits": {
                "max_operations": max(10_000, len(incumbent_value["operations"]) * 4)
            },
        }
    else:
        target_value = json.loads(target_path.read_text(encoding="utf-8"))
        target = Target.from_dict(target_value)
        if target.status != "frozen":
            raise ValueError("a supplied frontier target must have status 'frozen'")
        if incumbent_path is not None:
            incumbent_value = json.loads(incumbent_path.read_text(encoding="utf-8"))
            incumbent_result = verify_circuit(
                incumbent_value,
                target,
            )
            if incumbent_result.total_operations != target.incumbent_total:
                raise ValueError(
                    "incumbent circuit must verify at the target incumbent count"
                )
    Target.from_dict(target_value)

    seed = Path(__file__).with_name("seed")
    shutil.copytree(seed, destination)
    Path(destination, "control").mkdir()
    Path(destination, "incumbent").mkdir()
    Path(destination, "submissions/current").mkdir(parents=True)
    Path(destination, "submissions/archive").mkdir()
    Path(destination, "reports").mkdir()
    Path(destination, "tools").mkdir(exist_ok=True)
    shutil.copy2(
        Path(__file__).with_name("verifier.py"), destination / "tools/fft_verify.py"
    )
    shutil.copy2(
        Path(__file__).with_name("search.py"), destination / "tools/fft_search.py"
    )
    Path(destination, "submissions/current/README.md").write_text(
        "# Current submissions\n\nWrite this generation's candidate JSON files here.\n",
        encoding="utf-8",
    )
    if incumbent_value is not None and incumbent_result is not None:
        _write_json(destination / "incumbent/circuit.json", incumbent_value)
        _write_json(
            destination / "incumbent/verification.json",
            dataclasses.asdict(incumbent_result),
        )
    _write_json(destination / "control/target.json", target_value)
    _write_json(
        destination / "reports/preparation.json",
        {
            "target_status": target_value["status"],
            "warning": (
                "Calibration mode proves the pipeline only. It cannot establish a "
                "scientific record."
                if target_value["status"] == "calibration"
                else "The operator supplied a frozen frontier target."
            ),
        },
    )
    _init_git(destination)
    print(f"state_dir={destination}")
    print(f"target_status={target_value['status']}")


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _init_git(root: Path) -> None:
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="HyTorch",
        GIT_AUTHOR_EMAIL="hytorch@localhost",
        GIT_COMMITTER_NAME="HyTorch",
        GIT_COMMITTER_EMAIL="hytorch@localhost",
    )
    for args in (
        ("init", "--quiet", "--initial-branch=main"),
        ("add", "-A"),
        ("commit", "--quiet", "-m", "Initialize FFT discovery state"),
    ):
        subprocess.run(["git", "-C", root, *args], env=env, check=True)


if __name__ == "__main__":
    main()
