"""Generate a reproducible exact FFT incumbent circuit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .verifier import split_radix_circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    parser.add_argument("--n", type=int, default=32)
    args = parser.parse_args()

    destination = Path(args.destination)
    if destination.exists():
        raise FileExistsError(destination)
    circuit = split_radix_circuit(args.n)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(circuit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"candidate={destination.resolve()}")


if __name__ == "__main__":
    main()
