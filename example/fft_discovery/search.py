"""Executable exact simplification and novelty filter for FFT circuits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .verifier import (
        Cyclotomic,
        Target,
        VerificationError,
        _combine_forms,
        structural_fingerprint,
        verify_circuit,
    )
except ImportError:  # Standalone copy in a research statespace.
    from fft_verify import (  # type: ignore[no-redef]
        Cyclotomic,
        Target,
        VerificationError,
        _combine_forms,
        structural_fingerprint,
        verify_circuit,
    )


def simplify_circuit(candidate: dict[str, Any], target: Target) -> dict[str, Any]:
    """Merge exact equivalent registers, then remove dead operations."""
    verify_circuit(candidate, target)
    field = Cyclotomic(target.n)
    input_count = 2 * target.n
    forms: list[dict[int, tuple]] = [{index: field.one} for index in range(input_count)]
    form_register = {_form_key(form): index for index, form in enumerate(forms)}
    remap = list(range(input_count))
    kept: list[dict[str, Any]] = []

    for operation in candidate["operations"]:
        kind = operation["op"]
        rewritten = dict(operation)
        rewritten["a"] = remap[operation["a"]]
        left = forms[rewritten["a"]]
        if kind in {"add", "sub"}:
            rewritten["b"] = remap[operation["b"]]
            form = _combine_forms(
                field,
                left,
                forms[rewritten["b"]],
                subtract=kind == "sub",
            )
        elif kind == "neg":
            form = {index: field.neg(value) for index, value in left.items()}
        else:
            constant = field.parse(operation["constant"])
            form = {
                index: product
                for index, value in left.items()
                if (product := field.mul(constant, value)) != field.zero
            }
        key = _form_key(form)
        existing = form_register.get(key)
        if existing is None:
            existing = input_count + len(kept)
            kept.append(rewritten)
            forms.append(form)
            form_register[key] = existing
        remap.append(existing)

    outputs = [remap[register] for register in candidate["outputs"]]
    operations, outputs = _remove_dead(input_count, kept, outputs)
    result = {
        "format": candidate["format"],
        "n": candidate["n"],
        "description": "Exact semantic CSE and dead-code simplification.",
        "operations": operations,
        "outputs": outputs,
    }
    verify_circuit(result, target)
    return result


def known_fingerprints(paths: list[Path]) -> set[str]:
    """Load structural identities from files or directory trees."""
    fingerprints: set[str] = set()
    for path in paths:
        files = path.rglob("*.json") if path.is_dir() else (path,)
        for file in files:
            try:
                fingerprints.add(structural_fingerprint(json.loads(file.read_text())))
            except (OSError, json.JSONDecodeError, VerificationError):
                continue
    return fingerprints


def _form_key(form: dict[int, tuple]) -> tuple:
    return tuple(sorted(form.items()))


def _remove_dead(
    input_count: int, operations: list[dict[str, Any]], outputs: list[int]
) -> tuple[list[dict[str, Any]], list[int]]:
    live: set[int] = set()
    stack = list(outputs)
    while stack:
        register = stack.pop()
        if register < input_count:
            continue
        index = register - input_count
        if index in live:
            continue
        live.add(index)
        operation = operations[index]
        stack.append(operation["a"])
        if operation["op"] in {"add", "sub"}:
            stack.append(operation["b"])

    register_map = {index: index for index in range(input_count)}
    compact: list[dict[str, Any]] = []
    for index, operation in enumerate(operations):
        old_register = input_count + index
        if index not in live:
            continue
        rewritten = dict(operation)
        rewritten["a"] = register_map[operation["a"]]
        if operation["op"] in {"add", "sub"}:
            rewritten["b"] = register_map[operation["b"]]
        register_map[old_register] = input_count + len(compact)
        compact.append(rewritten)
    return compact, [register_map[register] for register in outputs]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simplify one exact FFT circuit and reject known structures."
    )
    parser.add_argument("target")
    parser.add_argument("candidate")
    parser.add_argument("output")
    parser.add_argument(
        "--known",
        action="append",
        default=[],
        help="known circuit file or directory; repeat as needed",
    )
    args = parser.parse_args()
    target = Target.load(args.target)
    source = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    before = verify_circuit(source, target)
    result = simplify_circuit(source, target)
    after = verify_circuit(result, target)
    fingerprint = structural_fingerprint(result)
    if fingerprint in known_fingerprints([Path(value) for value in args.known]):
        raise SystemExit(f"duplicate_structure={fingerprint}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"before_total={before.total_operations}")
    print(f"after_total={after.total_operations}")
    print(f"structural_sha256={fingerprint}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
