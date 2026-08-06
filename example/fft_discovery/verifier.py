"""Exact verifier for real-arithmetic circuits computing a complex DFT."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

TARGET_FORMAT = "hytorch-fft-target-v1"
CIRCUIT_FORMAT = "hytorch-fft-circuit-v1"
MAX_FILE_BYTES = 8_000_000
MAX_RATIONAL_CHARS = 256
TRANSFORM = "unscaled complex-input DFT with negative exponential sign"
INPUT_ORDER = "x0.real, x0.imag, x1.real, x1.imag, ..."
OUTPUT_ORDER = "X0.real, X0.imag, X1.real, X1.imag, ..."
COST_MODEL = {
    "add": 1,
    "sub": 1,
    "nontrivial_real_scale": 1,
    "negation": 0,
    "multiplication_by_one_or_minus_one": 0,
    "fused_operations": "not allowed",
}


class VerificationError(ValueError):
    """A target or candidate does not satisfy the declared format."""


@dataclasses.dataclass(frozen=True)
class Target:
    n: int
    status: str
    incumbent_total: int
    incumbent_name: str
    source: str
    max_operations: int

    @classmethod
    def load(cls, path: str | Path) -> Target:
        return cls.from_dict(_load_json(path))

    @classmethod
    def from_dict(cls, value: Any) -> Target:
        if not isinstance(value, dict) or value.get("format") != TARGET_FORMAT:
            raise VerificationError(f"target format must be {TARGET_FORMAT!r}")
        if value.get("transform") != TRANSFORM:
            raise VerificationError(f"target transform must be {TRANSFORM!r}")
        if value.get("input_order") != INPUT_ORDER:
            raise VerificationError(f"target input_order must be {INPUT_ORDER!r}")
        if value.get("output_order") != OUTPUT_ORDER:
            raise VerificationError(f"target output_order must be {OUTPUT_ORDER!r}")
        if value.get("cost_model") != COST_MODEL:
            raise VerificationError("target cost_model does not match the verifier")
        n = value.get("n")
        status = value.get("status")
        incumbent = value.get("incumbent")
        limits = value.get("limits", {})
        if not _is_power_of_two(n) or n < 4:
            raise VerificationError("target n must be a power of two of at least 4")
        if status not in {"calibration", "frozen"}:
            raise VerificationError("target status must be 'calibration' or 'frozen'")
        if not isinstance(incumbent, dict):
            raise VerificationError("target incumbent must be an object")
        total = incumbent.get("total_operations")
        name = incumbent.get("name")
        source = incumbent.get("source")
        maximum = limits.get("max_operations", max(10_000, total * 4 if total else 0))
        if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
            raise VerificationError("incumbent total_operations must be positive")
        if not isinstance(name, str) or not name.strip():
            raise VerificationError("incumbent name must be non-empty text")
        if not isinstance(source, str) or not source.strip():
            raise VerificationError("incumbent source must be non-empty text")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
            raise VerificationError("limits.max_operations must be positive")
        return cls(n, status, total, name.strip(), source.strip(), maximum)


@dataclasses.dataclass(frozen=True)
class Verification:
    valid: bool
    additions: int = 0
    multiplications: int = 0
    total_operations: int = 0
    depth: int = 0
    live_operations: int = 0
    dead_operations: int = 0
    candidate_sha256: str = ""
    error: str = ""

    @property
    def score(self) -> tuple[int, int, int]:
        return (self.total_operations, self.depth, self.multiplications)


class Cyclotomic:
    """Exact arithmetic in Q(ω), where ω is a power-of-two root of unity."""

    def __init__(self, n: int):
        if not _is_power_of_two(n) or n < 4:
            raise ValueError("cyclotomic order must be a power of two of at least 4")
        self.n = n
        self.degree = n // 2
        self.zero = (Fraction(0),) * self.degree
        self.one = self.rational(1)
        self.minus_one = self.rational(-1)

    def rational(self, value: int | Fraction) -> tuple[Fraction, ...]:
        return (Fraction(value),) + (Fraction(0),) * (self.degree - 1)

    def root(self, exponent: int) -> tuple[Fraction, ...]:
        exponent %= self.n
        sign = 1
        if exponent >= self.degree:
            exponent -= self.degree
            sign = -1
        values = [Fraction(0)] * self.degree
        values[exponent] = Fraction(sign)
        return tuple(values)

    def add(self, left, right):
        return tuple(a + b for a, b in zip(left, right, strict=True))

    def sub(self, left, right):
        return tuple(a - b for a, b in zip(left, right, strict=True))

    def neg(self, value):
        return tuple(-item for item in value)

    def scale_rational(self, value, scalar: int | Fraction):
        scalar = Fraction(scalar)
        return tuple(scalar * item for item in value)

    def mul(self, left, right):
        result = [Fraction(0)] * self.degree
        left_terms = [(i, value) for i, value in enumerate(left) if value]
        right_terms = [(i, value) for i, value in enumerate(right) if value]
        for i, a in left_terms:
            for j, b in right_terms:
                exponent = i + j
                if exponent >= self.degree:
                    result[exponent - self.degree] -= a * b
                else:
                    result[exponent] += a * b
        return tuple(result)

    def conjugate(self, value):
        result = self.zero
        for exponent, coefficient in enumerate(value):
            if coefficient:
                result = self.add(
                    result,
                    self.scale_rational(self.root(-exponent), coefficient),
                )
        return result

    def is_real(self, value) -> bool:
        return self.conjugate(value) == value

    def real_part(self, value):
        return self.scale_rational(
            self.add(value, self.conjugate(value)), Fraction(1, 2)
        )

    def imaginary_part(self, value):
        # ω = exp(-2πi/N), so q = ω^(N/4) = -i and
        # Im(z) = q(z - conjugate(z))/2.
        difference = self.sub(value, self.conjugate(value))
        return self.scale_rational(
            self.mul(self.root(self.n // 4), difference), Fraction(1, 2)
        )

    def parse(self, value: Any):
        if not isinstance(value, dict) or set(value) != {"basis"}:
            raise VerificationError("scale constant must contain one 'basis' object")
        basis = value["basis"]
        if not isinstance(basis, dict):
            raise VerificationError("constant basis must be an object")
        result = [Fraction(0)] * self.degree
        for raw_exponent, raw_coefficient in basis.items():
            try:
                exponent = int(raw_exponent)
            except (TypeError, ValueError) as exc:
                raise VerificationError("constant exponent must be an integer") from exc
            if str(exponent) != str(raw_exponent) or not 0 <= exponent < self.degree:
                raise VerificationError(
                    f"constant exponent must be between 0 and {self.degree - 1}"
                )
            try:
                if isinstance(raw_coefficient, bool) or not isinstance(
                    raw_coefficient, (int, str)
                ):
                    raise TypeError
                if len(str(raw_coefficient)) > MAX_RATIONAL_CHARS:
                    raise VerificationError("constant coefficient is too long")
                coefficient = Fraction(raw_coefficient)
            except (TypeError, ValueError, ZeroDivisionError) as exc:
                raise VerificationError(
                    "constant coefficient must be an integer or rational string"
                ) from exc
            result[exponent] += coefficient
        return tuple(result)

    def encode(self, value) -> dict[str, dict[str, str]]:
        return {
            "basis": {
                str(index): str(coefficient)
                for index, coefficient in enumerate(value)
                if coefficient
            }
        }

    def display(self, value) -> str:
        terms = []
        for index, coefficient in enumerate(value):
            if coefficient:
                suffix = "" if index == 0 else f"*omega^{index}"
                terms.append(f"{coefficient}{suffix}")
        return " + ".join(terms) if terms else "0"


def verify_file(path: str | Path, target: Target) -> Verification:
    candidate_path = Path(path)
    try:
        raw = candidate_path.read_bytes()
        if len(raw) > MAX_FILE_BYTES:
            raise VerificationError("candidate file is too large")
        value = json.loads(raw)
        result = verify_circuit(value, target)
        return dataclasses.replace(
            result, candidate_sha256=hashlib.sha256(raw).hexdigest()
        )
    except (OSError, json.JSONDecodeError, VerificationError) as exc:
        return Verification(valid=False, error=str(exc))


def structural_fingerprint(candidate: Any) -> str:
    """Return an identity for circuit structure, independent of prose metadata."""
    if not isinstance(candidate, dict):
        raise VerificationError("candidate must be an object")
    required = ("format", "n", "operations", "outputs")
    if any(name not in candidate for name in required):
        raise VerificationError("candidate has no complete circuit structure")
    structure = {name: candidate[name] for name in required}
    canonical = json.dumps(
        structure,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_circuit(candidate: Any, target: Target) -> Verification:
    if not isinstance(candidate, dict) or candidate.get("format") != CIRCUIT_FORMAT:
        raise VerificationError(f"candidate format must be {CIRCUIT_FORMAT!r}")
    if candidate.get("n") != target.n:
        raise VerificationError(f"candidate n must equal target n={target.n}")
    operations = candidate.get("operations")
    outputs = candidate.get("outputs")
    if not isinstance(operations, list):
        raise VerificationError("candidate operations must be a list")
    if len(operations) > target.max_operations:
        raise VerificationError(
            f"candidate has {len(operations)} operations; limit is {target.max_operations}"
        )
    if not isinstance(outputs, list) or len(outputs) != 2 * target.n:
        raise VerificationError(
            f"candidate outputs must contain {2 * target.n} registers"
        )

    field = Cyclotomic(target.n)
    input_count = 2 * target.n
    forms: list[dict[int, tuple[Fraction, ...]]] = [
        {index: field.one} for index in range(input_count)
    ]
    depths = [0] * input_count
    sources: list[tuple[int, ...]] = []
    costs: list[tuple[int, int]] = []

    for operation_index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise VerificationError(f"operation {operation_index} must be an object")
        kind = operation.get("op")
        register_count = len(forms)
        if kind in {"add", "sub"}:
            if set(operation) != {"op", "a", "b"}:
                raise VerificationError(
                    f"operation {operation_index} has invalid fields"
                )
            a = _register(operation["a"], register_count, operation_index)
            b = _register(operation["b"], register_count, operation_index)
            form = _combine_forms(field, forms[a], forms[b], subtract=kind == "sub")
            depth = max(depths[a], depths[b]) + 1
            sources.append((a, b))
            costs.append((1, 0))
        elif kind == "neg":
            if set(operation) != {"op", "a"}:
                raise VerificationError(
                    f"operation {operation_index} has invalid fields"
                )
            a = _register(operation["a"], register_count, operation_index)
            form = {index: field.neg(value) for index, value in forms[a].items()}
            depth = depths[a]
            sources.append((a,))
            costs.append((0, 0))
        elif kind == "scale":
            if set(operation) != {"op", "a", "constant"}:
                raise VerificationError(
                    f"operation {operation_index} has invalid fields"
                )
            a = _register(operation["a"], register_count, operation_index)
            constant = field.parse(operation["constant"])
            if not field.is_real(constant):
                raise VerificationError(
                    f"operation {operation_index} scale constant is not real"
                )
            form = {
                index: product
                for index, value in forms[a].items()
                if (product := field.mul(constant, value)) != field.zero
            }
            depth = depths[a] + (constant not in {field.one, field.minus_one})
            sources.append((a,))
            costs.append((0, int(constant not in {field.one, field.minus_one})))
        else:
            raise VerificationError(
                f"operation {operation_index} has invalid op {kind!r}"
            )
        forms.append(form)
        depths.append(depth)

    output_registers = [
        _register(value, len(forms), f"output {index}")
        for index, value in enumerate(outputs)
    ]
    expected = _expected_dft(field)
    for output_index, (register, expected_form) in enumerate(
        zip(output_registers, expected, strict=True)
    ):
        actual = forms[register]
        variables = sorted(set(actual) | set(expected_form))
        for variable in variables:
            actual_value = actual.get(variable, field.zero)
            expected_value = expected_form.get(variable, field.zero)
            if actual_value != expected_value:
                raise VerificationError(
                    f"output {output_index} coefficient for input {variable} differs: "
                    f"got {field.display(actual_value)}, expected "
                    f"{field.display(expected_value)}"
                )

    live = _live_operations(input_count, sources, output_registers)
    additions = sum(costs[index][0] for index in live)
    multiplications = sum(costs[index][1] for index in live)
    total = additions + multiplications
    return Verification(
        valid=True,
        additions=additions,
        multiplications=multiplications,
        total_operations=total,
        depth=max(depths[register] for register in output_registers),
        live_operations=len(live),
        dead_operations=len(operations) - len(live),
    )


def direct_dft_circuit(n: int) -> dict[str, Any]:
    """Return a correct direct DFT circuit for calibration and verifier tests."""
    field = Cyclotomic(n)
    operations: list[dict[str, Any]] = []
    input_count = 2 * n

    def emit(kind: str, **values) -> int:
        operations.append({"op": kind, **values})
        return input_count + len(operations) - 1

    def scaled(register: int, constant) -> int | None:
        if constant == field.zero:
            return None
        if constant == field.one:
            return register
        if constant == field.minus_one:
            return emit("neg", a=register)
        return emit("scale", a=register, constant=field.encode(constant))

    def sum_terms(terms: list[int]) -> int:
        if not terms:
            raise RuntimeError("direct DFT output unexpectedly has no terms")
        result = terms[0]
        for term in terms[1:]:
            result = emit("add", a=result, b=term)
        return result

    outputs = []
    for expected_form in _expected_dft(field):
        terms = []
        for variable, coefficient in expected_form.items():
            term = scaled(variable, coefficient)
            if term is not None:
                terms.append(term)
        outputs.append(sum_terms(terms))
    return {
        "format": CIRCUIT_FORMAT,
        "n": n,
        "description": "Direct exact DFT used as a calibration incumbent.",
        "operations": operations,
        "outputs": outputs,
    }


def split_radix_circuit(n: int) -> dict[str, Any]:
    """Return a conjugate-pair-cost split-radix circuit for a power-of-two DFT."""
    field = Cyclotomic(n)
    operations: list[dict[str, Any]] = []
    input_count = 2 * n

    def emit(kind: str, **values) -> int:
        operations.append({"op": kind, **values})
        return input_count + len(operations) - 1

    def add(left: int, right: int) -> int:
        return emit("add", a=left, b=right)

    def sub(left: int, right: int) -> int:
        return emit("sub", a=left, b=right)

    def neg(register: int) -> int:
        return emit("neg", a=register)

    def scale(register: int, constant) -> int | None:
        if constant == field.zero:
            return None
        if constant == field.one:
            return register
        if constant == field.minus_one:
            return neg(register)
        return emit("scale", a=register, constant=field.encode(constant))

    def combine_terms(terms: list[tuple[int, int | None]]) -> int:
        nonzero = [(sign, register) for sign, register in terms if register is not None]
        if not nonzero:
            raise RuntimeError("split-radix product unexpectedly has no terms")
        sign, register = nonzero[0]
        result = register if sign == 1 else neg(register)
        for sign, register in nonzero[1:]:
            result = add(result, register) if sign == 1 else sub(result, register)
        return result

    def multiply(pair: tuple[int, int], exponent: int) -> tuple[int, int]:
        twiddle = field.root(exponent)
        cosine = field.real_part(twiddle)
        sine = field.imaginary_part(twiddle)
        real, imaginary = pair
        if cosine == sine and cosine != field.zero:
            return (
                scale(sub(real, imaginary), cosine),
                scale(add(real, imaginary), cosine),
            )
        if cosine == field.neg(sine) and cosine != field.zero:
            return (
                scale(add(real, imaginary), cosine),
                scale(sub(imaginary, real), cosine),
            )
        return (
            combine_terms([(1, scale(real, cosine)), (-1, scale(imaginary, sine))]),
            combine_terms([(1, scale(real, sine)), (1, scale(imaginary, cosine))]),
        )

    def transform(indices: list[int]) -> list[tuple[int, int]]:
        size = len(indices)
        if size == 1:
            index = indices[0]
            return [(2 * index, 2 * index + 1)]
        if size == 2:
            first = (2 * indices[0], 2 * indices[0] + 1)
            second = (2 * indices[1], 2 * indices[1] + 1)
            return [
                (add(first[0], second[0]), add(first[1], second[1])),
                (sub(first[0], second[0]), sub(first[1], second[1])),
            ]

        even = transform(indices[0::2])
        odd_one = transform(indices[1::4])
        odd_three = transform(indices[3::4])
        quarter = size // 4
        outputs: list[tuple[int, int] | None] = [None] * size
        root_stride = n // size
        for k in range(quarter):
            first = multiply(odd_one[k], k * root_stride)
            third = multiply(odd_three[k], 3 * k * root_stride)
            pair_sum = (add(first[0], third[0]), add(first[1], third[1]))
            pair_difference = (
                sub(first[0], third[0]),
                sub(first[1], third[1]),
            )
            low = even[k]
            high = even[k + quarter]
            outputs[k] = (
                add(low[0], pair_sum[0]),
                add(low[1], pair_sum[1]),
            )
            outputs[k + size // 2] = (
                sub(low[0], pair_sum[0]),
                sub(low[1], pair_sum[1]),
            )
            outputs[k + quarter] = (
                add(high[0], pair_difference[1]),
                sub(high[1], pair_difference[0]),
            )
            outputs[k + 3 * quarter] = (
                sub(high[0], pair_difference[1]),
                add(high[1], pair_difference[0]),
            )
        if any(value is None for value in outputs):
            raise RuntimeError("split-radix circuit left an output unset")
        return [value for value in outputs if value is not None]

    outputs = [register for pair in transform(list(range(n))) for register in pair]
    return {
        "format": CIRCUIT_FORMAT,
        "n": n,
        "description": "Exact split-radix DFT incumbent.",
        "operations": operations,
        "outputs": outputs,
    }


def _expected_dft(field: Cyclotomic):
    expected: list[dict[int, tuple[Fraction, ...]]] = []
    for output_index in range(field.n):
        real_form = {}
        imaginary_form = {}
        for input_index in range(field.n):
            twiddle = field.root(output_index * input_index)
            cosine = field.real_part(twiddle)
            sine = field.imaginary_part(twiddle)
            real_input = 2 * input_index
            imaginary_input = real_input + 1
            if cosine != field.zero:
                real_form[real_input] = cosine
                imaginary_form[imaginary_input] = cosine
            if sine != field.zero:
                real_form[imaginary_input] = field.neg(sine)
                imaginary_form[real_input] = sine
        expected.extend((real_form, imaginary_form))
    return expected


def _combine_forms(field, left, right, *, subtract: bool):
    result = dict(left)
    for index, value in right.items():
        combined = (
            field.sub(result.get(index, field.zero), value)
            if subtract
            else field.add(result.get(index, field.zero), value)
        )
        if combined == field.zero:
            result.pop(index, None)
        else:
            result[index] = combined
    return result


def _live_operations(
    input_count: int, sources: list[tuple[int, ...]], outputs: list[int]
) -> set[int]:
    live: set[int] = set()
    stack = list(outputs)
    while stack:
        register = stack.pop()
        if register < input_count:
            continue
        operation = register - input_count
        if operation in live:
            continue
        live.add(operation)
        stack.extend(sources[operation])
    return live


def _register(value: Any, limit: int, location: int | str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < limit:
        raise VerificationError(f"{location} references invalid register {value!r}")
    return value


def _load_json(path: str | Path) -> Any:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise VerificationError(str(exc)) from exc
    if len(raw) > MAX_FILE_BYTES:
        raise VerificationError("JSON file is too large")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VerificationError(str(exc)) from exc


def _is_power_of_two(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        and value & (value - 1) == 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("candidate")
    args = parser.parse_args()
    target = Target.load(args.target)
    result = verify_file(args.candidate, target)
    print(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))
    raise SystemExit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
