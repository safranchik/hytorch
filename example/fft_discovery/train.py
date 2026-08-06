"""Run resumable HyTorch generations against an exact FFT verifier."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import hytorch
from hytorch._git import GitError
from hytorch.parameter import set_tree_writable

from .network import FFTDiscoveryNetwork
from .verifier import (
    Target,
    Verification,
    structural_fingerprint,
    verify_circuit,
    verify_file,
)

DEFAULT_MODEL = "gpt-5.6-terra"
DISCOVERY_TASK = """\
Improve the exact FFT circuit in the statespace. Inspect control/target.json,
CIRCUIT.md, prior reports, archived submissions, and the incumbent. Develop
reusable algebra and executable search programs. Run tools/fft_search.py or
extend it with stronger bounded local synthesis. Every submitted structure must
be new relative to incumbent/ and submissions/archive/. Put this generation's
final candidate JSON files under submissions/current/ with role-specific names.
Run python tools/fft_verify.py on every submitted candidate. Never edit control/
or incumbent/. A trusted external verifier will ignore changes to those paths.
An unverified numerical approximation or renamed duplicate is not a candidate.
"""
MAX_SUBMISSIONS = 128
RECOVERABLE_BACKWARD_ERRORS = ("agent produced no final text response",)


@dataclasses.dataclass(frozen=True)
class EvaluatedCandidate:
    path: str
    verification: Verification
    structural_sha256: str = ""
    duplicate_of: str = ""

    @property
    def novel(self) -> bool:
        return self.verification.valid and not self.duplicate_of


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", nargs="?", help="prepared standalone research state")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--max-total-tokens", type=int, default=500_000)
    parser.add_argument("--max-stagnant", type=int, default=10)
    parser.add_argument("--backward-tokens", type=int, default=4_000)
    parser.add_argument("--temp", type=float, default=0.4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default="openai-codex")
    parser.add_argument("--allow-calibration", action="store_true")
    parser.add_argument("--continue-after-record", action="store_true")
    args = parser.parse_args()
    if args.generations <= 0:
        parser.error("--generations must be positive")
    if args.max_hours <= 0:
        parser.error("--max-hours must be positive")
    if args.max_total_tokens <= 0:
        parser.error("--max-total-tokens must be positive")
    if args.max_stagnant <= 0:
        parser.error("--max-stagnant must be positive")
    if args.resume and args.state:
        parser.error("omit state when --resume is set")
    if not args.resume and not args.state:
        parser.error("state is required for a new run")

    run(
        state_path=Path(args.state).resolve() if args.state else None,
        run_dir=Path(args.run_dir).resolve(),
        resume=args.resume,
        generations=args.generations,
        max_hours=args.max_hours,
        max_total_tokens=args.max_total_tokens,
        max_stagnant=args.max_stagnant,
        backward_tokens=args.backward_tokens,
        temp=args.temp,
        model_type=args.model,
        provider=args.provider,
        allow_calibration=args.allow_calibration,
        stop_on_record=not args.continue_after_record,
    )


def run(
    *,
    state_path: Path | None,
    run_dir: Path,
    resume: bool,
    generations: int,
    max_hours: float,
    max_total_tokens: int,
    max_stagnant: int,
    backward_tokens: int,
    temp: float,
    model_type: str,
    provider: str,
    allow_calibration: bool,
    stop_on_record: bool,
) -> None:
    harness = hytorch.harness.PiHarness(provider=provider)
    model = FFTDiscoveryNetwork().to(harness, mtype=model_type)
    cumulative_tokens = 0
    prior_elapsed = 0.0
    stagnant = 0
    latest: dict | None = None

    if resume:
        latest = _load_latest(run_dir)
        generation = latest["generation"]
        checkpoint = run_dir / f"generation-{generation:04d}"
        model.load_state_dir(hytorch.load(checkpoint / "model"))
        state = hytorch.space(checkpoint / "state", harness=harness)
        cumulative_tokens = latest["cumulative_tokens"]
        prior_elapsed = latest["elapsed_seconds"]
        stagnant = latest["stagnant_generations"]
        start_generation = generation + 1
    else:
        if run_dir.exists():
            raise FileExistsError(run_dir)
        state = hytorch.space(state_path, harness=harness)
        start_generation = 1

    target_value = json.loads(state.repo.read_file(state.commit, "control/target.json"))
    target = Target.from_dict(target_value)
    if target.status == "calibration" and not allow_calibration:
        raise RuntimeError(
            "refusing to train on a calibration target; pass --allow-calibration "
            "for a bounded pipeline test"
        )
    canonical_target = json.dumps(target_value, indent=2, sort_keys=True) + "\n"
    incumbent_bytes = _read_optional(state, "incumbent/circuit.json")
    incumbent_verification = (
        verify_bytes(incumbent_bytes, target) if incumbent_bytes is not None else None
    )
    best_total = latest["best_total_operations"] if latest else target.incumbent_total
    if not isinstance(best_total, int) or isinstance(best_total, bool):
        raise RuntimeError("latest checkpoint has no valid best operation count")
    if incumbent_verification is not None:
        if not incumbent_verification.valid:
            raise RuntimeError(
                "stored incumbent circuit does not pass the trusted verifier"
            )
        if incumbent_verification.total_operations != best_total:
            raise RuntimeError("stored incumbent circuit does not match the best count")
    elif best_total < target.incumbent_total:
        raise RuntimeError("improved checkpoint has no stored incumbent circuit")
    if latest and latest.get("target_beaten") is True and stop_on_record:
        print("stop=verified_target_beaten_in_checkpoint", flush=True)
        return
    if not resume:
        run_dir.mkdir(parents=True)

    state = _activate_state(state, harness, run_dir)

    optimizer = hytorch.optim.DFM(
        model.parameters(), temp=temp, max_tokens=backward_tokens
    )
    if not resume:
        _checkpoint(
            run_dir,
            0,
            model,
            state,
            {
                "generation": 0,
                "cumulative_tokens": 0,
                "elapsed_seconds": 0.0,
                "stagnant_generations": 0,
                "best_total_operations": best_total,
                "target_beaten": False,
                "record_found": False,
                "target_status": target.status,
                "model": model_type,
                "provider": provider,
                "mutation_temperature": temp,
                "backward_tokens": backward_tokens,
                "target_sha256": hashlib.sha256(canonical_target.encode()).hexdigest(),
            },
        )

    started = time.monotonic()
    usage_before_run = harness.usage()
    final_generation = start_generation + generations - 1
    for generation in range(start_generation, final_generation + 1):
        elapsed = prior_elapsed + time.monotonic() - started
        if elapsed >= max_hours * 3600:
            print("stop=max_hours", flush=True)
            break
        if cumulative_tokens >= max_total_tokens:
            print("stop=max_total_tokens", flush=True)
            break
        if stagnant >= max_stagnant:
            print("stop=max_stagnant", flush=True)
            break

        optimizer.zero_feed()
        generation_usage_start = harness.usage()
        generation_started = time.monotonic()
        output = model(state, task=DISCOVERY_TASK)
        seen = known_structures(output.dir)
        evaluations = evaluate_submissions(output.dir, target, seen)
        target_changed = _target_changed(output.dir, canonical_target)
        best = best_valid(evaluations)
        improved = best is not None and best.verification.total_operations < best_total
        feedback = build_feedback(
            target,
            evaluations,
            best,
            current_best=best_total,
            improved=improved,
            target_changed=target_changed,
        )
        backward_error = ""
        try:
            hytorch.Loss(output, feedback=feedback).backward()
            optimizer.step()
        except RuntimeError as exc:
            if not any(message in str(exc) for message in RECOVERABLE_BACKWARD_ERRORS):
                raise
            backward_error = str(exc)
            optimizer.zero_feed()
            print(
                f"warning=recoverable_backward_error generation={generation}",
                flush=True,
            )

        if improved and best is not None:
            incumbent_bytes = Path(output.dir, best.path).read_bytes()
            incumbent_verification = best.verification
            best_total = best.verification.total_operations
            stagnant = 0
        else:
            stagnant += 1

        state = promote_generation_state(
            output,
            harness,
            generation,
            canonical_target,
            evaluations,
            incumbent_bytes,
            incumbent_verification,
            best_total,
            backward_error,
        )
        generation_usage = harness.usage() - generation_usage_start
        run_usage = harness.usage() - usage_before_run
        cumulative_tokens_at_start = cumulative_tokens
        cumulative_tokens = cumulative_tokens_at_start + _tokens(generation_usage)
        elapsed = prior_elapsed + time.monotonic() - started
        target_beaten = best_total < target.incumbent_total
        record_found = target.status == "frozen" and target_beaten
        metadata = {
            "generation": generation,
            "cumulative_tokens": cumulative_tokens,
            "elapsed_seconds": elapsed,
            "stagnant_generations": stagnant,
            "best_total_operations": best_total,
            "target_beaten": target_beaten,
            "record_found": record_found,
            "target_status": target.status,
            "model": model_type,
            "provider": provider,
            "mutation_temperature": temp,
            "backward_tokens": backward_tokens,
            "backward_error": backward_error,
            "target_sha256": hashlib.sha256(canonical_target.encode()).hexdigest(),
            "generation_seconds": time.monotonic() - generation_started,
            "generation_usage": dataclasses.asdict(generation_usage),
            "current_process_usage": dataclasses.asdict(run_usage),
            "valid_submissions": sum(
                evaluation.verification.valid for evaluation in evaluations
            ),
            "novel_submissions": sum(evaluation.novel for evaluation in evaluations),
            "duplicate_submissions": sum(
                bool(evaluation.duplicate_of) for evaluation in evaluations
            ),
            "total_submissions": len(evaluations),
        }
        _checkpoint(run_dir, generation, model, state, metadata)
        print(
            f"generation={generation} best_total={best_total} "
            f"target_beaten={target_beaten} record={record_found} "
            f"valid={metadata['valid_submissions']}/"
            f"{metadata['total_submissions']} tokens={cumulative_tokens} "
            f"seconds={metadata['generation_seconds']:.1f}",
            flush=True,
        )
        if target_beaten and stop_on_record:
            print("stop=verified_target_beaten", flush=True)
            break


def known_structures(root: str) -> dict[str, str]:
    """Return structural identities that predate the current generation."""
    base = Path(root)
    paths: list[Path] = []
    incumbent = base / "incumbent/circuit.json"
    if incumbent.is_file():
        paths.append(incumbent)
    archive = base / "submissions/archive"
    if archive.is_dir():
        paths.extend(sorted(archive.rglob("*.json")))
    seen: dict[str, str] = {}
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            fingerprint = structural_fingerprint(value)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        seen.setdefault(fingerprint, path.relative_to(base).as_posix())
    return seen


def evaluate_submissions(
    root: str, target: Target, seen: dict[str, str] | None = None
) -> list[EvaluatedCandidate]:
    current = Path(root, "submissions", "current")
    paths = sorted(current.rglob("*.json")) if current.is_dir() else []
    if len(paths) > MAX_SUBMISSIONS:
        paths = paths[:MAX_SUBMISSIONS]
    known = dict(seen or {})
    evaluations = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        verification = verify_file(path, target)
        fingerprint = ""
        duplicate_of = ""
        if verification.valid:
            try:
                fingerprint = structural_fingerprint(json.loads(path.read_text()))
                duplicate_of = known.get(fingerprint, "")
                known.setdefault(fingerprint, relative)
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        evaluations.append(
            EvaluatedCandidate(relative, verification, fingerprint, duplicate_of)
        )
    return evaluations


def best_valid(
    evaluations: list[EvaluatedCandidate],
) -> EvaluatedCandidate | None:
    valid = [value for value in evaluations if value.novel]
    return min(valid, key=lambda value: value.verification.score) if valid else None


def build_feedback(
    target: Target,
    evaluations: list[EvaluatedCandidate],
    best: EvaluatedCandidate | None,
    *,
    current_best: int,
    improved: bool,
    target_changed: bool,
) -> str:
    valid = [value for value in evaluations if value.verification.valid]
    invalid = [value for value in evaluations if not value.verification.valid]
    novel = [value for value in evaluations if value.novel]
    duplicates = [value for value in evaluations if value.duplicate_of]
    lines = [
        f"Trusted exact evaluation for N={target.n}: {len(valid)} of "
        f"{len(evaluations)} submissions are valid.",
        f"Structural novelty: {len(novel)} new and {len(duplicates)} duplicate.",
        f"The declared incumbent uses {target.incumbent_total} real arithmetic operations.",
    ]
    if current_best != target.incumbent_total:
        lines.append(
            f"The best preserved circuit now uses {current_best} real arithmetic "
            "operations."
        )
    if target_changed:
        lines.append(
            "The forward state changed control/target.json. Do not edit trusted control files."
        )
    if best is not None:
        score = best.verification
        lines.append(
            f"Best valid submission {best.path} uses {score.total_operations} total "
            f"operations: {score.additions} additions and {score.multiplications} "
            f"multiplications, with depth {score.depth}."
        )
    else:
        lines.append("No new valid candidate was submitted.")
    if improved:
        lines.append(
            "The best candidate is an exact improvement. Preserve the reusable methods "
            "that produced it and attempt independent simplification and validation."
        )
    else:
        lines.append(
            "No candidate improved the incumbent. Improve the reusable search, algebra, "
            "and simplification procedures. Submit fewer and stronger exact candidates."
        )
    for evaluation in invalid[:8]:
        lines.append(f"Invalid {evaluation.path}: {evaluation.verification.error}")
    for evaluation in duplicates[:8]:
        lines.append(
            f"Duplicate {evaluation.path}: same structure as {evaluation.duplicate_of}."
        )
    return "\n".join(lines)


def promote_generation_state(
    output: hytorch.Space,
    harness: hytorch.harness.Harness,
    generation: int,
    canonical_target: str,
    evaluations: list[EvaluatedCandidate],
    incumbent_bytes: bytes | None,
    incumbent_verification: Verification | None,
    best_total: int,
    backward_error: str,
) -> hytorch.Space:
    root = Path(output.dir)
    set_tree_writable(output.dir, True)
    shutil.rmtree(root / "control", ignore_errors=True)
    Path(root, "control").mkdir()
    Path(root, "control/target.json").write_text(canonical_target, encoding="utf-8")
    Path(root, "tools").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).with_name("verifier.py"), root / "tools/fft_verify.py")
    shutil.copy2(Path(__file__).with_name("search.py"), root / "tools/fft_search.py")

    shutil.rmtree(root / "incumbent", ignore_errors=True)
    Path(root, "incumbent").mkdir()
    if incumbent_bytes is not None:
        Path(root, "incumbent/circuit.json").write_bytes(incumbent_bytes)
    if incumbent_verification is not None:
        _write_json(
            root / "incumbent/verification.json",
            dataclasses.asdict(incumbent_verification),
        )

    current = root / "submissions/current"
    archive = root / f"submissions/archive/generation-{generation:04d}"
    if current.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise RuntimeError(f"submission archive already exists: {archive}")
        shutil.move(current, archive)
    current.mkdir(parents=True)
    Path(current, "README.md").write_text(
        "# Current submissions\n\nWrite this generation's candidate JSON files here.\n",
        encoding="utf-8",
    )
    _write_json(
        root / f"reports/generation-{generation:04d}.json",
        {
            "generation": generation,
            "best_total_operations": best_total,
            "backward_error": backward_error,
            "submissions": [
                {
                    "path": value.path,
                    "structural_sha256": value.structural_sha256,
                    "duplicate_of": value.duplicate_of,
                    "novel": value.novel,
                    "verification": dataclasses.asdict(value.verification),
                }
                for value in evaluations
            ],
        },
    )
    _git(root, "add", "-A")
    _git(root, "commit", "-m", f"Evaluate FFT generation {generation}")
    return hytorch.space(root, harness=harness)


def verify_bytes(value: bytes, target: Target) -> Verification:
    try:
        result = verify_circuit(json.loads(value), target)
        return dataclasses.replace(
            result, candidate_sha256=hashlib.sha256(value).hexdigest()
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return Verification(valid=False, error=str(exc))


def _checkpoint(
    run_dir: Path,
    generation: int,
    model: FFTDiscoveryNetwork,
    state: hytorch.Space,
    metadata: dict,
) -> None:
    destination = run_dir / f"generation-{generation:04d}"
    if destination.exists():
        raise FileExistsError(destination)
    temporary_checkpoint = run_dir / f".generation-{generation:04d}.tmp"
    shutil.rmtree(temporary_checkpoint, ignore_errors=True)
    temporary_checkpoint.mkdir()
    hytorch.save(model.state_dir(), temporary_checkpoint / "model")
    _clone_state(state, temporary_checkpoint / "state")
    _write_json(temporary_checkpoint / "metadata.json", metadata)
    os.replace(temporary_checkpoint, destination)
    temporary = run_dir / ".latest.json"
    _write_json(temporary, metadata)
    os.replace(temporary, run_dir / "latest.json")


def _clone_state(state: hytorch.Space, destination: Path) -> None:
    _git(
        state.repo.root,
        "clone",
        "--quiet",
        "--no-local",
        "--no-checkout",
        "--no-tags",
        state.repo.root,
        str(destination),
    )
    _git(destination, "checkout", "--quiet", "-B", "main", state.commit)
    _git(destination, "remote", "remove", "origin")


def _activate_state(
    state: hytorch.Space, harness: hytorch.harness.Harness, run_dir: Path
) -> hytorch.Space:
    """Clone input state and install the current trusted executable tools."""
    destination = run_dir / ".active-state"
    shutil.rmtree(destination, ignore_errors=True)
    _clone_state(state, destination)
    set_tree_writable(destination, True)
    tools = destination / "tools"
    tools.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).with_name("verifier.py"), tools / "fft_verify.py")
    shutil.copy2(Path(__file__).with_name("search.py"), tools / "fft_search.py")
    _git(destination, "add", "tools/fft_verify.py", "tools/fft_search.py")
    if _git(destination, "status", "--porcelain"):
        _git(destination, "commit", "-m", "Install trusted FFT search tools")
    return hytorch.space(destination, harness=harness)


def _load_latest(run_dir: Path) -> dict:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    try:
        value = json.loads(Path(run_dir, "latest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("run directory has no valid latest checkpoint") from exc
    integer_fields = {
        "generation": 0,
        "cumulative_tokens": 0,
        "stagnant_generations": 0,
        "best_total_operations": 1,
    }
    for field, minimum in integer_fields.items():
        item = value.get(field)
        if not isinstance(item, int) or isinstance(item, bool) or item < minimum:
            raise RuntimeError(f"latest checkpoint has invalid {field}")
    elapsed = value.get("elapsed_seconds")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or elapsed < 0
    ):
        raise RuntimeError("latest checkpoint has invalid elapsed_seconds")
    for field in ("record_found", "target_beaten"):
        if not isinstance(value.get(field), bool):
            raise RuntimeError(f"latest checkpoint has invalid {field}")
    return value


def _read_optional(state: hytorch.Space, path: str) -> bytes | None:
    try:
        return state.repo.read_file(state.commit, path)
    except GitError:
        return None


def _target_changed(root: str, canonical_target: str) -> bool:
    try:
        return (
            Path(root, "control/target.json").read_text(encoding="utf-8")
            != canonical_target
        )
    except OSError:
        return True


def _tokens(usage: hytorch.harness.Usage) -> int:
    return usage.input_tokens + usage.output_tokens


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git(root: str | Path, *args: str) -> str:
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="HyTorch",
        GIT_AUTHOR_EMAIL="hytorch@localhost",
        GIT_COMMITTER_NAME="HyTorch",
        GIT_COMMITTER_EMAIL="hytorch@localhost",
    )
    result = subprocess.run(
        ["git", "-C", root, *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


if __name__ == "__main__":
    main()
