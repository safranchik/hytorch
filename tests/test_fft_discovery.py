import copy
import json
import os
import shutil
import threading
from pathlib import Path

import pytest
from conftest import (
    commit_agent_changes,
    find_agent_workspace,
    merge_agent_inputs,
)

import hytorch
from example.fft_discovery.prepare import prepare
from example.fft_discovery.search import simplify_circuit
from example.fft_discovery.train import evaluate_submissions, known_structures, run
from example.fft_discovery.verifier import (
    CIRCUIT_FORMAT,
    COST_MODEL,
    INPUT_ORDER,
    OUTPUT_ORDER,
    TARGET_FORMAT,
    TRANSFORM,
    Target,
    direct_dft_circuit,
    split_radix_circuit,
    structural_fingerprint,
    verify_circuit,
)


def target(n=8, incumbent=10_000):
    return Target.from_dict(
        {
            "format": TARGET_FORMAT,
            "status": "calibration",
            "n": n,
            "transform": TRANSFORM,
            "input_order": INPUT_ORDER,
            "output_order": OUTPUT_ORDER,
            "cost_model": COST_MODEL,
            "incumbent": {
                "name": "test incumbent",
                "total_operations": incumbent,
                "source": "generated test fixture",
            },
            "limits": {"max_operations": 100_000},
        }
    )


def test_direct_dft_circuit_passes_exact_verification():
    for n in (4, 8):
        result = verify_circuit(direct_dft_circuit(n), target(n))

        assert result.valid
        assert result.total_operations == result.additions + result.multiplications
        assert result.live_operations > 0
        assert result.dead_operations == 0


def test_independent_n4_formula_passes_exact_verification():
    operations = []

    def linear(terms):
        register = terms[0][1]
        for sign, source in terms[1:]:
            operations.append(
                {"op": "add" if sign == 1 else "sub", "a": register, "b": source}
            )
            register = 8 + len(operations) - 1
        return register

    outputs = [
        linear(terms)
        for terms in (
            ((1, 0), (1, 2), (1, 4), (1, 6)),
            ((1, 1), (1, 3), (1, 5), (1, 7)),
            ((1, 0), (1, 3), (-1, 4), (-1, 7)),
            ((1, 1), (-1, 2), (-1, 5), (1, 6)),
            ((1, 0), (-1, 2), (1, 4), (-1, 6)),
            ((1, 1), (-1, 3), (1, 5), (-1, 7)),
            ((1, 0), (-1, 3), (-1, 4), (1, 7)),
            ((1, 1), (1, 2), (-1, 5), (-1, 6)),
        )
    ]
    candidate = {
        "format": CIRCUIT_FORMAT,
        "n": 4,
        "operations": operations,
        "outputs": outputs,
    }

    result = verify_circuit(candidate, target(4))

    assert result.valid
    assert result.total_operations == 24


def test_split_radix_matches_published_operation_formula():
    for n in (4, 8, 16, 32):
        result = verify_circuit(split_radix_circuit(n), target(n))

        assert result.valid
        assert result.total_operations == 4 * n * (n.bit_length() - 1) - 6 * n + 8
        assert result.dead_operations == 0


def test_frozen_n32_target_matches_generated_incumbent():
    path = (
        Path(__file__).parents[1] / "example" / "fft_discovery" / "targets" / "n32.json"
    )
    frozen = Target.load(path)

    result = verify_circuit(split_radix_circuit(32), frozen)

    assert frozen.status == "frozen"
    assert result.valid
    assert result.additions == 372
    assert result.multiplications == 84
    assert result.total_operations == frozen.incumbent_total == 456


def test_verifier_rejects_an_incorrect_output():
    candidate = direct_dft_circuit(4)
    candidate["outputs"][0] = 0

    with pytest.raises(ValueError, match="output 0 coefficient"):
        verify_circuit(candidate, target(4))


def test_verifier_counts_only_live_operations():
    candidate = direct_dft_circuit(4)
    baseline = verify_circuit(candidate, target(4))
    candidate["operations"].append({"op": "add", "a": 0, "b": 1})

    result = verify_circuit(candidate, target(4))

    assert result.total_operations == baseline.total_operations
    assert result.dead_operations == 1


def test_structural_fingerprint_ignores_description_and_json_layout():
    first = direct_dft_circuit(4)
    second = copy.deepcopy(first)
    second["description"] = "A renamed circuit is still the same structure."
    second["untrusted_note"] = {"anything": True}

    assert structural_fingerprint(first) == structural_fingerprint(second)
    second["outputs"] = list(reversed(second["outputs"]))
    assert structural_fingerprint(first) != structural_fingerprint(second)


def test_exact_search_removes_semantically_duplicate_and_dead_operations():
    candidate = direct_dft_circuit(4)
    input_count = 8
    duplicate = copy.deepcopy(candidate["operations"][0])
    candidate["operations"].insert(1, duplicate)
    for operation in candidate["operations"][2:]:
        for name in ("a", "b"):
            if operation.get(name, -1) >= input_count + 1:
                operation[name] += 1
    candidate["outputs"] = [
        register + 1 if register >= input_count + 1 else register
        for register in candidate["outputs"]
    ]
    candidate["operations"].append({"op": "add", "a": 0, "b": 1})

    simplified = simplify_circuit(candidate, target(4))
    result = verify_circuit(simplified, target(4))

    assert result.valid
    assert len(simplified["operations"]) < len(candidate["operations"])
    assert result.dead_operations == 0


def test_submission_evaluation_rejects_prior_and_same_generation_duplicates(tmp_path):
    root = tmp_path
    (root / "incumbent").mkdir()
    (root / "submissions/current").mkdir(parents=True)
    circuit = direct_dft_circuit(4)
    (root / "incumbent/circuit.json").write_text(json.dumps(circuit))
    renamed = copy.deepcopy(circuit)
    renamed["description"] = "renamed"
    for name in ("a.json", "b.json"):
        (root / "submissions/current" / name).write_text(json.dumps(renamed))

    evaluations = evaluate_submissions(
        str(root), target(4), known_structures(str(root))
    )

    assert len(evaluations) == 2
    assert all(value.verification.valid for value in evaluations)
    assert all(value.duplicate_of == "incumbent/circuit.json" for value in evaluations)


def test_verifier_rejects_non_real_and_float_scale_constants():
    non_real = direct_dft_circuit(4)
    non_real["operations"].append(
        {"op": "scale", "a": 0, "constant": {"basis": {"1": "1"}}}
    )
    floating = copy.deepcopy(non_real)
    floating["operations"][-1]["constant"] = {"basis": {"0": 0.5}}

    with pytest.raises(ValueError, match="not real"):
        verify_circuit(non_real, target(4))
    with pytest.raises(ValueError, match="integer or rational string"):
        verify_circuit(floating, target(4))


def test_prepare_creates_a_committed_calibration_state(tmp_path):
    destination = tmp_path / "fft-state"

    prepare(destination, n=8)

    target_value = json.loads((destination / "control/target.json").read_text())
    prepared_target = Target.from_dict(target_value)
    incumbent = json.loads((destination / "incumbent/circuit.json").read_text())
    result = verify_circuit(incumbent, prepared_target)
    assert result.valid
    assert result.total_operations == prepared_target.incumbent_total
    assert (destination / ".git").is_dir()
    assert (destination / "submissions/current/README.md").is_file()
    assert (destination / "tools/fft_verify.py").is_file()
    assert (destination / "tools/fft_search.py").is_file()


class FFTTestHarness(hytorch.harness.Harness):
    def __init__(self, **kwargs):
        super().__init__("fft-test")
        self._lock = threading.Lock()
        self._next_session = 0

    def start(self, directory, prompt, mtype, **kwargs):
        with self._lock:
            session_id = f"session-{self._next_session}"
            self._next_session += 1
        if prompt.startswith("Update your persistent native state"):
            workspace = os.path.join(directory, "workspace")
            selected = find_agent_workspace(workspace)
            with open(
                os.path.join(selected, "verified-feedback.md"),
                "w",
                encoding="utf-8",
            ) as file:
                file.write("Use the trusted verifier.\n")
            session = hytorch.harness.Session(self.name, session_id, "")
            return hytorch.harness.Result("updated owner", session)
        statespace = os.path.join(directory, "statespace")
        merge_agent_inputs(statespace)
        source = os.path.join(statespace, "incumbent", "circuit.json")
        destination = os.path.join(
            statespace, "submissions", "current", f"{session_id}.json"
        )
        if os.path.isfile(source):
            shutil.copy2(source, destination)
        commit_agent_changes(statespace, "test agent: submit incumbent")
        session = hytorch.harness.Session(self.name, session_id, "")
        return hytorch.harness.Result("submitted incumbent", session)

    def resume(self, session, directory, prompt, mtype, **kwargs):
        workspace = os.path.join(directory, "workspace")
        selected = find_agent_workspace(workspace)
        with open(
            os.path.join(selected, f"lesson-{session.id}.md"),
            "w",
            encoding="utf-8",
        ) as file:
            file.write("Use the trusted verifier.\n")
        refs = os.path.join(
            directory, "statespace", ".git", "refs", "hytorch", "inputs"
        )
        input_count = len(os.listdir(refs))
        return hytorch.harness.Result(
            json.dumps(
                {
                    "update": "Use the trusted verifier.",
                    "feedback": ["Keep exact evidence."] * input_count,
                }
            ),
            session,
        )

    def close(self, session):
        pass

    def usage(self):
        return hytorch.harness.Usage()


def test_training_checkpoints_and_resumes_offline(tmp_path, monkeypatch):
    state = tmp_path / "fft-state"
    run_dir = tmp_path / "fft-run"
    prepare(state, n=4)
    monkeypatch.setattr(hytorch.harness, "PiHarness", FFTTestHarness)

    options = {
        "run_dir": run_dir,
        "generations": 1,
        "max_hours": 1,
        "max_total_tokens": 10_000,
        "max_stagnant": 5,
        "backward_tokens": 100,
        "temp": 0.1,
        "model_type": "test-model",
        "provider": "test-provider",
        "allow_calibration": True,
        "stop_on_record": True,
    }
    run(state_path=state, resume=False, **options)
    run(state_path=None, resume=True, **options)

    latest = json.loads((run_dir / "latest.json").read_text())
    assert latest["generation"] == 2
    assert latest["total_submissions"] > 0
    assert latest["valid_submissions"] == latest["total_submissions"]
    assert latest["novel_submissions"] == 0
    assert latest["duplicate_submissions"] == latest["total_submissions"]
    assert (run_dir / "generation-0000/model/MODEL.json").is_file()
    assert (run_dir / "generation-0001/state/reports/generation-0001.json").is_file()
    assert (run_dir / "generation-0002/state/reports/generation-0002.json").is_file()
