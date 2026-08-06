"""Run the FFT target-selection network once in inference mode."""

from __future__ import annotations

import argparse

import hytorch

from .network import FFTDiscoveryNetwork

DEFAULT_MODEL = "gpt-5.6-terra"
RESEARCH_TASK = """\
Select one machine-verifiable frontier problem in exact FFT algorithm discovery.
Start from the files in the statespace. Audit current primary literature before
choosing a target. The final state must define the transform, allowed constants,
equivalence domain, arithmetic cost model, published incumbent, exact certificate,
independent verifier, search budget, and stop conditions. Prefer a small fixed
transform that can support repeated local experiments. This run selects and
specifies the research problem. It does not claim a new FFT result.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", help="standalone Git directory with the research seed")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default="openai-codex")
    args = parser.parse_args()

    harness = hytorch.harness.PiHarness(provider=args.provider)
    model = FFTDiscoveryNetwork().to(harness, mtype=args.model)
    state = hytorch.space(args.state, harness=harness)
    with hytorch.inference_mode():
        output = model(state, task=RESEARCH_TASK)

    print(f"output_dir={output.dir}")
    print(f"output_commit={output.commit}")
    print(f"workspace_store={model._parameter_store.root}")


if __name__ == "__main__":
    main()
