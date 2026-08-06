"""A small research network for exact FFT algorithm discovery."""

from __future__ import annotations

import hytorch


class FFTDiscoveryNetwork(hytorch.mn.Module):
    """Three specialists, two reviewers, and one final curator."""

    def __init__(self) -> None:
        super().__init__()
        self.theory = hytorch.mn.Linear(
            1,
            1,
            bias=(
                "Act as the FFT algebra and literature specialist. Establish exact "
                "definitions, derive useful decompositions, and audit primary sources. "
                "Write owned artifacts under research/theory/. Do not claim a record "
                "without a traceable source and a matching cost model."
            ),
        )
        self.search = hytorch.mn.Linear(
            1,
            1,
            bias=(
                "Act as the algorithm-search engineer. Study executable search methods "
                "for FFT straight-line programs, including symbolic rewriting, common "
                "subexpression elimination, SAT or SMT search, and evolutionary search. "
                "Start by running tools/fft_search.py on each promising circuit. Extend "
                "it with bounded exact local synthesis when its built-in semantic CSE "
                "does not improve the circuit. Keep machine-readable search logs under "
                "research/search/. Put only new exact candidate circuits under "
                "submissions/current/search/. Verify each candidate with "
                "tools/fft_verify.py. Do not submit renamed copies of known circuits."
            ),
        )
        self.verification = hytorch.mn.Linear(
            1,
            1,
            bias=(
                "Act as the exact-verification and cost-model specialist. Define how to "
                "prove transform equivalence with exact algebraic arithmetic and how to "
                "count permitted operations. Write owned artifacts under "
                "research/verification/. Independently run tools/fft_verify.py on "
                "candidate circuits. Compare structural hashes with the archive. Reject "
                "ambiguous, duplicate, or floating-point-only claims."
            ),
        )
        self.proposer = hytorch.mn.Linear(
            3,
            1,
            bias=(
                "Synthesize the three specialist branches into one concrete discovery "
                "proposal. Select a bounded FFT target with a published incumbent, an "
                "exact certificate format, and a feasible CLI search plan. Write the "
                "proposal under proposals/primary/. Build an exact circuit when possible. "
                "Put it under submissions/current/proposer/. Never edit control/ or "
                "incumbent/. If no new candidate exists, submit no circuit."
            ),
        )
        self.critic = hytorch.mn.Linear(
            3,
            1,
            bias=(
                "Audit the specialist branches adversarially. Find unsupported novelty "
                "claims, mismatched cost models, verification gaps, and targets that are "
                "too expensive for the available compute. Write the audit under "
                "reviews/adversarial/. Try to repair or simplify candidates. Put each "
                "new verified alternative under submissions/current/critic/. Reject a "
                "candidate if its only change is prose, formatting, or a file name."
            ),
        )
        self.curate = hytorch.mn.Linear(
            2,
            1,
            bias=(
                "Reconcile the proposal and adversarial review. Produce TARGET.md with "
                "one precise research target, SOURCES.md with primary citations, and "
                "VERIFIER.md with the exact acceptance contract. Preserve contrary "
                "evidence. Inspect the machine target in control/target.json. Select or "
                "construct the strongest exact circuit and write it to "
                "submissions/current/curator.json. Run tools/fft_verify.py before the "
                "final commit. Do not resubmit the incumbent when no new structure was "
                "found. Do not report an improvement unless it verifies."
            ),
        )

    def forward(self, state: hytorch.Space, *, task: str) -> hytorch.Space:
        theory = self.theory(state, task=task)[0]
        search = self.search(state, task=task)[0]
        verification = self.verification(state, task=task)[0]
        evidence = (theory, search, verification)
        proposal = self.proposer(*evidence, task=task)[0]
        critique = self.critic(*evidence, task=task)[0]
        return self.curate(proposal, critique, task=task)[0]


__all__ = ["FFTDiscoveryNetwork"]
