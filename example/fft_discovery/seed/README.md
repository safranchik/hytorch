# Exact FFT discovery research state

This directory contains the reusable seed for a HyTorch FFT discovery state.
The preparation command adds a machine-readable target, a trusted-verifier
copy, an incumbent when available, and submission directories.

The research state should eventually contain:

- A frozen transform definition and arithmetic cost model.
- A catalog of published incumbents with primary sources.
- An executable candidate representation.
- An exact, independent verifier.
- A reproducible search program.
- A verified Pareto frontier of candidate circuits.

Write candidate JSON files under `submissions/current/`. Run
`tools/fft_verify.py` before the final commit. Do not edit `control/` or
`incumbent/`.

Keep generated evidence and rejected hypotheses. Do not keep credentials or
unlicensed copies of papers in this repository.
