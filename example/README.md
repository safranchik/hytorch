# HyTorch examples

[`terminal_bench/`](terminal_bench/) contains an experimental training and
evaluation harness for Terminal-Bench 2.1. It downloads the upstream benchmark
at a pinned revision. See its local README for requirements and commands.

[`fft_discovery/`](fft_discovery/) contains a resumable `1 → 3 → 2 → 1`
research network. It searches for smaller exact DFT circuits and evaluates them
with a trusted exact verifier.
