# Terminal-Bench 2.1 MVP

This experiment uses ten public Terminal-Bench 2.1 tasks. Five tasks provide
training feedback. Five tasks are held out for evaluation. It does not use the
private test set or submit results.

The network is `1 -> 3 -> 1`. It uses Pi with `gpt-5.6-terra`. All task forward
passes and all official verifiers run concurrently. Each backward pass updates
the same canonical workspace history, so DFM promotes those updates in a stable
order.

Run two epochs:

```sh
uv run python -m example.terminal_bench.train --epochs 2
```

The first run builds `hytorch-terminal-bench-mvp:latest` and downloads the
pinned public task images. The run prints the baseline, each training score,
and the held-out score after each epoch. It updates these files after each
measurement:

```text
example/terminal_bench/results/scores.csv
example/terminal_bench/results/learning_curve.png
```
