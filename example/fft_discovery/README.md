# Exact FFT circuit discovery

This example trains six persistent agents to search for a smaller exact DFT
circuit. A trusted verifier checks every candidate. Each completed generation
has a model checkpoint, a statespace checkpoint, an evaluation report, and
token-use metadata.

The example can prove a new upper bound under one fixed cost model. It cannot
prove that an algorithm is globally optimal. It also cannot guarantee that a
run will find an improvement.

## Network

The `1 → 3 → 2 → 1` graph has these roles:

```text
                         algebra and literature ─┐
                                                ├─ proposal ───────┐
research state ───────── search engineering ────┤                  ├─ curator
                                                ├─ adversarial ────┘
                         exact verification ────┘
```

The agents keep search code, algebra, and failure lessons in their persistent
workspaces. The statespace keeps candidates and reports. The controller gives
exact evaluation results to the graph as directional feedback.

## Executable search and novelty

Each statespace contains `tools/fft_search.py`. It performs exact semantic
common-subexpression elimination and dead-code removal. It verifies the result
before it writes a candidate. It also accepts repeated `--known` paths and
refuses to write a structure that is already known:

```sh
python tools/fft_search.py \
  control/target.json \
  incumbent/circuit.json \
  submissions/current/search/simplified.json \
  --known incumbent \
  --known submissions/archive
```

This is a safe first search primitive. Agents can extend it with bounded local
synthesis, SAT, SMT, or other exact methods. Numerical equality is not enough.

The controller computes a canonical SHA-256 identity from `format`, `n`,
`operations`, and `outputs`. It ignores descriptions, file names, formatting,
and other prose. It compares each valid submission with the incumbent, all
prior generations, and earlier submissions in the same generation. A duplicate
remains visible in the report, but it cannot become the generation winner.

## First calibration run

Create a standalone Git statespace with an `N=8` direct-DFT incumbent:

```sh
uv run python -m example.fft_discovery.prepare fft-calibration --n 8
```

This is a deliberately weak incumbent. Use it to test the complete pipeline.
It is not a scientific frontier. Training refuses this target unless you pass
`--allow-calibration`.

Run one generation:

```sh
uv run python -m example.fft_discovery.train \
  fft-calibration \
  --run-dir fft-calibration-run \
  --allow-calibration \
  --generations 1 \
  --max-hours 2 \
  --max-total-tokens 250000
```

Run a second generation from the last complete checkpoint:

```sh
uv run python -m example.fft_discovery.train \
  --run-dir fft-calibration-run \
  --resume \
  --allow-calibration \
  --generations 1 \
  --max-hours 4 \
  --max-total-tokens 500000
```

`--max-hours` and `--max-total-tokens` are cumulative run limits. The
controller checks them between generations. One active generation can exceed a
limit. An interrupted generation is discarded. The prior checkpoint remains
valid.

Inspect `fft-calibration-run/latest.json` after each generation. Inspect the
matching `generation-NNNN/state/reports/` directory for exact candidate
results.

## Frozen research target

A record attempt needs a separately audited target JSON file. It must use this
shape:

```json
{
  "format": "hytorch-fft-target-v1",
  "status": "frozen",
  "n": 16,
  "transform": "unscaled complex-input DFT with negative exponential sign",
  "input_order": "x0.real, x0.imag, x1.real, x1.imag, ...",
  "output_order": "X0.real, X0.imag, X1.real, X1.imag, ...",
  "cost_model": {
    "add": 1,
    "sub": 1,
    "nontrivial_real_scale": 1,
    "negation": 0,
    "multiplication_by_one_or_minus_one": 0,
    "fused_operations": "not allowed"
  },
  "incumbent": {
    "name": "audited published construction",
    "total_operations": 123,
    "source": "primary-source citation with theorem or table location"
  },
  "limits": {
    "max_operations": 10000
  }
}
```

Replace the example count and source with audited values. Do not infer the
count from a different transform, scaling convention, or arithmetic model.

The included [`targets/n32.md`](targets/n32.md) audit freezes a practical
frontier target at 456 operations. It also includes commands to generate and
verify the exact split-radix incumbent before training.

Prepare the target:

```sh
uv run python -m example.fft_discovery.prepare \
  fft-frontier-state \
  --target target.json
```

An incumbent circuit is optional. If one is available, add
`--incumbent incumbent.json`. Preparation requires it to pass the exact
verifier at the declared count.

## Bounded overnight run

First complete one or two calibration generations. Confirm that at least one
candidate is valid, checkpoints resume, and token use is acceptable. Then run
a frozen target with explicit limits:

```sh
uv run python -m example.fft_discovery.train \
  fft-frontier-state \
  --run-dir fft-frontier-run \
  --generations 50 \
  --max-hours 10 \
  --max-total-tokens 2000000 \
  --max-stagnant 12
```

The run stops after a trusted candidate beats the frozen count. Pass
`--continue-after-record` only if you want it to search for further reductions.
Use `--resume` after an interruption. Do not pass the source statespace during
a resumed run.

## Trust boundary

The candidate format is in `seed/CIRCUIT.md`. Verification uses exact rational
arithmetic in `Q(ω)`. It does not use floating-point tolerance.

Agents can edit files in their statespace. They cannot change the acceptance
result. The controller:

- Reads the canonical target from the input checkpoint.
- Runs the package copy of `verifier.py` outside the agent statespace.
- Restores `control/target.json`, `tools/fft_verify.py`, and
  `tools/fft_search.py` after each generation.
- Promotes only a candidate that passes exact transform equivalence.
- Rejects known circuit structures even if their prose or file name changed.
- Stores each generation in a new immutable checkpoint directory.

A verified lower count supports an arithmetic-count claim only. A practical
speed claim needs an optimized implementation, fixed hardware, and comparison
with current FFT libraries.
