# HyTorch Development

HyTorch is a Python library, package `hytorch`, for building and training
agent meta-networks with PyTorch-shaped composition. Values are Git-backed
statespaces, agents are neurons, directory-backed workspaces are Parameters,
directional feedback replaces gradients, and workspace mutations become Git
commits.
`SPEC.md` is the canonical design.

There is no CLI product surface. HyTorch is imported as a library.

## Documentation

Use standard Markdown that does not require MathJax, KaTeX, or another
renderer extension. Do not use LaTeX math delimiters such as `\(`, `\)`,
`\[`, `\]`, `$`, or `$$`. Write equations with regular Markdown and Unicode
characters, for example: `Yᵢ = Agent(Xᵢ; Wᵢ)`.

## PyTorch correspondence

Inspect current official PyTorch documentation and source before inventing a
public API. Match both its syntax and its ownership/composition logic where a
real text-agent equivalent exists.

| PyTorch | HyTorch |
|---|---|
| `torch.tensor(...)` | `hytorch.space(...)` |
| `Tensor` | statespace `Space` |
| `torch.nn` | `hytorch.mn` |
| neuron | agent |
| `nn.Module` | `mn.Module` |
| `nn.Parameter` | `mn.Parameter` |
| `nn.Linear` | `mn.Linear` |
| `torch.nn.init` | `hytorch.mn.init` |
| `torch.manual_seed` | `hytorch.manual_seed` |
| gradient `Tensor` | directional feedback |
| `.grad` | `.feed` |
| `optimizer.zero_grad()` | `optimizer.zero_feed()` |
| `torch.optim` | `hytorch.optim` |
| learning rate | mutation temperature `temp` |
| applied parameter update | committed Git mutation |

`Module.__setattr__` registers Parameters and child Modules. Registration owns
state; calls in `forward()` dynamically define topology. `model.parameters()`
is the only normal input to an optimizer.

`mn.Linear(in_features, out_features, bias=...)` owns one monolithic `weight`
workspace per output agent. Its physical shape is `(out_features,)`. Each
output receives every input statespace, so the logical graph remains dense.
The `bias` initializes the mutable `AGENTS.md` in each output workspace.

## Node state

Every executed output agent receives two sibling directory trees:

```text
node/
  statespace/    # activation X becomes Y; writable during forward
  workspace/     # persistent Parameter W; writable only during optimization
```

Forward gives every output agent an independent, writable `statespace/.git`
with one fetched local ref per input. The agent chooses the merge order,
resolves conflicts, edits the statespace, and commits before it finishes. It
uses a sparse,
read-only `workspace/` checkout that contains the global model history.
`loss.backward()` resumes the session with the permissions reversed. The agent
commits workspace changes directly on the global candidate lineage and returns
one directional feedback string per input. `DFM.step()` promotes the completed
candidate model branch.

`mn.Linear.reset_parameters()` delegates to `hytorch.mn.init`, just as
`torch.nn.Linear` delegates to `torch.nn.init`. Each workspace starts with an
`AGENTS.md` that contains seeded, randomly sampled, general phrases from
`mn.init.DEFAULT_PRIORS`. It can later contain arbitrary code, tools, examples,
and data.

Forward returns the complete committed statespace, never a special answer file.
Do not inject Space contents into the agent prompt. Mount complete directory
trees. `zero_feed()` clears accumulated feedback and discards an unpromoted
candidate branch. It never deletes canonical Git history.

## Git semantics

Each Space owns its statespace repository and forward history. The private,
global model workspace store records initialization and optimizer generations.
Feedback is transient text.
Workspace diffs are concrete mutations. DFM is evolutionary: it advances to
every valid child mutation and does not roll back because one immediate result
is worse.

Backward runs dependency-ready nodes with distinct workspace paths in
parallel. Each node commits against one global candidate revision. HyTorch
merges these commits into the global model history. Nodes that share a
workspace path run in sequence.

## Harnesses

`hytorch.harness.Harness` is the execution base class. Built-ins are
`hytorch.harness.pi`, `codex`, and `claude_code`; only Pi executes in v0. Pi
uses OpenAI through the operator's Codex login or `OPENAI_API_KEY`, with
`gpt-5.6-terra` as the default model.

One executed graph uses one harness. `model.to(harness)` moves the complete
model before a new forward pass. Docker remains external deployment
configuration. HyTorch uses the standard active Docker context and accepts
`HYTORCH_PI_IMAGE` as an image override. Agent variables come from
`.hytorch.env`, the global HyTorch secrets file, or `HYTORCH_ENV_FILE`. Never
load the ordinary project `.env` automatically.

## Layout

- `hytorch/mn/` — meta-network namespace and initialization functions.
- `hytorch/graph.py` — Module registration and dynamic execution boundary.
- `hytorch/parameter.py` — directory-backed workspace Parameters and their Git store.
- `hytorch/linear.py` — dense parallel agent layer.
- `hytorch/backward.py` — Loss and feed-Space propagation.
- `hytorch/optim/` — Optimizer base and DFM.
- `hytorch/space.py` — Space and lowercase `space` factory.
- `hytorch/runtime/` — Dockerized Pi runtime.
- `example/` — Terminal-Bench training and evaluation example.
- `tests/` — offline unit tests plus an opt-in real Pi integration test.

## Commands

```sh
uv sync
uv run pytest
HYTORCH_PI_TEST=1 uv run pytest tests/test_pi.py -v -s
uv run python -m example.terminal_bench.train --epochs 2
```
