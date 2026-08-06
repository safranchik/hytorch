# HyTorch Meta-Network Specification

## Purpose

HyTorch applies PyTorch-shaped ownership and training to agent networks.

```text
Tensor          -> statespace Space
Parameter       -> persistent native agent state
neuron          -> agent
autograd graph  -> retained execution graph
gradient        -> directional feedback
parameter delta -> accumulated owner mutation feed
```

Calls in `forward()` define the runtime graph. `model.parameters()` supplies
the optimizer. `loss.backward()` accumulates directional feed while it
propagates feedback. `optimizer.step()` reduces all feed into one update per
Parameter and atomically promotes the model generation.

## Spaces and Parameters

`hytorch.space(data, *, mtype=None, harness=None, requires_feed=False)` creates
a `Space` from one Git-backed directory. A list creates an ordered
`SpaceBatch`.

A Space is an activation state. It enters a node as `X`. The node transforms
it into output statespace `Y`.

An `mn.Parameter` is one agent's persistent native state `W`. It can include
the native transcript, compaction records, memories, instructions, skills,
settings, databases, tools, and other local files. The harness defines the
format. HyTorch treats the directory as opaque state.

A harness can reserve small metadata files inside native state. These files
can identify the stable local project or current session tip. The agent must
not change harness-owned metadata directly.

```text
Yᵢ = Agentᵢ(X₀, ..., Xₙ; Wᵢ)
```

Each model owns one private Git repository:

```text
hytorch/workspaces/<temporary-id>/
├── .git/
├── MODEL.json
└── layers/
    └── <qualified-layer>/
        └── <agent>/
            ├── AGENTS.md
            └── <harness-native state>
```

The private repository is a HyTorch implementation detail. The agent receives
a plain materialized directory. It does not receive the repository or model
history. The initial `AGENTS.md` contains the `bias` and seeded priors. A
harness or agent can replace it with any native state.

Credentials and live process state are not Parameters. A harness must keep API
keys, OAuth credentials, sockets, process IDs, bearer tokens, and runtime locks
outside the Parameter. It can attach a temporary credential overlay during a
turn. It must remove the overlay before HyTorch captures state.

Native agent state must contain normal files, directories, and internal
symlinks. It must not contain `.git`, escaping symlinks, sockets, or other
special files. This rule lets private Git capture the complete state.

## Linear

One output feature is one persistent agent. `mn.Linear(3, 4)` runs four agents
in parallel. Each agent receives all three input Spaces and owns one monolithic
state directory. `layer.weight.shape` is `(out_features,)`.

The logical graph is dense. A native state can implement nonlinear behavior
across all inputs.

`mn.Linear.reset_parameters()` delegates to `hytorch.mn.init`.
`hytorch.manual_seed(seed)` makes prior initialization reproducible.

## Harnesses

`hytorch.harness.Harness` defines three lifecycle operations:

```text
start(workspace, prompt)          -> Result(text, native session tip)
resume(session, workspace, prompt)-> Result(text, new native session tip)
close(session)                    -> release runtime resources only
```

`start()` continues an existing native session when the workspace contains
one. It creates a session only for a new agent. `resume()` can return a new
opaque session tip because native compaction can rotate a session ID.
`close()` must not delete persisted state.

Built-in harnesses are `pi`, `codex`, `claude-code`, `opencode`, `hermes`, and
`prime-agent`. Each harness owns its native profile layout and command-line
contract. One executed graph uses one harness. `model.to(harness)` moves the
complete model before a new forward pass. `model.to(mtype=...)` changes the
default model type.

Agent variables come from global `~/.config/hytorch/secrets.env`, project
`.hytorch.env`, and `HYTORCH_ENV_FILE`, in that order. Exported provider keys
override declared values. HyTorch ignores ordinary `.env`.

## Forward

Each output agent receives this node root:

```text
node/
├── statespace/     # writable, self-contained activation Git repository
├── parameter/      # read-only canonical native state
└── workspace/      # writable temporary episode fork
```

The canonical Parameter remains unchanged. The writable `workspace/` is a
disposable episode fork. Native session writes, transcript growth, compaction,
and memory updates can occur during forward. They are not Parameter updates.

For each output agent, forward does the following:

1. Create an independent repository in `statespace/`.
2. Create an empty integration commit.
3. Fetch each input commit into `refs/hytorch/inputs/<index>`.
4. Export canonical state into read-only `parameter/`.
5. Copy it into writable `workspace/` and start an episode session.
6. Let the agent merge every input, resolve conflicts, and commit each merge.
7. Let the agent transform and test the statespace.
8. Require a clean committed statespace.
9. Verify that every input is an ancestor of `HEAD`.
10. Create a HyTorch seal commit with node metadata.
11. Return the sealed repository as the output Space.
12. Retain the episode and native session tip for backward.

A forward pass can leave merged content unchanged. It must still integrate
every input ref. The agent owns statespace merge and work commits. HyTorch owns
validation and the seal commit.

Inference uses the same episode behavior. HyTorch closes the runtime and
discards the episode after forward. Inference never changes canonical
agent state.

## Loss and feedback

The core `Loss` contains one output Space and one non-empty directional
feedback string:

```python
loss = hytorch.Loss(
    output,
    feedback="Preserve the input schema and use fewer intermediate steps.",
)
```

Feedback is an imperative direction of change. It is not a score or a Git
state. An evaluator can inspect the output and synthesize this direction.

## Backward

`loss.backward()` traverses the retained graph from outputs to inputs. It
accumulates feed. The canonical model remains unchanged.

For each ready node, backward does the following:

1. Accumulate all directions from downstream consumers.
2. Resume the exact native session tip retained by forward.
3. Keep the complete `statespace/` repository read-only.
4. Keep the temporary episode `workspace/` writable.
5. Ask for one owner mutation proposal and one direction per input.
6. Accumulate the owner proposal in the Parameter's `.feed`.
7. Require one non-empty upstream direction for each input ref.
8. Accept the new opaque native session tip.
9. Validate that `statespace/` and `parameter/` did not change.
10. Record feed provenance and a stable content digest.
11. Deliver each upstream direction to its input producer.
12. Close runtime resources and discard the episode unless the graph is retained.

The operation is:

```text
Backward(feedback, X₀, ..., Xₙ, Y; W)
    -> feed(W), feedback₀, ..., feedbackₙ
```

A node with multiple consumers receives all feedback strings together. It
emits one owner proposal after all consumers finish. Multiple forward and
backward passes can add feed before one `step()`. A second backward through the
same graph requires `retain_graph=True`, as in PyTorch.

Dependency-ready episodes run in parallel. Repeated use of one Parameter
creates independent episode forks. HyTorch never merges those opaque forks.

## Optimizer

DFM means Directional Feedback Mutation.

```python
optimizer = hytorch.optim.DFM(model.parameters(), temp=0.7, max_tokens=10_000)
```

`temp` states the semantic mutation scale. A harness forwards it only when its
native runtime supports a sampling temperature. `max_tokens` limits a resumed
turn only when the runtime has a matching control.

The training lifecycle is:

```python
optimizer.zero_feed()
output = model(input_space)
loss = criterion(output)
loss.backward()
optimizer.step()
```

```text
zero_feed()  clear accumulated feed and discard an incomplete step candidate
backward()   accumulate owner feed and propagate per-input feedback
step()       reduce each Parameter once and atomically promote the generation
```

`step()` invokes the persistent owner agent once per Parameter. The owner sees
all sorted feed records and their provenance in a read-only evidence Space.
It updates a writable copy of its canonical native state. HyTorch promotes all
owner updates in one transaction. If one owner fails, HyTorch promotes none
and retains feed for retry or `zero_feed()`.

## Git semantics

Each activation Space owns an independent repository:

```text
git init                            create a node repository
git fetch                           import independent input commits
git merge --no-ff                  agent integrates input refs
git add -A && git commit           agent records statespace work
git commit --allow-empty           HyTorch seals node execution
```

One private repository owns all model Parameters:

```text
git worktree add                   create the model candidate
plain directory export            materialize one native agent state
filesystem copy                   capture the completed opaque state
git add -A && git commit           HyTorch records the candidate generation
git merge --no-ff                  step promotes the generation
```

Agents never receive model Git metadata and never create model commits. Git
gives canonical model generations stable identity, diffs, history, and atomic
promotion.

## Model state directories

Model checkpoints use directory-native PyTorch-shaped syntax:

```python
hytorch.save(model.state_dir(), path)
model.load_state_dir(hytorch.load(path))
```

`model.state_dir()` fixes one canonical model commit. It excludes optimizer
feed and temporary episodes. It includes every durable native agent
state, including sessions, transcripts, memories, compaction records, and
session artifacts that its harness stores in the Parameter.

It excludes temporary node trees, credential overlays, live processes,
sockets, locks, and other deployment state.

Loading validates the complete checkpoint before it changes any destination
Parameter. `strict=True` requires matching workspace keys. `strict=False`
permits missing and unexpected keys, but shape and module-type mismatches are
errors.

## Required invariants

1. One output feature owns one persistent agent state.
2. Every dense output agent receives every input feature.
3. A harness defines the content and format of its native state.
4. The agent never sees the private model repository.
5. Forward can modify only its episode workspace and statespace.
6. Backward can modify only its episode workspace.
7. Canonical state is immutable until `step()`.
8. `zero_feed()` clears feed and discards an incomplete step candidate.
9. Feedback is non-empty directional text.
10. Each node produces one upstream direction per input edge.
11. A node waits for all downstream feedback before backward runs once.
12. Repeated Parameter executions use isolated episode forks.
13. Harness credentials and process state never enter a Parameter.
14. HyTorch owns all model commits and promotion.
15. A StateDir identifies one committed, immutable model generation.
16. A failed load leaves every destination Parameter unchanged.
