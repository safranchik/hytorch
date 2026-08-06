# HyTorch Meta-Network Specification

## Purpose

HyTorch applies PyTorch-shaped ownership and training to Git-backed agent
systems.

```text
Tensor          -> statespace Space
Parameter       -> trainable workspace
neuron          -> agent
autograd graph  -> retained execution graph
gradient        -> directional feedback
parameter delta -> candidate workspace mutation
```

Calls in `forward()` define the runtime graph. `model.parameters()` supplies
the optimizer. `loss.backward()` creates candidate workspace updates while it
propagates feedback. `optimizer.step()` promotes the completed candidate model
branch.

## Spaces and workspaces

`hytorch.space(data, *, mtype=None, harness=None, requires_feed=False)` creates
a `Space` from one Git-backed directory. A list of directories creates an
ordered `SpaceBatch`.

A Space is an activation state. It enters a node as `X`. The node transforms
it into output statespace `Y`.

An `mn.Parameter` is a persistent workspace. It enters a node as `W`. Forward
cannot change it.

```text
Yᵢ = Agentᵢ(X₀, ..., Xₙ; Wᵢ)
```

Each model owns one Git repository under:

```text
hytorch/workspaces/<temporary-id>/
├── .git/
├── MODEL.json
└── layers/
    └── <qualified-layer>/
        └── <agent>/
            ├── AGENTS.md
            └── ...
```

Each numbered directory is one complete workspace. `AGENTS.md` is mutable
workspace state. The `bias` constructor value initializes it. Optimization can
change it and can create, update, move, or delete any other workspace file.

## Linear

One output feature is one agent. `mn.Linear(3, 4)` runs four agents in
parallel. Each output agent receives all three input Spaces and owns one
monolithic workspace. `layer.weight.shape` is `(out_features,)`.

The logical graph is dense. A workspace can implement nonlinear behavior
across all inputs.

`mn.Linear.reset_parameters()` delegates to `hytorch.mn.init`. Each initial
`AGENTS.md` contains the mutable bias and seeded input priors.
`hytorch.manual_seed(seed)` makes prior initialization reproducible.

## Forward

### Runtime placement

One executed graph uses one registered harness. Mixed per-layer harnesses are
invalid. `model.to(harness)` moves the complete model before a new forward
pass. `model.to(mtype=...)` changes only the default model type. Omitted values
preserve the existing setting. Forward and its matching backward retain the
same harness session.

Docker configuration is external to the model API. HyTorch uses the active
Docker context and standard `DOCKER_CONTEXT`, `DOCKER_HOST`, `DOCKER_CONFIG`,
and TLS settings. `HYTORCH_PI_IMAGE` overrides the packaged Pi image. HyTorch
resolves the chosen tag to one image ID when the harness first executes.
Local contexts use bind mounts. SSH and TCP contexts use temporary Docker
volumes to upload node state and download the writable result. Read-only
permissions remain volume-mount properties. HyTorch removes temporary remote
volumes after execution. Remote Pi execution requires `OPENAI_API_KEY` because
the local Pi OAuth directory is not available to the remote daemon.

Agent variables come from the optional global `~/.config/hytorch/secrets.env`,
project `.hytorch.env`, and `HYTORCH_ENV_FILE`, in that order. Exported shell
values override declared keys. HyTorch forwards known provider keys from the
shell without a file. It ignores ordinary `.env`. It supplies merged values
through a temporary mode-0600 Docker env file and deletes it after execution.
Secret values do not enter model state or Git.

Each output agent receives this node root:

```text
node/
├── statespace/     # self-contained Git state; read-write
└── workspace/      # sparse global model checkout; read-only
```

For each output agent, forward does the following:

1. Initialize a new, independent repository in `statespace/`.
2. Create an empty integration commit.
3. Fetch each input commit from its own Space repository into
   `refs/hytorch/inputs/<index>`.
4. Materialize the global model history as a read-only sparse checkout of the
   current node workspace.
5. Start a persistent harness session.
6. Let the agent inspect its workspace and choose an input merge order.
7. Let the agent merge every input, resolve conflicts, and commit each merge.
8. Let the agent transform and test the statespace.
9. Require the agent to commit all final statespace changes.
10. Treat the end of the harness turn as its completion signal.
11. Validate a clean repository and verify every input is an ancestor of `HEAD`.
12. Create an empty HyTorch seal commit with node metadata.
13. Return the sealed local repository as the output Space.
14. Retain the harness session for one backward pass.

A forward pass can leave the merged content unchanged. It must still integrate
every input ref and leave a clean committed state. The agent owns its merge and
work commits. HyTorch owns validation and the seal commit.

## Loss and feedback

The core `Loss` contains only one output Space and one non-empty directional
feedback string:

```python
loss = hytorch.Loss(
    output,
    feedback="Preserve the input schema and use fewer intermediate steps.",
)
```

Feedback is an imperative direction of change. It is not a score, objective,
metric, observation bundle, or Git-backed Space. A loss function can invoke an
evaluator agent, inspect the output statespace, use tools, and synthesize this
direction.

## Backward

`loss.backward()` traverses the retained graph from outputs to inputs. It uses
one model-wide candidate branch. The canonical model remains unchanged.

For each ready node, backward does the following:

1. Accumulate all directional feedback from downstream consumers.
2. Resume the exact harness session saved during forward.
3. Keep the complete `statespace/` repository read-only.
4. Materialize the latest global candidate as a writable sparse `workspace/`
   checkout. It retains model history from initialization but checks out only
   the selected node workspace.
5. Let the agent update the workspace, or leave it unchanged.
6. Require the agent to commit all workspace changes.
7. Require one non-empty directional feedback string for every input ref in
   the final JSON response.
8. Treat the end of the resumed harness turn as the completion signal.
9. Validate both Git repositories and the structured response.
10. Run dependency-ready nodes with distinct workspace paths in parallel from
    one candidate commit.
11. Merge their validated commits into the global candidate branch.
12. Deliver each feedback string to its corresponding input producer.
13. Close and delete the saved agent session.

The local operation is:

```text
UpdateAndBackward(feedback, X₀, ..., Xₙ, Y; W)
    -> W′, feedback₀, ..., feedbackₙ
```

A node that has multiple downstream consumers receives their feedback strings
separately. It updates its workspace once after all consumers finish. This is
the text-agent equivalent of gradient accumulation.

One Git repository owns all model workspaces. Each agent commit changes only
its registered workspace path. The global history records the exact trajectory
of every weight and every complete model generation. `step()` can promote all
workspace changes atomically.

Backward parallelism follows graph dependencies. Nodes in one ready frontier
can use separate Docker containers and commit from the same candidate base.
HyTorch merges commits for distinct workspace paths. It serializes executions
that refer to the same workspace path. An earlier layer waits until it receives
all downstream feedback.

Every forward session is single-use. HyTorch does not retain a session after
backward. A new model generation requires a new forward pass.

## Optimizer

DFM means Directional Feedback Mutation.

```python
optimizer = hytorch.optim.DFM(model.parameters(), temp=0.7, max_tokens=10_000)
```

`temp` controls the semantic scale and sampling temperature of backward
workspace changes. `max_tokens` limits the resumed agent turn.

The training lifecycle is:

```python
optimizer.zero_feed()
output = model(input_space)
loss = criterion(output)
loss.backward()
optimizer.step()
```

The operations have separate responsibilities:

```text
zero_feed()  clear old feedback and discard an unpromoted candidate
backward()   update candidate workspaces and propagate feedback
step()       atomically promote the completed candidate branch
```

`step()` does not invoke agents. It makes the already committed candidate
workspaces canonical. If backward fails, HyTorch discards the candidate branch
and leaves the canonical model unchanged.

## Git semantics

Each activation Space owns an independent repository:

```text
git init                            create a self-contained node repository
git fetch                           import each independent input commit
git merge --no-ff                  agent integrates input refs
git add -A && git commit           agent records statespace work
git commit --allow-empty           HyTorch seals the node execution
```

One global repository owns all model workspaces:

```text
git worktree add                   create a backward candidate checkout
git init && git commit             materialize the agent workspace repository
git add -A && git commit           agent records workspace work
git add -A && git commit           canonicalize W′ on the candidate branch
git merge --no-ff                  promote the completed model generation
```

Feedback is transient text and does not use a third Git repository. Git gives
Spaces and model generations stable identity, ancestry, diffs, audit history,
and atomic promotion.

## Model state directories

HyTorch serializes model state as a directory because each Parameter element is
already a complete directory. The public form follows PyTorch checkpoint
syntax:

```python
hytorch.save(model.state_dir(), path)
model.load_state_dir(hytorch.load(path))
```

`model.state_dir()` returns a `StateDir` fixed to the canonical model commit at
the time of the call. It does not include an unpromoted DFM candidate.
`hytorch.save()` creates a self-contained Git directory at that exact commit.
The saved state contains `MODEL.json`, all registered workspace directories,
and the canonical model history. It excludes feedback, active harness sessions,
temporary node trees, and optimizer candidates.

`hytorch.load()` validates the repository root, committed `MODEL.json`, format,
and workspace paths. It returns a `StateDir`; it does not modify a model.
`model.load_state_dir()` copies matching workspaces into an initialized model
and records one canonical load commit. The default `strict=True` requires the
saved and destination workspace keys to match exactly. `strict=False` permits
missing and unexpected keys, but shape and module-type mismatches remain
errors. The return value reports missing and unexpected keys in the same style
as PyTorch's `load_state_dict()`.

State capture and load require a clean canonical model worktree. Loading while
an optimizer candidate is pending is an error. Validation must finish before
HyTorch changes any destination workspace.

## Required invariants

1. One output feature executes one agent.
2. Every dense output agent receives every input feature.
3. Each numbered model directory is one complete trainable workspace.
4. `AGENTS.md` is mutable workspace state initialized by `bias`.
5. Forward can modify only `statespace/`.
6. Backward can modify only the candidate `workspace/`.
7. Feedback is non-empty directional text.
8. Each node produces one feedback string per input edge.
9. A node waits for all downstream feedback before it runs backward once.
10. Agents own local work commits. HyTorch owns seal and canonical commits.
11. Backward closes each resumed forward session.
12. Only `optimizer.step()` promotes candidate workspace commits.
13. `zero_feed()` never changes committed canonical workspace history.
14. A StateDir identifies one committed, immutable model generation.
15. Saved model state never includes transient feedback, sessions, or candidates.
16. A failed state load leaves every destination workspace unchanged.
