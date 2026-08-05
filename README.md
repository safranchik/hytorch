<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/safranchik/hytorch/raw/main/assets/hytorch-logo-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/safranchik/hytorch/raw/main/assets/hytorch-logo-light.png">
  <img alt="HyTorch recursive agent meta-network" src="https://github.com/safranchik/hytorch/raw/main/assets/hytorch-logo-light.png">
</picture>

---

HyTorch is a Python library for composing and training meta-networks of coding
agents with a PyTorch-shaped API. Define dynamic graphs of cooperating agents
in Python, give them versioned directories, and improve their behavior with
plain-language feedback.

## Quickstart

### Prerequisites

- Python 3.11 or later
- Git
- Docker

### Installation

Install HyTorch with uv or pip:

```sh
# uv (recommended)
uv add hytorch

# pip
pip install hytorch
```

## Examples

### Research and synthesize an answer

This example runs three research agents in parallel. A fourth agent combines
their work into one supported answer.

A model subclasses `mn.Module`. HyTorch uses
`mn.Linear(in_features, out_features)` to describe its agent layers. The first
number is the number of inputs. The second number is the number of agents that
run and produce outputs.

```python
import hytorch
import hytorch.mn as mn


class ResearchNetwork(mn.Module):
    def __init__(self):
        super().__init__()
        self.research = mn.Linear(1, 3, bias="Find independent evidence.")
        self.synthesize = mn.Linear(3, 1, bias="Resolve conflicts and cite sources.")

    def forward(self, state, question):
        evidence = self.research(state, task=question)
        return self.synthesize(evidence, task=question)[0]


model = ResearchNetwork().to("pi")
```

The `forward()` method defines a `1 → 3 → 1` network. The first layer receives
one input and runs three agents. The second layer receives all three results
and runs one synthesis agent. Each `bias` value gives those agents their
initial instructions. Each agent owns a persistent working directory that it
can improve during training. `model.to("pi")` selects the built-in Pi agent
runtime.

#### Give the model a task

HyTorch agents work on complete directories. Each directory uses Git so every
input and output has an exact version and history. Create a small input
directory with one committed question:

```sh
mkdir research
git -C research init -b main
echo "Compare the candidate designs." > research/question.md
git -C research add .
git -C research commit -m "Add research question"
```

Wrap that directory with `hytorch.space`. A Space is the value that moves
between HyTorch layers, just as a tensor moves between PyTorch layers.

```python
state = hytorch.space("research")
```

#### Run the model

Run the model in inference mode when you only need an answer:

```python
with hytorch.inference_mode():
    output = model(state, "Which design has the strongest evidence?")

print(output.dir)  # complete output directory
print(output.commit)  # immutable Git identity
```

The result is another complete Git-backed directory. The agents decide which
files to create or change, then commit their work. `output.dir` is the output
directory. `output.commit` identifies its exact contents. Inference mode
closes the agent sessions after the result is complete.

### Improve the network with feedback

An evaluator can inspect an output and return a plain-language direction for
improvement. HyTorch sends that direction backward through the executed
network. The `DFM` optimizer manages these agent updates as complete model
generations.

```python
optimizer = hytorch.optim.DFM(
    model.parameters(),
    temp=0.7,
    max_tokens=10_000,
)

for state, target in training_data:
    optimizer.zero_feed()
    output = model(state, target.question)
    feedback = evaluate_with_tools(output, target)
    loss = hytorch.Loss(output, feedback=feedback)
    loss.backward()
    optimizer.step()
```

Useful feedback is specific and imperative:

```text
Preserve source identifiers when you combine the reports.
Test malformed inputs before you select an implementation.
Keep contradictory evidence and explain how you resolved it.
```

This lifecycle mirrors PyTorch training. `zero_feed()` clears feedback from the
previous iteration. `backward()` resumes the agents and creates candidate
workspace changes. `step()` promotes all completed changes as one new model
generation.

## PyTorch-shaped composition

HyTorch follows PyTorch syntax and ownership where a direct agent equivalent
exists:

```python
# PyTorch
tensor = torch.tensor(data, requires_grad=True)
layer = torch.nn.Linear(3, 4, bias=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# HyTorch
state = hytorch.space(directory, requires_feed=True)
layer = hytorch.mn.Linear(3, 4, bias="Synthesize the inputs.")
optimizer = hytorch.optim.DFM(model.parameters(), temp=0.7)
```

Assignment registers child Modules and Parameters. `model.parameters()` is the
normal optimizer input. `mn.Linear(in_features, out_features)` owns one
directory-backed workspace for each output agent. Its physical weight shape is
`(out_features,)`. Every output receives every input Space, so the logical
layer is dense.

The `bias` argument initializes each workspace's mutable `AGENTS.md`.
Optimization can later add instructions, code, tools, examples, and data.

## How one agent runs

Each output agent receives two sibling directory trees:

```text
node/
├── statespace/    # activation: writable during forward
└── workspace/     # parameter: writable during backward
```

During forward, the agent merges every input statespace, transforms the merged
tree, and commits the result. Its workspace is read-only.

During backward, HyTorch resumes the same agent session. The statespace is now
read-only. The agent can mutate and commit its candidate workspace. Git records
each activation, workspace diff, and promoted model generation.

## Harnesses and environment

One executed graph uses one harness:

```python
model.to("pi")
model.to(harness="pi", mtype="gpt-5.6-terra")
```

The built-in harness identities are `pi`, `codex`, and `claude-code`. Only Pi
executes in 0.1.0. Pi uses `gpt-5.6-terra` by default.

Agent variables come from `~/.config/hytorch/secrets.env`, project
`.hytorch.env`, `HYTORCH_ENV_FILE`, and exported shell variables, in increasing
precedence. HyTorch never loads an ordinary `.env` file. It does not put secret
values in prompts or Git state.

## Full example

[Terminal-Bench](example/terminal_bench/README.md) trains and evaluates a
`1 → 3 → 1` HyTorch network on Terminal-Bench 2.1 tasks.

## Project status

HyTorch 0.1.0 is the first public alpha release. Run agents in isolated
environments and review agent-created changes before production use.

Version 0.1.0 includes Spaces, Parameters, dynamic Module graphs, dense Linear
layers, directional backward feedback, atomic DFM optimizer generations, and
the Dockerized Pi harness. It does not yet implement `state_dict()`. The
`codex` and `claude-code` harnesses are reserved but unavailable.

## Resources

- [Specification](SPEC.md) — canonical behavior and invariants
- [Glossary](GLOSSARY.md) — complete PyTorch correspondence
- [Contributing](CONTRIBUTING.md) — development and test workflow
- [Security](SECURITY.md) — trust boundaries and private reporting
- [Changelog](CHANGELOG.md) — release history

## License

HyTorch is released under the [Apache License 2.0](LICENSE).
