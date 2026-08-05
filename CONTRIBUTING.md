# Contributing to HyTorch

HyTorch is an early-stage research library. Contributions that make its core
model smaller, clearer, safer, or easier to test are welcome.

## Before you start

Read [SPEC.md](SPEC.md). It defines the required behavior. Open an issue before
you make a large API or architecture change. This helps maintain one coherent
PyTorch-shaped design.

HyTorch follows PyTorch syntax and ownership rules when a direct text-agent
equivalent exists. Link to the relevant current PyTorch documentation or source
when you propose a new public API.

## Development setup

Install [uv](https://docs.astral.sh/uv/), clone the repository, and run:

```sh
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

The default test suite is offline. It does not call an agent provider.

## Real Pi integration test

The real Pi test uses Docker and can consume paid model tokens. Run it only
when you intend to make an external model call:

```sh
HYTORCH_PI_TEST=1 uv run pytest tests/test_pi.py -v -s
```

Pi uses the operator's Codex login by default. To use an OpenAI API key, export
`OPENAI_API_KEY` and set `HYTORCH_PI_PROVIDER=openai`.

## Pull requests

Keep each pull request focused. Add tests for behavior changes. Update
`README.md`, `SPEC.md`, and `GLOSSARY.md` when a public concept changes. Do not
commit credentials, `.hytorch.env`, generated model workspaces, or agent
session data.

Use short commit subjects in the imperative form. Explain design decisions and
test results in the pull request description.
