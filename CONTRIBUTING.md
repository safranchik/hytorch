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
commit credentials, `.hytorch.env`, or generated model workspaces. Durable
agent sessions belong inside generated model state. Do not add them to the
source tree unless they are explicit test fixtures.

Use short commit subjects in the imperative form. Explain design decisions and
test results in the pull request description.

## Release process

HyTorch publishes to PyPI through GitHub Trusted Publishing. A push to `main`
runs CI but does not publish a package. Publishing starts only when a maintainer
publishes a GitHub Release.

For the first release, configure a pending publisher in the PyPI account:

- Project name: `hytorch`
- Owner: `safranchik`
- Repository: `hytorch`
- Workflow: `publish.yml`
- Environment: `pypi`

For each release:

1. Update the version in `pyproject.toml` and `hytorch/__init__.py`.
2. Add the dated release entry to `CHANGELOG.md`.
3. Run the complete release checks:

   ```sh
   uv lock --check
   uv run ruff check .
   uv run ruff format --check .
   uv run pytest -q
   uv build
   uvx --from twine twine check dist/*
   ```

4. Push the release commit to `main` and wait for all CI jobs to pass.
5. Create and push an annotated version tag:

   ```sh
   git tag -a v0.1.0 -m "HyTorch 0.1.0"
   git push origin v0.1.0
   ```

6. Publish the matching GitHub Release:

   ```sh
   gh release create v0.1.0 \
     --title "HyTorch 0.1.0" \
     --generate-notes
   ```

The `Publish` workflow builds the source distribution and wheel in an
unprivileged job. A separate job obtains a short-lived PyPI credential through
OpenID Connect and uploads the artifacts. The workflow stores no PyPI token.

After publication, confirm that the version appears on PyPI and install it in a
clean environment. PyPI release files are immutable. If an uploaded release is
wrong, fix the problem and publish a new version. Do not replace an existing
version or move its Git tag.
