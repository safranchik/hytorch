"""Initialization functions for workspace Parameters, mirroring torch.nn.init."""

from __future__ import annotations

from collections.abc import Sequence

from .._random import shuffled
from ..parameter import Parameter

DEFAULT_PRIORS = [
    "Approach this from a different angle.",
    "Track uncertainty explicitly.",
    "Start from concrete evidence.",
    "Challenge the first plausible interpretation.",
    "Look for edge cases and exceptions.",
    "Surface hidden assumptions.",
    "Check for internal consistency.",
    "Separate observations from inferences.",
    "Consider more than one interpretation.",
    "Identify important missing information.",
    "Preserve details that may matter later.",
    "Prioritize what is relevant.",
    "Verify names, values, dates, and identifiers carefully.",
    "Do not mistake correlation for evidence.",
    "Look for independent corroboration.",
    "Search for disconfirming evidence.",
    "Pay attention to context and scope.",
    "Summarize the distinct contribution before combining conclusions.",
    "Call out conflicts instead of silently resolving them.",
    "Try the simplest viable interpretation first.",
]


def uniform_(parameter: Parameter, priors: Sequence[str] = DEFAULT_PRIORS) -> Parameter:
    """Seed every workspace with randomly ordered input priors.

    Values are sampled without replacement until the pool is exhausted, then
    the pool is reshuffled. The trailing underscore follows torch.nn.init's
    convention for in-place initialization.
    """
    pool = _validate_priors(priors)
    choices: list[str] = []
    total = parameter.shape[0] * parameter.input_features
    while len(choices) < total:
        choices.extend(shuffled(pool))
    cursor = 0
    for view in parameter.views():
        sections = ["# Initial policy\n"]
        for input_index in range(parameter.input_features):
            sections.extend([f"## Input {input_index}\n", choices[cursor], "\n"])
            cursor += 1
        view._set_text("\n".join(sections))
    if parameter._store is not None:
        parameter._store.commit(f"hytorch: initialize {parameter.name}")
    return parameter


def constant_(parameter: Parameter, value: str) -> Parameter:
    """Set every workspace's root ``AGENTS.md`` to the same text."""
    if not isinstance(value, str):
        raise TypeError("hytorch.mn.init.constant_: value must be a string")
    for view in parameter.views():
        view._set_text(value)
    if parameter._store is not None:
        parameter._store.commit(f"hytorch: initialize {parameter.name}")
    return parameter


def _validate_priors(priors: Sequence[str]) -> list[str]:
    if isinstance(priors, (str, bytes)) or not priors:
        raise ValueError(
            "hytorch.mn.init.uniform_: priors must be a non-empty sequence"
        )
    values = []
    for prior in priors:
        if not isinstance(prior, str) or not prior.strip():
            raise ValueError(
                "hytorch.mn.init.uniform_: every prior must be non-empty text"
            )
        values.append(prior.strip())
    return values


__all__ = ["DEFAULT_PRIORS", "constant_", "uniform_"]
