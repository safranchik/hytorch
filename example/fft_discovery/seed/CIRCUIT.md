# Exact circuit format

Candidate files use `hytorch-fft-circuit-v1` JSON. The target defines `N`.
Registers `0` through `2N - 1` are complex input components in this order:

```text
x0.real, x0.imag, x1.real, x1.imag, ...
```

Each operation appends one register. The allowed operations are:

```json
{"op": "add", "a": 0, "b": 1}
{"op": "sub", "a": 0, "b": 1}
{"op": "neg", "a": 0}
{"op": "scale", "a": 0, "constant": {"basis": {"0": "1/2"}}}
```

Constants belong to `Q(ω)`, where `ω = exp(-2πi/N)`. The `basis` object gives
rational coefficients for `1, ω, ..., ω^(N/2 - 1)`. The relation
`ω^(N/2) = -1` applies. Every scale constant must be real under complex
conjugation.

The `outputs` list contains `2N` register numbers in this order:

```text
X0.real, X0.imag, X1.real, X1.imag, ...
```

Example container:

```json
{
  "format": "hytorch-fft-circuit-v1",
  "n": 8,
  "description": "Candidate description",
  "operations": [],
  "outputs": []
}
```

Run the statespace copy of the verifier before submission:

```sh
python tools/fft_verify.py \
  control/target.json submissions/current/candidate.json
```

The training process uses a separate trusted copy of the verifier. Editing the
statespace copy cannot change the external result.
