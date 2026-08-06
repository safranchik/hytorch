# Arithmetic cost model

The machine target in `control/target.json` is authoritative.

The verifier counts these live real-arithmetic operations:

- Real addition: 1.
- Real subtraction: 1.
- Multiplication by a nontrivial real constant: 1.

Real negation and multiplication by `1` or `-1` are free. Fused operations are
not available. Circuit depth is a secondary metric. Dead operations do not
contribute to the score.

The transform is an unscaled complex-input DFT with a negative exponential
sign. Inputs and outputs use interleaved real and imaginary components.

Do not compare counts from different transform conventions or cost models.
