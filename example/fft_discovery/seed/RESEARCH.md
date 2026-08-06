# Research objective

Find an exact FFT straight-line program that improves a published arithmetic
operation count for one fixed transform.

## Phase 1: target audit

Audit the prepared transform size and computational model. Establish that the
published incumbent uses the same model. Preserve precise primary-source
locations.

The initial target should be small enough for repeated local search. Prefer a
case with a clear gap between a published construction and a known lower bound.

## Phase 2: search

Implement multiple candidate generators. Preserve every verified improvement
and the complete evidence needed to reproduce it.

## Phase 3: validation

Verify the final circuit independently. Audit the operation count, novelty,
numerical stability, and reproducibility. Use measured runtime only for a
separate implementation-performance claim.
