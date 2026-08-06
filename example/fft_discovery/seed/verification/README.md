# Verification research

Store independent checks, proof notes, and verifier audits here.

The executable verifier is `tools/fft_verify.py`. The training controller uses
a separate trusted package copy. It proves exact linear equivalence and reports
the live operation count. Floating-point tests are useful diagnostics, but they
are not final correctness evidence.
