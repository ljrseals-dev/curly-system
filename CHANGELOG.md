# Changelog

## 1.1.0 — 2026-11-08
- GPU backend rewritten (`harmonia_gpu.py`); 1.0.0 root file kept at `Source/crypt-1.0.0/`.
- Triton kernel: explicit round-half-to-even (tl.math.round rounds ties away from zero); emits indices only; dequant shared with eager path.
- STE moved to the grid: gradient now reaches `to_fsq`, `from_fsq` and the encoder. **Behaviour change** for training; token output unchanged.
- Float32 grid math for fp16/bf16 inputs; no host->device copies inside graph capture.
- `GraphSafeQuantizer`: side-stream warmup, replay only with grad disabled and matching shape/dtype/device, cloned outputs.
- Restored SSRR (`SW2VExtractor`, `ssrr_loss`) dropped in the 1.0.0 rebuild.
- Added torch mixed-radix `pack_codes` / `unpack_codes`, golden vectors, `scripts/make_golden.py`, tests, `docs/GPU_VERIFICATION.md`, `docs/PAPER02_ERRATA.md`.
- GPU/Triton/graph paths remain UNVERIFIED (no GPU available).

## 1.0.0 — 2026-09-16
- Rebuilt `curly-octo-engine` as the LewSpEW Crypt engine.
- Added deterministic Instrument IR and synthesis graph compiler.
- Added spectral analysis with parabolic peak interpolation.
- Added Voodoo Water-style bounded nonlinear DSP with finite-value guard.
- Added corrected per-dimension FSQ reference implementation with `[8,5,5,5]` and 1000-code cardinality.
- Added SHA-256 Crypt manifest support.
- Added CLI, packaging metadata, examples, and regression tests.
- Kept optional ML/GPU dependencies out of the base install.
