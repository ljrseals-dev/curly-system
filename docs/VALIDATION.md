# Forensic Validation

| Test | Status | Meaning |
|---|---|---|
| Import/package | IMPLEMENTED | Package imports from clean source tree |
| FSQ cardinality | IMPLEMENTED | `[8,5,5,5]` = 1000 combinations |
| Per-dimension FSQ | IMPLEMENTED | Each dimension uses its own level count |
| Deterministic compiler | IMPLEMENTED | Same prompt + seed gives same patch |
| Finite output guard | IMPLEMENTED | NaN/Inf input is sanitized before DSP |
| Sample ceiling | IMPLEMENTED | Output is bounded at configured sample ceiling |
| Spectral peak interpolation | IMPLEMENTED | Quadratic interpolation is used around detected peaks |
| FSQ golden vectors (NumPy) | IMPLEMENTED | 4162 vectors incl. exact rounding ties; `tests/test_fsq_golden.py` |
| Kernel rounding algorithm (NumPy mirror) | IMPLEMENTED | Half-to-even via trunc+parity equals reference on 1M random + all ties. Verifies the algorithm, not Triton |
| Torch eager path, STE gradients, SSRR, pack/unpack | UNVERIFIED | Written in 1.1.0; no PyTorch in the authoring environment. Run `python harmonia_gpu.py` |
| CUDA/Triton | UNVERIFIED | Needs a GPU. Procedure: `docs/GPU_VERIFICATION.md` |
| True-peak compliance | REQUIRES-HOST-TESTING | Sample ceiling is not a certified dBTP measurement |
| DAW/plugin validation | BLOCKED | No native plugin target is included in this Python build |
