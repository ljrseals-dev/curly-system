# GPU verification runbook (codec 1.1.0)

Nothing on the GPU path has been executed yet. This is the exact procedure that moves it from UNVERIFIED to VERIFIED.

## Run (needs an NVIDIA GPU, CUDA PyTorch, Triton)

```bash
pip install -e '.[ml,test]'
python harmonia_gpu.py          # full protocol, prints VERIFIED / SKIPPED per check
python -m pytest -q tests       # same checks, plus NumPy golden vectors
```

Success on a GPU box prints, in order:

```
[VERIFIED] Part 2: per-dimension index ranges
[VERIFIED] eager path == NumPy reference on 4162 golden vectors (incl. exact ties), pack/unpack round-trip
[VERIFIED] decode(tokens) == from_fsq(dequant(tokens))
[VERIFIED] Part 1b: gradient reaches to_fsq, from_fsq and the encoder input
[VERIFIED] Part 1: SSRR gradient flow, extractor frozen
[VERIFIED] Part 3: Triton path bit-exact against eager (ties, fp16, non-contiguous, full forward)
[VERIFIED] CUDA graph replay ok (...)
harmonia_gpu: all checks passed
```

Any `[SKIPPED]` line means that path is still UNVERIFIED. Record in `docs/VALIDATION.md`: GPU model, driver, CUDA, torch and triton versions, and the date.

## If Part 3 fails

The assertion names the input set. Ties failing means rounding semantics; non-tie failures mean indexing (`off % dims`, contiguity). Keep `use_triton=False` until it prints zero mismatches. The eager path is the reference and is always safe.

## What is deliberately NOT checked

- Perceptual quality, real SW2V checkpoint behaviour (fallback is log-mel).
- NaN/Inf inputs: neither path defines them; guard upstream (Voodoo finite guard).
- Speed. No benchmark has been run.
