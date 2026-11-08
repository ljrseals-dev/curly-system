# Curly Octo Engine — LewSpEW Crypt 1.0.0

A complete reference-engine rebuild of the supplied `curly-octo-engine`, using the LewSpEW Crypt architecture as the organizing layer.

## What changed

The original delivery was essentially a single GPU codec module. Crypt turns it into a coherent engine:

- **Instrument IR** — stable representation for identity, archetype, spectral measurements and articulation.
- **Deterministic compiler** — `B + Tg + Tp + Tm + C` pipeline from prompt/IR to a validated synthesis graph.
- **Spectral analyzer** — high-resolution FFT, parabolic peak interpolation and harmonic-deviation measurements.
- **Voodoo Water DSP** — bounded high-pass, nonlinear character, controlled compression and sample ceiling.
- **FSQ codec** — corrected per-dimension `[8,5,5,5]` quantization with exactly 1000 possible scalar-grid combinations.
- **Crypt manifest** — SHA-256 file inventory for reproducible delivery auditing.
- **CLI + tests + packaging** — clean Python package rather than a loose script.

The earlier LewSpEW architecture separates source formats from the normalized IR, generated DSP and runtime. This build follows that separation rather than embedding an external synth implementation directly. fileciteturn1file6L470-L500

## Install

```bash
python -m pip install -e .
```

Optional GPU/ML layer:

```bash
python -m pip install -e '.[ml]'
```

## Run tests

```bash
python -m pytest -q
```

## CLI

```bash
lewspew-crypt compile "modern rock and roll flute" --seed 42
```

The compiler is deterministic for a fixed prompt and seed.

## Python

```python
import numpy as np
from lewspew_crypt import CryptEngine

engine = CryptEngine()
t = np.arange(48000) / 48000.0
x = 0.8*np.sin(2*np.pi*440*t)
y, report = engine.process_audio(x)
print(report)
patch = engine.compile_prompt("modern rock and roll flute", seed=42)
print(patch.dumps())
```

## FSQ correction

The supplied GPU module claimed per-dimension levels but its Triton kernel quantized every element using the maximum level. Crypt's reference FSQ implementation applies the actual level for each dimension and exposes mixed-radix code packing/unpacking. This is the authoritative CPU reference for future CUDA/Triton implementations.

## GPU boundary

The original GPU module is retained as a reference artifact. It is not silently presented as production-certified GPU code. A future CUDA/Triton backend must pass the CPU reference vectors, including exact per-dimension indices, code packing, decode, finite safety and deterministic replay.

## Validation status

See `docs/VALIDATION.md`. In particular, a sample-domain ceiling is **not** a certified true-peak/dBTP limit; external reconstructed-peak testing remains required. This distinction is consistent with the project's existing validation discipline. fileciteturn1file1L99-L106

The runtime design also keeps the real-time boundary separate from analysis/compiler work, matching the LewSpEW architecture direction. fileciteturn1file0L60-L68

## License

MIT. See `LICENSE`.
