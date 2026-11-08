```markdown
# HarmoniaCodec — GPU-Native Quantizer & SSRR Loss

A single-file Python 3 module (`harmonia_gpu.py`) providing the GPU-side
components of a neural audio codec: FSQ quantization, SSRR training loss,
a Triton kernel, and CUDA-graph-safe forward paths. It replaces `MERVQNN`
in `harmonia_neural.py` with an interface-compatible quantizer.

Two bugs in an earlier draft were found by direct execution and fixed here.
Both fixes are annotated in the file's docstring.

---

## What this is

- **`FSQQuantizer`** — Finite Scalar Quantization with per-dimension
  levels (default `[8, 5, 5, 5]`, 1000 effective entries). No codebooks,
  no usage buffers, no balance loss. Straight-through gradient through
  the rounding step.
- **`SW2VExtractor`** — log-mel feature extractor with frozen parameters,
  used as the "semantic teacher" for the SSRR loss. Replaceable with a
  real SW2V/W2V-BERT checkpoint.
- **`ssrr_loss`** — Self-Supervised Representation Reconstruction loss.
  L1 between source features and reconstruction features. Gradient flows
  through the reconstruction only.
- **`_fsq_forward_triton`** — Triton kernel applying FSQ per dimension,
  one kernel launch per dimension.

Interface compatibility with `MERVQNN`:
`forward(z, stage_mask=None)`, `decode(tokens, T, D, device, ...)`,
`balance_loss()`, `step_reset()`, `utilization()`.

---

## Install

```bash
pip install torch numpy
pip install triton     # optional; Linux only; gives the fused FSQ path
```

Triton is optional at import. If unavailable, the quantizer falls back
to a pure-PyTorch implementation with identical numerics.

---

Verify

```bash
python harmonia_gpu.py
```

Expected output on CPU:

```
device: cpu
triton available: False
NOTE: no GPU in this environment. Triton kernel path is
      UNVERIFIED here — only the pure-PyTorch path runs.
      Run this file on real GPU hardware before trusting
      the Triton path in production.
FSQ tokens: shape=(2, 16, 4), per-dim ranges verified correct
decode: shape=(2, 16, 64)
SSRR loss: 0.531842
SSRR gradient check: y.grad abs sum = 2.5800  OK
SSRR extractor params correctly frozen (no gradient)
harmonia_gpu: all checks passed (Triton path unverified on this CPU-only machine)
```

On GPU with Triton available, the same checks run plus the Triton kernel
path, and the environment note is skipped.

---

The two bugs, in one paragraph each

SSRR gradient flow. The extractor's forward was decorated
@torch.no_grad(). That decorator strips gradient tracking from the
output regardless of the input, so feat_y carried no gradient and the
entire SSRR term contributed zero to the loss — silently, no error. The
fix is to remove the decorator and freeze the extractor's parameters via
requires_grad_(False) instead. The extractor becomes differentiable
through (gradients reach whatever produced its input) while remaining
frozen as a model (its own weights never update). This is the same
pattern used by SemanticAnchor for frozen WavLM elsewhere in the
codebase.

Triton per-dimension levels. The original kernel call passed a single
scalar n_levels=max(levels) applied uniformly to every dimension.
For levels=[8, 5, 5, 5] that means three of four dimensions were
quantized against the wrong grid, silently producing incorrect codes.
The fix calls the kernel once per dimension with that dimension's own
level. n_dims is small (typically 4–8), so the extra Python loop is
cheap; the memory-bound cost is still in the per-dim kernel call.

---

What is verified and what is not

Verified by direct execution:

· The pure-PyTorch _quantize path: correct per-dim levels, correct
  integer ranges, correct dequantization round-trip.
· The SSRR gradient flow: loss.requires_grad is True, backward()
  produces a non-zero gradient on the reconstruction input, and the
  extractor's own parameters receive no gradient.
· The per-dimension level arithmetic, via a numpy simulation of the
  Triton kernel's math, bit-exact against the reference.

Not verified:

· The Triton kernel on real GPU hardware. This environment has no GPU.
  The kernel compiles in principle (it uses only tl.load, tl.store,
  tl.math.round, tl.arange, tl.program_id), but runtime behavior on
  a real device, memory layout of the scatter-write loop, and rounding
  edge cases at .5 boundaries have not been exercised.

Before trusting the Triton path in production, run this on a GPU machine:

```python
import torch
from harmonia_gpu import FSQQuantizer, FSQConfig

cfg_t = FSQConfig(dim=64, levels=(8, 5, 5, 5), use_triton=True)
cfg_e = FSQConfig(dim=64, levels=(8, 5, 5, 5), use_triton=False)

q_t = FSQQuantizer(cfg_t).cuda()
q_e = FSQQuantizer(cfg_e).cuda()
q_e.load_state_dict(q_t.state_dict())

# large-magnitude input to hit .5 grid boundaries
x = torch.randn(64, 128, 64, device="cuda") * 3

out_t = q_t(x)
out_e = q_e(x)

mismatches = (out_t["tokens"] != out_e["tokens"]).sum().item()
print(f"token mismatches: {mismatches} / {out_t['tokens'].numel()}")
```

If mismatches is 0, the Triton path is bit-exact against eager and safe
to enable. If it is non-zero, the cause is almost certainly tl.math.round
semantics (PyTorch rounds half-to-even, Triton may round half-away-from-zero),
and the fix is a single line in the kernel. Until that comparison prints 0,
keep use_triton=False for correctness.

---

Integration into harmonia_neural.py

Replace the quantizer constructor:

```python
# old
self.quant = MERVQNN(cfg.n_stages, cfg.n_experts, cfg.top_k,
                     cfg.codebook_size, cfg.codebook_dim,
                     cfg.harp_groups)

# new
from harmonia_gpu import FSQQuantizer, FSQConfig
self.quant = FSQQuantizer(FSQConfig(dim=cfg.codebook_dim))
```

Add the SSRR term to _one_step:

```python
from harmonia_gpu import SW2VExtractor, ssrr_loss

# at __init__ time
self.extractor = SW2VExtractor(sr=16000, out_dim=1024).to(device)

# inside _one_step, after l_mel is computed
l_ssrr = ssrr_loss(x, yt.squeeze(1), self.extractor, weight=1.0)
g_loss = g_loss + 0.5 * l_ssrr
```

The y_stages list returned by forward(return_stage_decodes=True) is
unaffected — FSQ has no stages, so the HARP loss will use a single-stage
list or be skipped entirely depending on your config.

---

Design notes

Why FSQ instead of RVQ. Residual Vector Quantization requires
codebooks, needs usage tracking to avoid collapse, and produces
sequential gradients across stages. FSQ replaces all of that with a
fixed element-wise grid. No codebook collapse by construction. No
balance loss. Straight-through estimator on round() is exact, not
approximate. The tradeoff is a smaller effective codebook for the same
dimensionality, which is why the NeMo Codec family uses 8 codebooks
and a low frame rate to compensate.

Why the SSRR loss. Standard waveform and mel losses reward acoustic
similarity, not intelligibility. A codec can produce a waveform that
sounds close to the source while scrambling phonemes, and mel loss will
not notice. SSRR closes that gap by comparing semantic features directly.

Why the Triton kernel is per-dimension. FSQ with non-uniform levels
is not a single-grid operation. The original mistake — a single kernel
call with a shared max(levels) — is the kind of thing that produces
plausible-looking output and never errors, so it can survive many
training runs before anyone notices the codes are wrong.

---

Files

```
harmonia_gpu.py       this module
```

Related, not included:

```
harmonia_neural.py    the training stack; MERVQNN is the class this replaces
harmonia.py           the NumPy reference codec
codec_probe.py        the benchmark harness
```

---

References

· NeMo Codec — NVIDIA. FSQ with [8, 5, 5, 5] levels, 8 codebooks,
  22.05 kHz, 86.1 fps, 6.9 kbps.
· NanoCodec / Low Frame-rate Speech Codec — NVIDIA. Variants at
  1.89 kbps (21.5 fps) and 0.6 kbps (12.5 fps). Uses WavLM as a
  training-time discriminator.
· JHCodec — MIT-licensed streaming Transformer codec. 50 Hz frame
  rate, 8-codebook RVQ, SSRR loss, Triton VQ kernel, single-GPU
  training budget.

---

License

The contents of harmonia_gpu.py are yours to license. Ideas attributed
to the references above are used for their published purpose. The FSQ
scheme is from NeMo Codec; check NVIDIA's license terms for redistribution
of anything derived from their weights. The SSRR loss and the Triton VQ
pattern are from JHCodec, MIT.

```
