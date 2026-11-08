#!/usr/bin/env python3
"""LewSpEW Crypt GPU codec backend, v1.1.0.

FSQ quantizer, SSRR loss, Triton index kernel, CUDA-graph wrapper, and the
Paper 02 verification protocol, in one file.  The NumPy implementation in
``lewspew_crypt/codec/fsq.py`` is the canonical numerical reference; this
module must agree with it bit-for-bit on token indices.

STATUS (read this before trusting the Triton / graph paths)
    CPU/eager path .... written against the reference; run `python harmonia_gpu.py`
    Triton kernel ..... UNVERIFIED until `python harmonia_gpu.py` prints
                        "Triton path bit-exact" on a CUDA machine
    CUDA graph ........ UNVERIFIED until the same run prints "graph replay ok"
    Nothing in this file was executed on a GPU when it was written.

CHANGES FROM 1.0.0 (each is covered by a check in verify())
    1. Rounding.  tl.math.round is round-half-away-from-zero; torch.round and
       np.rint are round-half-to-even.  They disagree on exact ties, e.g. a
       5-level dimension at u = +-0.25 / +-0.75.  The kernel now implements
       half-to-even explicitly (int truncation + parity), using only basic ops
       that exist in every Triton version.
    2. Dequantization is no longer done in the kernel.  The kernel emits int32
       indices only; q = idx/(L-1)*2-1 is computed by one torch function shared
       by forward() and decode().  This removes Triton's approximate fp32
       divide as a source of mismatch and makes decode(tokens) == forward's q.
    3. Straight-through estimator.  1.0.0 returned z + (zq - z).detach(), which
       is a constant offset: from_fsq received no gradient and to_fsq received
       gradient only from the commit term.  forward() now applies the STE at
       the grid (q.detach() + (t - t.detach())) so gradient reaches both
       projections and the encoder.  Forward values are unchanged.
    4. All grid math runs in float32 regardless of input dtype (fp16/bf16 safe).
    5. GraphSafeQuantizer: side-stream warmup, replay only when grad is
       disabled and shape/dtype/device match, outputs cloned so the next replay
       cannot overwrite what the caller holds.
    6. SSRR (SW2VExtractor, ssrr_loss) restored from Source/v2 -- the 1.0.0
       rebuild had dropped it.  No @torch.no_grad() on the extractor forward;
       parameters are frozen instead.
    7. Torch mixed-radix pack/unpack matching FSQ.encode_code/decode_code.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

CODEC_VERSION = "1.1.0"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "tests", "golden", "fsq_vectors.json")


# ══════════════════════════════════════════════════════════════════════
# 1. TRITON INDEX KERNEL
# ══════════════════════════════════════════════════════════════════════
if HAVE_TORCH and HAVE_TRITON:

    @triton.jit
    def _fsq_idx_kernel(x_ptr, idx_ptr, levels_ptr, n,
                        dims: tl.constexpr, BLOCK: tl.constexpr):
        """x: contiguous fp32 grid-domain values, last axis = `dims`.
        Writes int32 indices.  Per-element level via (flat index % dims), so
        every dimension uses its OWN level count (Paper 02 Bug 2)."""
        pid = tl.program_id(0)
        off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        x = tl.load(x_ptr + off, mask=mask, other=0.0)
        lv = tl.load(levels_ptr + (off % dims), mask=mask, other=2.0)
        x = tl.maximum(tl.minimum(x, 1.0), -1.0)
        # Same op order as the eager path.  No add follows a multiply, so
        # there is no FMA contraction, and *0.5 is exact, so any
        # reassociation would still give identical bits.
        s = (x + 1.0) * 0.5 * (lv - 1.0)            # s >= 0
        f = s.to(tl.int32)                          # trunc == floor for s >= 0
        d = s - f.to(tl.float32)
        # round-half-to-even: up if frac > .5, or frac == .5 and f is odd
        up = (d > 0.5) | ((d == 0.5) & ((f & 1) == 1))
        tl.store(idx_ptr + off, f + up.to(tl.int32), mask=mask)

    def fsq_triton_indices(t, levels):
        """t: CUDA tensor (..., len(levels)), already in the grid domain."""
        assert t.is_cuda, "Triton FSQ requires CUDA"
        n_dims = levels.numel() if isinstance(levels, torch.Tensor) else len(levels)
        assert t.shape[-1] == n_dims, "last dim must equal len(levels)"
        t32 = t.detach().to(torch.float32).contiguous()
        idx = torch.empty(t32.shape, dtype=torch.int32, device=t32.device)
        lv = _levels_tensor(levels, t32.device)  # defined in section 2; resolved at call time
        n, block = t32.numel(), 1024
        _fsq_idx_kernel[(triton.cdiv(n, block),)](
            t32, idx, lv, n, dims=int(lv.numel()), BLOCK=block)
        return idx
else:
    def fsq_triton_indices(t, levels):  # pragma: no cover
        raise RuntimeError("Triton (and CUDA PyTorch) not available")


# ══════════════════════════════════════════════════════════════════════
# 2. SHARED GRID MATH (eager indices, dequant, mixed-radix packing)
# ══════════════════════════════════════════════════════════════════════
if HAVE_TORCH:

    _LV_CACHE = {}

    def _levels_tensor(levels, device):
        """float32 level tensor on `device`.  A tensor is passed through, and
        tuples are cached, so nothing here does a host->device copy after the
        first call -- required inside CUDA-graph capture."""
        if isinstance(levels, torch.Tensor):
            return levels.to(device=device, dtype=torch.float32)
        key = (str(device), tuple(int(v) for v in levels))
        lv = _LV_CACHE.get(key)
        if lv is None:
            lv = _LV_CACHE[key] = torch.as_tensor(
                [float(v) for v in key[1]], dtype=torch.float32, device=device)
        return lv

    def fsq_indices_eager(t, levels):
        """Reference-equivalent indices for grid-domain input t (float32 math)."""
        lv = _levels_tensor(levels, t.device)
        t = t.detach().to(torch.float32).clamp(-1.0, 1.0)
        s = (t + 1.0) * 0.5 * (lv - 1.0)
        idx = torch.round(s)                         # half-to-even
        idx = torch.minimum(idx.clamp_min(0.0), lv - 1.0)
        return idx.to(torch.int32)

    def fsq_dequant(idx, levels):
        """The ONE dequantizer: used by forward(), decode() and the tests."""
        lv = _levels_tensor(levels, idx.device)
        return idx.to(torch.float32) / (lv - 1.0) * 2.0 - 1.0

    def pack_codes(idx, levels):
        """(..., D) int indices -> (...) int64 code; dim 0 least significant."""
        code = torch.zeros(idx.shape[:-1], dtype=torch.int64, device=idx.device)
        mul = 1
        for d, l in enumerate(levels):
            code = code + idx[..., d].to(torch.int64) * mul
            mul *= int(l)
        return code

    def unpack_codes(code, levels):
        rem = code.to(torch.int64).clone()
        out = torch.empty(tuple(code.shape) + (len(levels),),
                          dtype=torch.int32, device=code.device)
        for d, l in enumerate(levels):
            out[..., d] = (rem % int(l)).to(torch.int32)
            rem = torch.div(rem, int(l), rounding_mode="floor")
        return out


# ══════════════════════════════════════════════════════════════════════
# 3. FSQ QUANTIZER
# ══════════════════════════════════════════════════════════════════════
if HAVE_TORCH:

    @dataclass(frozen=True)
    class FSQConfig:
        levels: tuple = (8, 5, 5, 5)
        dim: int = 64
        use_triton: bool = True

    class FSQQuantizer(nn.Module):
        """Fixed scalar-grid quantizer with true per-dimension levels.

        Interface-compatible with MERVQNN: forward(z, stage_mask=None),
        decode(tokens, T, D, device, ...), balance_loss(), step_reset(),
        utilization().
        """

        def __init__(self, cfg: FSQConfig = FSQConfig()):
            super().__init__()
            self.cfg = cfg
            self.levels = tuple(int(v) for v in cfg.levels)
            self.register_buffer(
                "level_tensor",
                torch.tensor(self.levels, dtype=torch.float32), persistent=False)
            self.to_fsq = nn.Linear(cfg.dim, len(self.levels), bias=False)
            self.from_fsq = nn.Linear(len(self.levels), cfg.dim, bias=False)
            nn.init.orthogonal_(self.to_fsq.weight)
            nn.init.orthogonal_(self.from_fsq.weight)

        def _indices(self, t):
            lv = self.level_tensor
            if self.cfg.use_triton and HAVE_TRITON and t.is_cuda:
                return fsq_triton_indices(t, lv)
            return fsq_indices_eager(t, lv)

        def _quantize(self, x):
            """Back-compat (dequant, idx) from pre-tanh x; eager path, no grad."""
            t = torch.tanh(x.detach().to(torch.float32))
            idx = fsq_indices_eager(t, self.level_tensor)
            return fsq_dequant(idx, self.level_tensor), idx

        def forward(self, z, stage_mask=None):
            x = self.to_fsq(z).to(torch.float32)
            t = torch.tanh(x)                          # grid domain, [-1, 1]
            idx = self._indices(t)
            q = fsq_dequant(idx, self.level_tensor)    # no grad (from ints)
            # STE at the grid: value == q exactly, gradient flows to t.
            q_ste = q + (t - t.detach())
            zq = self.from_fsq(q_ste.to(z.dtype))
            return {
                "tokens": idx,
                "commit": F.mse_loss(t, q.detach()),
                "probs": torch.sigmoid(x.mean(dim=-1, keepdim=True)),
                "zq": zq,
            }

        def decode(self, tokens, T=None, D=None, device=None,
                   stage_mask=None, batch_size=None):
            idx = tokens["tokens"] if isinstance(tokens, dict) else tokens
            idx = idx.to(device or self.level_tensor.device)
            q = fsq_dequant(idx, self.level_tensor).to(self.from_fsq.weight.dtype)
            return self.from_fsq(q)

        def balance_loss(self):
            return self.level_tensor.new_zeros(())

        def step_reset(self):
            pass

        def utilization(self):
            return 1.0

    # ══════════════════════════════════════════════════════════════════
    # 4. CUDA-GRAPH WRAPPER (inference only)
    # ══════════════════════════════════════════════════════════════════
    class GraphSafeQuantizer(nn.Module):
        """Replays a captured FSQ forward when, and only when, it is safe.

        Every branch is decided by shape / dtype / device / grad-mode, never by
        tensor values.  Training (grad enabled) always runs eager so gradients
        exist.  Outputs are cloned: the captured buffers are overwritten by the
        next replay.
        """

        def __init__(self, fsq: "FSQQuantizer"):
            super().__init__()
            self.fsq = fsq
            self.graph = None
            self.replays = 0
            self._in = None
            self._out = None

        def capture(self, B, T):
            w = self.fsq.to_fsq.weight
            if not w.is_cuda:
                raise RuntimeError("CUDA required for graph capture")
            was_training = self.fsq.training
            self.fsq.eval()
            with torch.no_grad():
                self._in = torch.zeros(B, T, self.fsq.cfg.dim,
                                       device=w.device, dtype=w.dtype)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):                  # warmup + Triton compile
                        self.fsq(self._in)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._out = self.fsq(self._in)
                self.graph = g
            self.fsq.train(was_training)

        def forward(self, z):
            ok = (self.graph is not None
                  and not torch.is_grad_enabled()
                  and tuple(z.shape) == tuple(self._in.shape)
                  and z.dtype == self._in.dtype
                  and z.device == self._in.device)
            if not ok:
                return self.fsq(z)
            self._in.copy_(z)
            self.graph.replay()
            self.replays += 1
            return {k: v.clone() for k, v in self._out.items()}

    # ══════════════════════════════════════════════════════════════════
    # 5. SSRR LOSS (restored from Source/v2)
    # ══════════════════════════════════════════════════════════════════
    class SW2VExtractor(nn.Module):
        """Frozen log-mel feature extractor (fallback for a real SW2V
        checkpoint).  NO @torch.no_grad() on forward(): that would silently
        zero the SSRR gradient (Paper 02, Bug 1).  Parameters are frozen
        instead, so gradient flows through the input but the extractor never
        trains."""

        def __init__(self, sr: int = 16000, out_dim: int = 1024):
            super().__init__()
            self.sr, self.out_dim, self.using_fallback = sr, out_dim, True
            self.register_buffer(
                "mel_fb", self._mel_filterbank(sr, 80, 400, 8000, 512),
                persistent=False)
            self.fallback_proj = nn.Linear(80, out_dim, bias=False)
            nn.init.orthogonal_(self.fallback_proj.weight)
            for p in self.fallback_proj.parameters():
                p.requires_grad_(False)

        @staticmethod
        def _mel_filterbank(sr, n_mels, f_min, f_max, n_fft):
            hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
            mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
            hz = mel2hz(np.linspace(hz2mel(f_min), hz2mel(f_max), n_mels + 2))
            bins = np.floor((n_fft + 1) * hz / sr).astype(int)
            fb = np.zeros((n_mels, n_fft // 2 + 1))
            for m in range(1, n_mels + 1):
                l, c, r = bins[m - 1], bins[m], bins[m + 1]
                if r > c:
                    fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l)
                    fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c)
            return torch.from_numpy(fb).float()

        def forward(self, x):
            if x.ndim == 1:
                x = x.unsqueeze(0)
            n_fft, hop = 512, 160
            win = torch.hann_window(n_fft, device=x.device)
            frames = x.unfold(-1, n_fft, hop) * win
            spec = torch.fft.rfft(frames, dim=-1).abs().pow(2)
            mel = torch.einsum("btf,mf->btm", spec, self.mel_fb.to(x.device))
            return self.fallback_proj(torch.log(mel + 1e-6))

    def ssrr_loss(x, y, extractor, weight: float = 1.0):
        """L1 between extractor features of source x and reconstruction y.
        Source is detached; reconstruction is NOT."""
        with torch.no_grad():
            feat_x = extractor(x)
        feat_y = extractor(y)
        T = min(feat_x.shape[1], feat_y.shape[1])
        return weight * F.l1_loss(feat_y[:, :T], feat_x[:, :T])

else:  # pragma: no cover
    FSQConfig = FSQQuantizer = GraphSafeQuantizer = None
    SW2VExtractor = ssrr_loss = None


# ══════════════════════════════════════════════════════════════════════
# 6. VERIFICATION PROTOCOL  (python harmonia_gpu.py)
# ══════════════════════════════════════════════════════════════════════
def _load_golden():
    if not os.path.exists(_GOLDEN):
        return None
    with open(_GOLDEN) as f:
        return json.load(f)


def verify() -> bool:
    """Runs every check it can; prints VERIFIED / SKIPPED for each.  Returns
    True only if nothing failed.  SKIPPED means UNVERIFIED, not passed."""
    if not HAVE_TORCH:
        print("PyTorch unavailable: only the NumPy reference exists here.")
        print("  GPU backend: UNVERIFIED (nothing was run)")
        return False

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"codec {CODEC_VERSION}  device={dev}  triton={HAVE_TRITON}")
    cfg = FSQConfig(dim=64, levels=(8, 5, 5, 5))
    levels = cfg.levels
    q = FSQQuantizer(cfg).to(dev)

    # Part 2 -- per-dimension range
    z = torch.randn(2, 16, 64, device=dev)
    idx = q(z)["tokens"]
    assert idx.dtype == torch.int32 and idx.shape == (2, 16, 4)
    for d, L in enumerate(levels):
        assert int(idx[..., d].min()) >= 0 and int(idx[..., d].max()) <= L - 1, \
            f"dim {d} outside [0, {L - 1}]"
    print("[VERIFIED] Part 2: per-dimension index ranges")

    # Golden vectors (NumPy reference), including exact rounding ties
    g = _load_golden()
    if g is None:
        print("[SKIPPED]  golden vectors: tests/golden/fsq_vectors.json missing")
    else:
        u = torch.tensor(g["u"], dtype=torch.float32, device=dev)
        ref = torch.tensor(g["idx"], dtype=torch.int32, device=dev)
        got = fsq_indices_eager(u, levels)
        assert torch.equal(got, ref), \
            f"eager indices differ from reference on {(got != ref).any(-1).sum().item()} rows"
        qref = torch.tensor(g["q"], dtype=torch.float32, device=dev)
        assert (fsq_dequant(got, levels) - qref).abs().max() <= 2e-7, "dequant drift"
        codes = torch.tensor(g["codes"], dtype=torch.int64, device=dev)
        assert torch.equal(pack_codes(got, levels), codes), "pack_codes != reference"
        assert torch.equal(unpack_codes(codes, levels), got), "unpack_codes != reference"
        print(f"[VERIFIED] eager path == NumPy reference on {u.shape[0]} golden vectors "
              f"(incl. exact ties), pack/unpack round-trip")

    # decode(tokens) must reproduce forward's quantized value
    out = q(z)
    assert torch.equal(q.decode(out, 16, 64, dev), q.from_fsq(fsq_dequant(out["tokens"], levels)))
    print("[VERIFIED] decode(tokens) == from_fsq(dequant(tokens))")

    # Part 1b -- FSQ gradient flow (1.0.0 detached the quantized path)
    q.zero_grad()
    zr = torch.randn(2, 16, 64, device=dev, requires_grad=True)
    q(zr)["zq"].pow(2).mean().backward()
    for name, p in (("to_fsq", q.to_fsq.weight), ("from_fsq", q.from_fsq.weight)):
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} received no gradient"
    assert zr.grad is not None and zr.grad.abs().sum() > 0, "encoder input received no gradient"
    print("[VERIFIED] Part 1b: gradient reaches to_fsq, from_fsq and the encoder input")

    # Part 1 -- SSRR gradient flow
    ext = SW2VExtractor(sr=16000, out_dim=128).to(dev)
    x = torch.randn(2, 16000, device=dev)
    y = (torch.randn(2, 16000, device=dev) * 0.1).requires_grad_(True)
    loss = ssrr_loss(x, y, ext)
    assert loss.requires_grad, "SSRR loss carries no gradient"
    loss.backward()
    assert y.grad is not None and y.grad.abs().sum() > 0, "zero SSRR gradient on y"
    assert ext.fallback_proj.weight.grad is None, "extractor params received gradient"
    print("[VERIFIED] Part 1: SSRR gradient flow, extractor frozen")

    if dev != "cuda" or not HAVE_TRITON:
        print("[SKIPPED]  Part 3: Triton vs eager  -> Triton path UNVERIFIED")
        print("[SKIPPED]  CUDA graph replay        -> graph path UNVERIFIED")
        print("harmonia_gpu: eager checks passed; GPU paths NOT verified")
        return True

    # Part 3 -- Triton vs eager, on inputs that actually contain ties
    def tri_vs_eager(name, t):
        a, b = fsq_triton_indices(t, levels), fsq_indices_eager(t, levels)
        n = (a != b).sum().item()
        assert n == 0, f"Triton mismatch on {name}: {n} / {a.numel()} indices"
    if g is not None:
        tri_vs_eager("golden vectors (ties)", torch.tensor(g["u"], device=dev))
    tie = torch.tensor([-0.75, -0.25, 0.25, 0.75], device=dev).repeat(4, 1).T.contiguous()
    tri_vs_eager("5-level exact ties", tie)
    tri_vs_eager("uniform grid-domain", torch.empty(1 << 20, 4, device=dev).uniform_(-1.2, 1.2))
    tri_vs_eager("fp16 input", torch.empty(4096, 4, device=dev).uniform_(-1, 1).half())
    tri_vs_eager("non-contiguous", torch.empty(4, 4096, device=dev).uniform_(-1, 1).T)
    qt = FSQQuantizer(FSQConfig(dim=64, levels=levels, use_triton=True)).to(dev)
    qe = FSQQuantizer(FSQConfig(dim=64, levels=levels, use_triton=False)).to(dev)
    qe.load_state_dict(qt.state_dict())
    xl = torch.randn(64, 128, 64, device=dev) * 3
    ot, oe = qt(xl), qe(xl)
    assert torch.equal(ot["tokens"], oe["tokens"]), "forward tokens differ"
    assert torch.equal(ot["zq"], oe["zq"]), "forward zq differs"
    print("[VERIFIED] Part 3: Triton path bit-exact against eager "
          "(ties, fp16, non-contiguous, full forward)")

    # CUDA graph replay (grad disabled: that is the only mode that replays)
    gq = GraphSafeQuantizer(qt)
    gq.capture(4, 32)
    zin = torch.randn(4, 32, 64, device=dev)
    with torch.no_grad():
        e = qt(zin)
        r = gq(zin)
        assert gq.replays == 1, "graph was not replayed (silently ran eager)"
        assert torch.equal(r["tokens"], e["tokens"]), "graph tokens differ from eager"
        assert (r["zq"] - e["zq"]).abs().max() <= 1e-5, "graph zq differs from eager"
        keep = gq(zin)["tokens"]
        gq(torch.randn(4, 32, 64, device=dev))          # would overwrite aliased buffers
        assert torch.equal(keep, e["tokens"]), "replay overwrote a previous result"
        n_before = gq.replays
        assert gq(torch.randn(2, 7, 64, device=dev))["tokens"].shape == (2, 7, 4)
        assert gq.replays == n_before, "shape mismatch replayed a stale graph"
    zg = torch.randn(4, 32, 64, device=dev, requires_grad=True)
    gq(zg)["zq"].sum().backward()                       # grad enabled -> eager
    assert zg.grad is not None and gq.replays == n_before
    print("[VERIFIED] CUDA graph replay ok (matches eager, no aliasing, "
          "falls back on shape mismatch and when grad is enabled)")
    print("harmonia_gpu: all checks passed")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
