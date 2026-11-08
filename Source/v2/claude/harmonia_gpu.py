#!/usr/bin/env python3
"""
harmonia_gpu.py — GPU-native codec quantizer and loss stack.

Two bugs were found in the original pasted version by direct execution
and numerical derivation, and are fixed here — both verified before
being applied:

  FIX 1 (SSRR gradient flow): the original SW2VExtractor.forward() was
  decorated @torch.no_grad(), which strips gradient tracking from its
  output regardless of the input. Verified: loss.requires_grad was
  False and .backward() failed outright in isolation; in a realistic
  combined loss it would silently contribute zero gradient forever,
  never erroring. Fixed by removing the decorator and instead freezing
  the extractor's own parameters (same pattern as SemanticAnchor's
  frozen WavLM elsewhere in this codebase) -- differentiable through,
  not trained itself. Verified: loss.requires_grad is now True,
  backward() succeeds, y.grad is non-zero, extractor params stay frozen.

  FIX 2 (Triton per-dimension quantization): the original kernel call
  passed a single scalar n_levels=max(levels) applied uniformly to every
  dimension, instead of each dimension's own level. Verified
  numerically: for levels=[8,5,5,5], three of four dimensions were
  quantized against the wrong grid size, producing silently incorrect
  codes. Fixed by calling the kernel once per dimension with that
  dimension's own level (n_dims is small, 4-8 typically, so this is
  cheap) and reassembling. Verified bit-exact against the correct
  per-dimension reference.

Neither fix has been run on real GPU/Triton hardware -- this
environment has no GPU. The pure-PyTorch fallback path IS verified by
direct execution. Run the Triton path once on real hardware and compare
against the pure-PyTorch path (see _verify()) before trusting it.

Deps:
    pip install torch torchaudio triton
    # SSRR extractor weights: jhcodec/sw2v_60k (unverified citation --
    # this repo/checkpoint was never checked from this environment)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False


# ══════════════════════════════════════════════════════════════════════
# 1. TRITON VQ KERNEL — FIXED: per-dimension level, not a shared scalar
# ══════════════════════════════════════════════════════════════════════

if HAVE_TRITON:

    @triton.jit
    def _fsq_kernel_single_dim(
        x_ptr, out_ptr, idx_ptr,
        n_elements,
        level,              # this dimension's OWN level count, not a shared max
        BLOCK: tl.constexpr,
    ):
        """
        Quantizes ONE dimension's worth of elements against its own
        level count. Called once per dimension from the caller below --
        n_dims is small (4-8 typically), so this loop is cheap in
        Python, and the expensive axis (B*T, potentially large) still
        goes through the fused kernel per call.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        scaled = (x + 1.0) * 0.5 * (level - 1.0)
        rounded = tl.math.round(scaled)
        idx = rounded.to(tl.int32)
        dequant = rounded / (level - 1.0) * 2.0 - 1.0

        tl.store(out_ptr + offsets, dequant, mask=mask)
        tl.store(idx_ptr + offsets, idx, mask=mask)


    def _fsq_forward_triton(x: torch.Tensor, levels: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FIX: x is (..., n_dims). Calls the kernel once PER DIMENSION with
        that dimension's own level, instead of once for the whole
        flattened tensor with a single shared scalar level (the original
        bug -- verified to silently quantize non-max-level dimensions
        against the wrong grid).
        """
        assert x.is_cuda, "Triton FSQ requires CUDA"
        assert x.shape[-1] == len(levels)
        out = torch.empty_like(x)
        idx = torch.empty_like(x, dtype=torch.int32)
        BLOCK = 1024
        for d, level in enumerate(levels):
            xd = x[..., d].contiguous().reshape(-1)
            out_d = torch.empty_like(xd)
            idx_d = torch.empty_like(xd, dtype=torch.int32)
            n = xd.numel()
            grid = (triton.cdiv(n, BLOCK),)
            _fsq_kernel_single_dim[grid](xd, out_d, idx_d, n, float(level), BLOCK=BLOCK)
            out[..., d] = out_d.reshape(x.shape[:-1])
            idx[..., d] = idx_d.reshape(x.shape[:-1])
        return out, idx

else:
    def _fsq_forward_triton(x, levels):
        raise RuntimeError("Triton not installed")


# ══════════════════════════════════════════════════════════════════════
# 2. FSQ QUANTIZER
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FSQConfig:
    levels: tuple = (8, 5, 5, 5)
    dim: int = 64
    use_triton: bool = True


class FSQQuantizer(nn.Module):
    """
    Finite Scalar Quantization. Drop-in replacement for MERVQNN.

    No codebooks. No usage buffers. No balance loss. The pure-PyTorch
    _quantize() path is verified correct by direct execution -- it
    already applies levels per-dimension correctly via broadcasting.
    The Triton path (forward()) now matches it exactly (FIX 2).
    """

    def __init__(self, cfg: FSQConfig):
        super().__init__()
        self.cfg = cfg
        self.levels = list(cfg.levels)
        self.n_dims = len(self.levels)
        self.to_fsq = nn.Linear(cfg.dim, self.n_dims, bias=False)
        self.from_fsq = nn.Linear(self.n_dims, cfg.dim, bias=False)
        nn.init.orthogonal_(self.to_fsq.weight)
        nn.init.orthogonal_(self.from_fsq.weight)
        self.register_buffer(
            "level_tensor",
            torch.tensor(self.levels, dtype=torch.float32),
            persistent=False,
        )

    def _quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure-PyTorch path. Correctly applies per-dim levels via
        broadcasting -- verified by direct execution, unchanged."""
        x = torch.tanh(x)
        levels = self.level_tensor.to(x.device)
        scaled = (x + 1.0) * 0.5 * (levels - 1.0)
        idx = torch.round(scaled)
        dequant = idx / (levels - 1.0) * 2.0 - 1.0
        return dequant, idx.to(torch.int32)

    def forward(self, z: torch.Tensor, stage_mask: Optional[torch.Tensor] = None):
        B, T, D = z.shape
        x = self.to_fsq(z)

        if (self.cfg.use_triton and x.is_cuda and HAVE_TRITON):
            x_clamped = torch.tanh(x)
            # FIX: pass the real per-dim levels list, not a single scalar.
            dequant, idx = _fsq_forward_triton(x_clamped, self.levels)
        else:
            dequant, idx = self._quantize(x)

        quantized = self.from_fsq(dequant)
        quantized_st = z + (quantized - z).detach()
        commit = F.mse_loss(dequant.detach(), x)
        probs = torch.sigmoid(x.mean(dim=-1, keepdim=True))

        return {
            "tokens": idx,
            "commit": commit,
            "probs": probs,
            "zq": quantized_st,
        }

    def decode(self, tokens, T, D, device, stage_mask=None, batch_size=None):
        if isinstance(tokens, dict):
            idx = tokens["tokens"]
        else:
            idx = tokens
        idx = idx.to(device)
        levels = self.level_tensor.to(device)
        dequant = idx.float() / (levels - 1.0) * 2.0 - 1.0
        return self.from_fsq(dequant)

    def balance_loss(self):
        return torch.tensor(0.0, device=self.level_tensor.device)

    def step_reset(self):
        pass

    def utilization(self):
        return 1.0


# ══════════════════════════════════════════════════════════════════════
# 3. SSRR LOSS — FIX 1: extractor is differentiable-through, frozen params
# ══════════════════════════════════════════════════════════════════════

class SW2VExtractor(nn.Module):
    """
    FIX: no @torch.no_grad() decorator on forward() -- that stripped
    gradient tracking from the output regardless of the input, which
    is verified to make the entire SSRR loss silently carry zero
    gradient. Parameters are frozen instead (requires_grad_(False)),
    which is the correct way to get "differentiable through, but not
    trained itself" -- same pattern as SemanticAnchor's frozen WavLM
    elsewhere in this codebase.
    """

    def __init__(self, sr: int = 16000, out_dim: int = 1024):
        super().__init__()
        self.sr = sr
        self.out_dim = out_dim
        self.using_fallback = True

        self.register_buffer(
            "mel_fb",
            self._mel_filterbank(sr, 80, 400, 8000, 512),
            persistent=False,
        )
        self.fallback_proj = nn.Linear(80, out_dim, bias=False)
        nn.init.orthogonal_(self.fallback_proj.weight)
        # FIX: freeze parameters here instead of wrapping forward() in
        # no_grad. This is what actually makes it "frozen" without
        # blocking gradient flow through to the caller's input.
        for p in self.fallback_proj.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _mel_filterbank(sr, n_mels, f_min, f_max, n_fft):
        def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
        def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        mels = np.linspace(hz2mel(f_min), hz2mel(f_max), n_mels + 2)
        hz = mel2hz(mels)
        bins = np.floor((n_fft + 1) * hz / sr).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(1, n_mels + 1):
            l, c, r = bins[m - 1], bins[m], bins[m + 1]
            if r > c:
                fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l)
                fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c)
        return torch.from_numpy(fb).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FIX: no @torch.no_grad() here. Callers wrap the SOURCE call
        (x) in no_grad explicitly (see ssrr_loss below); the
        RECONSTRUCTION call (y) must stay differentiable, or the loss
        does nothing -- verified by direct execution."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        n_fft, hop = 512, 160
        win = torch.hann_window(n_fft, device=x.device)
        frames = x.unfold(-1, n_fft, hop) * win
        spec = torch.fft.rfft(frames, dim=-1).abs().pow(2)
        mel = torch.einsum("btf,mf->btm", spec, self.mel_fb.to(x.device))
        log_mel = torch.log(mel + 1e-6)
        return self.fallback_proj(log_mel)


def ssrr_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    extractor: SW2VExtractor,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Self-Supervised Representation Reconstruction loss.

    L1 between extractor features of source (x) and reconstruction (y).
    feat_x is explicitly detached (no_grad) since the source needs no
    gradient. feat_y is NOT detached -- gradient must flow through it
    to reach y, which is the entire point of this loss. Verified by
    direct execution: with the extractor's forward() no longer wrapped
    in no_grad, loss.requires_grad is True and .backward() produces a
    real non-zero gradient on y.
    """
    with torch.no_grad():
        feat_x = extractor(x)
    feat_y = extractor(y)
    T = min(feat_x.shape[1], feat_y.shape[1])
    return weight * F.l1_loss(feat_y[:, :T], feat_x[:, :T])


# ══════════════════════════════════════════════════════════════════════
# 4. SELF-TEST — now actually checks gradient flow, not just shapes
# ══════════════════════════════════════════════════════════════════════

def _verify():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"triton available: {HAVE_TRITON}")
    if device == "cpu":
        print("NOTE: no GPU in this environment. Triton kernel path is")
        print("      UNVERIFIED here -- only the pure-PyTorch path runs.")
        print("      Run this file on real GPU hardware before trusting")
        print("      the Triton path in production.")

    cfg = FSQConfig(dim=64, levels=(8, 5, 5, 5))
    q = FSQQuantizer(cfg).to(device)

    z = torch.randn(2, 16, 64, device=device)
    out = q(z)
    idx = out["tokens"]
    assert idx.dtype == torch.int32
    assert idx.shape == (2, 16, 4)
    # Per-dimension range check -- catches the original Triton bug if
    # it recurs, since a shared max-level bug would let dims with
    # smaller levels report indices above their own true range.
    levels_t = torch.tensor(cfg.levels)
    for d in range(idx.shape[-1]):
        assert idx[..., d].max() <= levels_t[d] - 1, \
            f"dim {d}: idx exceeds its own level {levels_t[d]} -- per-dim bug regressed"
    print(f"FSQ tokens: shape={tuple(idx.shape)}, per-dim ranges verified correct")

    zq = q.decode(out, 16, 64, device)
    assert zq.shape == z.shape
    print(f"decode: shape={tuple(zq.shape)}")

    # SSRR extractor -- now actually checks gradient flow, which the
    # original self-test never did (it only checked the loss value
    # printed, not whether it could train anything).
    ext = SW2VExtractor(sr=16000, out_dim=128).to(device)
    x = torch.randn(2, 16000, device=device)
    y = (torch.randn(2, 16000, device=device) * 0.1).requires_grad_(True)
    loss = ssrr_loss(x, y, ext)
    print(f"SSRR loss: {loss.item():.6f}")
    assert loss.requires_grad, "SSRR loss carries no gradient -- regression of FIX 1"
    loss.backward()
    assert y.grad is not None and y.grad.abs().sum() > 0, \
        "SSRR loss produced zero gradient on y -- regression of FIX 1"
    print(f"SSRR gradient check: y.grad abs sum = {y.grad.abs().sum().item():.4f}  OK")
    assert ext.fallback_proj.weight.grad is None, \
        "extractor params received gradient -- should stay frozen"
    print("SSRR extractor params correctly frozen (no gradient)")

    print("harmonia_gpu: all checks passed (Triton path unverified on this CPU-only machine)")


if __name__ == "__main__":
    _verify()
