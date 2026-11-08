#!/usr/bin/env python3
"""Generate golden FSQ vectors from the canonical NumPy reference.

Inputs are in the *grid domain* (post-tanh, nominally [-1, 1]), because that is
what both the eager path and the Triton kernel quantize.  Random inputs almost
never land exactly on a rounding tie, so the set deliberately includes them:
for a level count L, u_k = (2k+1)/(L-1) - 1 sits exactly on the k+0.5 boundary.
For L=5 these are +-0.25 / +-0.75, which are exact in binary32, so
half-to-even (torch.round, np.rint) and half-away-from-zero (libdevice round)
disagree on them.

Run:  python scripts/make_golden.py
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lewspew_crypt.codec.fsq import FSQ, FSQConfig  # noqa: E402

LEVELS = (8, 5, 5, 5)
SEED = 20261108


def build():
    f32 = np.float32
    rng = np.random.default_rng(SEED)
    fsq = FSQ(FSQConfig(levels=LEVELS))
    rows, tags = [], []

    # 1. tie / near-tie boundaries, one dimension at a time (others random)
    for d, L in enumerate(LEVELS):
        for k in range(L - 1):
            u = f32((2 * k + 1) / (L - 1) - 1.0)
            for eps in (0.0, 1, -1):
                row = rng.uniform(-1, 1, 4).astype(f32)
                v = u if eps == 0 else np.nextafter(u, f32(np.inf if eps > 0 else -np.inf), dtype=f32)
                row[d] = v
                rows.append(row)
                tags.append(f"tie d={d} L={L} k={k} eps={int(eps)}")
    # 2. endpoints, zero, and out-of-range (must clip)
    for v in (-1.0, 1.0, 0.0, -0.999999, 0.999999, -1.5, 1.5, -10.0, 10.0):
        rows.append(np.full(4, v, dtype=f32)); tags.append(f"const {v}")
    # 3. dense random
    for r in rng.uniform(-1, 1, (4096, 4)).astype(f32):
        rows.append(r); tags.append("rand")

    u = np.stack(rows).astype(f32)
    q, idx = fsq.quantize(u)
    lv = np.asarray(LEVELS, dtype=f32)
    # float32 dequant, op-for-op identical to the torch path: idx/(L-1)*2-1
    q32 = (idx.astype(f32) / (lv - f32(1)) * f32(2) - f32(1)).astype(f32)
    codes = fsq.encode_code(idx)
    payload = {
        "levels": list(LEVELS),
        "seed": SEED,
        "generator": "scripts/make_golden.py (NumPy reference, np.rint = half-to-even)",
        "numpy": np.__version__,
        "u": u.tolist(),
        "idx": idx.tolist(),
        "q": q32.tolist(),
        "codes": codes.tolist(),
        "tags": tags,
    }
    body = json.dumps(payload, sort_keys=True).encode()
    payload["sha256"] = hashlib.sha256(body).hexdigest()
    return payload


if __name__ == "__main__":
    p = build()
    out = os.path.join(ROOT, "tests", "golden", "fsq_vectors.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(p, f)
    print(f"wrote {out}: {len(p['u'])} vectors, sha256={p['sha256'][:16]}...")
