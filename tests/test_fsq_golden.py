"""NumPy-only checks (run anywhere).  Plain asserts: works with pytest or `python tests/test_fsq_golden.py`."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lewspew_crypt.codec.fsq import FSQ, FSQConfig

F32 = np.float32
G = json.load(open(os.path.join(ROOT, "tests", "golden", "fsq_vectors.json")))
LEVELS = tuple(G["levels"])


def test_golden_matches_reference():
    fsq = FSQ(FSQConfig(levels=LEVELS))
    u = np.array(G["u"], dtype=F32)
    _, idx = fsq.quantize(u)
    assert np.array_equal(idx, np.array(G["idx"], dtype=np.int32))


def test_per_dimension_ranges():
    idx = np.array(G["idx"])
    for d, L in enumerate(LEVELS):
        assert idx[:, d].min() >= 0 and idx[:, d].max() <= L - 1


def test_exact_ties_are_half_to_even():
    """5-level dim at +-0.25/+-0.75 sits exactly on k+.5. Half-to-even gives 0,2,2,4;
    half-away-from-zero (libdevice round, tl.math.round) would give 1,2,3,4."""
    fsq = FSQ()
    u = np.array([[0, -0.75, -0.25, 0.25], [0, 0.75, 0, 0]], dtype=F32)
    _, idx = fsq.quantize(u)
    assert idx[0, 1:].tolist() == [0, 2, 2] and idx[1, 1] == 4


def test_golden_contains_discriminating_ties():
    """The vector set must actually be able to tell the two rounding modes apart."""
    u = np.array(G["u"], dtype=F32); lv = np.array(LEVELS, dtype=F32)
    s = (np.clip(u, -1, 1) + F32(1)) * F32(.5) * (lv - F32(1))
    away = np.where(s >= 0, np.floor(s + F32(.5)), np.ceil(s - F32(.5))).astype(np.int32)
    assert (away != np.array(G["idx"])).any(), "golden set cannot detect half-away rounding"
    assert ((s - np.floor(s)) == F32(.5)).sum() >= 12


def kernel_algorithm(u, levels):
    """Line-for-line NumPy mirror of _fsq_idx_kernel in harmonia_gpu.py."""
    lv = np.asarray(levels, dtype=F32)
    x = np.minimum(np.maximum(u.astype(F32), F32(-1)), F32(1))
    s = (x + F32(1)) * F32(.5) * (lv - F32(1))
    f = s.astype(np.int32)                       # trunc
    d = s - f.astype(F32)
    up = (d > F32(.5)) | ((d == F32(.5)) & ((f & 1) == 1))
    return f + up.astype(np.int32)


def test_kernel_algorithm_equals_reference():
    """Verifies the ALGORITHM the Triton kernel uses. It does not verify Triton."""
    fsq = FSQ(FSQConfig(levels=LEVELS))
    rng = np.random.default_rng(1)
    u = np.concatenate([np.array(G["u"], dtype=F32),
                        rng.uniform(-1.3, 1.3, (1_000_000, 4)).astype(F32)])
    assert np.array_equal(kernel_algorithm(u, LEVELS), fsq.quantize(u)[1])


def test_pack_unpack_bijective():
    fsq = FSQ(FSQConfig(levels=LEVELS))
    codes = np.arange(1000)
    idx = fsq.decode_code(codes)
    assert np.array_equal(fsq.encode_code(idx), codes) and len({tuple(r) for r in idx}) == 1000
    assert np.array_equal(fsq.encode_code(np.array(G["idx"])), np.array(G["codes"]))


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            try: f(); print("PASS", n)
            except AssertionError as e: fails += 1; print("FAIL", n, e)
    sys.exit(1 if fails else 0)
