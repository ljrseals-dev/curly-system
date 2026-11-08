"""Torch / CUDA checks for harmonia_gpu.py.  Skipped (not passed) when torch / CUDA / Triton are absent."""
import os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")
import harmonia_gpu as H

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device: UNVERIFIED")
needs_triton = pytest.mark.skipif(not (torch.cuda.is_available() and H.HAVE_TRITON), reason="no CUDA+Triton: UNVERIFIED")


def test_verify_protocol_runs():
    assert H.verify()


def test_gradient_reaches_both_projections():
    q = H.FSQQuantizer(H.FSQConfig())
    z = torch.randn(2, 8, 64, requires_grad=True)
    q(z)["zq"].pow(2).mean().backward()
    assert q.to_fsq.weight.grad.abs().sum() > 0 and q.from_fsq.weight.grad.abs().sum() > 0
    assert z.grad.abs().sum() > 0


def test_forward_value_equals_dequantized_grid():
    q = H.FSQQuantizer(H.FSQConfig())
    z = torch.randn(2, 8, 64)
    o = q(z)
    assert torch.equal(o["zq"], q.decode(o["tokens"], 8, 64, "cpu"))


def test_pack_unpack_matches_numpy_reference():
    import numpy as np
    from lewspew_crypt.codec.fsq import FSQ
    codes = torch.arange(1000)
    idx = H.unpack_codes(codes, (8, 5, 5, 5))
    assert np.array_equal(idx.numpy(), FSQ().decode_code(codes.numpy()))
    assert torch.equal(H.pack_codes(idx, (8, 5, 5, 5)), codes)


@needs_triton
def test_triton_bit_exact_on_ties():
    t = torch.tensor([-0.75, -0.25, 0.25, 0.75], device="cuda").repeat(4, 1).T.contiguous()
    assert torch.equal(H.fsq_triton_indices(t, (8, 5, 5, 5)), H.fsq_indices_eager(t, (8, 5, 5, 5)))


@needs_triton
def test_triton_bit_exact_random():
    t = torch.empty(1 << 20, 4, device="cuda").uniform_(-1.2, 1.2)
    assert torch.equal(H.fsq_triton_indices(t, (8, 5, 5, 5)), H.fsq_indices_eager(t, (8, 5, 5, 5)))


@needs_cuda
def test_graph_replays_and_matches_eager():
    q = H.FSQQuantizer(H.FSQConfig(use_triton=H.HAVE_TRITON)).cuda()
    g = H.GraphSafeQuantizer(q); g.capture(2, 16)
    z = torch.randn(2, 16, 64, device="cuda")
    with torch.no_grad():
        r, e = g(z), q(z)
    assert g.replays == 1 and torch.equal(r["tokens"], e["tokens"])
