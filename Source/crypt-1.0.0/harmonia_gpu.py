#!/usr/bin/env python3
"""LewSpEW Crypt GPU codec backend.

Optional PyTorch/Triton acceleration for the Crypt reference FSQ codec.
The CPU NumPy implementation remains the canonical numerical reference.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH=True
except Exception:
    HAVE_TORCH=False

try:
    import triton
    import triton.language as tl
    HAVE_TRITON=True
except Exception:
    HAVE_TRITON=False

if HAVE_TORCH and HAVE_TRITON:
    @triton.jit
    def _fsq_kernel(x_ptr, q_ptr, idx_ptr, levels_ptr, n, dims: tl.constexpr, BLOCK: tl.constexpr):
        pid=tl.program_id(0)
        off=pid*BLOCK+tl.arange(0,BLOCK)
        mask=off<n
        x=tl.load(x_ptr+off,mask=mask,other=0.0)
        d=off % dims
        levels=tl.load(levels_ptr+d,mask=mask,other=2.0)
        x=tl.maximum(tl.minimum(x,1.0),-1.0)
        scaled=(x+1.0)*0.5*(levels-1.0)
        idx=tl.math.round(scaled).to(tl.int32)
        q=idx.to(tl.float32)/(levels-1.0)*2.0-1.0
        tl.store(q_ptr+off,q,mask=mask)
        tl.store(idx_ptr+off,idx,mask=mask)

    def fsq_triton(x, levels):
        x=x.contiguous(); out=torch.empty_like(x); idx=torch.empty_like(x,dtype=torch.int32)
        lv=torch.as_tensor(levels,device=x.device,dtype=torch.float32)
        n=x.numel(); dims=len(levels); block=1024
        _fsq_kernel[(triton.cdiv(n,block),)](x,out,idx,lv,n,dims=dims,BLOCK=block)
        return out,idx

if HAVE_TORCH:
    @dataclass(frozen=True)
    class FSQConfig:
        levels: tuple[int,...]=(8,5,5,5)
        dim:int=64
        use_triton:bool=True

    class FSQQuantizer(nn.Module):
        """Fixed scalar-grid quantizer with true per-dimension levels."""
        def __init__(self,cfg=FSQConfig()):
            super().__init__(); self.cfg=cfg
            self.register_buffer('levels',torch.tensor(cfg.levels,dtype=torch.float32),persistent=False)
            self.to_fsq=nn.Linear(cfg.dim,len(cfg.levels),bias=False)
            self.from_fsq=nn.Linear(len(cfg.levels),cfg.dim,bias=False)
            nn.init.orthogonal_(self.to_fsq.weight); nn.init.orthogonal_(self.from_fsq.weight)
        def _quantize(self,x):
            x=torch.tanh(x); lv=self.levels.to(x.device,dtype=x.dtype)
            scaled=(x+1)*0.5*(lv-1); idx=torch.round(scaled).clamp_min(0)
            idx=torch.minimum(idx,lv-1); q=idx/(lv-1)*2-1
            return q,idx.to(torch.int32)
        def forward(self,z,stage_mask=None):
            x=self.to_fsq(z)
            if self.cfg.use_triton and HAVE_TRITON and x.is_cuda:
                q,idx=fsq_triton(torch.tanh(x),tuple(int(v) for v in self.cfg.levels))
            else: q,idx=self._quantize(x)
            zq=self.from_fsq(q); zq_st=z+(zq-z).detach()
            return {'tokens':idx,'commit':F.mse_loss(q.detach(),torch.tanh(x)),'probs':torch.sigmoid(x.mean(-1,keepdim=True)),'zq':zq_st}
        def decode(self,tokens,T=None,D=None,device=None,stage_mask=None,batch_size=None):
            idx=tokens['tokens'] if isinstance(tokens,dict) else tokens
            idx=idx.to(device or self.levels.device)
            lv=self.levels.to(idx.device); q=idx.float()/(lv-1)*2-1
            return self.from_fsq(q)
        def balance_loss(self): return self.levels.new_zeros(())
        def step_reset(self): pass
        def utilization(self): return 1.0

    class GraphSafeQuantizer(nn.Module):
        def __init__(self,fsq):
            super().__init__(); self.fsq=fsq; self.graph=None; self.static_in=None; self.static_out=None
        def capture(self,B,T):
            if not self.fsq.to_fsq.weight.is_cuda: raise RuntimeError('CUDA required')
            self.static_in=torch.zeros(B,T,self.fsq.cfg.dim,device=self.fsq.to_fsq.weight.device)
            for _ in range(3): self.static_out=self.fsq(self.static_in)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph): self.static_out=self.fsq(self.static_in)
        def forward(self,z):
            if self.graph is None or tuple(z.shape)!=tuple(self.static_in.shape): return self.fsq(z)
            self.static_in.copy_(z); self.graph.replay(); return self.static_out
else:
    FSQQuantizer=None
    GraphSafeQuantizer=None

def verify():
    if not HAVE_TORCH:
        print('PyTorch unavailable: reference backend only'); return
    dev='cuda' if torch.cuda.is_available() else 'cpu'; q=FSQQuantizer().to(dev)
    z=torch.randn(2,16,64,device=dev); o=q(z)
    assert o['tokens'].shape==(2,16,4)
    assert int(o['tokens'][...,0].max())<=7
    assert int(o['tokens'][...,1:].max())<=4
    print(f'device={dev} triton={HAVE_TRITON} tokens={tuple(o["tokens"].shape)} cardinality=1000')
    print('harmonia_gpu: verification passed')
if __name__=='__main__': verify()
