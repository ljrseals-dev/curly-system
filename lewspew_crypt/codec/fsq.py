from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class FSQConfig:
    levels: tuple[int,...]=(8,5,5,5)
    clip: float=1.0

class FSQ:
    def __init__(self,cfg=FSQConfig()): self.cfg=cfg
    @property
    def dims(self): return len(self.cfg.levels)
    @property
    def cardinality(self):
        n=1
        for l in self.cfg.levels: n*=l
        return n
    def quantize(self,x):
        x=np.asarray(x,dtype=np.float32)
        if x.shape[-1]!=self.dims: raise ValueError(f"expected last dim {self.dims}, got {x.shape[-1]}")
        levels=np.asarray(self.cfg.levels,dtype=np.float32)
        u=np.clip(x,-self.cfg.clip,self.cfg.clip)
        idx=np.rint((u/self.cfg.clip+1)*0.5*(levels-1)).astype(np.int32)
        q=(idx/(levels-1)*2-1)*self.cfg.clip
        return q,idx
    def encode_code(self,idx):
        idx=np.asarray(idx,dtype=np.int64)
        code=np.zeros(idx.shape[:-1],dtype=np.int64); mul=1
        for d,l in enumerate(self.cfg.levels): code += idx[...,d]*mul; mul*=l
        return code
    def decode_code(self,code):
        code=np.asarray(code,dtype=np.int64); out=np.empty(code.shape+(self.dims,),dtype=np.int32); rem=code.copy()
        for d,l in enumerate(self.cfg.levels): out[...,d]=rem%l; rem//=l
        return out
