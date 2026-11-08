from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class VoodooConfig:
    hp_hz: float = 28.0
    presence_hz: float = 2800.0
    presence_db: float = 1.5
    saturation: float = 0.10
    compression: float = 0.18
    ceiling: float = 0.8912509381  # -1 dBFS sample ceiling

def _one_pole_hp(x, sr, fc):
    a=np.exp(-2*np.pi*fc/sr); y=np.empty_like(x); prev=0.0
    for i,v in enumerate(x):
        prev=a*prev+(1-a)*v; y[i]=v-prev
    return y

def process(x: np.ndarray, sr: int, cfg=VoodooConfig()):
    x=np.asarray(x,dtype=np.float64)
    mono=x.ndim==1
    y=x.reshape(-1) if mono else x.copy()
    if mono: y=_one_pole_hp(y,sr,cfg.hp_hz)
    else:
        for c in range(y.shape[1]): y[:,c]=_one_pole_hp(y[:,c],sr,cfg.hp_hz)
    # bounded soft saturation
    y=np.tanh(y*(1+4*cfg.saturation))/(np.tanh(1+4*cfg.saturation)+1e-12)
    # simple RMS-like static compression, deterministic and allocation-free per call
    rms=float(np.sqrt(np.mean(y*y)+1e-12)); gain=1.0/(1.0+cfg.compression*max(0.0,rms-0.18)/0.18)
    y*=gain
    peak=float(np.max(np.abs(y))) if y.size else 0.0
    if peak>cfg.ceiling: y*=cfg.ceiling/peak
    return y
