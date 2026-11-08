from __future__ import annotations
from dataclasses import dataclass,asdict
import numpy as np
from .dsp.voodoo import VoodooConfig,process
from .analysis.spectral import analyze_spectrum
from .compiler.compiler import Compiler,prompt_to_ir
from .codec.fsq import FSQ
@dataclass
class CryptConfig:
    sample_rate:int=48000
    ceiling_dbfs:float=-1.0
    saturation:float=.10
    fsq_levels:tuple=(8,5,5,5)
@dataclass
class ProcessReport:
    samples:int
    channels:int
    peak_in:float
    peak_out:float
    rms_in:float
    rms_out:float
    finite:bool
    ceiling_ok:bool
class CryptEngine:
    def __init__(self,cfg=CryptConfig()): self.cfg=cfg; self.compiler=Compiler(); self.fsq=FSQ()
    def process_audio(self,x):
        x=np.asarray(x,dtype=np.float64); y=process(x,self.cfg.sample_rate,VoodooConfig(saturation=self.cfg.saturation,ceiling=10**(self.cfg.ceiling_dbfs/20)))
        peak_in=float(np.max(np.abs(x))) if x.size else 0.; peak_out=float(np.max(np.abs(y))) if y.size else 0.
        return y,ProcessReport(x.shape[0] if x.ndim else 0,1 if x.ndim==1 else x.shape[1],peak_in,peak_out,float(np.sqrt(np.mean(x*x)+1e-12)),float(np.sqrt(np.mean(y*y)+1e-12)),bool(np.all(np.isfinite(y))),peak_out<=10**(self.cfg.ceiling_dbfs/20)+1e-9)
    def analyze(self,x): return analyze_spectrum(x,self.cfg.sample_rate)
    def compile_prompt(self,prompt,seed=None): return self.compiler.compile(prompt_to_ir(prompt),seed=seed)
