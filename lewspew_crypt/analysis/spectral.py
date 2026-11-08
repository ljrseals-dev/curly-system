from __future__ import annotations
import numpy as np
from ..ir import InstrumentIR, Partial

def parabolic_peak(y: np.ndarray, k: int) -> float:
    if k <= 0 or k >= len(y)-1: return float(k)
    a,b,c = float(y[k-1]),float(y[k]),float(y[k+1])
    den = a - 2*b + c
    return float(k) if abs(den) < 1e-20 else float(k + 0.5*(a-c)/den)

def analyze_spectrum(x: np.ndarray, sr: int, f0: float | None = None, n_fft: int = 65536) -> InstrumentIR:
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    if x.size == 0: raise ValueError("empty signal")
    x=np.nan_to_num(x, nan=0.0,posinf=0.0,neginf=0.0)
    n=min(n_fft,x.size); w=np.hanning(n); spec=np.abs(np.fft.rfft(x[:n]*w)); freqs=np.fft.rfftfreq(n,1/sr)
    if f0 is None:
        band=(freqs>=40)&(freqs<=2000)
        f0=float(freqs[band][np.argmax(spec[band])]) if np.any(band) else 440.0
    peaks=[]
    for k in range(1,min(64,int((sr/2)/max(f0,1.0)))+1):
        target=k*f0; lo=np.searchsorted(freqs,target-0.08*f0); hi=np.searchsorted(freqs,target+0.08*f0)
        if hi<=lo: continue
        p=lo+int(np.argmax(spec[lo:hi])); pp=parabolic_peak(spec,p); f=float(pp*sr/n); a=max(float(spec[p]),1e-12)
        expected=k*f0; dev=1200*np.log2(max(f,1e-12)/expected)
        peaks.append(Partial(k,f,20*np.log10(a/(spec.max()+1e-12)),dev,0.0,max(0.0,1.0-abs(dev)/100)))
    harmonicity=float(np.mean([p.confidence for p in peaks])) if peaks else 0.0
    harmonic_power=0.0
    for pp in peaks:
        k=int(np.argmin(np.abs(freqs-pp.frequency_hz)))
        lo=max(0,k-1); hi=min(len(spec),k+2)
        harmonic_power += float(np.sum(spec[lo:hi]**2))
    total_power=float(np.sum(spec**2))+1e-12
    noise_power=max(total_power-harmonic_power,1e-12)
    hnr=10*np.log10(max(harmonic_power,1e-12)/noise_power)
    return InstrumentIR(name="analyzed",f0_hz=f0,harmonicity=harmonicity,hnr_db=hnr,partials=peaks)
