"""LewSpEW Crypt — deterministic audio analysis, compilation, and DSP core."""
from .core import CryptEngine, CryptConfig, ProcessReport
from .ir import InstrumentIR, SynthGraph, Patch
from .vault import CryptManifest, hash_file, hash_bytes
__version__ = "1.1.0"
__all__ = ["CryptEngine", "CryptConfig", "ProcessReport", "InstrumentIR", "SynthGraph", "Patch", "CryptManifest", "hash_file", "hash_bytes"]
