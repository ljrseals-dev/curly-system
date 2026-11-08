from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
import json

@dataclass
class Partial:
    harmonic: int
    frequency_hz: float
    amplitude_db: float
    deviation_cents: float = 0.0
    snr_db: float = 0.0
    confidence: float = 0.0

@dataclass
class InstrumentIR:
    name: str
    family: str = "hybrid"
    archetype: str = "harmonic"
    f0_hz: float = 440.0
    harmonicity: float = 1.0
    hnr_db: float = 0.0
    attack_ms: float = 10.0
    decay_ms: float = 120.0
    sustain: float = 0.75
    release_ms: float = 180.0
    partials: List[Partial] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class GraphNode:
    id: str
    op: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SynthGraph:
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[List[str]] = field(default_factory=list)

@dataclass
class Patch:
    name: str
    seed: int
    source: InstrumentIR
    graph: SynthGraph
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
