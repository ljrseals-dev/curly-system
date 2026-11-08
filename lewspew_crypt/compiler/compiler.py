from __future__ import annotations
import hashlib, json, re
from ..ir import InstrumentIR, SynthGraph, GraphNode, Patch

ARCHETYPES={
 'harmonic':['oscillator_bank','envelope','filter','saturation'],
 'subtractive':['oscillator','filter','envelope','saturation'],
 'fm':['carrier','fm_operator','envelope','filter'],
 'physical':['exciter','resonator','damping','envelope'],
 'noise':['noise','filter_bank','envelope','saturation'],
 'formant':['oscillator','formant_bank','envelope'],
 'resonator':['exciter','resonator','damping'],
 'spectral':['fft','spectral_shape','ifft','envelope'],
 'granular':['grain_source','grain_window','grain_mixer','envelope'],
 'hybrid':['oscillator_bank','resonator','filter','saturation']}

def stable_seed(text:str)->int: return int(hashlib.sha256(text.encode()).hexdigest()[:8],16)

class Compiler:
    def compile(self, ir:InstrumentIR, genre='generic', role='lead', seed=None, constraints=None)->Patch:
        seed=stable_seed(ir.name+genre+role) if seed is None else int(seed)
        ops=ARCHETYPES.get(ir.archetype,ARCHETYPES['hybrid'])
        nodes=[GraphNode(f'n{i:02d}',op,{}) for i,op in enumerate(ops)]
        for i,n in enumerate(nodes):
            n.params={'f0_hz':ir.f0_hz,'role':role,'genre':genre,'seed':seed}
        edges=[[nodes[i].id,nodes[i+1].id] for i in range(len(nodes)-1)]
        graph=SynthGraph(nodes,edges)
        meta={'compiler':'LewSpEW Crypt','formula':'B+Tg+Tp+Tm+C','constraints':constraints or {},'deterministic':True}
        return Patch(ir.name,seed,ir,graph,meta)

def prompt_to_ir(prompt:str)->InstrumentIR:
    t=prompt.lower(); known=['flute','guitar','piano','violin','saxophone','drum','bass','synth','voice','bell','organ']
    name=next((x for x in known if x in t),'hybrid')
    arch='fm' if 'fm' in t or 'digital' in t else 'physical' if any(x in t for x in ['acoustic','wood','bowed']) else 'harmonic'
    return InstrumentIR(name=name,archetype=arch,tags=re.findall(r'\b[a-z]+\b',t)[:32])
