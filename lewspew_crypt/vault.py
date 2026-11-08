from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,time

def hash_bytes(data:bytes)->str: return hashlib.sha256(data).hexdigest()
def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
@dataclass
class CryptManifest:
    name:str='curly-octo-engine'
    version:str='1.1.0'
    algorithm:str='SHA-256'
    generated_utc:str=''
    files:dict=None
    def build(self,root):
        root=Path(root); self.generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()); self.files={}
        for p in sorted(root.rglob('*')):
            if p.is_file() and '.git' not in p.parts: self.files[str(p.relative_to(root))]=hash_file(p)
        return self
    def dumps(self): return json.dumps(asdict(self),indent=2,sort_keys=True)
