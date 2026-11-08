from __future__ import annotations
import argparse,json
import numpy as np
from .core import CryptEngine

def main():
 p=argparse.ArgumentParser(prog='lewspew-crypt'); s=p.add_subparsers(dest='cmd',required=True)
 c=s.add_parser('compile'); c.add_argument('prompt'); c.add_argument('--seed',type=int)
 a=s.add_parser('analyze'); a.add_argument('file'); a.add_argument('--sr',type=int,default=48000)
 args=p.parse_args(); eng=CryptEngine()
 if args.cmd=='compile': print(json.dumps(eng.compile_prompt(args.prompt,args.seed).to_dict(),indent=2))
 else:
  import wave
  with wave.open(args.file,'rb') as w:
   n=w.getnframes(); ch=w.getnchannels(); raw=w.readframes(n); x=np.frombuffer(raw,dtype=np.int16).astype(np.float64)/32768
  if ch>1: x=x.reshape(-1,ch).mean(1)
  print(json.dumps(eng.analyze(x)._asdict(),indent=2,default=lambda o:o.__dict__))
if __name__=='__main__': main()
