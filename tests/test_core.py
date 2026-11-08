import numpy as np
from lewspew_crypt import CryptEngine

def test_finite_and_ceiling():
 e=CryptEngine(); x=np.sin(2*np.pi*440*np.arange(48000)/48000)*2
 y,r=e.process_audio(x); assert r.finite and r.ceiling_ok

def test_deterministic_compile():
 e=CryptEngine(); a=e.compile_prompt('modern rock and roll flute',42).dumps(); b=e.compile_prompt('modern rock and roll flute',42).dumps(); assert a==b

def test_fsq():
 from lewspew_crypt.codec import FSQ
 q=FSQ(); z,i=q.quantize(np.zeros((4,4),np.float32)); assert q.cardinality==1000 and np.all(i[:,0]==4)
