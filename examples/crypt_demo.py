import numpy as np
from lewspew_crypt import CryptEngine

e=CryptEngine(); t=np.arange(48000)/48000; x=.8*np.sin(2*np.pi*440*t)+.2*np.sin(2*np.pi*880*t)
y,report=e.process_audio(x)
print(report)
print(e.compile_prompt('modern rock and roll flute',42).dumps())
