# LewSpEW Crypt Architecture

```text
Prompt / Audio
      |
      +--> Analysis ----------------------+
      |     FFT / peaks / harmonicity     |
      |                                    v
      +--> Identity ------------------> Instrument IR
                                           |
                                           v
                              B + Tg + Tp + Tm + C
                                           |
                                           v
                                      Synth Graph
                                           |
                              +------------+-----------+
                              |                        |
                           Render                  FSQ Codec
                              |                        |
                              v                        v
                         Voodoo Water             token stream
                              |
                              v
                         Validation
                              |
                              v
                         Crypt Manifest
```

The compiler boundary is deliberately separate from the DSP runtime. External synthesis formats can be parsed into the IR without copying their runtime implementation into Crypt. This follows the earlier LewSpEW architecture decision to separate source formats, IR, generated DSP and runtime. 

## Real-time boundary

The current Python implementation is a reference engine, not a DAW-certified real-time runtime. The DSP API avoids file I/O and networking in processing functions, but Python allocation and interpreter overhead remain. A production JUCE/SynthEdit port should preserve the IR and validation contracts while moving the hot path to C++.
