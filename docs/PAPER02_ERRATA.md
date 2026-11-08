# Paper 02 — errata and amendments found while continuing the GPU work

Found by reading the manuscript against the code and by direct execution of the NumPy parts. Suggested edits, not applied to the manuscript.

1. **§4.1 worked example is numerically wrong.** For `x = [0.3, 0.7, -0.2, 0.9]`, levels `[8,5,5,5]`, the correct per-dimension indices are `[5, 3, 2, 4]` (not `[4, 4, 1, 5]`). The shared-max bug gives `[5, 6, 3, 7]`. The snippet also still contains the line "wait, this matches. Let me redo." Replace the example with the values above.
2. **§4.1 "silent on average" overstates it.** Under the shared-max bug the indices on 5-level dimensions reach 7 (legal max 4), and the bug changes the codes on ~99.6% of random rows. It is invisible to the loss, but the Part 2 range check catches it immediately.
3. **§4.4 / Appendix B Triton test cannot detect the failure it names.** `randn * 3` through `tanh` produced 0 exact rounding ties in 32,768 values. A half-to-even vs half-away rounding difference would pass. Add inputs on exact ties (5-level dimension at ±0.25, ±0.75); `tests/golden/fsq_vectors.json` and `verify()` do.
4. **Part 1 omits FSQ gradient flow.** Only SSRR is checked. In 1.0.0 `zq = z + (zq - z).detach()` left `from_fsq` with no gradient (by reading; not run). Part 1b now asserts gradient on `to_fsq`, `from_fsq` and the encoder input.
5. **§3.4 describes a per-dimension kernel loop with scatter writes.** 1.1.0 uses one fused kernel with per-element level lookup (`off % dims`) that emits indices only; dequant is shared with the eager path. Update §3.4 and the "cheap loop" paragraph.
6. **§6.2 claims `.5`-boundary mismatch is "a fraction of a percent".** That was never measured; with tie-free random input the rate would be ~0, and on tie inputs it is up to 100% of the tie elements. Remove or measure.
