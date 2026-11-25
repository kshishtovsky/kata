# Experiment Log

| ID | Task | Status | Result |
|----|------|--------|--------|
| EXP-0001 | q4_0 x q8_0 AVX-512 VNNI GEMV vs ggml | PASSED (with caveat) | 1.26x faster while L3-resident, 0.88x when DRAM-bound |

## EXP-0001 — q4_0 x q8_0 GEMV kernel

**Status:** PASSED for the compute-bound regime, NEEDS_RESEARCH for DRAM-bound.

**Hypothesis.** Resolving the q4_0 nibble pairings directly with VNNI
(`_mm256_dpbusd_epi32`) over 4 weight rows per activation load, and reading the
f16 scales with F16C `_cvtsh_ss` instead of a scalar conversion, beats ggml's
AVX2 path on this Zen 4 CPU.

**Implementation.** `src/gemv.cpp`. Weight nibbles stay unsigned 0..15 through
VNNI and the -8 offset is corrected once per block as `sum(u*y) - 8*sum(y)`,
computed with `maddubs(1, qy)` + `slli 3`. Four rows share one activation load.
Reference: `gemv_q4_0_q8_0_ref` and `vec_dot_q4_0_q8_0_ref` in `src/quant.cpp`.

**Correctness.** 13/13 gtest cases. Max relative error `kata=2.77e-08` vs the
scalar reference, identical to ggml's `ggml_vec_dot_q4_0_q8_0` (`2.77e-08`) on
the same data — both are within fp32 accumulation noise, neither drifts.

**Measured** (Ryzen 9 7950X3D, single thread, median of 100-200 runs):

| Workload | kata | ggml (same loop) | Ratio |
|---|---|---|---|
| 4096x4096, 9 MB weights, L3-resident | 22.03 GB/s | 17.50 GB/s | 1.26x |
| 233016x4096, 512 MB weights, DRAM | 18.36 GB/s | 20.91 GB/s | 0.88x |

**Analysis.** Two separate findings.

1. VNNI + F16C is a genuine win while data is in cache, exactly as hypothesized.
2. When DRAM-bound, the 4-row interleave *loses* to ggml. Most likely cause: four
   weight streams ~18 KB apart (233016 rows x 18 B... the streams are `rows*nb*18`
   apart, i.e. megabytes) destroy TLB and page-level locality that ggml's
   sequential single-row walk keeps. Not yet proven.

**Machine bandwidth context** (`/tmp/bw2.c`, 1 GiB streaming read):
1 thread 12.56 GB/s, 4 threads 48.43, 8 threads 66.11, 16 threads 66.71 GB/s.
Single-thread decode is therefore capped near 12-19 GB/s regardless of kernel
quality. Threading, not SIMD, is the dominant remaining lever.

**Test coverage.** 13 tests over quantisation (f16 round trip, specials, q8 error
bound, zero block, nibble layout, ref-dot identity) and GEMV (ISA dispatch
consistency, all three kernels vs reference, odd/tail row counts, hand-computed
value). Coverage tooling not wired up yet — that is an open item.

**Conclusion.** Keep the AVX-512 kernel. Do not invest further in single-thread
SIMD tweaks; the ceiling is memory, not math.

**Next iteration (EXP-0002).** Multi-threaded GEMV via a static thread pool with
row-block partitioning, aiming at the 66 GB/s the machine can actually sustain.
Baseline to beat: 18.36 GB/s at 512 MB (8 threads -> theoretical ~5x).
