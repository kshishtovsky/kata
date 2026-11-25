# Experiment Log

| ID | Task | Status | Result |
|----|------|--------|--------|
| EXP-0001 | q4_0 x q8_0 AVX-512 VNNI GEMV vs ggml | PASSED (compute-bound) / NEEDS_RESEARCH (DRAM) | 1.26x faster L3-resident, 0.88x when DRAM-bound |
| EXP-0002 | Multi-threaded GEMV + thread pool | PASSED | 19.20 -> 82.77 GB/s (4.31x) at 32 threads; 90.84% coverage |

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

## EXP-0002 — Multi-threaded GEMV

**Status:** PASSED.

**Hypothesis.** EXP-0001 showed single-thread decode is capped near 12-19 GB/s while
the machine sustains ~66 GB/s at 8 threads, so partitioning rows across a static
pool should scale almost linearly. The open question was how far.

**Implementation.** `src/pool.cpp` (`ThreadPool`, generation-counter fan-out, caller
thread acts as worker 0), `gemv_q4_0_q8_0_mt` in `src/gemv.cpp` — contiguous row
block per worker so each thread streams one region of the weight matrix.

**Correctness.** 19/19 tests (6 new: 4 pool invariants incl. 100 repeated fan-outs
and disjoint-range writes, 2 GEMV: 37 rows across 1/2/3/8 threads vs reference,
tiny matrices). Coverage **90.84%** of kata sources (`scripts/coverage.sh`);
per file: pool.cpp 100%, gemv.cpp 93.55%, cpu.cpp 85.71%, quant.cpp 84.62%.

**Measured** (512 MB and 1 GiB weight buffers, DRAM-bound, median of 5 whole-run reps):

| Threads | GB/s (1 GiB) | vs 1 thread |
|---|---|---|
| 1 | 19.20 | 1.00x |
| 4 | 60.78 | 3.17x |
| 8 | 74.96 | 3.90x |
| 16 | 77.64 | 4.04x |
| 24 | 80.64 | 4.20x |
| 32 | 82.77 | 4.31x |

**Analysis — this CPU needs special treatment.** The 7950X3D has two CCDs, and only
CCD0 (cores 0-7) has the 96 MB 3D V-cache (`/sys/.../index3/shared_cpu_list` →
`0-7,16-23` and `8-15,24-31`). Measured per-CCD, 8 threads pinned:

| Affinity | GB/s |
|---|---|
| CCD0 (0-7), 8 threads | 67.5 (very stable, ~0.1% spread) |
| CCD1 (8-15), 8 threads | 59.6 |
| CCD0 + SMT (0-15), 16 threads | 76.5 |
| CCD1 + SMT (8-23), 16 threads | 76.7 |
| all 32 | 80.1-83.6 |

So SMT gives +13% (a memory-latency hiding win, not a compute win), and crossing
to the second CCD is what unlocks the last ~8%. Unpinned runs showed 10-30% run
spread, pinned runs 0.1-2.5% — **all future benchmarks must pin affinity**.

**Conclusion.** Keep. `default_threads()` returns hw-1 (31 here) which lands in the
plateau; 16 threads gets 94% of peak for half the threads and is the better default
for a decode loop that also has to do sampling and graph work. Revisit once a real
model runs end to end — the pool must not starve the rest of the pipeline.

**Next iteration (EXP-0003).** The 1.26x L3-resident lead from EXP-0001 vanished
into memory bandwidth, which means kernel quality is currently invisible. Before
more kernel work: measure q4_0/q8_0 *quantisation* and build a correctness harness
against llama.cpp on a real model tensor, so later kernel changes are validated
against real weights rather than synthetic noise.
