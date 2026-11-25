# Experiment Log

| ID | Task | Status | Result |
|----|------|--------|--------|
| EXP-0001 | q4_0 x q8_0 AVX-512 VNNI GEMV vs ggml | PASSED (compute-bound) / NEEDS_RESEARCH (DRAM) | 1.26x faster L3-resident, 0.88x when DRAM-bound |
| EXP-0002 | Multi-threaded GEMV + thread pool | PASSED | 19.20 -> 82.77 GB/s (4.31x) at 32 threads; 90.84% coverage |
| EXP-0003 | Real weights from the shipped models | PASSED | quantiser byte-identical to ggml; 1.11x L3, 0.88x DRAM at 1 thread |
| EXP-0004 | Load guard for benchmarks | PASSED | KATA_MAX_THREADS + duty cycle; noise 10-30% -> 0.2% |
| EXP-0005 | Where single-thread GEMV time goes | PASSED | instruction-throughput bound, not latency; 4-row interleave is a loss in DRAM regime |
| EXP-0006 | q4_K GEMV on real model weights (CPU-first format pivot) | PASSED | 1.006x ggml at 26.95 GB/s DRAM-bound; AVX-512 path 0.83x, rejected |

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

## EXP-0003 — Real weights from the shipped models

**Status:** PASSED (loader + quantiser) / REJECTED (kernel is slower DRAM-bound).

**Why.** Every number in EXP-0001/0002 came from synthetic buffers. The shipped
models contain no q4_0 tensors at all — LFM2.5-24B-A2B-Q4_K_M is 77.6% Q4_K
(10.49 GiB) and 22.3% Q6_K (3.01 GiB); Qwen3.6-35B-A3B is 61.6% IQ3_XXS. So the
test takes a real Q4_K tensor, dequantises it with ggml's own reference, and
requantises with kata's quantiser. That validates kernel *and* quantiser against
real weight distributions.

**Implementation.** `include/kata/gguf.h` + `src/gguf.cpp` (GGUF v2/v3 reader:
metadata, tensor table, mmap, bounds-checked slice access, block-size/type-size
tables for 30 ggml types). `bench/bench_real_weights.cpp` dlopens libggml-base
and libggml-cpu, calls `ggml_cpu_init()` (without it ggml's f16 table is all
zeros and every dot returns 0.00000000 — cost an hour), and runs three kernels
over the same bytes with alternating order so neither side inherits a warm cache.

**Found and fixed a real bug in the quantiser.** kata's q4_0 scale was `-maxv/amax`,
which looks right but is not ggml's rule: ggml uses `d = maxv / -8`, anchoring the
extreme weight onto the code -8. With our rule 12076 of 753664 blocks (1.6%) got a
different scale. Also switched f16 rounding from half-up to nearest-even, since
scale packing is exactly where ties land. After both fixes the quantiser is
**byte-identical to ggml on all 753664 blocks**: 0 scale diffs, 0 nibble diffs.

**Correctness.** 21/21 tests (2 new: `F16.MatchesRoundToNearestEven` compares
20000 random values against an independent RNE implementation, and
`Quant.Q4ScaleAnchorsExtremeWeight`). Coverage 90.33% of kata sources.
Numerical agreement on real weights, normalised by max|ref|: kata 2.87e-07,
ggml 3.44e-07 — same order, both well inside f32 noise.

**Measured**, LFM2.5 `blk.0.ffn_gate` (13 MiB, L3-resident) — same bytes both sides:

| threads | kata GB/s | ggml GB/s | speedup |
|---|---|---|---|
| 1 | 23.21 | 20.79 | 1.12x |
| 8 | 134.14 | 80.54 | 1.67x |
| 16 | 152.11 | 131.19 | 1.16x |
| 32 | 202.23 | 146.20 | 1.38x |

**Measured**, `token_embd` (205 MiB, genuinely DRAM-bound):

| threads | kata GB/s | ggml GB/s | speedup |
|---|---|---|---|
| 1 | 16.86 | 20.38 | **0.83x** |
| 8 | 119.37 | 153.56 | **0.78x** |
| 16 | 140.05 | 145.42 | 0.96x |
| 32 | 174.14 | 183.89 | 0.95x |

**Analysis.** The advantage inverts the moment the weights leave L3, and the
pattern matches EXP-0001 exactly. Tried the obvious fix — a one-stream variant
(`gemv_q4_0_q8_0_avx512_r1`) to test whether the 4-row interleave was splitting
the TLB/pagetable walk across four distant regions. It was **not** faster
(r1 137.78 GB/s vs 119.37 at 8 threads, still below ggml), so the interleave is
not the cause. Also tried `_mm_prefetch(wr + b + 8)`: no effect, so the hardware
prefetcher is already handling the sequential stream.

A measurement flaw was found and corrected mid-experiment: the first runs always
executed kata before ggml, giving ggml a warm cache on a buffer that half-fits L3.
Alternating the two removed a 5-15% bias. Run-to-run spread is 10-30% on this
machine at high thread counts even pinned, so only >=10% differences are read as
real.

**Conclusion.** Keep the loader and the fixed quantiser; treat the kernel as
*compute*-competitive and *memory*-inferior. Do not push further on the GEMV inner
loop — it is already at the machine's bandwidth ceiling (202 GB/s effective at 32
threads on 13 MiB, which only works because that tensor lives in L3). The honest
next target is a memcpy-style sequential read of the same 205 MiB to establish
the true single-thread DRAM ceiling, which tells us whether 16.8 GB/s is our
kernel or the machine.

**Incident.** During this experiment the machine hard-reset with no shutdown
sequence in the journal (last entry 17:33:16, reboot 17:37:25) — no OOM, no MCE,
no thermal trip logged. Preceding coredumps were ggml SIGSEGVs from a mistyped
`ggml_vec_dot_t` signature (fixed: it takes `(n, s, bs, x, bx, y, by, nrc)`),
not a hardware fault. Cause of the reset remains unidentified; if it repeats under
the same memory-heavy benchmark, treat it as a suspect PSU/RAM stability issue.

## EXP-0004 — Load guard for benchmarks (and a rejected micro-optimisation)

**Date:** 2026-11-25
**Status:** PASSED (guard) / REJECTED (correction hoist)
**Hypothesis:** (a) benchmark harnesses can be made safe on a box that hard-resets
under sustained all-core memory saturation; (b) hoisting the q4_0 `-8` bias
correction out of the inner loop would recover the 25% it appeared to cost.

### (a) Load guard — PASSED

`include/kata/throttle.h` adds an env-driven cap consulted by every harness:
`KATA_MAX_THREADS` (default `hardware_concurrency()/2`, i.e. the physical core
count) and `KATA_DUTY` (default 0.90). `throttle::pause_since(t0)` idles
`(1-duty)/duty` of the elapsed work time after each iteration.

- `bench_gemv` caps `--threads` and prints `threads 32 capped to 16`.
- `bench_real_weights` clamps its thread list through the same cap.
- `ThreadPool::default_threads()` now returns the capped physical core count
  instead of `hw-1`.
- `/tmp/safe_run.sh` wraps any run with `flock` so two benchmarks cannot overlap.

Verified: `--threads 32` → capped to 16, `run_spread` 6.5%; single-thread run at
90% duty completes with `run_spread=0.2%`.

### (b) Correction hoist — REJECTED

The `-8` bias correction depends only on the activation row, so it can be
precomputed once per call instead of `rows * nb` times. A microbenchmark measured
the no-correction variant at 22.48 GB/s against 17.97 for the full kernel, which
looked like 25% of headroom.

Implemented as `activation_corrections()` writing into a `thread_local
std::vector<int32_t>`, loaded back with `_mm256_loadu_si256` per block.

Measured A/B, same binary layout, single thread pinned, 288 MiB working set:

| variant | GB/s |
|---|---|
| recompute per block (baseline) | 16.63 |
| hoisted per call | 16.30 |

**Rejected — 2% slower.** The saved ALU work is invisible because the loop is
memory-bound, while the extra pass over the activation row costs a real
cache-resident read. Reverted with `git checkout src/gemv.cpp`.

**Lesson.** On this kernel the "-8 correction costs 25%" measurement was an
artifact of comparing against a kernel that also skipped a dependency chain; the
correction is free in the memory-bound regime. Do not micro-optimise ALU on a
kernel whose measured gap to `memcpy` is 4x.

### Incident log

Four hard resets during this session (17:33, 17:44, 17:53, 17:55). Three ended
with no shutdown sequence at all in the journal; one was a clean
`systemd-shutdown`. No OOM, no MCE, no thermal trip in any of them. Resets
cluster on multi-threaded memory-bandwidth benchmarks. Two additional "crashes"
turned out to be my own `abort()` from an uncaught `gguf: unsupported tensor type
18` exception — the GGUF reader lacked IQ-type sizes, now added for types
16-19, 21, 22, 29.

A third failure mode appeared: a reset during `cmake --build` left
`build/bench_real_weights`, `build/bench_gemv`, `build/bench_vs_ggml`,
`build/kata_tests` and one `.o` file at **0 bytes**, and ninja then reported
"no work to do" because the timestamps were newer than the sources. Always check
`find build -size 0 -type f` after a reset.

**Guard added to this log's procedure:** no benchmark may exceed
`KATA_MAX_THREADS`, and every harness must idle `1-KATA_DUTY` of wall clock.

## EXP-0005 — Where the single-thread GEMV time actually goes

**Date:** 2026-11-25
**Status:** PASSED (diagnosis only, no kernel change kept)
**Hypothesis:** the 1-thread gap (15.7 GB/s) against achievable bandwidth (43-79
GB/s) is a dependency-chain latency problem, so more independent accumulators
would close it.

**Setup.** Synthetic q4_0 weights generated from N(0,1) and quantised with kata's
own `quantize_row_q4_0`, so this isolates the kernel from weight distribution.
576 MB working set (L3 is 96 MB), single thread pinned with `taskset -c 0`, best
of 4 iterations, 50% duty cycle via `throttle`.

### Access-pattern ablation (no arithmetic, same bytes)

| shape | GB/s |
|---|---|
| sequential, 32-byte aligned | 79.07 |
| sequential, 18-byte blocks | 65.48 |
| 4-row interleaved tile walk | 35.58 |
| full GEMV kernel | 17.03 |

The cost splits roughly in half: the 4-row tile pattern gives up ~1.8x against a
sequential walk, and the arithmetic gives up another ~2.1x.

### Arithmetic ablation (4-row kernel, 576 MB)

| variant | GB/s | what it isolates |
|---|---|---|
| touch bytes only (`xor`) | 67.29 | the machine's 1-thread ceiling |
| integer accumulate, no fp | 35.61 | unpack + `vpdpbusd` cost 1.9x |
| `raw_qs`, no nibble unpack | 23.84 | |
| no `-8` bias correction | 22.04 | correction costs 1.33x |
| full kernel | 16.58 | |

### Chain-count experiment — hypothesis REJECTED

The dependent chain is `vpdpbusd`(5) -> `sub`(1) -> `cvtepi32_ps`(3) -> `fmadd`(4),
13 cycles. With 4 rows that is 3.25 cycles/block against 4.9 measured, so latency
looked like the limiter. Measured instead:

| shape (rows x K-chains) | GB/s |
|---|---|
| lib_best 4x1 | 15.31 |
| **1x1** | **16.94** |
| 4x1 | 15.54 |
| 4x2 | 15.01 |
| 4x4 | 11.04 |
| 8x2 | 13.42 |

More chains make it *worse*, and a single row single chain is the fastest of all.
So the kernel is **instruction-throughput bound, not latency bound**. Also note
the 4-row interleave, which EXP-0001 introduced and EXP-0003 kept, is a net loss
in the DRAM regime — the 1x1 shape wins.

### Two other candidate fixes, both rejected by measurement

- **Algebraic removal of the bias correction.** `sum((u-8)y) = sum(uy) - 8*sum(y)`,
  with the `-8*sum(y)` term moved to a second pass over the 2-byte scales. Verified
  numerically exact on a small matrix (`-6.388366` vs `-6.388367`), then measured:
  11.03 GB/s against 15.27. **Rejected** — reading 2 bytes out of every 18-byte
  block touches every cache line again, so the second pass costs a full extra
  memory traversal.
- **Software prefetch at distances 4/8/16/24/32/48/64 blocks.** Best case 16.88
  GB/s vs 15.26 unsignalled at 288 MB, but the ordering was not controlled and the
  repeat run inverted it (17.61 baseline vs 14.06 prefetch). **Not adopted** —
  effect is inside the noise floor and the hardware prefetcher already covers a
  sequential stream.

### Byte-level conclusion

| format | bytes/value | values per scale | scale ALU per byte |
|---|---|---|---|
| q4_0 | 18/32 = 0.5625 | 32 | 1 per 0.5625 B |
| q4_k | 144/256 = 0.5625 | 32 (8 sub-blocks of 32, 6-bit scales) | 1 per 4.5 B |

q4_0 and q4_k occupy **identical bytes per value**, but q4_k amortises scale
metadata over 8x more values. That is exactly the 1.33x the correction costs in
the ablation. And q4_k is what the user's models actually contain (77.6% of
LFM2.5-24B-A2B-Q4_K_M). Further q4_0 tuning is therefore capped: we and ggml sit
at 17-20 GB/s with the same instruction budget, against a 67 GB/s memory ceiling.

**Next:** implement `vec_dot_q8_0_q4_K` / q4_K GEMV. That is both the real workload
and the format whose metadata cost is 8x smaller.

## EXP-0006 — q4_K GEMV on real model weights

**Status:** PASSED
**Date:** 2026-11-25
**Hypothesis:** q4_K carries the same 0.5625 bytes/value as q4_0 but amortises
scale metadata over 8x more values (EXP-0005), so a q4_K kernel should reach the
same bandwidth as ggml's best AVX2 path while consuming the format the shipped
models actually contain — and unlike q4_0 it has no per-block bias correction to
pay for.

### Problem

All previous kernels were q4_0, a format that appears in **none** of the user's
models (LFM2.5-24B-A2B-Q4_K_M is 77.6% Q4_K / 22.3% Q6_K by bytes; Qwen3.6-35B
is 61.6% IQ3_XXS). Optimising q4_0 was therefore optimising a format we would
never load.

### Implementation

- `include/kata/types.h`: `QK_K = 256`, `K_SCALE_SIZE = 12`, `block_q4_K`
  (144 B, static_asserted) and `block_q8_K` (292 B, static_asserted).
- `src/quant.cpp`: `q4_K_scale_min`, `quantize_row_q8_K` (ggml's `-127/max`
  rule, byte-identical output verified), `dequantize_row_q4_K`,
  `vec_dot_q4_K_q8_K_ref`.
- `src/gemv.cpp`: `gemv_q4_K_q8_K_avx2_impl` — one row per pass, sequential
  weight stream, no bias correction in the hot loop (the min term rides on
  `block_q8_K.bsums`), `__attribute__((target(...)))` so one binary runs anywhere.
- `bench/bench_k4.cpp`: lifts the tensor **verbatim** out of the .gguf (no
  dequantise/requantise round trip), replicates whole rows past L3, and times
  kata against `ggml_vec_dot_q4_K_q8_K`.

### Bug found and fixed on the way

`f16_to_f32` was wrong for **every** f16 subnormal: it used `112 - shift` instead
of `113 - shift`, so each one came out at half its true value. Real weights hit
this — blocks 3, 4 and 6 of `blk.0.ffn_gate.weight` have super-block scales
`0x0350`, `0x0398`, `0x03b5`, i.e. exponent field 0. Symptom: 739 of 2048
dequantised elements differed from ggml and the dot came out `-0.755659` instead
of `-0.256831`. Regression test `F16.SubnormalsAreNotHalved` now walks all 1023
subnormals and round-trips them.

### Correctness

`max|kata-ggml| / max|ggml| = 5.007e-08` over 64 rows of a real Q4_K tensor.
Against an exact dequantised dot the residual is 4.1e-03, which is the q8_K
activation quantisation, not kernel error. 28/28 tests, coverage 82.89%.

### Benchmark

1 thread pinned to CPU 0, 776 MB working set (60 row-replications of the 12.9 MB
tensor), median of 7 reps.

| Metric | kata | ggml | ratio |
|---|---|---|---|
| Throughput (GB/s) | 26.95 | 26.78 | **1.006x** |
| p10 | 23.09 | 26.70 | |
| p90 | 27.35 | 27.18 | |

Ablation of the same instruction stream at 776 MB (what the ALU actually costs):

| variant | GB/s | share of peak |
|---|---|---|
| maddubs only (no scales, no min) | 36.43 | 1.00 |
| no per-sub-block shuffle | 36.35 | 1.00 |
| no scale unpack | 30.25 | 0.83 |
| no min term | 32.98 | 0.91 |
| **full kernel** | **24.79** | **0.68** |

### Rejected: AVX-512 VNNI variant

Written and measured. It replaces `maddubs`+`madd` with `dpbusd` and removes the
per-sub-block byte shuffle entirely (a sub-block scale is constant across the 64
values one zmm holds, so the shuffle degenerates to a broadcast `mullo`).
Result: **22.13 GB/s vs 26.83 for the AVX2 loop, 0.83x.** Zen 4 splits every
512-bit op into two 256-bit halves, so the two extra zmm inserts plus 32-bit
`mullo` cost more than the byte shuffle they removed. Kept in the log, code
deleted. Revisit only on a genuinely 512-bit-wide core.

### Conclusion

kata now matches ggml on the format that matters, at 1 thread. Against a 64-79
GB/s memory ceiling there is headroom, but the ablation says it is ALU-side, not
access-side: the full kernel runs at 0.68 of what the same bytes cost with the
scale arithmetic stripped. The min term and the scale unpack are the two
components worth attacking next.

### Next

Multi-row tiling for q4_K (the AVX2 kernel re-reads the activation block from L1
once per row; 2-4 rows would amortise it), plus a multi-threaded q4_K driver and
a real A/B against ggml at 8/16 threads.
