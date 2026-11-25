# Experiment Log

| ID | Task | Status | Result |
|----|------|--------|--------|
| EXP-0001 | q4_0 x q8_0 AVX-512 VNNI GEMV vs ggml | PASSED (compute-bound) / NEEDS_RESEARCH (DRAM) | 1.26x faster L3-resident, 0.88x when DRAM-bound |
| EXP-0002 | Multi-threaded GEMV + thread pool | PASSED | 19.20 -> 82.77 GB/s (4.31x) at 32 threads; 90.84% coverage |
| EXP-0003 | Real weights from the shipped models | PASSED | quantiser byte-identical to ggml; 1.11x L3, 0.88x DRAM at 1 thread |
| EXP-0004 | Load guard for benchmarks | PASSED | KATA_MAX_THREADS + duty cycle; noise 10-30% -> 0.2% |
| EXP-0005 | Where single-thread GEMV time goes | PASSED | instruction-throughput bound, not latency; 4-row interleave is a loss in DRAM regime |
| EXP-0006 | q4_K GEMV on real model weights (CPU-first format pivot) | PASSED | 1.006x ggml at 26.95 GB/s DRAM-bound; AVX-512 path 0.83x, rejected |
| EXP-0007 | q4_K multi-row tiling + multi-thread driver | PASSED (threads) / REJECTED (tiling) | tiling 13.5 vs 26.7 GB/s; 1-2 threads 1.05-1.08x, 4 threads 0.99x (DRAM saturated) |
| EXP-0008 | q6_K GEMV (22.3% of the shipped models) | PASSED | 1.03x ggml at 35.92 GB/s; caught a two-sub-block-per-maddubs scale bug |
| EXP-0009 | RMSNorm + backend boundary | PASSED | 5.5-6.3x over scalar, 97-112 GB/s; coverage tool was under-reporting by 19pp |
| EXP-0010 | RoPE | PASSED | 1.05-1.15x over scalar after caching the frequency table; both layouts supported |
| EXP-0011 | Decode attention + KV layout | PASSED | 2.2-2.9x over scalar; 17.5 ms at 32k context, becomes the dominant cost |
| EXP-0012 | Grouped-KV decode attention | FAILED | All 4 variants slower at 32k (24.2 vs 17.5 ms); reverted |

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

## EXP-0007 — q4_K multi-row tiling and the multi-thread ceiling

**Status:** PASSED for the thread driver, REJECTED for tiling
**Date:** 2026-11-25

### Hypothesis

The AVX2 q4_K kernel re-reads the activation block from L1 once per row. Two to
four rows per pass should amortise that decode (scale unpack, bsums, the
per-sub-block shuffle) across R rows.

### Tiling — REJECTED

`template <int R> gemv_q4_K_q8_K_tiled_impl` written for R = 2, 4, 8 and measured
at 776 MB, 1 thread pinned:

| variant | GB/s | vs ggml |
|---|---|---|
| untiled | 26.73 | 1.002x |
| tiled2 | 13.45 | 0.504x |
| tiled4 | 13.87 | 0.520x |
| tiled8 | 15.63 | 0.586x |

**Half speed, not faster.** The activation block was never the cost — it is 292
bytes re-read from L1, and L1 bandwidth is nowhere near the limit. What tiling
actually does is turn one sequential weight stream into R streams megabytes
apart, which costs TLB and page locality. This is the same mechanism that made
the 4-row interleave lose in EXP-0001 and EXP-0005. **Three independent
measurements now agree: on this machine, extra concurrent weight streams always
lose.** Code deleted.

### Multi-thread — PASSED

Bench extended with `--threads`; rows split into contiguous per-worker chunks;
ggml parallelised the same way (its dot is single-row by contract) so the
comparison is like-for-like. 512 MB working set, affinity pinned.

| threads | kata GB/s | ggml GB/s | ratio |
|---|---|---|---|
| 1 (cpu 0) | 26.01 | 24.84 | **1.047x** |
| 2 (cpu 0-1) | 50.50 | 46.78 | **1.079x** |
| 4 (cpu 0-3) | 69.46 | 69.95 | 0.993x |
| 4 (9 reps) | 68.08 | 77.15 | 0.882x |

At 4 threads both land on ~70 GB/s, which is the DRAM ceiling measured
independently (77-79 GB/s peak, ~70 GB/s achieved at this access pattern). The
4-thread ratio wanders between 0.88x and 0.99x with p10 spread of 59-65 GB/s on
both sides — that is run-to-run noise, not a kernel difference. **Conclusion:
kata is at parity or slightly ahead of ggml for q4_K up to saturation, and there
is nothing left to win in this kernel — the ceiling is memory.**

28/28 tests, coverage 82.89%.

### Next

Stop tuning GEMV. The remaining decode cost is elsewhere: q6_K is 22.3% of
LFM2.5-24B-A2B by bytes and every `attn_v`/`output` tensor in both shipped models
is Q6_K, so a q6_K kernel is required before any forward pass can run.

## EXP-0008 — q6_K GEMV

**Status:** PASSED
**Date:** 2026-11-25
**Hypothesis:** q6_K needs a different kernel shape than q4_K (6-bit values are
split across two bit-planes, `ql` and `qh`), but the same one-row-per-pass
streaming structure should reach ggml's AVX2 throughput. q6_K is not optional:
22.3% of LFM2.5-24B-A2B-Q4_K_M by bytes, and every `attn_v`, `ffn_down` and
`token_embd` tensor in both shipped models uses it.

### Format

`block_q6_K` is 210 B: `ql[128]` (low 4 bits), `qh[64]` (high 2 bits), `scales[16]`
(**signed** int8, one per 16 values), `d` (f16) — note `d` is **last**, unlike
q4_K and q4_0 where it comes first. Value = `d * scales[is] * q - 32`, and the
-32 is folded out through the accumulated `bsums` rather than applied per value.

### Bug found and fixed

ggml's own AVX2 loop gave me the shape but I mis-sized the scale shuffle: one
`maddubs` consumes 32 values, which spans **two** 16-value sub-blocks, so the low
and high 128-bit halves of its result need **different** scales. I duplicated a
single scale across both halves. Symptom on a real model tensor:
`max|kata-ggml| / max|ggml| = 3.065`, while the scalar reference matched ggml to
`2.0e-07` — i.e. the reference was right and only the vector path was wrong, which
is exactly what an independent `_ref` implementation is for. Fixed with a
`scale_pair(s)` mask writing scale `s` into the low 64 bits and `s+1` into the
high 64 bits. Regression test `GemvK6.PerSubBlockScalesAreNotDuplicated` forces
alternating scales of 100 and 1 so any half-block misweighting is unmissable.

### Correctness

All 26 Q6_K tensors in the real model: dequantiser **bit-exact** against ggml
(`deqmaxdiff = 0.000e+00` on every tensor), dot agreement `1.14e-07` relative.
31/31 tests, coverage 83.50%.

### Benchmark

1 thread pinned to CPU 0, 769 MB working set (`blk.2.attn_v.weight`, 937 row
replications), median of 5 reps:

| Metric | kata | ggml | ratio |
|---|---|---|---|
| Throughput (GB/s) | 35.92 | 34.88 | **1.030x** |
| p10 | 34.49 | 33.49 | |
| p90 | 36.04 | 35.41 | |

q6_K is notably faster than q4_K for both engines (35.9 vs 26.0 GB/s) because a
210-byte block carries 256 values — 0.82 B/value against q4_K's 0.5625 — so fewer
blocks and fewer scale unpackings per byte streamed.

### Conclusion

Both formats that the shipped models actually use now have kernels at or above
ggml's AVX2 path on this machine. Keeping an independent scalar reference per
format paid for itself twice (q4_K subnormals, q6_K scale pairing): in both cases
the reference was correct and the vector code was not.

### Next

A forward pass needs more than GEMV: RMSNorm, RoPE, attention with a KV cache,
and the MoE router/dispatch. Next iteration: RMSNorm + the block/backends
boundary, so subsequent kernels plug into a backend interface rather than growing
free functions (CPU now, ROCm later — per the m00800 direction).

## EXP-0009 — RMSNorm and the backend boundary

**Status:** PASSED
**Date:** 2026-11-25
**Hypothesis:** RMSNorm is memory-bound at decode (2048 floats in, 2048 out), so a
straightforward AVX2 reduction should already sit at the bandwidth ceiling and
needs no trickery. Separately: kernel free functions are about to multiply
(RoPE, attention, MoE router), so the second kernel family is the right moment to
introduce the backend interface rather than after five of them exist.

### Implementation

- `include/kata/backend.h` — `class Backend` with `name()`, `available()`,
  `rms_norm(const RmsNormArgs&)`, `gemv_q4_K_q8_K(const GemvArgs&)`,
  `gemv_q6_K_q8_K(const GemvArgs&)`, plus `const Backend & backend()` for
  process-wide selection. Argument structs are plain data so a GPU backend can
  copy them to a queue without changing the signature.
- `src/backend.cpp` — `CpuBackend`, the only implementation today, delegating to
  the existing `*_best()` kernel selectors and `kata::rms_norm`.
- `include/kata/norm.h`, `src/norm.cpp` — `rms_norm_ref` and `rms_norm_avx2`.
  The reduction accumulates in `double` in the scalar path; summing 4096 squared
  floats in float loses enough precision to move the last digits of every output.

**Deliberately NOT in the interface:** RoPE. The first draft had
`virtual void rope(const RopeArgs&)` with an empty CPU body, i.e. a silent no-op
that would have produced plausible-looking garbage the first time attention ran.
Removed before commit. A kernel joins the interface when it has a real
implementation, not in anticipation.

### Correctness

`rms_norm_avx2` matches `rms_norm_ref` to 1.4e-07 relative across widths
1/7/8/15/16/23/64/257 with and without a weight vector, i.e. every unaligned-tail
combination. Zero-input rows give zero, not NaN. 37/37 tests.

### Benchmark

1 thread pinned to CPU 0, median of 200-300 calls, bytes counted as read x +
read w + write out:

| case | scalar | vector | speedup |
|---|---|---|---|
| 1 row x 2048 (decode shape) | 1.39 us / 17.7 GB/s | **0.22 us / 111.7 GB/s** | 6.32x |
| 256 rows x 2048 (prefill shape) | 358.5 us / 17.6 GB/s | **64.7 us / 97.2 GB/s** | 5.54x |

The vector path exceeds the 78 GB/s DRAM ceiling because a single 8 KB row stays
in L1/L2 across iterations — this measures the kernel, not the memory system.

### Tooling bug found

Coverage was under-reporting by ~19 points and `src/quant.cpp` was listed twice
(once at 98.09% of 209 lines, once at 57.07% of **1810** lines). gcov prints a
trailing aggregate `Lines executed:` line with no `File` line in front of it, and
`scripts/coverage.py` charged it to whichever file came last, also leaving `cur`
set so a following file inherited the wrong name. Fixed by clearing `cur` after
each match. Real total went from 64.31% to **86.01%**; the same script bug had
been quietly under-reporting every previous iteration's number.

### Conclusion

Both cheap, both kept. The honest note: 111 GB/s is a cache-resident figure, and
RMSNorm at decode scale (2048 floats) is ~0.22 us, which is small next to a single
GEMV row (2048 wide, ~0.08 us/row at 78 GB/s but thousands of rows per layer). So
RMSNorm is not a bottleneck and must not be optimised further until profiling of a
real forward pass says otherwise.

### Next

RoPE with a real implementation, then a single transformer block wired through the
backend so a full forward pass can be profiled end to end.

## EXP-0010 — RoPE

**Status:** PASSED
**Date:** 2026-11-25
**Hypothesis:** RoPE at decode is ~4k floats in and out, so it is small next to a
GEMV row. The goal is correctness on both pairing layouts, not speed; if the
vector path were not measurably better it should not exist.

### The two layouts are not interchangeable

| layout | pairs | used by |
|---|---|---|
| `interleaved=true` | `(2i, 2i+1)` | `LLAMA_ROPE_TYPE_NORM`, older models |
| `interleaved=false` | `(i, i + head_dim/2)` | `LLAMA_ROPE_TYPE_NEOX` |

`LLM_ARCH_LFM2` / `LLM_ARCH_LFM2MOE` / `LLM_ARCH_LFM3` are listed under
`return LLAMA_ROPE_TYPE_NEOX;` in `/home/medovarov/lcpp-base/src/llama-model.cpp:4564-4590`,
so the shipped LFM2.5 models need the half-split form. Picking the wrong one keeps
every output norm valid, so nothing looks corrupt — generation quality just rots.

Frequencies come from the model, not a constant: `lfm2moe.rope.freq_base = 10000000`
(1e7, not the 10000 of older models). Read via a new `GgufFile::f32(key)`.

### Implementation

- `include/kata/rope.h`, `src/rope.cpp`: `rope_ref`, `rope_avx2`, `rope`,
  `rope_tables`. Added to `Backend` as `virtual void rope(const RopeArgs&)` — this
  time with a real body. Angles are computed in `double`, since at pos = 128k the
  float exponent is visibly wrong.
- Non-interleaved pairs are two contiguous halves, so the vector path is plain
  loads at `v+i` and `v+i+head_dim/2`. Interleaved needs a deinterleave: 8
  `_mm256_setr_ps` gathers, two rotations, then `_mm256_unpacklo_ps` /
  `_mm256_unpackhi_ps` / `_mm256_permute2f128_ps`. That costs ~10% vs neox.

### Bug caught by the tests

`angles()` divided the exponent by `half * 2` instead of `head_dim`. Identical for
even head_dim, wrong for odd — `Rope.MatchesDefinitionBothLayouts` failed at
`head_dim=7` against an independently written definition. Fixed by passing
`head_dim` and using `-2.0 * i / (double)head_dim`. Odd head dims are not real,
which is exactly why this would have survived unnoticed.

### Benchmark

1 thread pinned, 32 heads x 64 dim (LFM2.5 shape), median of 2000 calls:

| layout | scalar | vector | speedup |
|---|---|---|---|
| neox (LFM2) | 0.60 us / 27.3 GB/s | **0.53 us / 30.9 GB/s** | 1.13x |
| interleaved | 0.58 us / 28.2 GB/s | **0.55 us / 29.8 GB/s** | 1.05x |

Before caching the frequency table both paths sat at 0.87-0.91 us: `pow()` was
90% of the time and the rotation was noise. Caching `base^(-2i/head_dim)` per
(head_dim, base) in a `thread_local` cut it by 35-40%.

### Conclusion

Keep. The vector path is only 1.05-1.15x, which is honest for a 0.5 us operation,
but it is also 0.5 us per token out of a ~12 ms per-token budget — i.e. 0.004%.
**RoPE must not be optimised further.** What matters here is that both layouts
exist and are tested, because a wrong layout is a silent quality killer that no
benchmark would catch.

### Next

A single transformer block wired through the backend, so a forward pass can be
profiled end to end and the real bottleneck identified instead of guessed.

## EXP-0011 — Decode attention and the KV cache layout

**Status:** PASSED
**Date:** 2026-11-25
**Hypothesis:** decode attention is memory-bound on the KV cache, so a vectorised
3-pass softmax (scores, exp/sum, weighted v) should approach the bandwidth
ceiling and the gain over scalar should grow with context depth.

### Implementation

- `include/kata/attn.h`, `src/attn.cpp`: `AttnArgs`, `attention_decode_ref`,
  `attention_decode_avx2`, `attention_decode`, and `kv_slot(cache, pos, n_kv_heads, head_dim)`
  which returns the position-major slot. GQA is handled by
  `head_of(h) = h / (n_heads / n_kv_heads)`.
- Three passes rather than one: the running max has to be known before `exp`,
  so scores land in `scratch[n_kv]`, then exponentiate and sum, then accumulate v.
- Scoring uses `_mm256_fmadd_ps` on two accumulators plus a horizontal reduce.
- v accumulation writes the output row once and re-reads it per position instead
  of holding it in registers: `head_dim` can exceed the 16 ymm registers, and the
  row stays in L1, so this is free and keeps the loop simple.

### Benchmark

1 thread pinned, `32 heads / 8 kv heads x 64 dim` (LFM2.5 shape), median of 200:

| n_kv | scalar | vector | gain | KV size |
|---|---|---|---|---|
| 1 | 1.51 us | 0.77 us | 1.96x | 0 MiB |
| 128 | 141.3 us | 64.6 us | 2.19x | 0.5 MiB |
| 1024 | 1200 us | 438 us | 2.74x | 4 MiB |
| 4096 | 4816 us | 1734 us | 2.78x | 16 MiB |
| 16384 | 19247 us | 6925 us | 2.78x | 64 MiB |
| 32768 | 50600 us | 17544 us | 2.88x | 128 MiB |

The gain flattens at ~2.8x instead of approaching the memory ceiling, which says
the kernel is not yet bandwidth-bound. At 32768 the vector path reads 128 MiB of
KV in 17.5 ms = **7.3 GB/s**, against a 43 GB/s single-thread read ceiling. There
is a 5.9x gap sitting in this kernel.

### The number that matters

17.5 ms per token at 32k context, single-threaded. For scale: a q4_K GEMV row of
2048 takes ~0.08 us at 78 GB/s, so one layer's projections are ~0.5 ms of work
across all rows — meaning **past roughly 2-4k context, single-threaded decode
attention costs more than the entire matrix work of a layer**. Attention, not
GEMV, is the thing to fix for long context.

### Known limitation, deliberately not fixed here

`attention_decode` does not loop over KV in blocks, so the v-accumulation pass
re-reads the whole KV from a context that may exceed L3 (128 MiB) — at 32k it is
exactly at the L3 size. Blocked (tiled) attention over positions is the standard
answer and is the obvious next step; it was left out of this iteration to keep the
change to one thing, and because it needs its own benchmark to prove anything.

### Tests

`Attn.VectorMatchesRefAcrossShapes` (GQA, MHA, MQA, head_dim divisible by 8 but
not 16, and head_dim not divisible by 8 to exercise the scalar tail),
`Attn.FirstTokenReturnsExactlyV` (at n_kv = 1 softmax is 1, so the output must
equal v exactly — catches a wrong GQA mapping but not a missing normalisation),
`Attn.EqualScoresGiveMeanOfV` (replaced a broken "invariance to score offset"
test: scaling k does NOT leave softmax unchanged, it sharpens it),
`Attn.KvSlotIsDisjointAndPositionMajor`.

### Conclusion

Keep. Correct across shapes, 2.2-2.9x over scalar, and it told us where the real
long-context cost is. 88.39% coverage, 47/47 tests.

### Next

Blocked (tiled) decode attention — loop over KV in L2-sized chunks so the v pass
never re-reads DRAM, and measure whether the 5.9x gap to the read ceiling closes.

## EXP-0012 — Grouped-KV decode attention

**Status:** FAILED
**Date:** 2026-11-25
**Hypothesis:** at 32k the kernel read 512 MiB of KV per token because each of
the 4 query heads in a group re-read the shared kv head (4x amplification). Driving
the loop by kv head instead of query head should cut that to 128 MiB and land the
kernel near the read ceiling.

### Why the hypothesis was plausible

Traffic arithmetic checked out: 2 * 32 heads * 32768 * 64 * 4 B = 512 MiB today vs
2 * 8 * 32768 * 64 * 4 B = 128 MiB grouped, an ideal 12.5 ms -> 3.1 ms at the
43 GB/s single-thread read ceiling. Two more measurements supported it:

- A pure-read sweep of the same 128 MiB KV in both layouts reached
  **59.4 GB/s strided** (256 B used out of every 2048 B) and 50.2 GB/s contiguous —
  so the position-major layout is not a problem and the DRAM can feed this.
  (A first version of that sweep reported 12.6 GB/s for both layouts; the limit
  was a single dependent float-add chain, not memory. Independent accumulators
  fixed it. Lesson: a bandwidth probe with one accumulator measures the FPU.)
- `exp()` costs 2.7 ns/call, i.e. **2.9 ms** for 1.05M scores per token at 32k —
  17% of the kernel and not vectorisable without a polynomial approximation.

### What was tried, all measured, all reverted

| variant | 4k | 16k | 32k |
|---|---|---|---|
| baseline (per-head scoring, pos-major) | 1734 us | 6925 us | **17544 us** |
| kv-head outer, head inner, scores [g][t] | 1581 | 6317 | 18656 |
| ...scores transposed, [t][g] | 1628 | 6468 | 19594 |
| ...+ position-outer scoring | 1686 | 6826 | 23692 |
| ...+ register accumulators in the v pass | 1667 | 6638 | 21861 |
| ...+ stale `mx0` fix (incremental max removed) | 1635 | 6526 | 24198 |
| **final: reverted to baseline** | 1740 | 6948 | **17679** |

Every grouped variant helps at 4-16k (up to 10%) and loses at 32k, monotonically
worse the more of them are stacked. The traffic saving is real but it does not
survive 128 MiB: at that size the working set is exactly L3, and the grouped code
walks it in a different order (one kv head's 8 MB slab at a time, 8 slabs) which
appears to interact worse with L3/DRAM than the linear per-head walk. The 10%
win at 16k is inside run-to-run noise for this kernel.

### What was learned (worth keeping)

- The kernel is **not** bandwidth-bound at 32k: 128 MiB of KV in 17.5 ms is
  7.3 GB/s against 43-59 GB/s achievable. The gap is ALU and per-position
  overhead, so reducing bytes was attacking the wrong resource.
- `exp()` alone is ~17% of the cost and is scalar. A vectorised exp (or an
  `exp2`-based approximation with the base-2 scale folded into `a.scale`) is a
  bigger, cleaner lever than anything done here.
- Storing a value to the scratch array and immediately reading it back for the
  running max costs a store-forwarding stall per position. Keep it in a register.
- Blocking the v pass over positions (chunk KV to L2 size) remains untested and
  is the more promising direction than grouping, since it improves locality in
  the original loop order instead of changing it.

### Conclusion

Rejected and reverted with `git checkout src/attn.cpp`. Keeping a slower kernel
because the traffic arithmetic looked right would be exactly the "optimisation on
faith" this project forbids. `attention_scratch_floats()` is left in place with a
comment recording why the grouped shape was abandoned.

### Next

Vectorised exp (polynomial or `exp2` with a folded scale) as the measured 17%
target, then blocked v accumulation.
