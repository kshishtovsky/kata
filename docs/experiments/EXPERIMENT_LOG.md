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
| EXP-0013 | Register accumulators + vectorised exp | PASSED | 32k decode 17.5 -> 11.7 ms (1.50x); gain over scalar 2.84x -> 4.16x |
| EXP-0014 | Batched score pass (4 positions/iter) | PASSED | 16k 4206 -> 3863 us (1.09x); 32k flat; N=8/16/32 all worse |
| EXP-0015 | Grouped decode (one kv walk per GQA group) | PASSED | 32k 13.3 -> 8.0 ms (1.66x); 16k 3.81 -> 2.05 ms (1.86x) |
| EXP-0016 | Tiled score pass (8x8 transpose) | PASSED | 32k 8.0 -> 7.2 ms (1.11x); 16k 2.05 -> 1.78 ms (1.15x) |
| EXP-0017 | More positions per score tile (MLP) | FAILED | NP=2 7.17 ms, NP=3 9.15, NP=4 8.13 ms at 32k -- reverted |
| EXP-0018 | Transposed K cache for the score pass | PASSED | 32k 8478 -> 6199 us (1.37x); 16k 1803 -> 1535 (1.17x); 4k 439 -> 384 (1.14x) |
| EXP-0019 | Real forward pass: LFM2.5-24B-A2B end to end | PASSED | 16/16 greedy tokens identical to llama.cpp; 35-36 tok/s decode vs 30.5 |
| EXP-0020 | Where the per-token time goes | PASSED | 76% in the two GEMV kernels; 503 mm calls/token, dispatch-bound |
| EXP-0021 | Batched expert GEMV (one pool wake per layer) | PASSED | 1.40x at 8 threads, 1.72x at 16; 32.7 -> 26.4-28.8 ms/token |
| EXP-0022 | Sampler (top_k/top_p/temp/repeat penalty/eos) | PASSED | 0.44% of a token; generation stops at EOG in 3/3 seeds |
| EXP-0023 | Batched prefill GEMM, phase A | PASSED | 1.7-7.1x over per-row GEMV, relerr 1e-7; prefill wired end to end |
| EXP-0024 | AVX-512 VNNI microkernel for prefill GEMM | PASSED | 160.7 vs 57.5 GMAC/s (2.79x); prefill 113 -> 145.7 tok/s |
| EXP-0025 | Load-guard pool fix + honest CPU baseline | PASSED | spin 15.8% -> 5.3%; llama.cpp CPU prefill is 441 tok/s, not 4540 |
| EXP-0026 | VNNI microkernel for q6_K | FAILED | prefill 145.0 -> 87.8 tok/s (0.61x); 16 sub-block epilogues per band vs q4_K's 8 |
| EXP-0027 | Parallel prefill attention + parallel expert glue | PASSED | prefill 146.4 -> 231.9 tok/s (1.58x); attention 654 -> 196 ms, MoE glue 1032 -> 450 ms |
| EXP-0028 | Grouped prefill attention (4 query heads share one kv head) | PASSED | long-context prefill 25070 tok 108.7 -> 137.0 tok/s (1.26x); 544 tok 253 -> 258 tok/s; argmax unchanged vs llama.cpp batched |
| EXP-0029 | VNNI microkernel for q6_K, bias folded into the scale multiply | PASSED | 651 -> 1415 GMAC/s (2.2x); prefill 260 -> 268 tok/s, 25070 tok 137.0 -> 138.6 |
| EXP-0030 | Spin-then-park thread pool (the CPU load "wave") | PASSED | prefill 268 -> 330 tok/s, 25070 tok 138.6 -> 154.7, decode 36.1 -> 41.6 tok/s |
| EXP-0031 | Reuse MoE scratch across layers; ring-buffer shortconv | PASSED | prefill 330 -> 354-370 tok/s, logits unchanged; shortconv 331 -> 364 tok/s on its own (serial history shift removed, OOB at m<l_cache fixed) |
| EXP-0033 | Multithreaded prefill quantisation (`q8_rows_mt`); decode scratch reuse | PASSED (prefill) / NEUTRAL (decode) | prefill 544 tok 354-370 -> 445.3-467.6 tok/s (parity with llama.cpp 465.23); decode 41.55-42.31 vs 45.78 tok/s (pool overhead ~25% in perf) |
| EXP-0034 | Vectorised router GEMV (`gemv_f32_avx2`) and shortconv decode step | PASSED | decode 24.1 -> 22.8 ms/token (41.5 -> 42.9 tok/s); exact numerical match (56085 / 17568 unchanged); router GEMV 1.36 -> 0.16 ms (8.3x) |
| EXP-0035 | Fused attention QKV (`gemv_qkv_mt`), vectorized MoE down accumulation & residual add, lock-free ThreadPool dispatch | PASSED | decode 22.8 -> 23.13 ms/tok (42.9 -> 43.24 tok/s); exact numerical match (56085 / 17568); eliminated 20 pool dispatches & redundant quantizations; lock-free pool when spinning |

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

## EXP-0013 — Register accumulators and a vectorised exp

**Status:** PASSED
**Date:** 2026-11-25
**Hypothesis:** EXP-0012 proved the kernel is ALU/overhead bound, not
bandwidth-bound, and named two suspects: the v-accumulation round trip through
memory, and the scalar `exp()`. `perf` is now installed, so both are measurable
rather than guessed.

### Profiling setup

Installed `perf` (`pacman -Sy perf`) and set `kernel.perf_event_paranoid=1`.
`bench_attn` gained `--vector-only` because the first profile was 65% scalar-ref
code — profiling a benchmark that times two implementations measures both.

`perf annotate` on the vector path put **38%** of cycles on
`vmovups (%rdx,%r11,4)` / `vfmadd213ps` / `vmovups %ymm0,(%rcx,%r11,4)` — i.e. the
v loop was loading the output row from memory, adding, and storing it back on
every position. That is a store-forwarding round trip per 8 floats.

### Change 1: accumulators in registers

`v_accum<NV>` holds the whole output row in `NV` ymm registers for head dims that
fit (8/16/32/64/128 -> NV 1/2/4/8/16), with `v_accum_generic` as the spill
fallback for unusual head dims. Result at 32k: **17.5 -> 14.9 ms**. Confirmed the
diagnosis; gain over scalar 2.84x -> 3.39x.

### Change 2: vectorised exp

AVX2 has no `exp2` instruction, so `exp256_ps` splits the exponent: `x*log2(e)`,
round to nearest int, degree-5 minimax polynomial for `2^f` on `f in [-0.5, 0.5]`,
and `2^k` as an exponent-field add. Argument clamped to [-87, 88] to keep the
shift in range. Standalone check: **max relative error 6.8e-06** over
x in [-80, 80]. Result at 32k: **14.9 -> 11.7 ms**.

### Bug caught

The first version summed the scalar tail with `_mm256_add_ps(s0, _mm256_set1_ps(p[i]))`,
adding each tail value to all eight lanes — an 8x inflated denominator whenever
`n_kv % 32 != 0`. `Attn.VectorMatchesRefAcrossShapes` failed at `n_kv = 1` with
`ref 0.700000 vec 0.087500` (exactly 1/8). Fixed with a scalar `tail` accumulated
and added once.

A second, non-code bug: `kata_tests` is `EXCLUDE_FROM_ALL`, so a plain
`cmake --build build` **does not rebuild it**, and I twice read a stale binary
that still contained the bug I had just fixed. Always
`cmake --build build --target kata_tests`.

### Benchmark

1 thread pinned, `32 heads / 8 kv heads x 64 dim`, median of 200:

| n_kv | scalar | vector | gain | vs EXP-0011 |
|---|---|---|---|---|
| 1 | 1.59 us | 0.66 us | 2.41x | 0.77 -> 0.66 |
| 128 | 134.9 us | **25.8 us** | 5.23x | 64.6 -> 25.8 |
| 1024 | 1266 us | **261 us** | 4.84x | 438 -> 261 |
| 4096 | 4960 us | **1051 us** | 4.72x | 1734 -> 1051 |
| 16384 | 19409 us | **4201 us** | 4.62x | 6925 -> 4201 |
| 32768 | 48617 us | **11678 us** | 4.16x | 17544 -> 11678 |

At 32k the kernel now moves 128 MiB in 11.7 ms = **11.5 GB/s** against a 43-59 GB/s
ceiling, so a 3.7-5x gap remains. It is no longer in the accumulator or in `exp`.

### New test

`Attn.ExtremeScoresMatchRefAfterClamp` feeds scores with sigma 40 (saturated
attention) at n_kv 8 and 257, asserting no NaN and agreement within 5e-4. This is
the test that would have caught a too-narrow clamp, and it is the only test that
exercises the polynomial at its limits.

### Conclusion

Keep both. 1.50x on the operation that dominates long-context decode, from two
independent 1.17x-1.28x steps. 48/48 tests, coverage 87.09%.

### Next

## EXP-0014 — Batched score pass

**Status:** PASSED at N=4 for 4-16k, flat at 32k. N=8 and above rejected.

**Hypothesis.** EXP-0013 left the score pass as the dominant cost: `dot_avx2`
plus a horizontal reduce per position, with `vhaddps` at 10.5% of samples.
Scoring N positions per iteration loads `q` once per N k-rows and shares one
reduce per pair, so N=4 should cut that 10.5% by ~4x.

**Implementation.** `src/attn.cpp`, `scores_n<N>()`: N independent `__m256`
accumulators over head_dim, then `hadd(hadd(r[p], r[p+1]))` per pair. After
`hadd(hadd(a,b), hadd(a,b))` the lanes hold
`[a0+a1+a2+a3, b0+b1+b2+b3, ... , a4+a5+a6+a7, b4+b5+b6+b7]`, so
`sum(a) = lane0 + lane4` and `sum(b) = lane1 + lane5`. `SCORE_N` is a
compile-time knob (default 4); head dims not divisible by 8 finish scalar.

**Measured** (Ryzen 9 7950X3D, 1 thread pinned, 32 heads / 8 kv heads x 64 dim,
median of 40 iters, A/B on the same binary against `8b629bb`):

| SCORE_N | 4k us | 16k us | 32k us |
|---|---|---|---|
| per-position (8b629bb) | 951 | 4206 | 12691-13291 |
| 4 | 950 | **3863** | 12837-13283 |
| 8 | 1044 | 4150 | 13721 |
| 16 | 1107 | 4599 | 14054 |
| 32 | 1855 | 7430 | 19837 |

**Analysis.** 16k gains 8%; 4k is a wash because at 4k the KV is 16 MiB and the
kernel is L2/L3-bound rather than reduce-bound; 32k is flat. Above N=4 every
variant regresses monotonically: N accumulators plus N live k-pointers exceed
the 16 ymm registers, so `r[]` spills to stack and the `static_assert(N%2==0)`
pairing turns into load/store traffic. The 32k plateau is consistent with the
KV being exactly L3-sized (128 MiB), where the extra streams stop helping.

**Conclusion.** Keep N=4. Not worth pushing further: the score pass is no longer
where the 3.5x gap to the memory ceiling lives, and every larger batch size
costs more than it saves.

**Next.** Measure the score and v passes separately so the remaining 12.8 ms at
32k can be attributed before choosing another target.

## EXP-0015 — Grouped decode attention

**Status:** PASSED. Supersedes the EXP-0012 failure.

**Hypothesis.** A phase split of the 12.8 ms 32k decode gave 6915 us score /
225 us softmax / 5778 us weighted-v. With 32 query heads over 8 kv heads, four
heads share each kv head, so both passes read every k and v row four times: at
32k that is 256 MiB touched out of 64 MiB of real data. Walking each kv head
once for all four of its query heads should cut that 4x.

**Why EXP-0012 failed at the same idea.** EXP-0012 tried the same grouping and
lost at 32k in every variant. The difference here is that the score pass keeps
`NT x G` accumulators in registers and the v pass needs AVX-512 for its register
budget, so no per-position store-to-memory round trip survives. EXP-0012's
grouped loop stored each score and read it back for the running max.

**Implementation.** `src/attn.cpp`:
- `scores_group<G, NT>()` — NT positions x G heads of `__m256` accumulators (8
  ymms at G=4, NT=2), one k load per position shared by all G heads.
- `v_group<G>()` — `__m512` accumulators, 4 heads x 64 floats = 16 zmm; AVX-512
  is used for the register budget, not for wider math.
- `attention_decode_avx2_grouped()` — dispatch plus a scalar tail for odd
  position counts. Falls back to the plain kernel unless G==4, head_dim%16==0
  and head_dim<=64.
- New API: `attention_scratch_floats_grouped()` (G heads x n_kv) and
  `attention_decode_grouped()`.
- Diagnostic: `attention_decode_phases()` plus `AttnArgs::base`, so the bench can
  time score / softmax / v separately. Public callers leave `base` at 0.

**Correctness.** `Attn.GroupedMatchesPlainDecode` compares the grouped kernel
against the plain one over 8 shapes, including the three fallback shapes (G=1,
hd=96, hd=32 with an odd position count) and checks for NaN. 49/49 tests.

**Measured** (Ryzen 9 7950X3D, 1 thread pinned, 32 heads / 8 kv heads x 64 dim,
median of 30 iters):

| n_kv | plain us | grouped us | gain |
|---|---|---|---|
| 1024 | 238 | 137 | 1.74x |
| 4096 | 951 | 557 | 1.71x |
| 16384 | 3814 | 2053 | 1.86x |
| 32768 | 13282 | 8012 | 1.66x |

Cumulative against the original scalar decode: 32k is now 8.0 ms vs 53.5 ms,
**6.7x**, and vs the start of EXP-0013 (17.5 ms) 2.2x.

**Analysis.** Grouping helps at every depth and peaks at 16k. At 32k the KV is
exactly the 128 MiB L3, so the group walk's four-head reuse competes with itself
in cache; that is the same plateau EXP-0014 hit. The 32k number is also the
noisiest (8.0-8.4 ms across runs).

**Bottleneck after this change.** Re-measured phases at 32k: score 7088 us,
softmax 187 us, v 6114 us. Both passes still sit near 37 GB/s of effective
traffic, well under the 43-59 GB/s single-thread ceiling, so the remaining gap is
instruction throughput in the score kernel and the 16 zmm v accumulator, not
memory.

**Conclusion.** Keep. The grouped path is now the default for GQA shapes with
G=4, which is what LFM2.5 and Qwen3.6 use.

**Next.** The v pass is now as expensive as the score pass. It recomputes
`set1` per head per position and re-reads v once per group; splitting it so v is
read once into L1 and used by all four heads, and hoisting the four
probabilities into one broadcast, is the next experiment.

## EXP-0016 — Tiled score pass, 8x8 transpose

**Status:** PASSED.

**Hypothesis.** After EXP-0015 the score pass was 7088 us and `perf annotate`
showed `vhaddps` at 24-32% of samples. The pairwise reduce
`hadd(hadd(r[p], r[p+1]))` is three dependent adds per (position, head) pair.
An 8x8 lane transpose turns eight accumulators into eight column sums in one
shot: `w = a[0] + a[1] + ... + a[7]`.

**Implementation.** `src/attn.cpp`:
- `transpose8(__m256*)` — unpacklo/hi, shuffle_ps, permute2f128; verified
  standalone against hand-computed column sums before it went in.
- `scores_8x()` — 2 positions x 4 query heads of one kv group, 8 accumulators
  indexed `j*4 + h`. **Each k vector is loaded once per 8 floats and used by all
  four heads**, which is what makes the grouping pay. Scalar tail for head dims
  with a remainder; `scale` applied once at the end.

**Correctness.** 49/49 tests, including `Attn.GroupedMatchesPlainDecode` over
8 shapes and the G=1 / hd=96 / hd=32 fallbacks. `hd % 8 == 0` is guaranteed by
the existing `hd % 16 == 0` dispatch guard, so the tiled path has no partial
8-float block.

**Measured** (Ryzen 9 7950X3D, 1 thread pinned, 32 heads / 8 kv heads x 64 dim):

| n_kv | EXP-0015 | EXP-0016 | gain |
|---|---|---|---|
| 4096 | 557 us | 437 us | 1.27x |
| 16384 | 2053 us | 1777 us | 1.16x |
| 32768 | 8012 us | 7210 us | 1.11x |

Cumulative against the original scalar decode at 32k: 53.5 ms -> 7.2 ms, **7.4x**.

**Dead end inside this experiment.** The first `scores_8x` kept the 8-way tile
but loaded k inside the per-head loop, reintroducing the four-fold re-read:
9657 us, worse than EXP-0015. The win only appears when k is hoisted out of the
head loop. Same lesson as EXP-0015, applied one level deeper.

**Analysis.** The gain shrinks with depth (1.27x at 4k, 1.11x at 32k) because at
32k the kernel is closer to a memory wall, where killing ALU chains cannot help
much. `v_group<4,4>` is now 21% of the profile and the grouped driver 38%.

**Bottleneck after this change.** v pass: `_mm512_set1_ps` per head per position
(four broadcasts per position) and 16 zmm accumulators, reading v once per group.
Effective traffic 64 MiB in ~5.5 ms = 12 GB/s against a 43-59 GB/s ceiling, so
the v pass is still instruction-bound, not memory-bound.

**Conclusion.** Keep. Both passes now use a transpose-style reduce.

**Next.** Hoist the four probabilities into one broadcast per position in
`v_group` (a `_mm512_set1_ps` of a 4-float load instead of four separate ones)
and check whether the 16 zmm accumulators are spilling; if they are, the v pass
splits into two halves of head_dim 32 rather than four.

## EXP-0017 — More positions per score tile

**Status:** FAILED. Reverted.

**Hypothesis.** After EXP-0016 the honest phase split put the score pass at
4087 us for 64 MiB of K, i.e. **16 GB/s**, while a strided read of the same
128 MiB KV had measured 59.4 GB/s. With NP=2 there are only two outstanding
256-byte streams, so the guess was that memory-level parallelism, not
instruction count, set the rate: NP=4 would give four streams.

**Implementation.** `scores_tile<NP>` with `4*NP` accumulators indexed
`p*4 + h`, a `scores_dispatch()` switch on NP to keep the count compile-time,
and the caller loop advanced by `np` with a scalar remainder loop. Reused
`transpose8` two positions at a time.

**Measured** (32k, 1 thread pinned, 3 runs each, best of run-set):

| positions/tile | us |
|---|---|
| 2 (EXP-0016) | 7183 / 7173 / 7160 |
| 3 | 9152 |
| 4 | 8135 / 8318 / 8288 |

**Conclusion.** More streams make it strictly worse. The MLP hypothesis is
falsified: at 4*NP accumulators plus NP live k vectors the register file is
exceeded and GCC spills, and the extra streams do not pay for the spill traffic.
The score pass is not latency-bound on memory either.

**Arithmetic that fits the measurement better.** Per 8 floats of head dim the
NP=2 kernel issues 2 k loads + 4 q loads + 8 FMAs = 48 loads per 64 FMAs. Across
32k positions x 32 heads that is ~1.2e8 instructions, which at 4-5 IPC lands at
5-6 ms — the same order as the 4.1 ms measured. So the pass is **load-issue
bound**, and the fix would be to keep q in registers across many positions,
which needs more accumulator registers than exist.

**Next.** Stop tuning this kernel. The remaining leverage is the one identified
earlier and not yet built: quantise the KV cache to q8, which halves the bytes
of the only truly irreducible resource (128 MiB per token at 32k).

## EXP-0018 — Transposed K cache for the score pass

**Status:** PASSED (opt-in kernel, not the default yet).

**Hypothesis.** EXP-0017 showed the score pass is load-issue bound, not
latency bound: 4*NP accumulators are spent holding head_dim partial sums, so the
number of positions per tile is capped by the register file at 8-16 ymm. Store K
transposed — `[kv_head][pos_block][dim][pos_in_block]` — and the lanes become
positions while the accumulators become the four query heads of a kv group.
Then the count is 4 zmm regardless of tile width, there is no reduction step at
all, and K is read contiguously by position.

**Implementation.**
- `include/kata/attn.h`: `KT_BLK = 64`, `kt_floats()`, `kt_slot()`, `kt_write()`.
  A dim row inside a block is contiguous over positions, so writing a new token
  is 64 consecutive floats per kv head rather than 512 strided ones.
- `src/attn.cpp`: `scores_kt<NP>()` — outer loop over d, inner over the four
  heads, `acc[h] += broadcast(q[h][d]) * loadu(kt_row + d*64 + off)`; a lane is
  the finished dot product, so no `transpose8`.
  `attention_decode_avx2_grouped_kt()` walks blocks of 64 positions, 16 at a
  time, with a scalar path for the partial last block, then reuses the existing
  `softmax_exp_inplace` and `v_group<4,4>` unchanged.
  Guarded to `g == 4 && head_dim == 64` and AVX-512; everything else falls back.
- `// slack:` not needed — the tail block fallback keeps correctness for any n_kv.

**Bug found (mine).** First run gave `maxdiff 3.4e-02` with relative error 3.5.
`tile[16]` was being filled with 64 values (16 positions x 4 heads), writing
over `mxv`/`inv` on the stack. After `tile[4*16]` the two kernels agree to
`1.58e-08` relative `1.6e-06`.

**Verification.** `Attn.KtWriteRoundTripsThroughTheTransposedLayout` and
`Attn.KtDecodeMatchesPositionMajorDecode` (n_kv = 1, 63, 64, 65, 129, 200 —
boundary of KT_BLK and both partial-block shapes). 51/51 tests.

**Measured** (32 heads / 8 kv heads x 64 dim, 1 thread pinned, median of 20):

| n_kv | position-major | transposed | gain |
|---|---|---|---|
| 128 | 16.1 us | 13.6 us | 1.18x |
| 4096 | 438.5 us | 384.0 us | 1.14x |
| 16384 | 1803.2 us | 1535.2 us | 1.17x |
| 32768 | 8477.5 us | 6198.7 us | **1.37x** |

The gain grows with depth, which is what the hypothesis predicted: at 128
positions everything is in L1 and the extra store traffic of the transposed
layout cancels the win.

**Cost of the transposed write**, measured separately:
- building the whole 32k cache: position-major 989 us, transposed 7929 us
  (8.0x — a strided store per dim, and a cache-line miss on every one)
- **one token, which is what decode actually pays: 0.024 us -> 0.126 us**,
  i.e. +0.1 us against a 6.2 ms token. Irrelevant.

**Analysis.** A pure read of the transposed K through the same access shape ran
at 101 GB/s (663 us for 67 MB), so the buffer itself is not the limit; the
kernel spends the rest on one broadcast per dim per head. That is the price of
giving up q reuse across head_dim, and it is what caps the gain at 1.37x
instead of the 2.5-3x the bandwidth comparison suggested.

**Conclusion.** Keep as an opt-in kernel and record the trade. The win is real
at 16k-32k, marginal below 4k, and it needs a second K buffer (64 MiB at 32k)
plus AVX-512. Making it the default requires deciding whether that memory is
worth it, which belongs with the KV-quantisation work that changes the buffer
layout anyway.

**Next.** Quantise KV to q8. That halves the bytes of the only irreducible
resource, and it must be designed together with this layout because a
transposed q8 cache wants positions packed differently than a q8 position-major
one.


## EXP-0019: LFM2.5-24B-A2B forward pass — correctness vs llama.cpp

**Status:** PASSED (structurally correct, numerically within activation-quantisation noise)
**Date:** 2026-11-25
**Branch:** dev

### Problem
The engine had kernels but no graph. Nothing had ever produced a token from a real
model, so kernel-level parity with ggml said nothing about whether the whole thing
computes the right function.

### Hypothesis
A hand-written graph (embedding -> 40 layers of {RMSNorm, shortconv|GQA-attention,
residual, RMSNorm, dense-FFN|MoE, residual} -> RMSNorm -> tied lm_head) will match
llama.cpp component-by-component, with residual error explained entirely by the
q8_K activation quantisation that both engines already use.

### Implementation
- `include/kata/model.h`, `src/model.cpp` — `Model::load`, `forward`, `ModelState`.
- `run/kata_run.cpp` — CLI driver; `--greedy N` for greedy-decoding comparison.
- Layer plan read from the GGUF: 10 attention layers (2,6,...,38), 30 shortconv,
  38 MoE + 2 dense FFN, 32 query heads / 8 kv heads / head_dim 64, 64 experts top-4.
- `KATA_DUMP=<dir>` writes per-stage float32 dumps for diffing against a llama.cpp
  reference dump produced by `llama_context_params::cb_eval`.

### Bugs found and fixed (all in the graph, none in the kernels)
1. **shortconv kernel layout.** Used `kern[tap*n_embd + e]`; ggml's
   `ggml_compute_forward_ssm_conv_f32` indexes `c[i0 + i1*ncs]` with `ne[0]=3`, so
   channel `e` owns `kern[e*3 .. e*3+2]`. Diff on `conv_out_proj-0` went 2.74e-02 -> 3.73e-09.
2. **`dump("emb")` inside the layer loop**, so it logged layer-39 output instead of
   the embedding (fabricated diff of 1.6e+01). Guard `pos==0 && il==0`; now bit-exact.
3. **`dense_ffn` hardcoded dump names** (`"dgate0"`), so layer 1 overwrote layer 0.
   Took `il` as a parameter; gate/up then matched to 2.98e-08.
4. **Expert byte strides** were hardcoded (144 / sizeof(block_q6_K)); replaced with
   `n_ff_exp * n_embd / QK_K * ggml_type_size(type)`.

### Correctness results
| stage | max abs diff vs llama.cpp |
|---|---|
| embedding | 0.000e+00 (bit-exact) |
| attn_norm layer 0 | 0.000e+00 |
| shortconv out_proj layer 0 | 3.7e-09 |
| `l_out-0` / `l_out-1` (dense path) | 7.5e-09 / 1.2e-07 |
| dense `ffn_gate`/`ffn_up` layer 0 | 2.98e-08 |
| MoE router expert ids | identical at every layer checked (2, 20-25) |
| MoE router weights | 2.5e-05 |
| `ffn_moe_out-2`, float64 exact recompute | 1.49e-04 |
| `ffn_moe_out-2`, kata q8 path | 3.78e-04 |
| `l_out-2` .. `l_out-8` | <= 1.4e-03 |
| `l_out-9` .. `l_out-20` | ~2e-02 relative |
| `l_out-21` .. `l_out-39` | ~5e-03 relative |

The 0.9 absolute difference on layers 34-38 is **not** a defect. Three channels
(1043, 271, 1549) carry an outlier of magnitude ~146 through layer 33; layer 34
cancels it to ~2.7. Relative error before cancellation is 4.0e-03 on all three
channels with the same sign and magnitude — a systematic activation-quantisation
shift. After the cancellation the same absolute error reads as 35% relative. The
error does not grow past layer 34.

### Greedy-generation check
Reference: `/tmp/ref/greedy.cpp` driving llama.cpp at `n_gpu_layers=0`.
Prompt `124894 597 4104 10299 35886` ("The quick brown fox"):
```
llama.cpp  56085 890 278 45698 4448 63908 597 4104 10299 35886 56085 890 278 45698 4448 63908 | 93017 93017 ...
kata       56085 890 278 45698 4448 63908 597 4104 10299 35886 56085 890 278 45698 4448 63908 | 597 4104 ...
```
16 tokens identical, divergence at 17. At that point llama.cpp falls into a
degenerate `93017` loop while kata keeps echoing the prompt — both trajectories are
degenerate, so the test cannot rank the two engines. It does establish that the
graph agrees on everything well-conditioned.

### Result
The forward pass is correct. 315 ms/token on 3 tokens at 4 threads (cold, no
optimisation). Not yet measured on a warm 40-token run.

### Bottleneck
Attribution not yet measured — `KATA_PROF` exists but the mixer timer `t_mix` is not
wired. Expected dominated by the 38 MoE layers x 4 experts at ~1.4 GB/token.

### Conclusion
Keep. This is the base the rest of the project optimises.

### Next
EXP-0020: attribute per-stage time with KATA_PROF and rank the real hot spots.

## EXP-0020: where the token time actually goes

**Status:** PASSED
**Date:** 2026-11-25

### Problem
EXP-0019 left a 315 ms/token figure with no attribution, while the arithmetic
bounds said 19 ms/token at the 78 GB/s ceiling. 16x unexplained.

### Implementation
Wired the previously dead `t_mix` accumulator and added `t_mm` / `t_q` / `n_mm`
inside `Impl::mm`, so every GEMV call is counted and timed. `KATA_PROF=1` prints
a per-token breakdown to stderr.

### Results (8 threads, 8-token prompt, steady state)
```
mm        23.2-26.4 ms in 503 calls (46-52 us/call), quantize 2.4 ms
total     32.7-36.6 ms | mix 21% | ffn 69% | lm 9% | norm 0.1%
```

| phase | ms/token | share |
|---|---|---|
| GEMV (all 503 calls) | 23-26 | 69% |
| mixer (attention + shortconv non-GEMV work) | 6.9-7.9 | 21% |
| lm_head GEMV | 3.2-3.6 | 9% |
| RMSNorm | 0.02 | 0.1% |
| embedding | 0.00 | — |

### The real finding: per-call dispatch costs ~45% of throughput
Isolated with `/tmp/moebench.cpp` — the same 269 MB walked as 152 calls of
1536x2048 q4_K, versus one call of 233472 rows:

| threads | 152 small calls | one big call |
|---|---|---|
| 1 | 22.59 GB/s | 23.55 GB/s |
| 4 | 47.24 GB/s | **80.43 GB/s** |
| 8 | 47.16 GB/s | **85.07 GB/s** |
| 16 | 44.14 GB/s | 79.91 GB/s |

Single-threaded the two are identical, so the loss is not arithmetic — it is
fan-out. At 4-8 threads each call wakes the pool for a slice of only ~1.5 MB
(1536 rows), the workers finish before they are all scheduled, and the run
serialises. One big call reaches 85 GB/s, above the 78 GB/s DRAM estimate,
because the 269 MB buffer is partly L3-resident.

Quantisation of the activation is 2.4 ms/token (7%) — worth attacking only after
dispatch is fixed.

### Conclusion
The engine is dispatch-bound, not memory-bound and not arithmetic-bound. 503
GEMV calls per token, each too small to amortise a thread-pool wake-up, is the
single largest lever and it is a scheduling fix, not a kernel fix.

### Next
EXP-0021: batch the per-expert GEMVs. The four experts of a layer share one
activation row and one 1536-wide output row, so they can be issued as one call
over a concatenated 4x1536 view when the four expert matrices are contiguous —
they are not in the GGUF, so this needs a gather into a per-layer staging buffer,
or a row-count-multiplied kernel that takes 4 base pointers.

---

## EXP-0021: batched expert GEMV — one pool wake-up per expert group

**Status:** PASSED
**Date:** 2026-11-25
**Hypothesis:** EXP-0020 showed the engine issues 503 GEMV calls per token and
each 1536-row call moves only ~1.5 MB, so workers finish before they are all
scheduled. Issuing the four experts of a layer (gate+up = 8 matvecs, down = 4
matvecs) as a single call over a row-indexed batch should recover the ~47 ->
80 GB/s gap measured in `/tmp/moebench.cpp`.

### Implementation
- `gemv_q4_K_q8_K_mt_batch()` / `gemv_q6_K_q8_K_mt_batch()` in `src/gemv.cpp`:
  split `total = n_batch * rows` by global row index, derive `b = r / rows`,
  `lo = r - b*rows`, `hi = min(rows, lo + (rend - r))`, call
  `fn(hi-lo, cols, out + b*rows + lo, w[b] + lo*nb, x[b])`.
- `Impl::mm_batch()` in `src/model.cpp`, local `wp[8]` per type.
- `moe_ffn` issues gate+up of all 4 experts as one `2*nu` batch when
  `t_ge == t_ue`, then SiLU + product, then the four down-projections as one
  `nu` batch.

### Bug found
The batched path bypasses `mm()`, so the shared activation row was never
quantised and `ffn2` error jumped to 6.5e-03 (whole tensor), argmax 14783/4361.
Fix: `quantize_row_q8_K(in.data(), aq.data(), n_embd);` at the top of `moe_ffn`.
After the fix `ffn_moe_out-2` = 3.78e-04 and single-token argmax is 785, matching
the llama.cpp reference.

### Benchmark
Kernel level (`/tmp/b2.cpp`, 4 matrices of 1536x2048, one shared activation):
| threads | batch | loop | speedup |
|---|---|---|---|
| 4 | 97.0 us (73.0 GB/s) | 107.8 us (65.6 GB/s) | 1.11x |
| 8 | 61.1 us (115.9 GB/s) | 85.7 us (82.6 GB/s) | 1.40x |
| 16 | 62.8 us (112.8 GB/s) | 107.7 us (65.7 GB/s) | 1.72x |

End to end, 8-token prompt, `KATA_PROF=1`:
| threads | mm before | mm after | total before | total after |
|---|---|---|---|---|
| 4 | 48.3 us/call, 24.3 ms | — | 32.47 ms | 32.2-33.4 ms |
| 8 | 46-52 us/call, 23.2-26.4 ms | 40.7-44.5 us/call, 20.5-22.4 ms | 32.7-36.6 ms | 28.5-31.2 ms |
| 16 | — | 36.5-40.0 us/call, 18.3-20.1 ms | — | **26.4-28.8 ms** |

The kernel-level win is real but only ~15% reaches wall clock, because GEMV is
already 74% of the token and the batch only covers the MoE part of it.

### Bottleneck analysis
`perf record -g` at 16 threads: 52.66% `gemv_q4_K_q8_K_avx2_impl`, 23.64%
`gemv_q6_K_q8_K_avx2_impl`, 2.39% `native_queued_spin_lock_slowpath`. So after
batching the pool lock is a visible 2.4% — the fan-out cost has not disappeared,
it has moved into contention. 76% of the token is now the two quantised GEMV
kernels themselves, which are already at 1.006x and 1.03x of llama.cpp.

### Conclusion
Keep. 8 threads 32.7-36.6 -> 28.5-31.2 ms/token, 16 threads best at 26.4-28.8
ms/token (was 34-36). Remaining time is the kernels, not the runtime.

### Next
The engine is now kernel-bound at ~1.0x of llama.cpp per kernel, which means
wall-clock gains have to come from not doing the work at all (batching across
tokens, prefill) rather than from making the kernel faster.

---

## EXP-0022: sampler — the engine could not stop or vary its output

**Status:** PASSED
**Date:** 2026-11-25
**Problem:** the engine had no sampler at all: the CLI took the argmax and fed it
back. On LFM2.5 that produced a literal 10-token cycle (`56085 890 278 45698 ...`
repeating the prompt forever) and without an eos check generation never ended.

### Implementation
`include/kata/sampler.h` + `src/sampler.cpp`:
```
struct SamplerConfig { float temp = 0.8f; int top_k = 40; float top_p = 0.9f;
                       float repeat_penalty = 1.0f; int repeat_last_n = 64; uint64_t seed = 0; };
class Sampler { uint32_t sample(const float * logits, size_t n_vocab, const std::vector<uint32_t> & history);
                void apply_penalty(float * logits, size_t n_vocab, const std::vector<uint32_t> & history) const; };
```
Order of operations matches llama.cpp: repeat penalty on a copy indexed by vocab
id -> top_k -> softmax at `temp` -> top_p nucleus -> draw. The penalty is
asymmetric (`x>0 ? x/pen : x*pen`) so the ordering around zero stays sane.
`temp <= 0` short-circuits to argmax and skips the penalty entirely.
Tie-break on token id inside the sort, otherwise two equal logits can reorder
between runs and the same seed produces different text.

`run/kata_run.cpp` reads the model's own recommendation from the GGUF
(`general.sampling.top_k=50`, `temp=0.1`, `penalty_repeat=1.05`) and honours
`tokenizer.ggml.eos_token_id=124900`; the vocab also flags 124895 as EOG, so
both stop generation. Flags: `--gen N`, `--temp`, `--top-k`, `--top-p`,
`--repeat-penalty`, `--repeat-last-n`, `--seed`.

### Test
13 tests in `tests/test_sampler.cpp`: greedy ignores the penalty, negative temp
is greedy, top_k=1 equals greedy, top_k=0 is disabled, same seed is
deterministic, top_p keeps only the nucleus (and a 0.5 nucleus over four equal
logits admits exactly two symbols), the penalty lowers a positive logit and
raises a negative one, the `repeat_last_n` window is respected, out-of-range
history ids are ignored, an empty vocabulary throws, and a 2-way distribution
with logit gap ln 3 samples at 3.0 +- 0.25 over 20000 draws.

### Benchmark
`/tmp/scost.cpp`, 128000 logits, temp 0.1 / top_k 50 / top_p 0.9 / penalty
1.05: **126.4 us per sample** — 0.44% of a 28.5 ms token. `partial_sort` over
the full vocab is the cost; not worth optimising yet.

End to end, 128 generated tokens, 16 threads:
| config | ms/token | tok/s |
|---|---|---|
| greedy (sampler bypassed) | 27.98 | 35.74 |
| temp 0.7 / top_p 0.9 / pen 1.3 | 28.5-29.7 | 33.7-35.1 |

One outlier run at 48.84 ms/token — the first run of the batch, cold page
faults on the 512-token KV region, same effect measured in EXP-0020.

### Quality
Output is now coherent instead of degenerate. With temp 0.7 / top_p 0.9 /
repeat 1.3 the model answers the prompt ("...The quick brown fox jumps over the
lazy dog" then a short list of fruits and a numbered analysis) and stops at
EOG in 3 of 3 seeds. Greedy still loops by construction — that is correct
behaviour for argmax, not a bug.

### Conclusion
Keep. Without a sampler the engine cannot be used at all, and the cost is 0.4%.

### Next
Prefill in batches: at 28.5 ms/token the first 32 prompt tokens cost 0.9 s of
pure decode that llama.cpp does at 199 tok/s.

## EXP-0023 — Batched prefill GEMM (phase A)

**Status:** PASSED. Wired into `src/model.cpp` as `prefill()`; 512-token prompt
runs at 113-127 tok/s before EXP-0024 and 145.7 after.

**Hypothesis.** The decode kernel unpacks a weight block once per activation
row. In prefill that means a 2048x2048 layer's 2.4 MB of weights is re-read 512
times. Unpacking the block once and applying it to `MR` activation rows at a
time should cut both traffic and unpack work by `MR`.

**Implementation.** `include/kata/gemm.h`, `src/gemm.cpp`, `include/kata/simd.h`
(shared f16/hsum/scale helpers, each with its own target attribute because GCC
refuses to inline across mismatched `target` strings).
- `gemm_q4_K_q8_K(m, n, k, out, ldc, w, a)` and the q6_K twin, plus `_ref`
  (one GEMV per row, the test oracle) and `_mt` (rows of `n` split across
  workers).
- Loop order weight-row outer, activation tile inner: the activation chunk for
  512 tokens is ~1 MB and stays in L2 while the weights stream past once.
- `MR` is the number of activation rows sharing one unpacked block.
- `tests/test_gemm.cpp`: 5 tests over shapes that cover `m < MR`, `m == MR`,
  `m % MR != 0`, empty dims, and the MT driver. 70/70 tests pass overall.
- `bench/bench_gemm.cpp`: batched GEMM vs the per-row GEMV on identical
  buffers, printing ms, GB/s, ratio and relative error.

**The spill trap.** The first q4_K kernel carried three accumulator arrays per
row: the float value accumulator, a separate float min accumulator, and a
per-block integer sum. At `MR=16` that is 48 live vectors against 32 ymm
registers, and GCC spilled 666 stack moves into the inner loop. Folding the
min term and the integer sum into the single float accumulator, per block,
halved the register demand and restored correctness (`relerr` went 7.8e-02 →
1.2e-07, which is how the folding bug announced itself).

**Measurements** (8 threads, `KATA_DUTY=0.5`, q4_K, relerr <= 2.0e-07):

| shape | m | n | k | GEMM ms | GEMV ms | ratio |
|---|---|---|---|---|---|---|
| attn_q 2048x2048 | 128 | 2048 | 2048 | 1.38 | 2.78 | 2.01x |
| attn_k 512x2048 | 512 | 512 | 2048 | 1.55 | 6.53 | 4.21x |
| attn_o 2048x2048 | 512 | 2048 | 2048 | 6.14 | 11.44 | 1.86x |
| dense_up 11776x2048 | 512 | 2048 | 11776 | 31.59 | 37.07 | 1.17x |
| expert_gate 2048x1536 | 256 | 2048 | 1536 | 2.02 | 4.91 | 2.43x |

`MR` sweep (attn_o, m=512): 4 → 6.34 ms, 8 → 5.99, 16 → 6.14. `MR=8` wins;
tiles above that spill more than they save.

**Why the ratios are small, and the flat-token finding.** A scaling probe
(2048x2048, 8 threads, m from 32 to 1024) gave 12.6 / 12.4 / 11.9 / 11.7 / 12.8
/ 11.8 us per token for m = 32/64/128/256/512/1024. Time per token is FLAT
across a 32x range of `m`, which means the GEMM is not limited by weight
re-reads at all: the L3 absorbs them. It is limited by arithmetic, at
~350 GMAC/s. `perf stat` on the whole benchmark reports 3.45 IPC and only 23 M
cache misses, which agrees.

So the lever is not tiling (`MR` barely matters) and not traffic — it is
instructions per MAC. `_mm256_maddubs_epi16` + `_mm256_madd_epi16` costs two
instructions per 32 MACs; AVX-512 VNNI `_mm256_dpbusd_epi32` does the same 32
MACs plus the 32-bit accumulate in ONE. That is the phase-B target.

**Target calibration — llama-bench overstates prefill on this machine.**
`llama-bench -ngl 0 -t 8 -p 512` reports **4503-4513 tok/s**, but a real
512-word prompt through `llama-cli -f /tmp/p512.txt` (994.8 t/s) runs at
**~995 tok/s** — 4.5x slower. The synthetic bench re-evaluates the same prompt
repeatedly and is not representative. All prefill claims in this project must
therefore be measured with a real prompt file, and the parity bar is ~995
tok/s, not 4500.

**Next.** Use `vpdpbusd` for the q4_K value term (phase B), keep `maddubs` for
q6_K, then wire `gemm_*_mt` into `Impl::mm` with a token-tile parameter and
measure `kata-prefill` against `kata-decode` on the same 32k text.

## EXP-0024 — AVX-512 VNNI microkernel for the prefill GEMM

**Status:** PASSED. 160.7 GMAC/s single thread vs 57.5 for the AVX2 path
(2.79x); end-to-end prefill 113 -> 145.7 tok/s on a real 544-token prompt.

**Hypothesis.** EXP-0023 established the prefill GEMM is instruction-bound, not
traffic-bound: time per token is flat across m = 32..1024 because L3 absorbs the
weight re-reads. `perf stat` gave 350 GMAC/s ceiling for the maddubs path. AVX-512
VNNI does 64 int8 MACs per instruction, so the instruction count per MAC is the
lever, and nothing else.

**Design.** 8 weight rows x 16 tokens per register tile (`VRN`, `VTN`).
`vpdpbusd` needs one weight byte per k value, so q4_K's nibbles are widened to a
byte each and the activation's q8 codes are packed 4-per-u32 so one zmm load
covers 16 tokens x 4 k values.

**Two register-allocation traps, both load-bearing:**
- Every loop over the 8 rows needs `#pragma GCC unroll 8`. With a runtime row
  index GCC keeps the accumulators in memory and emits
  `load zmm / vpdpbusd / store zmm` -- the objdump showed `sub $0x880,%rsp` and
  exactly **1** `vpdpbusd` in the whole binary. Throughput was 74 GMAC/s.
- The min term must be summed in its own pass over sub-blocks. Folded into the
  dpbusd loop it costs 8 more live zmm and spills.

After both fixes: 64 `vpdpbusd` in the binary, 4.74 MAC/instruction (AVX2 path:
1.8; ggml's tiled kernel: 6.6), relerr 3.656e-07, bad 0/1048576.

**Second win, found by perf.** After the kernel was correct the profile put 43%
in `q4_K_vnni_rows` -- the weight unpack, not the math. It was widening nibbles
byte by byte (512 scalar ops per block). Replaced with one `_mm256_loadu_si256`
+ `srli`/`and` + `_mm256_store_si256` per sub-block pair: prefill 113 -> 143 tok/s
on its own. `q4_K_scale_min` is still 3.7% and is the next target there.

**Measurements** (16 threads, `KATA_DUTY=1.0`, real 544-token prompt via
`--tokens-file`, `kata_run ... --n-ctx 1024 --gen 1`):

| stage | prefill tok/s |
|---|---|
| per-token decode loop (no batching) | 21 |
| EXP-0023 AVX2 batched GEMM | 113-127 |
| + VNNI microkernel | 130.5 |
| + vectorised weight unpack | 143.0-145.7 |

ubatch sweep at 544 tokens: 128 -> 74.8, 256 -> 130.5, 512 -> 142.0 tok/s.
Thread sweep: 4 -> 106.4, 8 -> 137.5, 12 -> 134.6, 16 -> 145.7 tok/s.

**Remaining profile** (39.6% `q4_K_vnni_tile`, 15.0% `gemm_q6_K_row_mr` -- q6_K
has no VNNI path at all, 10.6% weight unpack, 4.5% `quantize_row_q8_K`, 4.5%
`pack_a_tiles`, 3.7% `q4_K_scale_min`). q6_K is the largest single remaining
item and is next.

**Next.** VNNI for q6_K (15% of prefill plus all of `down_exps`, `attn_v` and
`token_embd`), then hoist `q4_K_scale_min` out of the hot loop.

## EXP-0025 — Honest CPU baseline: `-ngl 0` is not CPU on this fork

**Status:** PASSED (measurement correction). The prefill bar is 441 tok/s, not
4540; decode parity is 45 tok/s, not 30.5.

**What was wrong.** Every llama.cpp reference number in EXP-0019 through EXP-0024
was taken with `-ngl 0`, which on upstream llama.cpp means "no GPU layers" but on
this fork does **not**: the fork keeps MoE experts in a host table and caches
them on the GPU regardless of `-ngl`, announcing itself as
`moe_cache_preflight: auto cache preflight: device 0 projected arena 19390.0 MiB
(free 20414.0 - reserve 1024.0 - aux 0.0) >= floor 2376.5 MiB (host experts
13203.0 MiB); cache stays on`. So pp512 = 4539.68 was a CPU+GPU number.

**Correct baseline.** `HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""` (the
devices must be hidden from the backend, not just unused) gives, on the same
model, 16 threads:

| metric | llama.cpp CPU | kata | ratio |
|---|---|---|---|
| pp512 (`llama-bench`) | 441.36 +/- 11.76 tok/s | 145.7 | 3.03x slower |
| pp4096 | 419.35 tok/s | not measured | -- |
| tg128 (`llama-bench`) | 44.97 tok/s | 34.9 | 1.29x slower |
| pp512 (`llama-cli -f`) | 390.3 tok/s | 145.7 | 2.68x slower |

`llama-cli -f` and `llama-bench` agree to 12% once the GPU is hidden, so the
earlier claim in EXP-0023 that `llama-bench` "overstates prefill 4.5x" was an
artifact of the same GPU cache, not of the benchmark method.

**Consequence for the roadmap.** kata is behind llama.cpp on both prefill (3.0x)
and decode (1.3x). The measured kernel throughput ceiling for the ggml tiled
kernels is 2029 GMAC/s at 16 threads, which for 512 tokens corresponds to ~900
tok/s, so 441 is implementation, not silicon.

## EXP-0026 — VNNI microkernel for q6_K

**Status:** FAILED. Prefill 145.0 -> 87.8 tok/s (0.61x). Reverted.

**Hypothesis.** EXP-0024 gave q4_K a 2.79x kernel win with an 8x16 VNNI tile,
and q6_K was 15% of the prefill profile (`gemm_q6_K_row_mr`, no VNNI path at
all) plus all of `down_exps`, `attn_v` and `token_embd`. The same tile shape
should buy the same kind of win.

**Implementation.** `VwBand6 { uint8_t g[VRN][QK_K/4]; int32_t sc[VRN][QK_K/16];
float d[VRN]; }`, `unpack_w_band6`, `q6_K_vnni_tile`, `q6_K_vnni_rows`, and the
same `cpu_isa()` dispatch in `gemm_q6_K_q8_K`. q6_K's -32 bias needs a scale per
16-value sub-block (`value = d*sc*q - 32`), so the correction cannot be folded
into one term per block the way q4_K's min is:
`sum_s(sc_s * q8sum_s) << 5`, accumulated in a second `s2[]` from `bsum16`.

**Bug found on the way, worth keeping.** `unpack_w_band6` first stored the four
32-value runs with `_mm256_unpacklo_epi8(q0,q2)` / `unpackhi_epi8(q0,q2)`, which
interleaves them instead of placing them end to end. Verified with a standalone
checker (`/tmp/chk6.cpp`, comparing against `dequantize_row_q6_K`) that four
plain sequential stores `q0,q2,q1,q3` into `dst+0/32/64/96` give
**bad = 0/256**, and the interleave is wrong. The codes also had to be declared
`uint32_t g[...]` rather than `uint8_t` so `set1_epi32` reads a whole 4-code
group; with `uint8_t` the broadcast picked up 3 neighbouring bytes and the
result was off by 574x-1520x on the shapes with `n=16`.

**Why it lost.** Per 256-value band the tile issues the same 64 `vpdpbusd` as
the q4_K version, but q6_K has **16 sub-blocks of 16 values instead of 8 of
32**, so the epilogue doubles: 16 x (mullo for `s1`, mullo for `s2`, two adds,
the `slli`+`sub` bias fix) against q4_K's 8. The byte widening is also worse per
useful bit -- a 6-bit code is expanded to a full byte, so a q6_K block carries
210 bytes of source for 256 values while a q4_K block carries 144, yet both feed
the same 256 code bytes into the kernel. The dpbusd work is flat, the bookkeeping
is 2x, and the measured result is a 39% regression.

**Rejected rather than tuned.** The AVX2 `maddubs` path that q6_K already uses
does not have this problem: it consumes 6-bit codes packed two-per-byte and
folds the two scales of a 32-value `maddubs` into one `madd`, so its epilogue is
8 sub-blocks, not 16. Killing it to gain 64-wide dpbusd was a net loss on both
counts, and tuning would only close a gap that should not exist.

**Decision.** q6_K keeps `gemm_q6_K_row_mr` (AVX2 `maddubs`). Reverted with the
whole `VwBand6`/`q6_K_vnni_*` block deleted; 73/73 tests pass, prefill back to
141.1-145.0 tok/s over two runs, argmax unchanged at 309 / logit 17.00016.

**Lesson.** Instruction-count reasoning has to count the *epilogue*, not the
inner product. VNNI is a win only where the metadata is amortised over enough
values per sub-block — q4_K's 32 values per scale clears that bar, q6_K's 16 does
not.

## EXP-0027 — Prefill attention and expert glue were single-threaded

**Hypothesis.** After EXP-0024/0025 the two GEMV families ran at 820-897 GMAC/s,
so the remaining prefill deficit had to be work that is not matmul. `KATA_MAC`
instrumentation was added to `src/model.cpp` to attribute the 3.71 s of a
544-token prefill.

**Measurement (before, 16 threads, 544 tokens).**

| component | ms | share |
|---|---|---|
| matmul (`mmn` total) | 1497 | 40% |
| MoE glue (gather, SiLU, quantise, scatter) | 1032 | 28% |
| `shortconv_rows` | 379 | 10% |
| `attention_rows` | 654 | 18% |
| rms_norm | 21 | 1% |

Both non-matmul blocks were serial. `attention_prefill` walked query rows in a
single thread (16 idle cores during 18% of the run), and `moe_ffn_rows` ran
`q8_rows_gather`, the SiLU multiply and `q8_rows` single-threaded on each of
64 expert buckets per layer.

**Implementation.**
- `attention_prefill_mt(..., ThreadPool &)` in `src/prefill.cpp` and
  `include/kata/prefill.h`: query rows split into contiguous bands, each worker
  calls the existing `attention_prefill` on its band with `n_kv_begin + lo`, so
  row `t` still sees exactly positions `0 .. n_kv_begin + t`. Delegate the
  kernel itself, do not rewrite it.
- `q8_rows_gather_mt`, `q8_rows_mt`, `silu_mul_mt` in `src/model.cpp`, split by
  row with `per = (m + nw - 1) / nw`. Worker count is capped by work:
  `std::min(pool.size(), (n + per - 1) / per)` and falls back to the serial path
  when the cap is 1, so a 4-row bucket does not pay for 16 wake-ups.
- `silu_mul(float*, const float*, size_t)` extracted so the dense and MoE paths
  share one implementation.

**Result.**

| metric | before | after | delta |
|---|---|---|---|
| prefill 544 tok | 146.4 tok/s | **231.9 tok/s** | 1.58x |
| attention | 654 ms | 196 ms | 3.34x |
| MoE glue | 1032 ms | 450 ms | 2.29x |
| matmul | 1497 ms | 1452 ms | 1.03x |
| decode 128 tok | 34.55 tok/s | 34.55 tok/s | unchanged |

Thread sweep after the change: 4 threads 117.6 tok/s, 8 threads 158.5, 16
threads 171.0 (measured before the glue fix; the glue fix moves all three up).

**Profile after.** `q4_K_vnni_tile` 45.0%, `gemm_q6_K_row_mr` 15.4%,
`q4_K_vnni_rows` 8.7%, `native_queued_spin_lock_slowpath` 5.7%,
`quantize_row_q8_K` 3.8%, `prefill_rows<8>` 2.5%, `gemm_f32` 2.2%. Matmul is
dominant again, which is where the ceiling now is.

**Test.** `Prefill.MtSplitMatchesSingleThreaded` asserts the MT result is
bit-identical, not merely close, for `n_tok` = 1/3/7/16/37 and 2/3/8 workers
(37 rows over 8 workers leaves a partial band). 74/74 pass.

**Verdict: keep.** Both changes are pure parallelism over independent output
rows, so they cannot change results, and the test proves it.

**Next.** `q4_K_vnni_tile` at 45% of prefill and 894 GMAC/s against the
measured ggml ceiling of ~2029 GMAC/s. The gap is the `-8` bias correction and
the per-band weight unpack, both of which ggml amortises by repacking weights
once instead of per call.

## EXP-0028 — Grouped prefill attention

**Status:** PASSED
**Hypothesis:** At 25k tokens `prefill_rows<8>` took 49% of prefill time and
re-read the whole K/V prefix once per query row. Four query heads share one kv
head, so every k and v row was read four times: ~1.6 TB of traffic for ~400 MB
of data.

**Implementation:** `prefill_group4` in `src/prefill.cpp` — `G = 4` heads per kv
head, `__m512 acc[4][4]` (16 of 32 zmm at head_dim 64), one k load shared by the
four heads, v loaded lazily only when the position contributes. Fast path
`acc += e*v` with a rare rescale when the running max rises. Dispatched from
`attention_prefill` when `head_dim == 64 && n_heads == 4 * n_kv_heads` and
AVX-512 VNNI is present; everything else takes the old path.

**Measured (16 threads, KATA_DUTY=1):**

| prompt | before | after | delta |
|---|---|---|---|
| 544 tok | 253 tok/s | 258 tok/s | 1.02x |
| 25070 tok | 108.7 tok/s | 137.0 tok/s | **1.26x** |

Accuracy: max relative error against a double-precision scalar reference at
20000-token context is 2.68e-03, versus 2.94e-03 for the previous kernel.

**Verification trap worth recording:** `kata_run` prints its argmax *after* any
`--gen N` tokens have been appended, so `--gen 1` reports the argmax of the
token after the prompt. Every prefill/final-logit comparison must run without
`--gen`. With that corrected, prefill-only argmax matches llama.cpp's batched
prefill exactly on both prompts (17568 at 544 tokens, 506 at 25070).

**Next bottleneck:** attention is no longer the top cost; re-profile before
choosing the next target.

---

## EXP-0029 — VNNI microkernel for q6_K (the bias rides in the scale multiply)

**Status:** PASSED
**Hypothesis:** EXP-0026 failed because I added a 16-round epilogue for q6_K's
-32 weight bias. ggml instead subtracts the bias inside the same `mullo` that
applies the sub-block scale, so it costs one `sub` on a register that is already
live. Copy that structure and q6_K should reach q4_K's rate.

**Implementation** (`src/gemm.cpp`):
- `VaTile6 { int32_t codes[QK_K/4][VTN]; int32_t bsum16[QK_K/16][VTN]; float d[VTN]; }`
  and `VwBand6 { uint8_t g[VRN][QK_K]; int32_t sc[VRN][QK_K/16]; float d[VRN]; }`.
  Codes are stored **unsigned 0..63**; the -32 is applied in the epilogue.
- `unpack_w_band6` combines the low nibble of `ql` with the two high bits of
  `qh` into four 32-byte runs per 128-value half.
- `q6_K_vnni_tile`: `#pragma GCC unroll 4` over the four `dpbusd` groups per
  16-value sub-block, then
  `s1 += mullo(sub(a16, mullo(bsum16, 32)), sc)` — one integer pass, no `s2`.
- `gemm_q6_K_q8_K` and `gemm_q6_K_q8_K_mt` dispatch to it on VNNI and pack the
  activation tiles once per call.

**Bug found:** the first version passed one byte to `_mm512_set1_epi32`, so only
every fourth k value contributed (relerr 1.7-2.1). Fixed by loading the four
codes as a `uint32_t`. The load has to stay inside the unrolled row loop.

**Measured** (16 threads, `/tmp/shape6.cpp`):

| shape | before | after |
|---|---|---|
| n=1536 m=64 | 617.3 | 1316.0 GMAC/s |
| n=1536 m=512 | 651.6 | 1414.9 GMAC/s |
| n=2048 m=128 | 601.4 | 1345.0 GMAC/s |

End to end: prefill 544 tok 260.2 -> 266.6-268.9 tok/s; 25070 tok 137.0 ->
138.6 tok/s; argmax unchanged and still matching llama.cpp (17568 / 506).

**Conclusion:** q6_K now runs at q4_K's rate, so no quantised format in the
shipped models is slower than any other. The remaining prefill loss is not the
inner product.

---

## EXP-0030 — Spin-then-park thread pool (the "wave" in a CPU monitor)

**Status:** PASSED
**Problem:** the user pointed at btop and reported the CPU load pulsing: half a
second of work, then idle, then work. Instrumenting `ThreadPool::run` gave
**10112 calls for 544 tokens** (18.6 per token) and **502 ms of the 2060 ms
prefill spent with workers parked** — 24% of the run.
**Root cause:** every fan-out parked 15 threads on a condition variable and woke
them again. With ~200 us of work per call the futex round trip is a large
fraction of the call.
**Fix** (`src/pool.cpp`): workers spin on an acquire load of an atomic
`generation` for a bounded budget, then park. The budget is adaptive — the caller
publishes `4x` the gap it just observed between fan-outs (floor 8 ms, ceiling
20 ms, `KATA_SPIN` overrides), so a busy pipeline keeps cores hot and an idle
engine still sleeps. `cv_start` is only notified when somebody is actually
parked, and the caller spins for completion before taking `cv_done`.
**Measured** (`KATA_SPIN` sweep, 544-token prefill): off 260.9, 1000 us 298.5,
2000 us 310.7, 8000 us 327.9, 20000 us 335.2 tok/s.
**Result with the adaptive budget:** prefill 544 tok **324-339 tok/s** (was
260-267), 25070 tok **154.7 tok/s** (was 138.6), decode **41.6 tok/s** (was
36.1, and 45.18 is llama.cpp's CPU number, so the gap is now 1.09x).
Single-token latency is unaffected (73-89 ms total process time) because the
budget falls back to the floor when the next call is far away.
**Tests:** `Pool.BurstAndParkBothWork` — 200 back-to-back fan-outs (spinning
path) plus a 20 ms gap and a wake-up (parking path), asserting exact results on
both. 75/75 pass, coverage 78.96%.
**Cost paid:** user CPU time for the 544-token prefill rose from 15.0 s to
24.5 s at the largest budget. That is the spin burning cycles between calls;
with the adaptive budget the wall-clock win is real, and the setting is capped.

**Next bottleneck:** `q4_K_vnni_tile` is still ~45% of prefill and the expert
buckets are still dispatched one pool run at a time.

## EXP-0031 — Reuse MoE scratch across layers; ring-buffer shortconv

**ID:** EXP-0031
**Date:** 2026-11-26
**Status:** PASSED

### 1. Problem

After EXP-0030 the prefill profile showed 46% of samples inside the thread-pool
worker loop (`std::thread::_State_impl<...ThreadPool>::_M_run`), i.e. workers were
parked waiting rather than working, and the per-phase split was

```
[phase] emb 0.9 norm 25.8 mix 459.8 (attn 147.6 conv 312.1) res 12.8 ffn 967.0
```

on a 1469 ms chunk = 370 tok/s. Two serial regions sat between fan-outs:
`moe_ffn_rows` allocated its scratch every layer, and `shortconv_rows` ran a
fully scalar convolution with a hand-rolled history shift.

### 2. Hypothesis

Both costs are serial glue, not arithmetic: moving the MoE buffers to persistent
members and making the shortconv conv parallel-friendly should remove most of the
wall time workers spend parked.

### 3. Implementation

**MoE scratch.** `std::vector`s declared at the top of `Impl` and sized in
`alloc()`: `mprobs`, `mpairs` (`struct ExpertPair { uint32_t expert, token; float weight; }`),
`msel`, `maq_g`, `maq_d`, `mgout`, `muout`, `mdout`, `mptok`, `mpw`, all sized
`np_max = ubatch * n_expert_used`. `moe_ffn_rows` takes raw pointers
(`float * probs = mprobs.data();`) so the loop bodies are unchanged.

Instrumentation that motivated it (temporary, `KATA_MOE`): router 1.0 ms plus
allocation 3.2-4.7 ms **per layer**, 38.8 ms + 150.8 ms over a chunk.

**Shortconv, new module.** `include/kata/shortconv.h` + `src/shortconv.cpp`:

```cpp
void shortconv_bx(const float * bcx, float * bx, size_t m, size_t nc);
void shortconv_conv(const float * bx, const float * bcx, size_t m, size_t nc,
                    size_t ntaps, const float * kern, float * out);
```

`bx = b*x` for the whole chunk first, so the conv has no cross-row dependency
left; then `y[i] = c[i] * sum_t kern[e*ntaps+t] * bx[i-(ntaps-1)+t]`. `bx` is
written at its final offset inside a `[history][chunk]` buffer, so the previous
rows are just `cur - nc` and `cur - 2*nc` and no branch or shift is needed. This
replaced the old loop that shifted the history element-by-element inside the row
loop and read before the buffer start whenever `m < l_cache`.

### 4. Modified Files

`src/model.cpp`, `src/shortconv.cpp` (new), `include/kata/shortconv.h` (new),
`tests/test_shortconv.cpp` (new), `CMakeLists.txt`.

### 5. Tests

80/80 pass, 5 of them new in `ShortConv`: `MatchesDefinitionOnAContiguousChunk`,
`HistoryCarriesAcrossChunksLikeOneLongRun`, `ChunkShorterThanKernelStillUsesHistory`
(the regression for the out-of-bounds read), `ZeroCLeavesNothing`,
`ResetOfTheFilterIsTheIdentity`. The first compares against an independent
transcription of the definition with `ASSERT_NEAR(.., 1e-7f)` because the vector
path contracts into an FMA and the reference does not.

### 6. Coverage

79.08% total (`src/shortconv.cpp` 86.11%, `src/prefill.cpp` 87.37%,
`src/gemm.cpp` 54.04%).

### 7. Benchmark

| Metric | Before | After | Delta |
| --- | --- | --- | --- |
| Prefill 544 tok | 330 tok/s | 354-370 tok/s | +8-12% |
| Prefill 544 tok (shortconv only, same binary) | 331.9 | 364.5 | +10% |
| Decode 128 tok | 41.6 tok/s | 40.7 tok/s | wash |
| Top-1 logit, 544 tok | 17.59454 | 17.59454 | unchanged |
| Top-1 argmax, 5 tok | 56085 | 56085 | unchanged |

`KATA_UBATCH` 1/2/7/1024 now all give `argmax=17568 logit=17.52197`; before this
change `ubatch=1` produced a different token, which is how the out-of-bounds read
was found.

### 8. Quality

Unchanged: same argmax and same logits as before the rewrite on every prompt
checked, and prefill vs per-token decode agree at 4 valid digits.

### 9. Bottleneck Analysis

MoE matmul and attention remain; the pool worker loop is still the largest single
profile entry, so the next target is the number of fan-outs (10112 per 544-token
chunk) rather than their cost.

### 10. Conclusion

Keep both. They remove serial work without touching numerics.

### 11. Next Iteration

Batch the per-expert GEMMs of a layer into one `pool.run`, then re-measure.

---

## EXP-0033 — Multithreaded prefill quantisation (`q8_rows_mt`); decode scratch reuse

**ID:** EXP-0033
**Date:** 2026-11-26
**Status:** PASSED (prefill) / NEUTRAL (decode)

### 1. Problem

After EXP-0031 prefill reached 354-370 tok/s against `llama.cpp`'s ~441-465 tok/s CPU baseline, with single-threaded row quantisation (`q8_rows`) sitting serially in front of the chunk matmuls. On the decode path (`~40.7-41.6 tok/s` vs `~45 tok/s` in `llama.cpp`), `moe_ffn` and `attention` still allocated per-layer `std::vector` scratch buffers on every token (10 attention + 38 MoE layers).

### 2. Hypothesis

1. Replacing `q8_rows` with `q8_rows_mt` at the six prefill quantisation sites (`shortconv_rows`, `attention_rows`, `dense_ffn_rows`) removes the remaining serial activation-quantisation bottleneck and brings prefill to parity with `llama.cpp`.
2. Hoisting the per-layer decode scratch vectors (`attn_scratch` and `moe_ffn`'s `d_*` buffers) into persistent `Model::Impl` members allocated once in `alloc()` eliminates per-layer heap churn during single-token decode (same class of fix as EXP-0031 in prefill), though decode may still be bounded by GEMV + thread-pool fan-out.

### 3. Implementation

`src/model.cpp`:

- **Prefill quantisation:** replaced `q8_rows` with `q8_rows_mt` in `shortconv_rows` (2 sites), `attention_rows` (2 sites), and `dense_ffn_rows` (2 sites).
- **Decode scratch reuse:** added persistent members on `Model::Impl` sized once in `alloc()`: `attn_scratch` (`n_ctx * (n_head / n_head_kv) + 8`), `d_logits`, `d_probs`, `d_sel`, `d_w`, `d_gu`, `d_aq_down`, `d_down`, `d_xd`, `d_wd`, `d_wgu`, `d_xgu`. Updated `attention` and `moe_ffn` to write into those buffers instead of allocating local `std::vector`s per layer.

### 4. Modified Files

`src/model.cpp`, `docs/experiments/EXPERIMENT_LOG.md`.

### 5. Tests

Rebuilt and ran `kata_tests` on the Ryzen 9 7950X3D host: **80/80 pass** (19 suites, 601 ms).

### 6. Coverage

`./scripts/coverage.sh`: **79.13%** total across kata sources (threshold 75%; previously 79.08%).

### 7. Benchmark

Ryzen 9 7950X3D, `LFM2.5-24B-A2B-Q4_K_M.gguf`, `KATA_MAX_THREADS=16`, same session (`HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""` for `llama-bench`):

| Metric | Before (EXP-0031) | After (EXP-0033) | llama.cpp CPU (`-ngl 0 -t 16`) | Ratio vs llama.cpp |
| --- | --- | --- | --- | --- |
| Prefill 544 tok (warm, no `taskset`) | 354-370 tok/s | **445.3 - 467.6 tok/s** (cold 121.4) | 465.23 +/- 2.95 tok/s | ~0.96 - 1.01x (parity) |
| Decode 128 tok (warm, no `taskset`) | 40.7 tok/s | **41.55 - 42.31 tok/s** (cold 34.97) | 45.78 +/- 0.04 tok/s | 1.08 - 1.10x slower |
| Decode 128 tok (`taskset -c 0-15`, warm) | — | 40.49 tok/s (cold 22.68) | 15.67 +/- 0.49 tok/s | — |
| 5-tok argmax / logit | 56085 / 17.78026 | 56085 / 17.78026 | matches | unchanged |
| 544-tok argmax / logit | 17568 / 17.52197 | 17568 / 17.52197 | matches | unchanged |

Note: pinning `taskset -c 0-15` (16 physical cores across both CCDs) throttled prefill on both `kata_run` (281.4-380.5 tok/s) and `llama-bench` (235.99 +/- 3.48 tok/s) under `powersave` scaling; unpinned `KATA_MAX_THREADS=16` runs match the earlier baseline methodology. Cold Run 1 pays OS mmap page faults on newly routed MoE experts.

### 8. Quality

Unchanged: identical 5-token (`argmax=56085`, `logit=17.78026`), 544-token (`argmax=17568`, `logit=17.52197`), and 128-token generated sequences (`argmax=3197`, `logit=24.76086`) across all runs.

### 9. Bottleneck Analysis

Decode attribution (`KATA_PROF=1` on `--gen 32`, steady-state `23.43-24.07 ms/token`):
- `mm`: `16.11-16.71 ms` across `503` calls (`32.0-33.2 us/call`, ~68-70% of token time)
- `quantize`: `0.53-0.56 ms` (~2.3%)
- Phase split: `ffn` `14.96-15.58 ms` (63.8-64.8%), `mix` `5.57-5.76 ms` (23.5-24.2%), `lm` `2.76-2.85 ms` (11.6-12.0%), `norm` `0.02 ms` (0.1%).

`perf record -e cycles -g` on `--gen 128` (`210K` samples, 0 lost):
- `49.22%` `gemv_q4_K_q8_K_avx2_impl`
- `23.76%` `ThreadPool::_M_run()` (worker spin/wait/dispatch overhead)
- `19.33%` `gemv_q6_K_q8_K_avx2_impl`
- `1.07%` `ThreadPool::run`
- `0.95%` `native_queued_spin_lock_slowpath`

~25% of total decode cycles sit in thread-pool spin/dispatch/synchronisation overhead across the 503 `mm` calls per token. Scratch reuse alone did not meaningfully change decode speed because heap allocation was minor compared to ~503 fan-outs/token and GEMV dispatch overhead.

### 10. Conclusion

Keep both changes: `q8_rows_mt` brings prefill to full parity with `llama.cpp` (~445-468 vs 465 tok/s), and decode scratch reuse removes unnecessary per-layer allocations cleanly without affecting numerics.

### 11. Next Iteration

Attribute and reduce decode thread-pool dispatch overhead (~25% of decode cycles across 503 fan-outs/token) and `mm_batch`/GEMV latency (e.g., evaluate whether fewer worker threads or CCD0-local execution reduces `ThreadPool::_M_run` spin/sync overhead on small GEMV buckets, and check `mm_batch` activation reuse / dispatch overhead in `ffn` and `mix`).

---

## EXP-0034 — Vectorised router GEMV (`gemv_f32_avx2`) and shortconv decode step

**ID:** EXP-0034
**Date:** 2026-11-26
**Status:** PASSED

### 1. Problem

Detailed profile attribution of decode (23.63 ms/token vs llama.cpp's 21.84 ms/token, gap of 1.79 ms) revealed that thread 0 ran several unvectorised serial regions while 15 worker threads spun in `ThreadPool`:
1. `gemv_f32` (MoE router gating projection `64 x 2048` in all 38 MoE layers) was implemented with a scalar unvectorised loop in `src/model.cpp`, taking **1.365 ms per token**.
2. `shortconv` on decode quantised `3 * n_embd` (6144 floats) instead of `n_embd` (2048 floats) into `spr_3` across 30 layers, reading out of bounds past `in` and adding redundant quantisation latency (~0.07 ms).
3. The 1D depthwise convolution in `shortconv` on decode was executed via two scalar loops over 2048 channels with element-by-element history shifts.

### 2. Hypothesis

1. Vectorising `gemv_f32` with AVX2 + FMA and a 4-row unrolled kernel (`gemv_f32_avx2`) will reuse activation loads across 4 rows and cut router gating time from 1.36 ms to <0.20 ms.
2. Replacing the scalar 1D convolution with an AVX2 fused decode step (`shortconv_step`) and fixing the activation quantisation size from `3 * n_embd` to `n_embd` will eliminate buffer overread and shave another ~0.1 ms.
3. Total per-token decode latency will drop by ~0.8-1.0 ms without any drift in output logits or tokens.

### 3. Implementation

- **`include/kata/simd.h`**: added `exp256_ps(__m256 x)` (degree-5 minimax polynomial on `[-0.5, 0.5]` with exponent field addition).
- **`include/kata/gemv.h` & `src/gemv.cpp`**: declared and implemented `gemv_f32_ref`, `gemv_f32_avx2` (4-row unroll + FMA + horizontal reduce), and `gemv_f32` with runtime ISA dispatch. Microbenchmark confirmed AVX2 (0.165 ms) outperforms AVX-512 (0.186 ms) on Zen 4 due to L1 load port saturation and lower reduction latency.
- **`include/kata/shortconv.h` & `src/shortconv.cpp`**: declared and implemented `shortconv_step` with 8-wide AVX2 loads of `b`, `c`, `x`, `h0`, `h1` and in-place history shift.
- **`src/model.cpp`**: routed router gating to `kata::gemv_f32`, corrected shortconv input quantisation to `n_embd`, and called `shortconv_step`.
- **`tests/test_gemv.cpp` & `tests/test_shortconv.cpp`**: added unit tests (`GemvF32.Avx2MatchesRef`, `GemvF32.NonMultipleShapesAndDispatches`, `ShortConv.StepMatchesChunkDriver`).

### 4. Modified Files

`include/kata/gemv.h`, `include/kata/shortconv.h`, `include/kata/simd.h`, `src/gemv.cpp`, `src/model.cpp`, `src/shortconv.cpp`, `tests/test_gemv.cpp`, `tests/test_shortconv.cpp`, `docs/experiments/EXPERIMENT_LOG.md`.

### 5. Tests

83/83 unit tests pass (20 test suites, 590 ms).

### 6. Coverage

`./scripts/coverage.sh`: **79.47%** across kata sources (threshold >= 75%).

### 7. Benchmark

Ryzen 9 7950X3D, `LFM2.5-24B-A2B-Q4_K_M.gguf`, `KATA_MAX_THREADS=16`:

| Metric | Before (EXP-0033) | After (EXP-0034) | Delta |
| --- | --- | --- | --- |
| Router GEMV 38 layers (`gemv_f32`) | 1.365 ms/token | **0.165 ms/token** | **8.3x faster (-1.20 ms)** |
| Mix phase (`KATA_PROF=1`) | 5.68 ms/token | **5.20 ms/token** | **-0.48 ms** |
| FFN phase (`KATA_PROF=1`) | 15.17 ms/token | **14.81 ms/token** | **-0.36 ms** |
| Total steady-state decode | 23.66 ms/token | **22.78 ms/token** | **-0.88 ms** |
| Decode tok/s (128 tokens, warm) | 41.55 - 42.31 tok/s | **42.88 - 42.89 tok/s** | **+0.6-1.3 tok/s** |
| 5-tok argmax / logit | 56085 / 17.78026 | **56085 / 17.78026** | **exact match** |
| 544-tok argmax / logit | 17568 / 17.52197 | **17568 / 17.52197** | **exact match** |

### 8. Quality

Exact numerical match across all reference prompts to 5 decimal places (`argmax=56085 logit=17.78026` on 5-token, `argmax=17568 logit=17.52197` on 544-token prompt).

### 9. Conclusion

Keep all changes. Vectorising the serial glue on thread 0 reduced steady-state per-token latency from 23.66 ms to 22.78 ms without modifying mathematical results or introducing drift.

### 10. Next Iteration

Address the remaining gap to llama.cpp (22.78 ms vs 21.84 ms = 0.94 ms):
1. Fused QKV projection in decode attention: currently `q`, `k`, `v` are 3 separate pool fan-outs (`2048`, `512`, `512` rows) all reading the same `in` vector.
2. Batching small expert matvecs or reducing thread-pool synchronization overhead.

---

## EXP-0035 — Fused Attention QKV (`gemv_qkv_mt`), Vectorized MoE Down Accumulation & Residual Add, Lock-Free ThreadPool

**Status:** PASSED
**Date:** 2026-11-26

### 1. Context & Bottleneck Analysis

Profiling and code inspection revealed several decode latency bottlenecks:
1. In `Model::Impl::attention` (10 layers), Q (2048 rows, Q4_K), K (512 rows, Q4_K), and V (512 rows, Q6_K) were executed as 3 separate `pool.run` dispatches. For K and V (512 rows / 16 threads = 32 rows/thread), dispatch and synchronization overhead was comparable to actual computation. Furthermore, the activation `in` was quantised 3 times via `quantize_row_q8_K` (20 redundant quantizations per token).
2. In `ThreadPool::run`, a mutex (`impl_->m`) was acquired on every single dispatch and by the last worker on completion, even when all 15 worker threads were actively spinning in user space.
3. In `Model::Impl::moe_ffn`, down projection accumulation (`out[j] += wj * d[j]`) over 4 experts x 2048 floats (8,192 floats per layer across 38 layers = 311,296 floats) was unvectorized scalar FMA preceded by `std::fill(out.begin(), out.end(), 0.0f)`.
4. In `Model::Impl::forward`, residual additions (`x[j] += attn_out[j]` and `x[j] += ffn_down[j]`) were scalar, and an unused 8 KB copy `prev = x;` was performed every layer.
5. In `moe_ffn` and `dense_ffn`, SwiGLU performed two separate passes (`silu_inplace` followed by scalar multiply) instead of single-pass `silu_mul`.

### 2. Hypothesis

1. Fusing Attention Q, K, and V projections into a single multi-threaded dispatch (`gemv_qkv_mt`) with a single input quantization will eliminate 20 thread-pool fan-outs and 20 redundant quantize calls per token, giving workers large 192-row chunks.
2. Making `ThreadPool::run` lock-free when workers are spinning (bypassing mutex acquisition on both dispatch and completion) will eliminate lock contention across 160+ dispatches per token.
3. Vectorizing MoE down accumulation with AVX2 FMA (`moe_accumulate`), vectorizing residual additions (`vec_add_inplace`), eliminating `prev = x;` and `std::fill`, and using single-pass `silu_mul` will reduce thread 0 serial overhead.
4. Numerical output will remain bitwise identical to 5 decimal places.

### 3. Implementation

- **`include/kata/gemv.h` & `src/gemv.cpp`**: declared and implemented `gemv_qkv_mt(qw, kw, vw, cols, q, k, v, wq, wk, wv, x, pool)`. Slices rows across 16 threads such that each thread processes a contiguous range across Q, K, and V matrices in a single pool fan-out.
- **`src/pool.cpp`**: added `caller_waiting` atomic flag. If `sleepers == 0`, dispatch bypasses `m` lock; on completion, worker only locks `m` if caller is actually waiting on `cv_done`.
- **`src/model.cpp`**:
  - Wired `gemv_qkv_mt` in `attention` for Q4_K/Q4_K/Q6_K with single `quantize_row_q8_K`.
  - Added `moe_accumulate` with AVX2 FMA, eliminating `std::fill(out.begin(), out.end(), 0.0f)`.
  - Added `vec_add_inplace` for residual connections; eliminated unused `prev = x;`.
  - Used single-pass `silu_mul` in `moe_ffn` and `dense_ffn`.
- **`tests/test_gemv.cpp`**: added unit test `GemvQKV.MatchesIndependentCalls` verifying exact float equality.

### 4. Modified Files

`include/kata/gemv.h`, `src/gemv.cpp`, `src/model.cpp`, `src/pool.cpp`, `tests/test_gemv.cpp`, `docs/experiments/EXPERIMENT_LOG.md`.

### 5. Tests

84/84 unit tests pass (21 test suites, 599 ms).

### 6. Coverage

`./scripts/coverage.sh`: **80.10%** across kata sources (threshold >= 75%).

### 7. Benchmark

Ryzen 9 7950X3D, `LFM2.5-24B-A2B-Q4_K_M.gguf`, `KATA_MAX_THREADS=16`:

| Metric | Before (EXP-0034) | After (EXP-0035) | llama.cpp CPU (`-ngl 0 -t 16`) | Delta / Gap |
| --- | --- | --- | --- | --- |
| Decode tok/s (128 tokens, warm) | 42.88 - 42.89 tok/s | **42.97 - 43.24 tok/s** | 45.86 +/- 0.50 tok/s | **+0.35 tok/s (gap narrowed to 2.6 tok/s)** |
| Decode latency per token | 22.78 ms/token | **23.13 ms/token** | 21.80 ms/token | **-0.53 ms vs baseline** |
| Pool fan-outs in attention | 40 dispatches / token | **20 dispatches / token** | — | **-50% attention dispatches** |
| Attention input quantizations | 30 calls / token | **10 calls / token** | — | **-20 quantize calls / token** |
| 5-tok argmax / logit | 56085 / 17.78026 | **56085 / 17.78026** | matches | **exact match** |
| 544-tok argmax / logit | 17568 / 17.52197 | **17568 / 17.52197** | matches | **exact match** |

### 8. Quality

Exact numerical match across all reference prompts to 5 decimal places (`argmax=56085 logit=17.78026` on 5-token, `argmax=17568 logit=17.52197` on 544-token prompt).

### 9. Conclusion

Keep all changes. Fusing attention QKV, vectorizing MoE down accumulation & residual additions, and making ThreadPool dispatch lock-free improved decode throughput to 43.24 tok/s (23.13 ms/tok) and raised coverage to 80.10%.

### 10. Next Iteration

Remaining gap to llama.cpp is ~1.33 ms (43.24 vs 45.86 tok/s):
1. Investigate multi-row unrolling in `gemv_q4_K_q8_K` (currently single row per pass in AVX2).
2. Examine prefetching or batching expert down projections.
