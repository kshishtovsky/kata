# Batched prefill for kata — design

Status: draft for approval. Date 2026-11-25.

## Intent and why

kata today has no prefill. `run/kata_run.cpp` feeds prompt tokens one at a time through the
decode path, so a 32k-token prompt costs 32768 × ~28 ms ≈ 15 minutes. llama.cpp on the same
machine and the same model (LFM2.5-24B-A2B-Q4_K_M, 13.52 GiB, 23.97 B params, 4 B active)
does a 32k prompt in 10.5 s (3123 tok/s). The user's real workload is "empty chat, 32k tokens
of context", so this is the single most valuable thing the engine can get.

The user's ask (m02822, m02842): make prefill unreasonably fast, approach A (batched GEMM in
the shape llama.cpp uses) first, then B (kernels that approach the machine's VNNI ceiling),
explicitly CPU-only — "pretend there is no GPU". C (GPU prefill) is out of scope and belongs
to the later GPU roadmap.

## Goals

1. `kata_run` prefills a 32k prompt in one pass, with the same logits as the token-at-a-time
   path on short prompts (bit-exactness not required; the same q8_K activation quantisation
   must be used so the numeric story does not change).
2. 32k prompt at **≥ 3123 tok/s** — parity with llama.cpp on this machine — as the first
   target, and **≥ 4500 tok/s** as the B-phase target.
3. Generated tokens after prefill are identical to what the decode path produces from the
   same KV cache state.
4. No regression: decode speed stays within noise of 35 tok/s, and `kata_run --greedy`
   still reproduces the verified llama.cpp trajectory for the 5-token "The quick brown fox"
   prompt.

## Non-goals

- GPU/HIP work of any kind.
- Continuous batching / multi-request serving.
- KV quantisation (that is a separate experiment; prefill only has to leave the door open).
- Speculative decoding, prefix caching.
- Beating llama.cpp's arithmetic; only the plumbing and the kernel shape are in scope.

## Anti-goals

- A second inference path that produces different numbers from the decode path and cannot be
  diffed against it.
- Prefill speed bought by skipping work (e.g. dropping the causal mask, or rounding K/V).
- Thread counts above `KATA_MAX_THREADS`; the machine hard-resets under sustained all-core
  loads and that is not negotiable.

## Measured ground truth

All on this box, `LFM2.5-24B-A2B-Q4_K_M`, CPU only:

| what | value |
|---|---|
| `llama-bench -ngl 0 -t 16 -p 512 -ub 512` | 4360 tok/s |
| pp1024 / pp4096 / pp16384 / pp32768 | 4541 / 4247 / 3720 / 3123 tok/s |
| pp32768 with `-fa 0` (no flash attention) | 1759 tok/s |
| `-ngl 99` pp512 (GPU) | 6045 tok/s |
| kata decode, batched expert GEMV, 16 threads | 26.4–28.8 ms/token |
| kata decode, per-token traffic | ≈1.4 GB |

Two facts drive the design:

- Flash attention is worth **1.86x** at 32k even for llama.cpp, so prefill attention is not a
  footnote at this length — it is roughly half the win.
- Per-token FLOPs are ~4 GFLOP (2 B active × 2). 32768 tokens ≈ 131 TFLOP of int8 MACs ≈
  65 T-MAC. Zen 4 with `vpdpbusd` at 16 cores gives roughly 20 T-MAC/s, so the compute floor
  is ≈3.3 s = 9900 tok/s. llama.cpp sits at 3123, i.e. about a third of the floor. There is
  room, but not 10x room — the ceiling on this CPU is around 6–7k tok/s realistic.

## Approaches considered

**A. Shape-for-shape GEMM (llama.cpp's plan).** One batched GEMM per matmul, tokens bucketed
per expert for MoE, chunked at 512 tokens, plus a tiled causal attention. Target 3.5–4.5k
tok/s. Moderate work: a `gemm_q4_K_q8_K` / `gemm_q6_K_q8_K` pair, a bucketing pass, a tiled
attention kernel, and a batched forward path next to the existing decode one.

**B. A + kernels aimed at the VNNI ceiling.** The decode kernels re-read weights for every
token, so they were optimised for a single stream. Prefill is the opposite regime: a weight
block loaded into registers can be reused across all B activation rows. B is A with a
different inner loop (weight-stationary, activations streamed) plus K-blocking for L2.
Target 6–7k tok/s.

**C. GPU prefill.** 20–50k tok/s but a whole new backend. Rejected by the user's scoping.

Chosen: **A first, then B**, because A is also the correctness harness — without a working
batched GEMM there is nothing to compare a faster kernel against, and the batched path has to
prove it matches the decode path before it is worth making fast.

## The how

### 1. Chunking

`prefill(tokens, n)` processes the prompt in chunks of `B = 512` (env `KATA_UBATCH`), because
a 32k activation matrix at 2048 floats is 256 MB per intermediate and the attention score
block for a 32k chunk does not fit anywhere. Position counter advances per chunk; the last
chunk may be shorter. Metrics reported per chunk so a slow chunk is visible.

### 2. Activation layout

Per chunk, activations are `[B][n_embd]` row-major, one row per token. Every projection
consumes `X[B][K]` and produces `Y[B][N]`. Quantisation to q8_K is per row, once per
consumed matrix, cached for the whole chunk — this is the single biggest structural win
versus the decode path, where the same activation row is re-quantised once per matvec (152
times per layer).

### 3. GEMM kernels (phase A)

```
void gemm_q4_K_q8_K(size_t m, size_t n, size_t k, float * out, size_t ldc,
                    const block_q4_K * w, const block_q8_K * a);
void gemm_q6_K_q8_K(...same shape...);
```
- `m` = tokens (≤512), `n` = output rows, `k` = inner dim, `w` row-major `[n][k]`,
  `a` row-major `[m][k]`.
- Parallelism over `n` and `m` both: split `n` into tiles of 4–8 rows across threads, and if
  `n` is small (attention q/k/v: 2048/512/512) split `m` too.
- Inner loop, phase A: for each 256-element K superblock, hold one weight block (144 B q4_K)
  in registers and loop over the activation rows in a register tile of 8–16. `vpdpbusd`
  accumulates int32, scale applied per 64-element sub-block, `dmin` term via the block
  `bsums` exactly as the decode kernel does.
- Fallback scalar/AVX2 paths behind the same interface, selected by `cpu_isa()`, so the
  correctness tests run everywhere and the AVX-512 path is one of several.

### 4. MoE in prefill

Per MoE layer, per chunk:
1. Router for all B tokens at once (a 64-wide GEMV per token; keep the existing per-token
   code, it is 0.1% of the layer).
2. Count tokens per expert, prefix-sum into an offset table.
3. Scatter each token's `ffn_norm` row into `[total_tokens][n_embd]` grouped by expert
   (`expert_offsets[e]`).
4. For each expert with a non-empty group: one `gemm` of `group_size × 1536 × 2048` for
   gate, one for up, one `1536→2048` for down. Groups smaller than a threshold (say 8) reuse
   the existing batched GEMV path instead, to avoid GEMM setup cost.
5. Scatter-add weighted results back to the token rows.

Expert group sizes at B=512: 512×4/64 = 32 tokens per expert on average, which is small for
a GEMM — this is the known weak spot and the reason phase B matters. At B=512 chunking, a
trigger to consider is raising the chunk to 1024 for MoE layers only; measure before doing it.

### 5. Attention in prefill (tiled causal, flash-style)

New `attention_prefill(q, k, v, n_tokens, n_kv, ...)`:
- Query tile of 64 rows, key tile of 64, online softmax (running max and sum), never
  materialising a full score row.
- Causal mask applied per tile: diagonal tiles masked, strictly-lower tiles unmasked.
- GQA head grouping as in the decode kernel (`n_heads / n_kv_heads = 4`).
- Multi-threaded over query tiles — this is embarrassingly parallel, unlike decode.

Rationale for flash-style rather than "scores then softmax": at 32k a full score row is
128 KB in fp32 for one head, and llama.cpp's own `-fa 0` → `-fa 1` delta of 1.86x is direct
evidence that the two-pass form loses badly at this length.

### 6. Wiring

`run/kata_run.cpp` gets `prefill(tokens)` before the decode loop. `Model` gains
`void prefill(const uint32_t * tokens, size_t n, ModelState & st)`. The decode path is
untouched; both write into the same `k_cache`/`v_cache`/`conv_state`.

### 7. Verification

- `tests/test_gemm.cpp`: GEMM vs per-row GEMV for `m` = 1, 7, 64, 128, 512 with
  `n`/`k` drawn from the real shapes (2048×2048, 512×2048, 1536×2048, 11776×2048), tolerance
  set from the observed q8_K error, not a guess.
- `tests/test_attn.cpp`: `attention_prefill` vs a reference loop that calls the existing
  decode kernel once per query position on the same cache — same numbers, different shape.
  This is the strongest available test because the decode path is already verified against
  llama.cpp.
- End-to-end: prefill "The quick brown fox" (5 tokens) and check the logits against
  `/tmp/ref/d9` (NTOK=1 single-token reference, argmax 785) and against the decode path.
- Then 32k: prefill a 32k prompt and check that token N+1's logits equal what the decode
  path produces after decoding the same 32k one at a time (sampled positions, not all 32768,
  since that takes 15 minutes).
- A benchmark `bench/bench_prefill.cpp` reporting tok/s at 512/4096/16384/32768 next to
  `llama-bench` numbers in the same table.

## Resolved after the builder check

1. **What "identical" means.** Argmax identity on short prompts and a numeric tolerance at 32k,
   not bit-exactness. Contract: `|prefill_logits - decode_logits| <= 1e-3 * max|decode_logits|`
   at the checked positions, and the same argmax token. Bit-exactness is impossible once the
   accumulation order changes and would forbid register tiling entirely. A 32k divergence
   does **not** block phase B; it is recorded as a finding, the way EXP-0019's 16-token
   agreement was.
2. **Which tokens need logits.** Only the last token of the last chunk. `prefill` returns
   nothing and leaves `logits` populated for that token, which is exactly what
   `run/kata_run.cpp:100` needs to start decoding. Per-chunk logits are produced only behind
   `KATA_PROF`-style debug (`KATA_DUMP`), never in the timed path — the 215 MB lm_head would
   otherwise dominate the run by an order of magnitude.
3. **K/V layout for the tiles.** Eat the stride: `attention_prefill` reads the existing
   position-major cache with 256 B contiguous rows per position, which is a fine hardware
   stride. `kt` stays a decode-only optimisation and prefill does not maintain it. A
   prefill-only blocked layout is out of scope until a measurement says the stride costs more
   than 10%.
4. **Prompt source.** New `--tokens-file <path>` flag reading whitespace-separated ids, plus
   `/tmp/p2000.txt`-style generation for synthetic lengths. Ground truth at 32k is
   kata-prefill vs kata-decode on sampled positions (the decode path is already verified
   against llama.cpp to 785 argmax at short length); a llama.cpp 32k reference is not
   required and would cost 15+ minutes per sample.
5. **Throttle.** Prefill reports **work time** (sum of measured chunk times, pauses excluded)
   as the headline number, because that is what `llama-bench` reports and comparing wall
   clock against it would be dishonest in the other direction. Wall clock including duty idle
   is printed alongside. `bench_prefill` runs at `KATA_DUTY=1.0` under `KATA_MAX_THREADS`,
   the same guard every existing harness uses.

## Decisions I made as the builder's calls

- **B = 512** chunk default, matching llama.cpp's `-ub 512`, because it is the measured sweet
  spot there and matches the score-tile size.
- **Flash-style attention is in phase A**, not deferred to B, because the 1.86x measured at
  32k makes it a first-order term at the user's actual prompt length.
- **q8_K activations for prefill too**, not q8_0 or fp32, so prefill and decode share one
  numeric definition and the diff tests mean something.
- **Expert groups below 8 tokens fall back to batched GEMV.** Unmeasured; flagged as
  `shortcut:` in the code and to be re-measured once the GEMM exists.
- **New files rather than extending the GEMV files**: `include/kata/gemm.h`, `src/gemm.cpp`,
  `include/kata/attn_prefill.h`, `src/attn_prefill.cpp` — the existing kernels are verified
  and should not churn.

## What is left to the builder

- Exact register tiling in the GEMM inner loop; that is measurement work, not design.
- Whether to split by `n` tiles alone or by both `n` and `m` for the narrow projections.
- The threshold for the GEMV fallback on small expert groups.
- Whether the MoE scatter/gather needs an AVX-512 path; start with a plain loop and profile.

## Risks

- **The MoE groups are small.** 32 tokens per expert at B=512 means a GEMM with `m=32`,
  which wastes the weight-reuse the whole plan depends on. If this dominates, the fix is a
  larger chunk or an expert-major weight layout, and the measurement to make that decision is
  one profile of the MoE layer at B=512 vs B=1024.
- **Attention at 32k may outgrow the estimate.** 32k²/2 × 32 heads × 10 layers is 172 G
  score elements; at the machine's fp32 peak that is under a second, but a naive
  implementation will not be within 5x of peak.
- **Memory**: a 32k chunk of activations plus scores has to stay bounded; the 512 chunking and
  64×64 tiles are what keep it so.
- **Hard resets**: prefill is exactly the sustained all-core load that reset this machine
  three times. Every benchmark goes through `/tmp/safe_run.sh`, `KATA_MAX_THREADS` is
  respected, and a reset is treated as a stop signal.

## Order of work

1. `gemm_q4_K_q8_K` / `gemm_q6_K_q8_K` + tests vs GEMV (correctness first, phase A speed).
2. Batched forward: embedding, norms, shortconv, dense FFN, MoE with bucketing.
3. Tiled causal prefill attention + test vs decode kernel.
4. `kata_run` wiring, 5-token diff against llama.cpp refs, then 32k end-to-end.
5. `bench_prefill` table; EXP-0023 write-up; compare against the 3123 tok/s bar.
6. Phase B: weight-stationary inner loop, L2 blocking, prefetch; re-measure; EXP-0024.