# ADR 0001 — KV cache layout: tiled-transposed K, head-major V, one buffer

Status: **SUPERSEDED before implementation** — see "Why this was not built"
Date: 2026-11-27

## Context

Prefill attention at 32k was 42% of wall time, and its QK pass ran at 50 GMAC/s
against a measured 165 GMAC/s fp32 peak. The published explanation had been
"instruction bound". Counters disproved it: the head-major score pass touches
**4.29 GB of K** for a 32k prefix when 64 query rows share each tile, while a
tiled-transposed pass touches **67 MB**. That is a 64x traffic difference, and
it is the actual mechanism.

## Why this was not built (added 2026-11-27, before any product code)

The 2.14x microbenchmark was an oracle, and the arithmetic that killed it is one
line. In the microbench arm (c) the online-softmax state was **not carried
across tiles**: each tile was scored independently and no `acc[4][4]`, `m[4]`,
`s[4]` had to survive. The real kernel must keep 4 heads x 64 dims of
accumulator plus the running max and sum for every query row while it walks
tiles. With 1024 query rows and 1024 tiles per chunk that state is 1 KB per
row:

    acc spill traffic = 1024 rows x 1024 tiles x 1 KB x 2 (read+write) = 2.15 GB

against 4.29 GB of head-major K traffic today. So row-blocking saves **~2x
traffic, not 64x**, and the 64x figure never described a kernel that carries
softmax state. The honest expected whole-run gain drops from 1.10-1.20x to
roughly 1.02-1.05x on prefill — not worth a global cache layout change that
also touches the verified decode path.

Also measured, and kept because they are true regardless:

- tiled-transposed K helps decode only 1.12x at 16k and 1.16x at 32k, below the
  1.15x bar at 16k, and decode runs at 77.6 of ~78 GB/s — at the DRAM ceiling
  where layout cannot help.
- `pp`-style measurements of the tile write (0.04 -> 0.12 ms per 1024-token
  chunk) were correct; the write was never the problem. The problem was that
  the win only exists for a kernel that does not carry online-softmax state,
  i.e. a kernel that computes the wrong thing.
- Golden points pinned in this ADR (argmax 440 / 17778, logits 16.21150 /
  16.12790 for ub 512/1024; 27.95969 / 28.45286 / 28.03733 for ub
  512/1024/2048 at 16k) remain valid and should be used by any future
  attention work. Argmax is ubatch-stable; logits are not, so a golden gate is
  argmax plus a tolerance, never logit equality.
- The `nq >= 7` break-even and the `bench_qk_layout --selftest` guards stay
  useful for the score-pass kernel itself, which is a real 2.14-2.48x
  improvement **on a single query row that fits its state in registers**.

## Original decision (not implemented)

K is stored tiled-transposed, `[layer][kv_head][tile][dim][pos_in_tile]` with
`tile = 16` positions (one zmm). V stays head-major, because PV reduces over
positions with `dim` in lanes — the shape V already has. Both live in the same
allocation: there is no second buffer and no second copy path.

The tile is written **once, at token-write time**, by `write_kv_tiled`, straight
into its slot. It is never rebuilt from a head-major copy. Measured cost of the
strided store: 0.04 -> 0.12 ms per 1024-token chunk per 10 layers (17.8 GB/s,
2.85x the contiguous store). Full 16k prefill: prep (norm+rope+KV write) is
19.4 ms per chunk, 310 ms of 40646 ms = 0.76%. Even at 3x for the strided
store that is +620 ms = **+1.5%**, against ~42% of the run removed.

## Expected effect, recorded before the change

Corrected arithmetic (the first version of this ADR over-claimed 1.4-1.5x by
attributing the whole core win to the whole run):

| quantity | value |
|---|---|
| QK share of attention core | 76% (phase-mode probe) |
| core share of prefill | 42% |
| QK speedup, measured, frozen | 2.14-2.16x |
| **expected whole-run speedup** | **1.10-1.20x** (1.10x if PV is additive as 26 s, 1.20x if PV is the 24% remainder) |
| expected 32k prefill | 141 s -> **117-128 s** |

The range exists because the two probe formulas disagree about PV's share; the
before/after per-phase split is the measurement that settles it, not this table.
Do not quote 1.4x for this iteration.

## Thresholds

- The tiled path wins only when query rows reuse a tile: break-even at
  `n_query ~= 7`, measured 0.23x / 0.44x / 0.74x / 1.23x / 1.71x / 2.48x /
  2.75x for `nq` = 1 / 2 / 4 / 8 / 16 / 64 / 128.
- **Decode does not use the tiled path.** Re-measured against the current
  head-major cache (not against EXP-0018's obsolete position-major baseline):
  0.97x at 512, 1.03x at 2048, 1.04x at 8192, 1.12x at 16384, 1.16x at 32768 —
  below the 1.15x bar at 16k, and decode already runs at 77.6 of ~78 GB/s, i.e.
  at the DRAM ceiling where layout cannot help. Decode keeps reading the same
  buffer through its existing `stride_pos`/`stride_head` view, unchanged.
- The threshold is a function of `head_dim`, `G` and `TP`, not a constant:
  break-even `nq ~= (TP * head_dim * 4 bytes) / (bytes saved per row)`. It must
  be **re-measured for any `head_dim != 64`** — at `head_dim = 128` the per-tile
  transpose doubles and the break-even `nq` moves. Do not apply 2.14x blindly.

## Golden points (pinned before the change)

Confirmed on commit `e6714c1`, three ubatch values each, because fp
reassociation across chunk boundaries moves logits even without a layout change:

| prompt | ub=512 | ub=1024 | ub=2048 |
|---|---|---|---|
| 522 tok (prefill, layout touches this) | argmax 440, logit 16.21150 | argmax 440, 16.12790 | — |
| 16384 tok (prefill) | argmax 17778, 27.95969 | argmax 17778, 28.45286 | argmax 17778, 28.03733 |

Argmax is ubatch-stable at both lengths; logits are not, so **the golden gate is
argmax plus a logit tolerance, never a logit equality across ubatch**. The 522
case exercises prefill at small `nq` (the path where the layout loses); the
512-context decode case exercises the path the layout does not touch and guards
against regression there.

## Consequences

- One code path for one buffer; no duplicated layout, which is what killed the
  two earlier attempts (per-head vs grouped, `kt` vs head-major).
- PV becomes the dominant term in the attention core. The next lever is
  fusing PV with the online softmax in the same lane-parallel form, which is a
  separate iteration and must not be bundled into this one.
- `bench_qk_layout --selftest` guards the three structural errors that produced
  wrong numbers in this area (per-head k load, row-stride tile walk, per-head
  tile rebuild).
