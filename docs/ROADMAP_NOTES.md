# Draft backlog (not scheduled, captured so it is not lost)

## After the current computation is optimised: shrink the data

Once per-token traffic is close to the memory ceiling, the next lever is moving
fewer bytes at all. Target list, ordered by measured share of per-token traffic
(LFM2.5-24B-A2B, 1490 MB):

| component | traffic | share | ideas |
|---|---|---|---|
| MoE experts (38 x 4 of 64) | 930 MB | 62% | expert-aware quantisation, per-expert bit allocation by activation frequency, low-rank factorisation of rarely-routed experts, expert pruning/merging, shared-expert extraction |
| shortconv (30 layers) | 283 MB | 19% | the 6144x2048 in_proj is 3x the width of everything else; low-rank or structured factorisation, sharing across adjacent layers |
| lm_head (128000x2048, tied) | 215 MB | 14% | vocabulary pruning, clustered/centroid softmax, low-rank head, sparse logits |
| attention (10 layers) | 62 MB | 4% | quantise KV to q8/q4, MLA-style latent compression, layer-selective precision |

Constraint recorded by the user: shrink data **without quality loss**. Every entry
here therefore needs a measurement of both speed and quality (perplexity delta or
agreement with the full-precision model) before it is adopted, not just a size win.

Related already-known facts:
- q4_K and q4_0 both cost 0.5625 B/weight; q6_K costs 0.82 B/weight.
- 4 of 64 experts active per token = 6.25% of expert weights are touched, so the
  930 MB is already "sparse"; the win must come from those 4 being cheaper.
