# ADR 0002 — Placement and execution configuration

Status: **ACCEPTED**, implementation split across the GPU backend work
Date: 2026-11-27

## Context

The user's requirement: the engine must be configurable the way llama.cpp is —
how much work goes to the GPU, how much to the CPU, and which tensors live where.
Today that is impossible because two things are hard-coded and one does not exist:

1. **Nothing is configurable at runtime.** Threads, ubatch, KV quantisation,
   profiling and the router are read from `KATA_*` environment variables
   (`src/model.cpp`: `KATA_PROF`, `KATA_ROUTER`, `KATA_KVQ`, `KATA_DUMP`,
   `KATA_DUMP_POS`, `KATA_UBATCH`; `include/kata/throttle.h`: `KATA_MAX_THREADS`,
   `KATA_DUTY`; `src/pool.cpp`: `KATA_SPIN`). Environment variables were chosen
   during measurement iterations because they let one binary be A/B'd without a
   rebuild. They are a research instrument, not a user interface: they are
   invisible, they are read once at construction so they cannot be changed per
   request, and they are exactly the mechanism that produced the false 171 vs 146
   tok/s A/B in EXP-0027.
2. **There is no backend selection at all.** `Model::Impl` holds one `ThreadPool`
   and calls `gemv_quant(...)` / `mm(...)` directly. A HIP backend cannot be
   attached without editing every call site, and the placement decision (which
   layer or tensor runs where) has nowhere to live.

The GPU experiments (EXP-0072..0088) say what the configuration must be able to
express, because each measured a different placement with a different cost:

| evidence | what it says the config must express |
|---|---|
| EXP-0078 | moving the fixed 674 MB of per-token weights and the KV cache to the device is 2.5x on its own; expert placement adds 1.6x more at 2 GB..8 GB |
| EXP-0080 | the win survives the layer dependency up to ~8 GB but degrades at 12 GB+, so the split must be tunable per size class, not fixed |
| EXP-0083/0087 | expert weights are read as one contiguous stream; placement granularity for experts must be per expert, not per layer |
| EXP-0088 | a host router costs 17.2 us per layer, so the router's device must be a configurable choice, and mixing a GPU expert set with a CPU router must be measurable |
| EXP-0085 | the GPU kernel is compute-bound from B≈4, so batch size and sequence count belong in the config too |

## Decision

**One config struct, resolved once, validated, and never read from the
environment inside a hot path.**

```cpp
namespace kata {

enum class Device { Cpu, Gpu };

struct Placement {
    // Tensors that are read for every token and are small enough to keep
    // resident: norms, attention projections, shortconv, dense FFN, lm_head.
    Device fixed = Device::Cpu;
    // The KV cache. Independent of `fixed` because it grows with context and is
    // the only term that changes during a run.
    Device kv = Device::Cpu;
    // Experts: either all on one device, or the N most-routed per layer on the
    // GPU (EXP-0078: 117 resident experts at 2 GB is already 2.5x). An explicit
    // per-[layer][expert] bitmap is the escape hatch for odd shapes.
    Device experts = Device::Cpu;
    size_t experts_on_gpu = 0;            // 0 = all of them follow `experts`
    std::vector<uint8_t> expert_bitmap;   // empty = derive from the fields above
    // Where the router runs. Cpu is correct for every placement; Gpu only makes
    // sense when the expert set of a layer is on the GPU (EXP-0088: 17.2 us per
    // layer otherwise).
    Device router = Device::Cpu;
};

struct Config {
    int threads = 0;            // 0 = physical core count, capped by KATA_MAX_THREADS
    size_t ubatch = 0;          // 0 = 1024 capped by n_ctx (EXP-0033 measured best)
    size_t batch = 1;           // sequences decoded together
    Placement placement;
    bool kv_int8 = false;       // EXP-0049: on from a position, off by default
    size_t kv_int8_min_kv = 0;
    bool profile = false;
    bool low_overhead = false;  // EXP-...: disable per-kernel timers
    unsigned spin_us = 0;       // 0 = the pool's adaptive default (EXP-0030)
};

// Resolved from CLI args, then defaults, then (for research only) KATA_* env.
Config resolve_config(const Config & explicit_cfg);
// Throws with a message naming the offending field and the reason.
void validate(const Config & c, const ModelInfo & info, const DeviceCaps & caps);
}
```

Rules this ADR fixes:

1. **The environment is read exactly once, inside `resolve_config`.** Precedence
   is explicit argument > `KATA_*` > default. Anything on a hot path takes its
   values from the resolved struct. This keeps the research workflow (which
   depends on env A/B) while making env-invisible behaviour impossible in the
   engine.
2. **Validation is loud.** A placement that cannot be satisfied must throw with
   the reason, not silently fall back to the CPU. The measured failure mode this
   prevents is EXP-0049: a `v_int8 = true` gate shipped on, every A/B compared
   float against float for weeks, and nothing complained.
3. **The placement is derived, not guessed, from device capacity.** `--n-gpu-layers`
   in llama.cpp is a layer count because its layers are homogeneous; LFM2.5's are
   not (10 attention, 30 shortconv, 2 dense, 38 MoE). So the knob that maps onto
   llama.cpp's is `--gpu-fixed` (a byte budget the resolver fills by cost, using
   the same per-tensor accounting as `tools/kata_place.cpp`) plus
   `--gpu-experts N` (a count, because experts are one homogeneous matrix stack).
   A byte budget is what the user actually has; a layer count is an accident of
   llama.cpp's uniformity.
4. **Every placement must be measurable with the existing stage reconciler.**
   `include/kata/prof.h` already reports per-stage time and refuses to reconcile
   when the parts do not sum to the whole. A placement that moves a stage to the
   device must keep that invariant, per device.
5. **No placement code until a backend exists.** This ADR is a decision, not a
   scaffold: `Placement` gets implemented together with the first real device
   backend, and no stub is committed in the meantime (the deleted `rope()` stub
   in EXP-0009 is the precedent — a no-op that looks like an implementation is
   worse than an absent one).

## Consequences

- The CLI grows: `--threads`, `--ubatch`, `--batch`, `--gpu-fixed MB`,
  `--gpu-experts N`, `--gpu-kv`, `--gpu-router`, `--kv-int8`, `--profile`,
  `--no-throttle`. Each maps onto a field above, and `--help` prints the resolved
  defaults so a user never has to read the source to know what happened.
- Two modes fall out of the same config and match the user's stated goal for
  batching: B=1 interactive (batch 1, minimal sync) and server mode (batch 8-16,
  latency traded for throughput). On the CPU the measured gain is 1.75x (EXP-0071);
  on the GPU it is 1.3-1.5x (EXP-0085), so batch is a throughput knob, not a
  default.
- The env variables stay for research but are documented as such, and the ones
  that change *results* (`KATA_KVQ`, `KATA_UBATCH`) get a warning printed to
  stderr when set, so a benchmark log always says which configuration produced it.
- Rejected alternative: per-backend subclasses of `Model` (a `CpuModel` and a
  `HipModel`). Both would duplicate the whole forward pass, and the interesting
  configurations are mixed, not one-or-the-other — EXP-0078/0080 are entirely
  about a single pass split across two devices.

## Open questions this ADR does not answer

- Whether the resolver should pick a placement automatically from a VRAM budget
  and a model profile, or always require the user to state it. The measurements
  support an automatic default for the fixed set (2 GB buys the whole first 2.5x,
  EXP-0078) but not for experts, where the best count depends on the traffic mix.
- Whether `batch` belongs in the same struct as placement. It is an execution
  parameter, not a placement one, but the two interact (GPU kernels are
  compute-bound from B≈4, EXP-0085) and separate structs invite invalid
  combinations.