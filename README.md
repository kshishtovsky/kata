# kata

LLM inference engine for weak and memory-limited hardware. C++20, no external
runtime dependencies.

## Status

Phase B (minimal working inference), iteration 001: q4_0 x q8_0 GEMV kernel.

## Build

    cmake -S . -B build -G Ninja
    cmake --build build --target bench_gemv kata_tests
    ./build/kata_tests
    ./build/bench_gemv --mb 512        # force DRAM-bound measurement

## Layout

    include/kata/   public headers: types, quant, cpu, gemv
    src/            implementation
    bench/          benchmark harnesses (bench_gemv, bench_vs_ggml)
    tests/          gtest suites
    docs/           experiment log, research, architecture

## Rules

Every change: one task, built, tested, measured, documented. See
docs/experiments/EXPERIMENT_LOG.md.
