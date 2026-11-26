cmake_minimum_required(VERSION 3.20)
project(kata LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(KATA_COVERAGE "Build with gcov instrumentation" OFF)

add_library(kata_core
  src/backend.cpp
  src/cpu.cpp
  src/norm.cpp
  src/gguf.cpp
  src/gemv.cpp
  src/gemm.cpp
  src/pool.cpp
  src/attn.cpp
  src/prefill.cpp
  src/model.cpp
  src/rope.cpp
    src/sampler.cpp
  src/quant.cpp
  src/shortconv.cpp
)
target_include_directories(kata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(kata_core PRIVATE -Wall -Wextra -Wpedantic -O3)

# Kernels carry their own target attributes so one binary runs on any x86 CPU
# and picks the fastest path at runtime.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_definitions(kata_core PRIVATE KATA_X86=1)
endif()

if(KATA_COVERAGE)
  target_compile_options(kata_core PRIVATE --coverage -O0 -g)
  target_link_options(kata_core PUBLIC --coverage)
endif()

add_executable(kata_ppl tools/pata_ppl.cpp)
target_link_libraries(kata_ppl PRIVATE kata_core)

add_executable(kata_heat tools/kata_heat.cpp)
target_link_libraries(kata_heat PRIVATE kata_core)

add_executable(bench_gemv bench/bench_gemv.cpp)
target_link_libraries(bench_gemv PRIVATE kata_core)

add_executable(bench_attn bench/bench_attn.cpp)
target_link_libraries(bench_attn PRIVATE kata_core)
add_executable(bench_rope bench/bench_rope.cpp)
target_link_libraries(bench_rope PRIVATE kata_core)
add_executable(bench_norm bench/bench_norm.cpp)
target_link_libraries(bench_norm PRIVATE kata_core)

add_executable(kata_run run/kata_run.cpp)
target_link_libraries(kata_run PRIVATE kata_core)

add_executable(bench_real_weights bench/bench_real_weights.cpp)
add_executable(bench_k4 bench/bench_k4.cpp)
add_executable(peak_bench tools/peak_bench.cpp)
target_link_libraries(peak_bench PRIVATE pthread)

add_executable(bench_msweep bench/bench_msweep.cpp)
target_link_libraries(bench_msweep PRIVATE kata_core)

add_executable(bench_gemm bench/bench_gemm.cpp)
target_link_libraries(bench_real_weights PRIVATE kata_core ${CMAKE_DL_LIBS})
target_link_libraries(bench_k4 PRIVATE kata_core ${CMAKE_DL_LIBS})
target_link_libraries(bench_gemm PRIVATE kata_core)

# EXP-0046: draft acceptance and verification cost for speculative decoding.
add_executable(bench_spec bench/bench_spec.cpp)
target_link_libraries(bench_spec PRIVATE kata_core)

# Optional : compares our kernels against a local llama.cpp build.
add_executable(bench_vs_ggml bench/bench_vs_ggml.cpp)
target_link_libraries(bench_vs_ggml PRIVATE kata_core ${CMAKE_DL_LIBS})

find_package(GTest QUIET)
if(NOT GTest_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GTEST REQUIRED gtest_main)
  add_executable(kata_tests EXCLUDE_FROM_ALL)
  target_sources(kata_tests PRIVATE tests/test_quant.cpp tests/test_gemv.cpp tests/test_shortconv.cpp
    tests/test_pool.cpp tests/test_norm.cpp tests/test_rope.cpp
    tests/test_sampler.cpp tests/test_attn.cpp tests/test_gemm.cpp tests/test_prefill.cpp)
  target_include_directories(kata_tests PRIVATE ${GTEST_INCLUDE_DIRS})
  target_link_libraries(kata_tests PRIVATE kata_core ${GTEST_LIBRARIES})
else()
  enable_testing()
  add_executable(kata_tests EXCLUDE_FROM_ALL tests/test_quant.cpp tests/test_gemv.cpp tests/test_shortconv.cpp tests/test_pool.cpp tests/test_norm.cpp tests/test_rope.cpp
    tests/test_sampler.cpp tests/test_attn.cpp tests/test_gemm.cpp tests/test_prefill.cpp)
  target_link_libraries(kata_tests PRIVATE kata_core GTest::gtest_main)
endif()

if(KATA_COVERAGE)
  target_compile_options(kata_tests PRIVATE --coverage -O0 -g)
endif()

add_test(NAME unit COMMAND kata_tests)
