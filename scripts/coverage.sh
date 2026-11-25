#!/usr/bin/env bash
# Line coverage for kata's own sources only (system headers excluded).
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build-cov -G Ninja -DKATA_COVERAGE=ON >/dev/null
cmake --build build-cov --target kata_tests >/dev/null
find build-cov -name "*.gcda" -delete   # stale counters trip checksum errors on recompile
(cd build-cov && ./kata_tests >/dev/null)
cd build-cov
gcov -o CMakeFiles/kata_core.dir/src CMakeFiles/kata_core.dir/src/*.gcda 2>&1 | python3 ../scripts/coverage.py
