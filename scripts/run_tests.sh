#!/usr/bin/env bash
# QBTM test driver: configure, build, then run every CTest and pytest test.
#
# Usage:
#   scripts/run_tests.sh                 # configure (if needed) + build + test
#   scripts/run_tests.sh --clean         # wipe build/ first (forces full rebuild)
#   scripts/run_tests.sh --no-build      # only run tests; skip cmake/build
#   scripts/run_tests.sh --no-python     # skip pytest
#   scripts/run_tests.sh --no-cpp        # skip ctest
#   BUILD_DIR=other-build scripts/run_tests.sh
#
# Exit code is non-zero if any phase fails.  Designed for both local use and
# minimal CI (no test framework dependencies beyond cmake + python3 + pytest).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"

CLEAN=0
DO_BUILD=1
DO_CPP=1
DO_PY=1

for arg in "$@"; do
    case "$arg" in
        --clean)     CLEAN=1 ;;
        --no-build)  DO_BUILD=0 ;;
        --no-python) DO_PY=0 ;;
        --no-cpp)    DO_CPP=0 ;;
        -h|--help)
            grep -E '^# ' "$0" | sed -e 's/^# \?//'
            exit 0
            ;;
        *)
            echo "unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
red()   { printf '\033[1;31m%s\033[0m\n' "$*"; }

if [ "$CLEAN" -eq 1 ]; then
    bold "[clean] removing $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

if [ "$DO_BUILD" -eq 1 ]; then
    bold "[configure] cmake -S $REPO_ROOT -B $BUILD_DIR"
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
          -DBUILD_PYTHON_BINDINGS=ON \
          -DBUILD_TESTING=ON
    bold "[build] cmake --build $BUILD_DIR -j"
    cmake --build "$BUILD_DIR" -j
fi

FAIL=0

if [ "$DO_CPP" -eq 1 ]; then
    bold "[ctest] running C++ tests (excluding pytest target)"
    if (cd "$BUILD_DIR" && ctest -E '^pytest$' --output-on-failure); then
        green "[ctest] PASSED"
    else
        red "[ctest] FAILED"
        FAIL=1
    fi
fi

if [ "$DO_PY" -eq 1 ]; then
    bold "[pytest] running Python tests"
    export PYTHONPATH="${BUILD_DIR}:${REPO_ROOT}/python:${PYTHONPATH:-}"

    # Use the same Python interpreter CMake selected when building the
    # extension; otherwise the .so ABI tag (e.g. cpython-312) won't match the
    # interpreter (e.g. python3 -> 3.11) and the example-imports tests skip.
    # pybind11 stores Python_EXECUTABLE; some CMake configs also set
    # Python3_EXECUTABLE.  Try both, then fall back to whatever python3 is.
    PY_BIN="$(grep -E '^Python(3)?_EXECUTABLE:FILEPATH=' \
              "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null \
              | head -n1 | cut -d= -f2)"
    if [ -z "${PY_BIN:-}" ] || [ ! -x "$PY_BIN" ]; then
        PY_BIN="$(command -v python3)"
    fi
    bold "[pytest] using interpreter: $PY_BIN"
    if (cd "$REPO_ROOT" && "$PY_BIN" -m pytest -q tests/python); then
        green "[pytest] PASSED"
    else
        red "[pytest] FAILED"
        FAIL=1
    fi
fi

if [ "$FAIL" -ne 0 ]; then
    red "FAILED: at least one test phase did not pass"
    exit 1
fi
green "ALL TESTS PASSED"
