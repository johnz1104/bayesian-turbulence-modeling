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
#   PYTHON=python3.11 scripts/run_tests.sh   # pin the build+test interpreter
#
# Exit code is non-zero if any phase fails.  Designed for both local use and
# minimal CI (no test framework dependencies beyond cmake + python3 + pytest).
#
# ONE interpreter is used for both the binding build and pytest (default
# python3, override with PYTHON=...), so the compiled rans_sst_py .so ABI tag
# (cpythonXYZ) always matches the interpreter that imports it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"

# Resolve the single pinned interpreter (absolute path) used for build + test.
PYTHON="${PYTHON:-python3}"
PY_BIN="$(command -v "$PYTHON" || true)"
if [ -z "$PY_BIN" ]; then
    echo "error: interpreter '$PYTHON' not found on PATH" >&2
    exit 2
fi

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
    # If a prior configure pinned a different interpreter, the cached binding ABI
    # won't match $PY_BIN; wipe so the rebuilt .so matches the test interpreter.
    # Guard the cache read: on a fresh checkout there is no CMakeCache.txt, and a
    # failing grep under `set -euo pipefail` would otherwise abort the script.
    CACHED_PY=""
    if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
        CACHED_PY="$(grep -E '^Python_EXECUTABLE:FILEPATH=' \
                     "${BUILD_DIR}/CMakeCache.txt" | head -n1 | cut -d= -f2 || true)"
    fi
    if [ -n "${CACHED_PY:-}" ] && [ "$CACHED_PY" != "$PY_BIN" ]; then
        bold "[configure] interpreter changed (${CACHED_PY} -> ${PY_BIN}); wiping ${BUILD_DIR}"
        rm -rf "$BUILD_DIR"
    fi
    bold "[configure] cmake -S $REPO_ROOT -B $BUILD_DIR (Python=$PY_BIN)"
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
          -DBUILD_PYTHON_BINDINGS=ON \
          -DBUILD_TESTING=ON \
          -DPython_EXECUTABLE="$PY_BIN" \
          -DPython3_EXECUTABLE="$PY_BIN"
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

    # Same pinned interpreter as the build above, so the .so ABI tag matches.
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
