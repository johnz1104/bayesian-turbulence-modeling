#!/usr/bin/env bash
# QBTM reproduce-all driver.
#
# Runs the full pipeline from build through every example script in logical
# phase order.  Use --quick to activate each script's fast mode (coarser
# meshes, smaller MCMC budgets).  All outputs land in results/.
#
# Usage:
#   scripts/reproduce_all.sh             # full run (may take 30+ minutes)
#   scripts/reproduce_all.sh --quick     # fast mode (~5 minutes)
#   scripts/reproduce_all.sh --no-build  # skip cmake/make, run examples only
#   BUILD_DIR=other-build scripts/reproduce_all.sh
#
# Exit code is non-zero if any step fails.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
RESULTS_DIR="${REPO_ROOT}/results"

QUICK=0
DO_BUILD=1

for arg in "$@"; do
    case "$arg" in
        --quick)    QUICK=1 ;;
        --no-build) DO_BUILD=0 ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

QUICK_FLAG=""
[ "$QUICK" -eq 1 ] && QUICK_FLAG="--quick"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
red()   { printf '\033[1;31m%s\033[0m\n' "$*"; }
step()  { bold ""; bold "=== $* ==="; }

# Resolve the Python interpreter CMake used so the ABI tag matches.
_py_bin() {
    local p
    p="$(grep -E '^Python(3)?_EXECUTABLE:FILEPATH=' \
         "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null \
         | head -n1 | cut -d= -f2)"
    if [ -z "${p:-}" ] || [ ! -x "$p" ]; then
        p="$(command -v python3)"
    fi
    echo "$p"
}

# ---------------------------------------------------------------------------
# 0. Build
# ---------------------------------------------------------------------------
if [ "$DO_BUILD" -eq 1 ]; then
    step "0/13  Configure and build"
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
          -DBUILD_PYTHON_BINDINGS=ON \
          -DBUILD_TESTING=ON
    cmake --build "$BUILD_DIR" -j
    green "Build OK"
fi

export PYTHONPATH="${BUILD_DIR}:${REPO_ROOT}/python:${PYTHONPATH:-}"
PY="$(_py_bin)"
bold "Python interpreter: $PY"
mkdir -p "$RESULTS_DIR"

FAIL=0

run_example() {
    # run_example <step_label> <script_relpath> [extra_args...]
    local label="$1"; shift
    local script="$1"; shift
    step "$label  $script"
    if "$PY" -u "${REPO_ROOT}/python/examples/${script}" "$@"; then
        green "$label OK"
    else
        red "$label FAILED"
        FAIL=1
    fi
}

# ---------------------------------------------------------------------------
# 1. Incompressible channel calibration (baseline)
# ---------------------------------------------------------------------------
run_example "1/13" "channel_flow_example.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/channel_flow"

# ---------------------------------------------------------------------------
# 2. BFS synthetic calibration (standard Gaussian likelihood)
# ---------------------------------------------------------------------------
run_example "2/13" "bfs_example.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/bfs_example"

# ---------------------------------------------------------------------------
# 3. KOH model-form uncertainty demo
# ---------------------------------------------------------------------------
run_example "3/13" "koh_example.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/koh_example"

# ---------------------------------------------------------------------------
# 4. Compressible channel flow
# ---------------------------------------------------------------------------
run_example "4/13" "compressible_channel.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/compressible_channel"

# ---------------------------------------------------------------------------
# 5. Compressible Bayesian calibration
# ---------------------------------------------------------------------------
run_example "5/13" "compressible_bayesian_calibration.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/compressible_calibration"

# ---------------------------------------------------------------------------
# 6. Wall functions demo
# ---------------------------------------------------------------------------
run_example "6/13" "wall_functions_demo.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/wall_functions"

# ---------------------------------------------------------------------------
# 7. Active learning BFS
# ---------------------------------------------------------------------------
run_example "7/13" "active_learning_bfs.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/active_learning"

# ---------------------------------------------------------------------------
# 8. Multi-case joint calibration
# ---------------------------------------------------------------------------
run_example "8/13" "multi_case_calibration_demo.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/multi_case_calibration_demo"

# ---------------------------------------------------------------------------
# 9. Scramjet KOH calibration demo
# ---------------------------------------------------------------------------
run_example "9/13" "scramjet_calibration_demo.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/scramjet_calibration"

# ---------------------------------------------------------------------------
# 10. Multi-fidelity calibration study
# ---------------------------------------------------------------------------
run_example "10/13" "multi_fidelity_calibration_demo.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/multi_fidelity_calibration"

# ---------------------------------------------------------------------------
# 11. Sensitivity analysis
# ---------------------------------------------------------------------------
run_example "11/13" "sensitivity_analysis_demo.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/sensitivity_analysis"

# ---------------------------------------------------------------------------
# 12. Active learning: scramjet adaptive ensemble
# ---------------------------------------------------------------------------
run_example "12/13" "active_learning_scramjet.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/active_learning_scramjet"

# ---------------------------------------------------------------------------
# 13. Adjoint-readiness diagnostic
# ---------------------------------------------------------------------------
run_example "13/17" "adjoint_readiness_demo.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/adjoint_readiness"

# ---------------------------------------------------------------------------
# 14-17. Infrastructure dry-run examples
# ---------------------------------------------------------------------------
run_example "14/17" "precomputed_high_mach_ensemble_demo.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/precomputed_ensemble_demo"

run_example "15/17" "high_mach_data_pathway_dry_run.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/high_mach_data_pathway_dry_run"

run_example "16/17" "observable_ablation_demo.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/observable_ablation"

run_example "17/17" "external_cfd_queue_dry_run.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/external_cfd_queue_dry_run"

# ===========================================================================
# High-dimensional calibration studies
# High-dimensional calibration: surrogate breakdown, gradients, model evidence,
# identifiability, and discrepancy.  Studies are
# checkpointed; --quick activates each driver's fast mode.
# ===========================================================================
bold ""
bold "===== High-dimensional calibration studies ====="

# Surrogate breakdown vs. dimensionality
run_example "surrogate scaling/breakdown" "surrogate_scaling_study.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/scaling_study"

# Model evidence + Bayes factors, term-toggled closures
run_example "model evidence (M_full/M_nolim/M_kw)" "model_evidence_study.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/model_evidence"

# High-D identifiability: active subspace + ARD + posterior cov
run_example "identifiability (3-way)" "identifiability_study.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/identifiability"

# Irreducible discrepancy δ(x) along the wall
run_example "discrepancy δ(x)" "discrepancy_study.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/discrepancy"

# True-model gradient engines: warm-FD (default,
#     == cold full FD, ~13x cheaper) vs frozen-pressure tangent; + active-subspace Gram matrix
#     + KOH ∂L/∂θ composition.  The held (∂R/∂U)ᵀ adjoint core stays parked.
run_example "gradient engine (warm-FD)" "gradient_engine_demo.py" \
    ${QUICK_FLAG} -o "${RESULTS_DIR}/gradient_engine"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
bold ""
if [ "$FAIL" -ne 0 ]; then
    red "REPRODUCE-ALL: at least one example failed — see output above"
    exit 1
fi
green "REPRODUCE-ALL: all examples completed successfully"
echo "Results written to: $RESULTS_DIR"
