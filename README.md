# Bayesian Uncertainty Quantification for RANS Turbulence Models

Reynolds-averaged Navier-Stokes (RANS) models make turbulent-flow simulations
computationally practical by replacing unresolved turbulent stresses with an
approximate closure. The resulting model-form error varies across Reynolds
number, geometry, Mach number, and flow type, and is not generally represented
by uncertainty in the closure coefficients alone.

This project develops probabilistic methods for separating parameter uncertainty
from structural closure error and for evaluating predictive uncertainty under
distribution shift. A RANS observable is represented as

$$
y(x) = G(x,\theta) + \delta(x) + \varepsilon,
$$

where $G$ is the CFD forward model, $\theta$ contains turbulence-model
parameters, $\delta$ is model discrepancy, and $\varepsilon$ is observation
error. Bayesian calibration updates the parameter distribution according to

$$
p(\theta \mid D) \propto p(D \mid \theta) p(\theta).
$$

Parameter inference is combined with Gaussian-process surrogates, generalized
Bayes, Gaussian and normalizing-flow models of Reynolds-stress discrepancy, and
split-conformal calibration. Uncertainty models are evaluated both against
held-out direct numerical simulation (DNS) data and through propagation in the
CFD solvers. The numerical stack consists of C++ flow solvers developed in this
repository and a Python inference and evaluation layer.

## Research program

| Area | Scope |
|---|---|
| Attached incompressible flows | Bayesian calibration and predictive coverage across Reynolds numbers and flow types |
| Separated flows | Conditional Reynolds-stress discrepancy models and a-posteriori propagation through the RANS solver |
| Compressible heat transfer | Turbulent heat-flux uncertainty, wall-cooling effects, and transfer across Mach number |
| Shock interaction | Density-based shock-capturing RANS, wall-thermal transfer, and model-form uncertainty for shock-boundary-layer interaction |

The research program uses fixed fitting, calibration, and testing roles. Test
cases do not enter model fitting or uncertainty calibration. Reported
a-posteriori statistics include only converged solver members, and propagated
stress corrections are checked for realizability. The shock-interaction study is
ongoing; no coupled finding is stated before its registered campaign is complete.

## Repository structure

| Path | Contents |
|---|---|
| [core/](core/) | Meshes, fields, linear solvers, boundary conditions, and the SST turbulence closure |
| [incompressible/](incompressible/) | Incompressible SIMPLE RANS solver and forward models |
| [compressible/](compressible/) | Pressure-based, low-Mach compressible SIMPLE solver |
| [dbns/](dbns/) | Density-based, shock-capturing compressible RANS solver |
| [python/UQ/](python/UQ/) | Bayesian inference, surrogates, discrepancy models, conformal calibration, and evaluation |
| [UQ-RANS_research/](UQ-RANS_research/) | Curated research protocols, finding memos, machine-readable results, and selected figures |
| [tests/](tests/) | C++ and Python unit, regression, and physics tests |
| [viz/](viz/) | Reproducible analysis and figure-generation scripts |
| [scripts/](scripts/) | Build, test, and reproduction entry points |

## Evidence and reproducibility

The repository supports an ongoing working paper. Completed studies are archived
under [UQ-RANS_research/](UQ-RANS_research/) with their study-specific protocol,
fixed seeds, finding memo, numbers file, and selected figures. Bulk solver
outputs and other artifacts that can be regenerated from fixed-seed drivers are
kept out of version control.

All DNS datasets are third-party reference data rather than outputs of this
project. Raw fields are stored locally and are not redistributed. Dataset
sources, citations, and usage terms are recorded in the
[research archive](UQ-RANS_research/README.md).

## Build and test

The validated environment uses a C++17 compiler, CMake 3.14 or newer, and Python
3.11. Use the same Python interpreter for dependency installation, CMake
configuration, and testing.

```bash
# Reference macOS interpreter; replace with your Python 3.11 path.
QBTM_PYTHON=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3

"$QBTM_PYTHON" -m pip install -r requirements.txt
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON \
  -DPython_EXECUTABLE="$QBTM_PYTHON"
cmake --build build -j
ctest --test-dir build --output-on-failure
PYTHONPATH=build:python "$QBTM_PYTHON" -m pytest
```

NumPy remains pinned at 1.25.2 for compatibility with the validated GPy stack.
The normalizing-flow studies additionally require PyTorch 2.10.0. Full
reproduction runs require the local DNS datasets and may be computationally
expensive; each evidence package identifies its driver and protocol.
