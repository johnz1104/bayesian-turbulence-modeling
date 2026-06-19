# Bayesian Turbulence Modeling

Bayesian uncertainty quantification for RANS turbulence modeling, aimed at
stochastic, non-local closures for the separated and high-speed flows where
conventional models are least reliable.

## Overview

RANS turbulence models drive the majority of production CFD, yet their closure
coefficients are empirical constants fixed decades ago against a handful of
canonical flows. On a new geometry, Reynolds-number regime, or separated flow they
carry systematic error with no indication of magnitude. The broader research
question behind this project is how to quantify, and ultimately reduce, that
closure uncertainty in a principled, data-driven way.

This research repo aims to explore that: a forward solver and a Bayesian
calibration pipeline that reframe coefficient selection as statistical inference.
Re-tuning coefficients to fit reference data returns a single point estimate, which
says nothing about how strongly the data constrains each coefficient, can overfit
measurement noise or compensating errors between coefficients, and offers no way to
propagate parametric uncertainty into predictions. Bayesian inference replaces that
point estimate with a posterior,

```
p(θ | data) ∝ p(data | θ) · p(θ)
```

combining a prior over physically reasonable coefficient ranges with the likelihood
of the data under the RANS model. The posterior, rather than a single tuned vector,
is the deliverable: marginal means and credible intervals per coefficient,
inter-parameter correlations, and an explicit read on which coefficients the data
informs (those shift and tighten relative to the prior; the rest stay near it).

## Theory

Averaging the Navier-Stokes equations leaves the Reynolds-stress tensor unclosed,
and every RANS model is a recipe for that tensor in terms of mean-flow
quantities [1]. The recipe used by SST and nearly all production closures is the
Boussinesq hypothesis: the Reynolds stress is aligned with the mean strain-rate
tensor and proportional to it through a single scalar eddy viscosity. Collapsing an
unknown symmetric tensor field onto one positive scalar is what makes linear
eddy-viscosity models cheap, robust, and ubiquitous.

That same collapse is where their accuracy is spent. A scalar eddy viscosity forces
the Reynolds-stress anisotropy to track the local strain instantaneously, so the
model is deterministic, local, and memoryless by construction. Those assumptions
hold in thin, attached, near-equilibrium shear layers and degrade together where
the turbulence leaves equilibrium: strong adverse pressure gradients, curvature,
separation, and reattachment [2], and, most severely, the shock-boundary-layer
interaction (SBLI) that organizes high-speed and hypersonic flow [3]. There the
stress anisotropy and upstream history, not the local strain, set the Reynolds
stresses, and no calibrated scalar viscosity can represent them. Generalized,
nonlinear and tensor effective-viscosity hypotheses were introduced to relax this
alignment [4], but they reintroduce coefficients whose values are uncertain and
case-dependent.

This is the motivation for the research direction. Rather than search for one more
fixed coefficient set, the aim is a closure that is:

- **Stochastic**, represented by a distribution rather than a single value, so that
  closure uncertainty is carried through the simulation instead of hidden.
  Structural and parametric uncertainty estimates are now an active and necessary
  complement to RANS models themselves [5].
- **Non-local and history-aware**, informed by upstream and surrounding state
  rather than the pointwise strain, matching the physics that breaks the Boussinesq
  assumption [4].
- **Data-driven and uncertainty-quantified by construction**, learned and
  constrained by data with model-form error represented explicitly. Data-driven
  turbulence modeling has made this tractable [6], and Bayesian inference supplies
  the formal machinery to fuse data with physical priors while quantifying what
  remains uncertain.

Bayesian calibration of the closure coefficients is the first concrete instrument
in that program. Treating the coefficients as data-constrained distributions rather
than constants turns calibration into inference and yields the parametric component
of the uncertainty directly [7]. A Kennedy-O'Hagan model-form discrepancy term [8]
then separates structural model inadequacy from parameter uncertainty, which is
essential when the underlying closure is known to be imperfect in the target regime.
The rest of this repository implements that instrument. Its compressible solver is
low-Mach; the high-speed regimes that motivate the work are the research target, not
a current capability.

## Method

A full likelihood evaluation requires a CFD solve, so direct MCMC over the
posterior is infeasible. The pipeline is surrogate-accelerated: a Gaussian-process
emulator trained on a modest ensemble of solver runs stands in for the solver
during sampling.

1. **Prior.** A truncated normal centered on Menter's (1994) SST defaults [9], with
   per-coefficient standard deviation a fixed fraction of the mean (15% by
   default). Support is truncated to physical bounds that enforce positivity and
   stability, so coefficient regions that would diverge the solver or violate
   realizability receive zero prior mass. (`priors.py`)
2. **Ensemble.** Latin Hypercube Sampling [10] draws space-filling coefficient
   vectors across the prior support; each is pushed through the C++ forward model,
   which unpacks the SST coefficients, runs the SIMPLE solver, extracts observables,
   and evaluates a Gaussian log-likelihood. Diverged or invalid solves are
   classified and filtered out. (`design.py`, `calibration.py`)
3. **Surrogate.** A Gaussian process with an ARD (Automatic Relevance
   Determination) RBF kernel [11] is fit to the surviving ensemble. Its
   per-dimension lengthscales double as a sensitivity readout: short lengthscales
   mark influential coefficients, long ones mark inert directions. (`gp_surrogate.py`,
   built on GPy [12])
4. **Sampling.** The `emcee` affine-invariant ensemble sampler [13, 14] explores the
   posterior using the surrogate for likelihood evaluations, with standard chain
   diagnostics (autocorrelation time, effective sample size, acceptance fraction).
   (`parallel_mcmc.py`)
5. **Analysis.** Posterior summaries (mean, standard deviation, credible intervals),
   shift-from-prior per coefficient, corner plots of the joint and marginal
   structure, and posterior predictive checks that re-run the actual solver at
   sampled coefficients to validate without surrogate error.
   (`inference_visualizer.py`)

An optional Kennedy-O'Hagan mode implements the model-form discrepancy term, with
diagonal and physical-GP variants. (`koh_likelihood.py`, `koh_calibration.py`)

The surrogate is only trustworthy where the sampler relies on it, and several
choices keep it so. Calibration targets a small, physically chosen subset of the 11
coefficients, so the emulated response stays low-dimensional and clear of the curse
of dimensionality; the ARD kernel down-weights directions the likelihood ignores;
the truncated prior and the sigma-clipped sampling box hold the ensemble inside the
solver-convergent region, so the surrogate interpolates rather than extrapolates;
classified failures are excluded so they cannot corrupt the fit; and the GP's
predictive variance, together with the posterior predictive checks above, keeps
surrogate error observable rather than assumed away.

## What's implemented

**C++ finite-volume solver**

- Menter SST k-ω turbulence model [9] with all 11 closure coefficients
  (σ_k1, σ_w1, β1, α1, σ_k2, σ_w2, β2, α2, β\*, a1, κ) exposed as first-class
  objects, plus runtime closure variants (full SST, no shear-stress limiter, pure
  k-ω) for model comparison.
- Incompressible SIMPLE [15] pressure-velocity coupling with Rhie-Chow
  interpolation [16] for steady RANS.
- Low-Mach compressible SIMPLE (subsonic, Ma ≲ 0.5) with an ideal-gas equation of
  state and Sutherland viscosity, reusing the same observation operator.
- Linear solvers: preconditioned conjugate gradient, BiCGSTAB, Gauss-Seidel, and an
  algebraic-multigrid (AMG) preconditioned CG for the pressure system.
- Observation operator mapping CFD fields to experimental observables with a
  Gaussian likelihood: drag, skin friction (Cf), pressure tap (Cp), velocity
  profile, separation point, and reattachment length.
- Forward model orchestration with warm-start caching, convergence classification
  (Converged, Unconverged, Diverged, InvalidParameters), and MCMC-ready
  log-likelihood output.
- Inference-parameter presets selecting which coefficients to calibrate (for
  example a1 and β\*, a near-wall four, all 11, or an arbitrary subset).
- CLI validation cases: turbulent channel flow (against the Dean correlation [17])
  and flat-plate boundary layer (against the Schoenherr correlation [18]).

**Python inference layer**

- pybind11 bindings (`rans_sst_py`) [19] exposing the incompressible and
  compressible forward models, the parameter sets, the observation operator, and
  parameter-sensitivity gradients.
- Truncated-normal prior, LHS ensemble driver, GPy ARD-RBF surrogate (scalar and
  multi-output), `emcee` sampling (serial and `multiprocess`-parallel), and the
  Kennedy-O'Hagan model-form discrepancy mode.
- Posterior visualization (chain traces, corner plots, ARD-lengthscale and
  surrogate-quality diagnostics), plus optional PyVista 3D flow-field rendering.
- Runnable end-to-end demonstrations under `python/examples/` (channel,
  backward-facing step, and Kennedy-O'Hagan calibration).

## Repository contents

A fresh clone contains the program only: the solver, the inference layer, and the
build configuration.

```
core/            physics-agnostic infrastructure: Mesh, Field, LinearSolver,
                 SSTModel, ObservationOperator, InferenceParameters
incompressible/  incompressible SIMPLE solver, ForwardModel, ParameterSensitivity
compressible/    low-Mach compressible SIMPLE solver and forward model
python/          inference layer (prior, LHS, GP surrogate, emcee, KOH), the
                 pybind11 bindings (Bindings.cpp), and runnable examples
include/, src/   an earlier flat header and source layout, retained alongside the
                 layered modules above
CMakeLists.txt   build for the static libraries, both CLIs, and the Python module
```

Local result and output trees, working notes, scratch caches, and build artifacts
are intentionally kept out of version control.

## Build & run

**Prerequisites**: a C++17 compiler (GCC 9+, Clang 10+, or Apple Clang),
CMake 3.14 or newer, Git (CMake fetches pybind11 automatically), and Python 3.8+
with `numpy`, `scipy`, `matplotlib`, `emcee`, `GPy`, and `multiprocess`. PyVista is
optional, for 3D flow rendering.

**Build** the static libraries, both CLIs, and the Python module:

```bash
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON
cmake --build build -j
```

**Validate the solver** from the build directory:

```bash
./build/rans_sst --validate-channel   # channel flow vs. Dean correlation
./build/rans_sst --validate-plate     # flat plate vs. Schoenherr correlation
./build/rans_sst --demo               # forward-model demo
./build/rans_sst --all                # all of the above
```

**Run the inference pipeline** (the bindings build into `build/`, the Python layer
lives in `python/`):

```bash
PYTHONPATH=build:python python3 python/examples/channel_flow_example.py
```

## References

1. S. B. Pope, Turbulent Flows. Cambridge University Press, 2000.
2. P. R. Spalart, "Strategies for turbulence modelling and simulations,"
   International Journal of Heat and Fluid Flow, vol. 21, no. 3, pp. 252-263, 2000.
3. C. J. Roy and F. G. Blottner, "Review and assessment of turbulence models for
   hypersonic flows," Progress in Aerospace Sciences, vol. 42, no. 7-8,
   pp. 469-530, 2006.
4. S. B. Pope, "A more general effective-viscosity hypothesis," Journal of Fluid
   Mechanics, vol. 72, no. 2, pp. 331-340, 1975.
5. H. Xiao and P. Cinnella, "Quantification of model uncertainty in RANS
   simulations: A review," Progress in Aerospace Sciences, vol. 108, pp. 1-31, 2019.
6. K. Duraisamy, G. Iaccarino, and H. Xiao, "Turbulence modeling in the age of
   data," Annual Review of Fluid Mechanics, vol. 51, pp. 357-377, 2019.
7. W. N. Edeling, P. Cinnella, R. P. Dwight, and H. Bijl, "Bayesian estimates of
   parameter variability in the k-ε turbulence model," Journal of Computational
   Physics, vol. 258, pp. 73-94, 2014.
8. M. C. Kennedy and A. O'Hagan, "Bayesian calibration of computer models," Journal
   of the Royal Statistical Society: Series B, vol. 63, no. 3, pp. 425-464, 2001.
9. F. R. Menter, "Two-equation eddy-viscosity turbulence models for engineering
   applications," AIAA Journal, vol. 32, no. 8, pp. 1598-1605, 1994.
10. M. D. McKay, R. J. Beckman, and W. J. Conover, "A comparison of three methods for
    selecting values of input variables in the analysis of output from a computer
    code," Technometrics, vol. 21, no. 2, pp. 239-245, 1979.
11. C. E. Rasmussen and C. K. I. Williams, Gaussian Processes for Machine Learning.
    MIT Press, 2006.
12. GPy, "GPy: A Gaussian process framework in Python," since 2012.
    https://github.com/SheffieldML/GPy
13. J. Goodman and J. Weare, "Ensemble samplers with affine invariance,"
    Communications in Applied Mathematics and Computational Science, vol. 5, no. 1,
    pp. 65-80, 2010.
14. D. Foreman-Mackey, D. W. Hogg, D. Lang, and J. Goodman, "emcee: The MCMC
    Hammer," Publications of the Astronomical Society of the Pacific, vol. 125,
    no. 925, pp. 306-312, 2013.
15. S. V. Patankar and D. B. Spalding, "A calculation procedure for heat, mass and
    momentum transfer in three-dimensional parabolic flows," International Journal of
    Heat and Mass Transfer, vol. 15, no. 10, pp. 1787-1806, 1972.
16. C. M. Rhie and W. L. Chow, "Numerical study of the turbulent flow past an airfoil
    with trailing edge separation," AIAA Journal, vol. 21, no. 11, pp. 1525-1532,
    1983.
17. R. B. Dean, "Reynolds number dependence of skin friction and other bulk flow
    variables in two-dimensional rectangular duct flow," Journal of Fluids
    Engineering, vol. 100, no. 2, pp. 215-223, 1978.
18. K. E. Schoenherr, "Resistance of flat surfaces moving through a fluid,"
    Transactions of the Society of Naval Architects and Marine Engineers, vol. 40,
    pp. 279-313, 1932.
19. W. Jakob, J. Rhinelander, and D. Moldovan, "pybind11: Seamless operability
    between C++11 and Python," 2017. https://github.com/pybind/pybind11
