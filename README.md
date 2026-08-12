# Bayesian Uncertainty Quantification for RANS Turbulence Models

Reynolds-averaged Navier-Stokes (RANS) models make turbulent-flow simulations
affordable, but they replace unknown turbulent stresses with an approximate
closure. That approximation creates systematic model error, and the error can
change when a prediction moves to a new Reynolds number, geometry, Mach number,
or flow type.

This project studies how to represent that model error and how to determine
whether the resulting uncertainty intervals can be trusted outside the
conditions used for calibration. It combines C++ CFD solvers developed in this
repository with Python methods for Bayesian inference, surrogate modeling,
generative modeling, and statistical calibration.

The research progresses from attached incompressible flows to separated flows,
high-speed heat transfer, and shock-boundary-layer interaction. Predictions are
evaluated against direct numerical simulation (DNS) data using fixed
fit/calibration/test splits, so an unseen case does not influence the method
being evaluated.

## Research questions

- When does Bayesian calibration become overconfident because parameter
  uncertainty does not capture structural model error?
- Can generalized Bayes or conformal calibration produce reliable intervals on
  flow conditions excluded from fitting?
- Can learned stress-error models transfer across flow regimes and improve a
  full CFD prediction while remaining physically valid?

## Research approach

A RANS prediction can be written as

$$
y(x) = G(x,\theta) + \delta(x) + \varepsilon,
$$

where $G$ is the CFD solver, $\theta$ contains turbulence-model parameters,
$\delta$ is systematic model error, and $\varepsilon$ is observation error.
Standard Bayesian calibration updates the parameter distribution:

$$
p(\theta \mid D) \propto p(D \mid \theta)\,p(\theta).
$$

If the likelihood treats systematic model error as ordinary noise, more data can
shrink the posterior around a biased prediction. This produces narrow intervals
that miss new flow conditions.

Gaussian-process surrogates replace expensive solver calls during inference.
Gaussian and normalizing-flow models represent the remaining stress discrepancy.
The flow model can capture correlated, non-Gaussian errors that a Gaussian model
cannot.

## Conformal calibration

Conformal calibration uses errors on reserved calibration data to choose an
interval width. For prediction $\hat y_i$ and scale $w_i$, the calibration
scores are

$$
s_i = \frac{|y_i - \hat y_i|}{w_i}, \qquad
q = Q_{1-\alpha}(s_1,\ldots,s_n).
$$

Here, $Q_{1-\alpha}$ is the empirical quantile of the reserved calibration
scores.

The resulting interval is

$$
C_{1-\alpha}(x) =
[\hat y(x) - q w(x), \quad \hat y(x) + q w(x)].
$$

The basic method uses $w=1$. The high-speed study also scales errors by the
predicted wall heat flux. Conformal calibration adjusts uncertainty width; it
does not change the mean CFD prediction.

## Current findings

Across the completed studies, standard Bayesian intervals were consistently too
narrow outside the conditions used for calibration. The degree of
overconfidence, and how fully it could be corrected, depended on the type of
distribution shift.

Coverage is the fraction of DNS reference values inside an interval. A reliable
90% interval should cover about 90% over repeated cases. The table reports
empirical coverage for nominal 90% intervals.

| Evaluation | Standard Bayes | Generalized Bayes | Split conformal |
|---|---:|---:|---:|
| Channel to unseen Couette flow, four Reynolds numbers | 5.3% to 21.1% | 42.1% to 52.6% | **89.5% to 100%** |
| Held-out channel locations, five Reynolds numbers | 0% to 20% | 60% to 80% | **80% pooled** |
| Channel transfer to an unseen Reynolds number | 9.5% to 33.3% | 81.0% to 95.2% | **90.5% to 100%** |

The clearest cross-flow result comes from calibrating on plane-channel flow and
evaluating on an unseen Couette flow type. Standard Bayes covered only 1 to 4 of
19 DNS targets per case. Conformal calibration used a separate channel case,
without fitting or calibrating on Couette data, and covered 17 to 19 targets.

Generalized Bayes widened the intervals and improved coverage but did not reach
the nominal level on the unseen flow type. On held-out locations across the
channel cases, conformal coverage improved to 80% but also remained below
nominal. These results use audited solver physics, converged runs, and separate
data for fitting, calibration, and testing.

<p align="center">
  <img src="UQ-RANS_research/step1_plane_channel/figures/coverage_in_distribution_corrected.png" alt="Coverage on held-out channel locations" width="47%">
  <img src="UQ-RANS_research/step2_couette/figures/crossflow_coverage_corrected.png" alt="Coverage on unseen Couette cases" width="47%">
</p>
<p align="center"><sub>Held-out channel locations (left) and channel-to-Couette transfer (right). The dashed line marks the nominal 90% target.</sub></p>

Full protocols and corrected numbers are in the
[channel finding](UQ-RANS_research/step1_plane_channel/channel_finding.md) and
[Couette finding](UQ-RANS_research/step2_couette/couette_finding.md).

## Additional results

### High-speed heat transfer

Across two noise settings, scaling conformal errors by predicted wall heat flux
raised held-out thermal coverage from 11.4% to 60.8% and from 10.8% to 64.8%,
without refitting the model. The result remains below 90%, showing that heat-flux
scale explains much, but not all, of the transfer error. The remaining error
changes with Mach number.

<p align="center">
  <img src="UQ-RANS_research/heatflux_modelform/figures/heatflux_conformal_scores.png" alt="Heat-flux coverage under three conformal score choices" width="68%">
</p>

See the [heat-flux finding](UQ-RANS_research/heatflux_modelform/heatflux_modelform_finding.md)
for the registered test and complete results.

### Separated flows

Conditional normalizing flows were compared with Gaussian and standard
stress-perturbation baselines. On the corrected periodic-hills runs, every
available 90% reattachment interval missed the DNS value. The corrections stayed
physically valid but were too small. The current method changes stress anisotropy
without correcting turbulence energy, so the solver's modeled energy limits how
far the mean flow can move. This is a measured method limitation, not a software
failure.

See the [separated-flow finding](UQ-RANS_research/separated_modelform/hills_crossgeom_finding.md).

### Shock interaction

The current study extends the same methods to shock and turbulent-boundary-layer
interaction. Its coupled solver campaign is ongoing, so no coupled result is
reported here. The protocol was fixed in advance in the
[shock-interaction pre-registration](UQ-RANS_research/shock_interaction/PRE_REGISTRATION.md).

## Implementation

The C++ solvers in this repository are implemented as part of the project:

- [incompressible/](incompressible/) contains the SIMPLE RANS solver used for
  channel, Couette, and separated-flow studies.
- [compressible/](compressible/) contains the low-Mach compressible SIMPLE solver.
- [dbns/](dbns/) contains the density-based, shock-capturing solver library.
- [core/](core/) contains meshes, fields, linear solvers, and the SST closure
  shared by the flow solvers.

The Python layer provides:

- Gaussian-process surrogates and Bayesian sampling;
- generalized Bayes and split-conformal calibration;
- conditional normalizing flows and Gaussian discrepancy baselines;
- fixed-seed evaluation, scoring, and figure generation.

Curated evidence lives in [UQ-RANS_research/](UQ-RANS_research/). Each completed
study includes a finding memo, machine-readable numbers, and selected figures.
Large solver caches and third-party DNS fields are not committed. Dataset
attribution and provenance are recorded in the
[research archive](UQ-RANS_research/README.md).

## Build and test

Use one Python 3.11 interpreter for installation, CMake, and tests. The project
pins NumPy 1.25.2 because newer releases are incompatible with the validated GPy
stack.

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

The normalizing-flow studies also require PyTorch 2.10.0. Reproduction drivers
are under [python/UQ/](python/UQ/) and use fixed seeds. The full studies require
the local DNS datasets and can be computationally expensive; each finding memo
names its driver and exact protocol.
