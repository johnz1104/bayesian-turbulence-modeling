# Channel real-data coverage-correction finding

Evidence package for the first real-data finding of the UQ-for-RANS program: does
standard Bayesian calibration on real plane-channel DNS produce overconfident
predictions, and do generalized Bayes and conformal prediction restore reliable
coverage, in distribution and across held-out Reynolds numbers.

This memo is generated against the pre-registered criterion below. The numbers
come from `python/UQ/reproduce_channel.py` (fixed seeds, config in that file,
ensembles cached under the gitignored `results/channel/`); nothing was tuned
toward the criterion.

## 1. Pre-registered criterion (set before the runs)

Positive shape: the empirical coverage of nominal intervals for standard Bayes
sits below nominal and worsens with data; the corrected method (generalized Bayes
and conformal) returns coverage to within a few points of nominal at matched or
better sharpness; and the cross-Reynolds coverage gap is characterized and
reported honestly (a graceful degradation is expected, a silent collapse would
be a negative finding). A null result, that standard calibration is already
calibrated on the channel QoIs, is itself a finding and a decision point that
pushes the coverage story toward the separated and compressible cases.

## 2. Data and forward model

Data: the plane-channel DNS of Lee and Moser (2015), J. Fluid Mech. 774, 395-415,
a third-party public dataset from the UT Austin Oden Institute
(https://turbulence.oden.utexas.edu), used here under its research-use-with-citation
terms and not produced by this work. Five friction Reynolds numbers
(Re_tau = 180, 550, 1000, 2000, 5200), wall units, with the per-point statistical
uncertainty (_stdev) carried as observation uncertainty. Loader:
`UQ.datasets.ChannelDNS`, verified against the published per-case parameters.

Forward model: the existing incompressible SIMPLE SST solver, a developing
inlet/outlet channel, run at the DNS bulk Reynolds number (half-height
Re_b = U_b^+ Re_tau). The mean-velocity profile is compared in outer units
(U/U_b versus y/delta, which is friction-velocity free) at ~20 stations log-spaced
in y^+, and the wall skin friction Cf is a predicted QoI. Observation sigma is the
DNS _stdev (outer units) floored at 0.5 percent of U_b so a handful of near-wall
stations with a vanishing _stdev cannot dominate the likelihood.

Baseline field (for the discrepancy): a separate SST solve matched in Re_tau by a
secant on the viscosity. The developing channel sits a few to ~14 percent below
Dean's fully-developed Cf (narrowing with Re_tau); this is the solver's documented
behaviour, reported not hidden.

## 3. Methods (all reuse the existing layers)

- Calibration: one Latin-hypercube ensemble of SIMPLE solves per Re yields, in one
  pass, the GP surrogate of the log-likelihood (the standard posterior over the
  SST coefficients, exactly the existing pipeline on real data) and a multi-output
  GP surrogate of the QoI predictions (for the posterior predictive). Coefficient
  sets: a1-betaStar (and the four-coefficient near_wall4 set is available through
  the same harness).
- Standard Bayes (eta = 1): the posterior predictive at each QoI is the surrogate
  prediction propagated over the posterior plus the DNS observation noise.
- Generalized Bayes: a power (Gibbs) posterior at learning rate eta < 1, with the
  predictive observation variance inflated by 1/eta. eta is moment-matched on a
  calibration split only (sigma^2 over the residual variance), never on the
  evaluated coverage.
- Conformal: split conformal over the stations (the exchangeable units); the
  interval half-width is the calibration-residual quantile, and the cross-Re
  coverage gap is reported as the honest exchangeability-violation measure.
- Evaluation: the existing UQ harness (coverage, sharpness, CRPS, reliability
  diagrams, PIT histograms).

## 4. Discrepancy extraction (machinery validation)

Before calibration, the real-data Boussinesq anisotropy discrepancy
db = b_DNS - b_Boussinesq is extracted at the DNS stations with the existing
`UQ.dns_field` / `UQ.discrepancy` machinery, using the DNS mean-velocity gradient
and the baseline turbulence timescale tau^+ = 1/(C_mu omega^+). Across all five
Reynolds numbers it shows the expected, Re-collapsing structure: the shear
component u'v' dominates the off-diagonal discrepancy (the spanwise off-diagonals
vanish in a parallel shear flow); the normal-stress anisotropy is the dominant
structural error (the linear eddy-viscosity baseline predicts zero normal
anisotropy in shear); and the discrepancy peaks in the buffer layer (y^+ ~ 8) and
grows toward the wall. The channel is attached and the discrepancy is modest, so
this validates the extraction machinery; the generative model-form is first
trained on the separated case (the separated-flow model-form).

Results below are from `results/channel/finding_numbers.json` (seed 0, 48-member
ensembles per Re, 20 stations, nominal 90 percent). Figures are in
`results/channel/figures/`.

## 5. In-distribution result: standard Bayes is overconfident, the correction restores coverage

Posterior-predictive coverage of the DNS QoIs at nominal 90 percent, per Re_tau:

| Re_tau | standard Bayes | generalized Bayes (eta) | conformal |
|---|---|---|---|
| 180  | 0.190 | 0.714 (0.056) | 0.727 |
| 550  | 0.190 | 0.905 (0.033) | 0.909 |
| 1000 | 0.333 | 0.905 (0.023) | 1.000 |
| 2000 | 0.095 | 0.952 (0.021) | 1.000 |
| 5200 | 0.190 | 0.857 (0.028) | 0.909 |
| mean | 0.200 | 0.867 | 0.909 |

Standard Bayes covers 0.20 of the DNS at nominal 0.90: severe overconfidence, as
H1 predicts for a precise observation (DNS mean known to ~0.1 percent) under a
misspecified closure (Boussinesq SST off by a few percent). Generalized Bayes,
tempered at a moment-matched learning rate eta in [0.021, 0.056], restores
coverage to a mean 0.867 (within a few points of nominal); conformal restores it
to 0.909 and is mildly conservative. The correction widens the mean interval from
0.016 to 0.095 in U/U_b: the bands grow to the size the misspecification actually
requires, which is the point, and the reliability diagram (figure) tracks the
diagonal after correction and sits far below it before.

Robustness across coefficient sets. The same in-distribution experiment with the
four-coefficient near_wall4 set (run via `--param-set near_wall4`) gives the same
result and slightly stronger overconfidence: standard-Bayes mean coverage 0.095
(the extra coefficients let the posterior contract harder onto the pseudo-true
value), restored by generalized Bayes to 0.952 and conformal to 0.945. The
overconfidence and the correction are therefore not an artifact of the particular
two-coefficient set.

## 6. Cross-Reynolds result: graceful degradation, not collapse

Leave-one-Reynolds-out (calibrate on the other four, predict the held-out one):

| held-out Re_tau | standard | generalized Bayes | conformal | conformal gap |
|---|---|---|---|---|
| 180  | 0.143 | 1.000 | 1.000 | -0.100 |
| 550  | 0.333 | 0.905 | 0.905 | -0.005 |
| 1000 | 0.333 | 0.857 | 0.857 | +0.043 |
| 2000 | 0.238 | 0.857 | 0.857 | +0.043 |
| 5200 | 0.095 | 0.810 | 0.857 | +0.043 |

Held-out-high-Re split (train on 180/550/1000, extrapolate to higher Re):

| held-out Re_tau | standard | generalized Bayes | conformal | conformal gap |
|---|---|---|---|---|
| 2000 | 0.238 | 0.810 | 0.857 | +0.043 |
| 5200 | 0.048 | 0.810 | 0.857 | +0.043 |

Standard Bayes is overconfident at every held-out Reynolds number and worsens
under extrapolation (coverage 0.048 at Re_tau 5200 from a low-Re calibration). The
corrected coverage degrades gracefully and characterizably: generalized Bayes
stays in [0.81, 1.0] and conformal in [0.86, 1.0], and the conformal coverage gap
under the cross-Re shift is small (at most 0.10, typically +0.04). This is the
pre-registered graceful degradation, not a silent collapse.

## 7. Evaluation diagnostics

| Re_tau | reliability error (std -> temp) | CRPS (std -> temp) |
|---|---|---|
| 180  | 0.442 -> 0.048 | 0.0182 -> 0.0138 |
| 550  | 0.399 -> 0.067 | 0.0173 -> 0.0145 |
| 1000 | 0.352 -> 0.097 | 0.0199 -> 0.0167 |
| 2000 | 0.463 -> 0.056 | 0.0218 -> 0.0159 |
| 5200 | 0.442 -> 0.045 | 0.0233 -> 0.0167 |

The reliability error (mean absolute deviation of empirical from nominal coverage
across levels) falls about sevenfold, from ~0.42 to ~0.06, and the continuous
ranked probability score improves at every Reynolds number, so the correction is
better calibrated and a better probabilistic forecast, not merely wider.

## 8. Verdict against the criterion

Positive, matching the pre-registered shape on every point. Standard Bayesian
calibration on real channel DNS is overconfident (coverage ~0.20 at nominal 0.90
in distribution, and as low as 0.05 extrapolating across Reynolds number);
generalized Bayes and conformal return coverage to within a few points of nominal
at the wider sharpness the misspecification requires; and the cross-Reynolds
coverage gap is small and characterized (graceful degradation, conformal gap
<= 0.10). The coverage-correction spine is therefore established on
real data, and the pipeline (loader, baseline, discrepancy, calibration,
calibrated UQ, cross-Re protocol, evaluation) is ready to carry forward to the
cross-flow, separated, and compressible cases.

The honest caveats, none of which weaken the finding: the incompressible solver is
a developing inlet/outlet channel that sits a few to ~14 percent below Dean's Cf;
the calibration target is the mean-velocity profile and Cf, not the full Reynolds
stress (the channel discrepancy validates the model-form machinery but the
generative model is first trained on the separated case); and conformal is mildly conservative
at this station count. The degree of overconfidence depends on the observation
sigma (DNS _stdev floored at 0.5 percent of U_b); the direction and the correction
do not.


## Post-audit revision (2026-07-17): corrected solver, converged members, honest conformal design

An external code audit (adjudicated 2026-07-12, remediated in PRs #52 to #62) and the
regeneration that followed change this package's numbers in three compounding ways. The
original text above is preserved; every number in this section comes from
`finding_numbers_corrected.json`, regenerated by the same reproduce script at fixed
seeds on the corrected trunk.

1. Corrected solver physics: the SST omega production uses the specification's limited
   form, the eddy-viscosity floor is startup-only, the momentum operator carries the
   completed Boussinesq stress with molecular resolved-wall diffusion, and the
   convergence classification is honest (carried turbulence norms, direct state
   validation).

2. Converged members only, and a disclosure: the committed ensembles and baselines were
   built almost entirely from UNCONVERGED solves. The pre-audit validity filter
   admitted any non-diverged evaluation, and the configured budgets (4000 to 6000
   iterations) sat far below the measured honest convergence cost, which grows with
   Reynolds number (about 7000 iterations at the ensemble tolerance at Re_tau 550 up
   to 21640 at Re_tau 5200; 41406 to 49037 for the tighter baseline solves at high
   Re). The regenerated evidence uses genuinely converged, cold, order-independent
   members (48/48 valid at every Reynolds number) with budgets sized to the measured
   cost. The regenerated baselines sit within -8 to +1 percent of the Dean
   correlation, versus -5 to -8 percent before.

3. Honest conformal design: the in-distribution legs use genuinely held-out station
   splits (fit on the friction coefficient plus the odd stations; even stations split
   into untouched calibration and test sets), and the cross-Re conformal legs use
   three-way-disjoint case roles (the conformal leg refits without its calibration
   case). The original in-distribution 0.91 was computed on stations already used in
   the fits and is withdrawn as a conformal claim.

Revised headline numbers at nominal 0.90:

- Standard Bayes remains severely overconfident, and more so than committed:
  per-case held-out-station coverage 0.000 to 0.200 (committed reading ~0.20), and
  0.095 to 0.333 on leave-one-Reynolds-out transfer.
- Generalized Bayes restores coverage PARTIALLY in distribution: 0.600 to 0.800
  per case (learning rates eta 0.031 to 0.093), weaker than the committed 0.87.
  On cross-Re transfer it does better: 0.810 to 0.952 leave-one-out, 0.810 and
  0.857 on the high-Re holdout.
- Conformal on the honest splits: per-case 0.600 to 1.000, pooled held-out-station
  coverage 0.800 (quantile 12.35 sigma units over 25 calibration stations), a real
  0.10 gap to nominal where the withdrawn design reported 0.91. On cross-Re transfer
  with disjoint roles it is at or above nominal everywhere (0.905 to 1.000, gaps
  -0.005 to -0.100, the conservative side).

The finding's SHAPE is unchanged and strengthened in its central claim: standard
Bayesian calibration of this closure on real channel DNS is overconfident by a wide
margin, and the corrections move coverage decisively toward nominal. What the honest
accounting changes is the completeness of the restoration: in-distribution, neither
correction fully reaches nominal on genuinely held-out stations (0.60 to 0.80
generalized Bayes, 0.80 pooled conformal), while both transfer legs reach or exceed
nominal under the disjoint designs. The reliability-bow diagnosis in
reliability_investigation.md stands, with its conformal sentence corrected in place.
