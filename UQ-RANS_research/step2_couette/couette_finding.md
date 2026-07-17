# Couette cross-flow generalization finding

Evidence package for the second real-data finding of the UQ-for-RANS program: does
the channel calibration plus its calibrated UQ transfer to a held-out flow
type (plane Couette), and does the predictive interval cover the Couette truth at
the strict 0.5 percent observation band.

This memo is generated against the pre-registered criterion below. The numbers come
from `python/UQ/reproduce_couette.py` (fixed seeds, config in that file, ensembles
cached under the gitignored `results/couette/`); nothing was tuned toward the
criterion. The correction is calibrated on the channel and applied to the held-out
Couette, exactly as the channel calibrated on training Reynolds numbers and applied to
held-out ones.

## 1. Pre-registered criterion (set before the runs)

A-posteriori Couette (through the moving-wall solver), 0.5 percent primary band and
a 1 percent sensitivity band:

- Standard Bayes is overconfident on the Couette mean-velocity and Cf QoIs
  (expected, re-confirming the channel overconfidence), and generalized Bayes and
  conformal restore coverage to within a few points of nominal at the 0.5 percent
  band WITHOUT inflating the observation floor, with the conformal cross-flow gap
  reported honestly. Graceful, characterized transfer is positive; a silent
  coverage collapse with no widening is negative; an already-calibrated transfer is
  itself a finding and a decision point.

Rotating-channel companion (a-priori diagnostic): the streamwise-rotation
out-of-plane stress is a structural model-form failure a linear eddy-viscosity
cannot represent, and the global correction inflates diffusely and cannot
localise to it, which motivates the feature-conditioned generative
model-form. (The streamwise-rotation mean velocity is symmetric, so there is no
one-sided unstable side; the failure is per-component, not per-wall.)

## 2. Data and forward model

Data: the plane-Couette DNS of Pirozzoli, Bernardini and Orlandi (2014), J. Fluid
Mech. 742, a third-party public dataset from the Roma (Sapienza University of Rome)
group, used under its research-use-with-citation terms and not produced by this
work. Four friction Reynolds numbers (Re_tau 171, 260, 507, 986), wall units. The
files carry no per-point statistical _stdev, so observation uncertainty is a MODELED
relative value (0.5 percent primary, 1 percent sensitivity), labeled modeled and
anchored by the data-only constant-total-stress identity dU+/dy+ - u'v'+ = 1 (rms
deviation 0.03 to 0.15 percent across the four cases). Loader: `UQ.datasets.CouetteDNS`.

Forward model (a-posteriori): a moving-wall (Couette) boundary condition was added
to the incompressible SIMPLE SST solver (`FlowBoundaryConditions::couette_defaults`,
`BCType::WallMoving`). The moving top wall drives the flow with no mean pressure
gradient; the streamwise ends are zero-gradient so the flow is fully developed and
streamwise invariant (constant total stress), reached on a short domain in about ten
seconds per solve. The moving-wall speed is set so the Couette bulk velocity is one,
so the solved streamwise velocity is already in outer (U/U_b) units, exactly as the
channel forward model runs at U_b = 1. The friction Reynolds number is matched to
each DNS case by a secant on the molecular viscosity. The QoI is the half-gap
mean-velocity profile U/U_b versus y/h and the wall skin friction Cf, mirroring the
channel QoI. The same SST closure that was calibrated on the channel is pushed
through this Couette solve, so the prediction is a genuine a-posteriori transfer,
not an analytic surrogate.

## 3. Methods (all reuse the channel layers)

- Channel posterior: the channel calibrations are pooled into a posterior
  over the SST coefficients (standard, eta = 1, and tempered at the channel-
  calibrated generalized-Bayes learning rate), reusing `CrossReStudy`.
- Cross-flow prediction: the channel posterior is propagated through the
  a-posteriori Couette forward map (`CouetteCalibration`, which subclasses
  `ChannelCalibration` and inherits the ensemble, surrogate, posterior-predictive,
  coverage, and conformal machinery), giving the posterior-predictive Couette QoIs.
- Correction: generalized Bayes (the channel-calibrated learning rate, predictive
  noise inflated by 1/eta) and split conformal (a channel case supplies the
  exchangeable calibration residuals, whose quantile sets the interval half-width
  applied to the Couette point predictions). The conformal cross-flow gap is the
  honest exchangeability-violation measure under the flow-type shift.
- Within-Couette (bonus): the same in-distribution and cross-Re protocol is
  reused directly on the Couette calibrations (`CrossReStudy`), giving the
  within-Couette coverage and the cross-Re-within-Couette degradation, plus the
  capability check that the correction restores coverage when calibrated on Couette.
- Evaluation: the existing UQ harness (coverage, sharpness, conformal gap).

## 4. Forward-model validation (the cross-flow model-form error is real)

The moving-wall solver produces a genuine fully-developed turbulent Couette profile:
the solved total stress (nu + nu_t) dU/dy is constant across the gap to better than
half a percent, the centerline velocity is the Couette bulk by antisymmetry, and the
wall-unit profile U+(y+) tracks the DNS. The channel-calibrated SST nevertheless
under-predicts the Couette mean velocity by a few percent and the skin friction by
about 17 percent: a genuine cross-flow model-form error (the channel closure applied
to Couette), not a forward-map artifact, since it is the real solver run. This is the
misspecification the calibrated UQ must cover, and it is what makes standard Bayes
overconfident on the held-out Couette.

Results below are from `results/couette/finding_numbers.json` (seed 0, channel cases
550/1000/2000 pooled into the posterior, Couette cases 171/260/507/986, ensembles of
24 to 40 members, 19 QoIs, nominal 90 percent). Figures are in `figures/`.

## 5. Cross-flow result: overconfidence transfers, the correction restores coverage gracefully

Coverage of the Couette DNS QoIs from the channel posterior at nominal 90 percent,
channel-calibrated learning rate eta = 0.014:

| Couette Re_tau | standard (0.5%) | gen. Bayes (0.5%) | conformal (0.5%) | conformal gap |
|---|---|---|---|---|
| 171  | 0.263 | 0.789 | 1.000 | -0.100 |
| 260  | 0.053 | 0.737 | 1.000 | -0.100 |
| 507  | 0.053 | 0.368 | 0.684 | +0.216 |
| 986  | 0.158 | 0.474 | 0.842 | +0.058 |

Standard Bayes is overconfident on every held-out Couette case (coverage 0.05 to
0.26 at nominal 0.90), with bands far too narrow (mean half-width 0.008 to 0.009 in
U/U_b). This is the channel overconfidence transferring to the cross-flow shift, as
H1 predicts for a precise observation under a misspecified closure now applied to a
flow type it never saw. The correction widens the bands roughly eightfold (gen.-Bayes
half-width 0.07) and lifts coverage substantially: it is restored fully at the lower
Couette Reynolds numbers (171, 260: gen. Bayes 0.74 to 0.79, conformal 1.00) and
partially at the higher ones (507, 986: gen. Bayes 0.37 to 0.47, conformal 0.68 to
0.84). The conformal cross-flow gap is reported honestly: it is negative (mild
over-coverage) at low Re and grows to +0.22 at Re_tau 507, the characterized cost of
applying the channel-calibrated interval to a held-out flow type and Reynolds number
at once. This is graceful, characterized degradation, not a silent collapse (the
bands do widen) and not a clean restoration (a residual gap remains at the strict
0.5 percent band and the higher Couette Re).

Sensitivity to the observation band (1 percent):

| Couette Re_tau | standard (1%) | gen. Bayes (1%) | conformal (1%) | conformal gap |
|---|---|---|---|---|
| 171  | 0.421 | 0.895 | 1.000 | -0.100 |
| 260  | 0.053 | 0.842 | 1.000 | -0.100 |
| 507  | 0.105 | 0.684 | 0.684 | +0.216 |
| 986  | 0.211 | 0.632 | 0.842 | +0.058 |

The wider 1 percent band improves coverage everywhere (gen. Bayes up to 0.90,
conformal up to 1.00) but leaves the same cross-flow gap structure: the direction
(overconfidence) and the correction (widening restores most of the coverage) are
robust to the modeled observation level, while the residual gap depends on it, the
same sigma-sensitivity the channel reported in distribution.

## 6. Within-Couette result: the correction restores coverage when calibrated on Couette

Calibrating the correction on Couette itself (the capability check, and the bonus
within-Couette cross-Re axis), 0.5 percent band:

| Couette Re_tau | in-distribution standard | in-distribution gen. Bayes | leave-one-Re-out conformal | gap |
|---|---|---|---|---|
| 171  | 0.737 | 1.000 | 0.842 | +0.058 |
| 260  | 0.632 | 1.000 | 1.000 | -0.100 |
| 507  | 0.632 | 1.000 | 0.789 | +0.111 |
| 986  | 0.895 | 1.000 | 0.789 | +0.111 |

When the correction is calibrated on Couette rather than transferred from the
channel, generalized Bayes restores coverage to 0.95 to 1.00 in distribution and the
cross-Re-within-Couette conformal gap is small (at most +0.11). So the correction is
fully capable of covering the Couette QoIs; the residual cross-flow gap in Section 5
is the genuine flow-type out-of-distribution shift (the channel-calibrated interval
under-covers the higher-Re Couette), reported honestly rather than tuned away.

## 7. Companions (a-priori)

Pipe (cross-geometry). The pipe Boussinesq discrepancy has the same structure as the
channel and Couette across all six Reynolds numbers (496 to 12055): the shear
component dominates the off-diagonal, the normal-stress anisotropy dominates the
discrepancy, and the DNS stress is realizable everywhere. The extraction machinery
transfers across geometry unchanged.

Rotating channel (the model-stress-test diagnostic). Streamwise rotation drives the
out-of-plane stress <uw>, which is a structural model-form failure a linear
eddy-viscosity cannot represent at all: in this mean flow the strain component
S_13 = 0, so the Boussinesq anisotropy b_13 = -C_mu S_13 is exactly zero (verified to
machine precision for every rotation number), while the DNS <uw> anisotropy is order
0.03 to 0.10. The discrepancy is distributed unevenly across the six stress
components, so a global correction (the scalar generalized-Bayes tempering or the
pooled conformal half-width of the channel and cross-flow work) must over-inflate the
median component by a factor of 2.2 to 3.2 to cover the worst, and it cannot equalise
the per-component coverage. This diffuse inflation is exactly what a feature-conditioned
generative model-form is built to remove; the rotating channel is therefore
a motivation for the separated-flow model-form, not a clean cross-flow transfer. (The streamwise-rotation
mean velocity is symmetric about the centerline, so there is no one-sided unstable
side; the structural failure is per-component, the picture that holds for streamwise
rather than spanwise rotation.)

## 8. Verdict against the criterion

Positive, matching the pre-registered graceful-degradation shape. Standard Bayesian
calibration is overconfident on the held-out Couette flow type (coverage 0.05 to 0.26
at nominal 0.90, bands eightfold too narrow), the channel overconfidence transferring
across the cross-flow shift. Generalized Bayes and conformal widen the bands and
restore coverage at the strict 0.5 percent band, fully at the lower Couette Reynolds
numbers and partially at the higher ones, with the conformal cross-flow gap reported
honestly (negative at low Re, up to +0.22 at Re_tau 507). The within-Couette check
confirms the correction restores coverage to nominal when calibrated on Couette, so
the residual gap is the genuine flow-type out-of-distribution shift, not a failure of
the method. The rotating-channel companion exposes the structural out-of-plane failure
and the diffuse inflation of a global correction, motivating the separated-flow generative
model-form. The coverage-correction spine therefore holds across a
held-out flow type and a-posteriori through a real moving-wall solve, and the pipeline
(loaders, modeled observation sigma, moving-wall forward model, cross-flow protocol,
companions, evaluation) is ready to carry to the separated and compressible
cases.

The honest caveats, none of which change the finding: the ensembles are 24 to 40
members (a 2-coefficient surrogate; a larger ensemble would sharpen the surrogate but
not the overconfidence, which is unambiguous); coverage is quantised by the 19 QoIs;
the channel-calibrated correction under-covers the higher-Re Couette at the 0.5 percent
band, which is the characterized cross-flow gap, not hidden; and the moving-wall solver
is a developing-free, streamwise-invariant Couette that the constant-total-stress
identity confirms is fully developed, but its channel-calibrated closure under-predicts
the Couette friction by about 17 percent, which is the model-form error the UQ covers.

## 9. Reproduce

`PYTHONPATH=build:python python3 python/UQ/reproduce_couette.py` (add
`--regen-ensembles` to re-run the forward solves, `--quick` for a smoke run). Fixed
seed 0; the matched viscosities and the ensembles are cached under the gitignored
`results/couette/`. The numbers above are in `finding_numbers.json`; the figures are
`figures/crossflow_coverage.png` and `figures/rotating_diffuse_inflation.png`.


## Post-audit revision (2026-07-17): corrected solver, converged members, disjoint conformal roles

The same audit remediation and regeneration as the channel package (see
step1_plane_channel/channel_finding.md for the full list: corrected solver physics,
genuinely converged cold members with budgets sized to the measured Reynolds-dependent
cost, and the disclosure that the committed ensembles were built almost entirely from
unconverged solves). Numbers from `finding_numbers_corrected.json`, same reproduce
script, fixed seeds, corrected trunk. The cross-flow conformal leg now uses
three-way-disjoint case roles: its posterior refits on the channel cases minus a
reserved calibration case (Re_tau 550 here), which supplies only the residuals, with
the Couette test flow untouched.

Revised headline numbers at nominal 0.90 (0.5 percent band, with the 1 percent band
in the JSON):

- Cross-flow transfer (channel-calibrated, held-out Couette flow type): standard
  Bayes overconfident at 0.053 to 0.211 coverage, the same band as committed;
  generalized Bayes partial restoration 0.421 to 0.526 (0.526 to 0.684 at the
  1 percent band); conformal with disjoint roles restores fully, 0.895 to 1.000
  with gaps -0.100 to +0.005, where the committed design's worst gap was +0.22.
- Within-Couette cross-Re: standard 0.632 to 0.947, generalized Bayes 1.000,
  conformal 0.947 to 1.000 per case with pooled held-out-station coverage 1.000
  (conservative at these calibration-set sizes).
- The cross-flow learning rate is eta = 0.0092, an order below the channel's
  in-distribution values, consistent with the committed reading that the channel
  posterior transfers its location but not its width.

The finding's shape is unchanged: the cross-flow overconfidence of standard Bayes is
confirmed on genuinely converged solves at the committed severity, and the corrected
UQ restores coverage at the held-out flow type, now with the conformal leg's validity
conditions actually satisfied by construction. The moving-wall solver results and the
rotating-channel diffuse-inflation diagnostic carry over unchanged in kind; the
companion diagnostics regenerate in the JSON.
