# Separated-flow model-form: baseline and injection operationalization

Dated addendum to PRE_REGISTRATION.md, fixed 2026-07-01 BEFORE any model-form
result was computed on the real separated data. PRE_REGISTRATION.md names the
methods and the criterion shape; this file pins down how the two comparison
baselines are turned into scoreable distributions and how every method enters
the solver, so no operational choice is left to be made after results exist.
It changes no criterion.

## 1. How the eigenspace-perturbation baseline is scored

The eigenspace-perturbation method (Emory, Larsson and Iaccarino 2013;
Iaccarino, Mishra and Gorle 2017) moves the anisotropy eigenvalues toward the
one-, two- and three-component limiting states of the barycentric triangle
(eigenvectors kept, per the 2013 formulation), giving a small set of perturbed
solves, not a probability distribution. Two pre-registered readings:

1. Envelope check (the method's own claim): does the [min, max] band over the
   perturbed solves contain the truth per quantity of interest. Reported as
   contain / not-contain per quantity.
2. Uniform-ensemble reading (for CRPS and energy-score comparability): the
   corner solves, equally weighted, are treated as samples of a predictive
   distribution. This is a charitable probabilistic reading (the method itself
   assigns no probabilities) and is labeled as such wherever scored.

Configuration: perturbations toward each corner (1C, 2C, 3C) at relative
magnitude Delta_B = 1.0 (full projection to the corner, the bounding envelope),
with Delta_B = 0.5 as a moderation sensitivity. The perturbed anisotropy is
propagated through the SAME injection mechanism as the generative model (section
3), so the comparison is solver-for-solver.

## 2. How the Gaussian Kennedy-O'Hagan baseline is built

The Gaussian model-form baseline is a heteroscedastic Gaussian conditional
model over the same five independent anisotropy-discrepancy components,
conditioned on the same five invariant features as the generative model (mean
and log-variance networks fit by maximum likelihood on the same training
pairs). It differs from the generative model in exactly one respect, the
distribution family (Gaussian versus normalizing flow), which is the property
under test. Its samples pass through the same realizability projection and the
same injection; the projection grants the Gaussian samples realizability they
do not have by construction, which is charitable to this baseline and is noted
where reported.

## 3. Injection conventions (shared by every method)

- The sampled target anisotropy is b_target = b_baseline + db, projected into
  the barycentric realizable set before injection and re-checked in the running
  solve; the realizability check is separate from the Galilean-invariant
  feature construction.
- The injected momentum source is the explicit deferred-correction force
  -div( 2 k b_target - 2 nu_t S ), with the eddy-viscosity diffusion kept
  implicit (PRE_REGISTRATION.md scope decision 3). k is the RUNNING k of the
  solve, so the injected stress scales with the local turbulence energy, and
  the correction vanishes identically when b_target equals the solver's own
  Boussinesq anisotropy (the baseline solve is recovered exactly; the injected
  difference from baseline is exactly -div(2 k db)).
- The training target db is formed against the limiter-consistent baseline
  anisotropy b_B = -(nu_t_phys/(k tau)) dev(S) with nu_t_phys = min(nu_t,
  k/omega): the SST Bradshaw limiter is honoured where it binds (separated
  shear layers), and the solver's numerical nu_t floor (0.1 nu), which would
  inflate b_B beyond the realizable bound near walls when divided by the
  vanishing local k, is excluded. Training target and a-posteriori injection
  therefore add db to the same baseline.

## 4. Quantities of interest scored a-posteriori

Reattachment length x_r/h (truth: the published 6.28), wall skin friction and
pressure at the six profile stations, and the mean-velocity profiles at the six
stations. Region-resolved reporting follows the pre-registration (recirculation,
reattachment, recovery); coverage, sharpness, CRPS and the multivariate energy
score come from the existing evaluation harness.

## 5. A-priori checkpoint protocol (added 2026-07-01, before any full fit ran)

The precursor checkpoint (PRE_REGISTRATION.md scope decision 1: "the a-priori
discrepancy fit and its coverage are the precursor") is scored as follows,
fixed before any conditional model was fit to the real separated data at full
training length:

- Held-out unit: a profile STATION. The six wall-normal profiles are
  internally correlated, so a point-level split would leak the test station
  into training; the protocol is leave-one-station-out over the six stations,
  plus a train-on-all in-distribution machinery check.
- Target: the DNS anisotropy at the held-out station. Predictions are
  b_baseline + db_model with every draw projected into the barycentric
  realizable set, the same projection the a-posteriori injection applies.
- Metrics: per-component coverage and sharpness of the central 90 percent
  band, per-component CRPS, the multivariate energy score over the five
  independent components, and the realizable fraction (1.0 by construction).
- Both the generative flow and the Gaussian model-form baseline are scored
  through the identical fit / sample / project / score path, isolating the
  distribution family. Fixed seed 0, 128 samples per point, 400 training
  epochs; no learning rate, feature set, or model is tuned toward any test
  number.

## 6. A-posteriori ensemble construction (added 2026-07-01, before any
## a-posteriori ensemble result was computed)

How a per-cell conditional sampler becomes a field realization one coupled
solve consumes, fixed before any ensemble ran:

- Each ensemble member is a COHERENT closure realization: one shared latent
  draw z (a single standard-normal vector over the five components) is pushed
  through the conditional model at every cell, so the member perturbs the
  whole field consistently, exactly as an eigenspace corner does. Sampling an
  independent latent per cell instead yields a spatially white stress
  perturbation whose divergence largely cancels; that variant is reported only
  as a labeled sensitivity diagnostic, never as the primary band.
- Both the generative flow and the Gaussian baseline use the same shared-
  latent construction (the Gaussian member is mu(x) + sigma(x) * z).
- Ensemble size 24 per probabilistic method, fixed seed 0, at the production
  grid; the eigenspace family contributes its corner solves (Delta_B = 1.0,
  with 0.5 as the moderation sensitivity) per section 1.
- Every member's target is projected into the realizable set before injection
  and re-checked in the running solve (section 3); members that fail to
  converge are reported by count and excluded from scores only with that
  exclusion stated.
- Scored quantities per section 4 against the DNS truths (reattachment 6.28
  and the measured station Cf), with coverage, CRPS and the energy score from
  the standard harness.

## 7. Post-result diagnostics and the calibration overlay (added 2026-07-02,
## after the first-geometry a-posteriori result, BEFORE any of these ran)

The backward-facing-step a-posteriori result (bfs_aposteriori_finding.md) is
already recorded; the three follow-ups below were requested on review and their
protocols are fixed here before any of them is computed. None changes the
recorded result or the criterion; they attribute it and apply the standing
correction layer.

1. Grid attribution of the baseline error. The corrected baseline is solved on
   a grid refined 1.5x in each direction (nx_up 60, nx_down 72, ny_up 36,
   ny_down 27, same inlet profile with the SAME inlet_delta = 0.6784 so only
   resolution changes), and the reattachment and delta_999 at x/h = -3 are
   reported for coarse / production / fine. The purpose is attribution of the
   0.33 baseline gap between discretization and closure, not accuracy tuning:
   whatever fraction moves with the grid is numerical and is reported alongside
   every claim that normalizes against the baseline error. No further
   refinement chasing.

2. Region attribution of the ensemble shift. The generative model's expected
   correction (the per-cell mean of its conditional over db, deterministic) is
   injected three ways on the production grid: full field; separated region
   only (the recirculation, shear layer and reattachment zone, cells with
   0 <= x/h <= 10 and y/h <= 1.5); attached complement only. The three
   reattachment shifts attribute the systematic upward displacement of the
   ensembles. This is a diagnostic with no retraining and no new criterion; a
   scope change it might motivate (for example region-restricted training) is a
   separate decision taken before any such training runs.

3. Calibration overlay (the standing coverage-correction layer), conformal
   form. Calibration data are the five measured downstream wall-friction
   stations, NOT the reattachment length being judged: per converged ensemble
   member the Cf at the stations is recorded; the nonconformity score at each
   station is |Cf_measured - median_ensemble| / MAD_ensemble (scale-normalized
   so scores pool across quantities); the split-conformal multiplier q is the
   ceil((n+1)(1-alpha))/n quantile of the five station scores at alpha = 0.10.
   The corrected reattachment band is median +/- q * MAD of the reattachment
   ensemble. Applied identically to the generative and Gaussian ensembles.
   Honest caveats stated with the result: five calibration points quantize the
   achievable quantile; exchangeability across quantity types (Cf stations to
   reattachment) is an approximation this diagnostic tests rather than assumes
   proven. The overlay corrects COVERAGE only; it cannot and does not repair
   the proper-score clauses already recorded.

## 8. Periodic-hills a-priori and cross-geometry protocol (added 2026-07-02,
## before any hills discrepancy or transfer result was computed)

The dense-field second geometry and the cross-geometry clause
(PRE_REGISTRATION.md positive-shape clause 4), fixed before any of the
following was run:

- Point set: interior fluid points of the dense field (the loader's interior
  mask, a clean gradient stencil), subsampled at a FIXED stride of 3 in each
  grid direction (a compute-budget choice made now; the dense field
  oversamples smooth regions). The discrepancy recipe matches the first
  geometry: b_DNS from the DNS stress at those points, b_baseline and the five
  invariant conditioning features from the converged RANS baseline field
  interpolated to them, with the limiter-consistent, floor-capped Boussinesq
  convention of section 3 and the wall-adaptive gradient step measured from
  the local hill surface and the top wall.
- Within-hills held-out unit: six equal streamwise bands (the analog of the
  first geometry's stations; wall-normal columns are internally correlated).
  Leave-one-band-out over the six bands plus the train-on-all in-distribution
  machinery check; nominal level 0.90, fixed seed 0, 400 training epochs, 128
  samples per point, exactly as section 5.
- Cross-geometry transfer (the pre-registered clause): train on ALL valid
  points of one geometry, score on ALL valid points of the other, BOTH
  directions, the generative flow and the Gaussian baseline through the
  identical fit / sample / project / score path. The conditioning features are
  the same five Galilean-invariant scalars with no geometry label, which is
  the property that makes the transfer question meaningful. Metrics as in
  section 5; graceful degradation is the pre-registered expectation, silent
  collapse the negative signal.
- The steepness family (alpha 0.5 to 1.5) is available through the same
  loaders; the primary protocol uses the alpha = 1.0 case (the benchmark
  configuration), with the family reserved for the parametric axis if the
  phase verdict motivates it.
