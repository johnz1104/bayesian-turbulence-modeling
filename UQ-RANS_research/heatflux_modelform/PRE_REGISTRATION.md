# Compressible heat-flux model-form UQ: pre-registration

Fixed 2026-07-07, before any generative model-form result was computed on the
compressible heat-flux discrepancy. This file records the success and null
criteria in advance, so they are timestamped ahead of every result and are not
tuned toward afterwards (the no-tuning-toward-the-number rule). It is
committed first.

## The work

The direct test of the requirement the compressible attached-flow study
measured. That study (PRE_REGISTRATION.md and compressible_attached_finding.md
under UQ-RANS_research/compressible_attached/) found standard Bayesian
calibration severely overconfident on the held-out thermal block (coverage
0.17 in distribution and 0.01 under the cross-Mach shift, at nominal 0.90),
and found the global coverage correction (scalar generalized-Bayes tempering
plus constant-width split conformal) able to restore the likelihood block
(0.955) but not the thermal block (0.42 in distribution, 0.18 under
transfer), with the conformal transfer collapsing to 0.006 because the
thermal residual scale grows two orders of magnitude along the Mach axis (the
wall-flux parameter B_q spans -0.002 to -0.200 across the matrix). Three
independent legs (that thermal block, the separated-flow injection diagnosis,
and the rotating-channel component diagnosis) fix one requirement: a usable
correction must reach the heat-flux quantities themselves and carry a
Mach-aware scale.

This study tests whether a conditional generative model of the heat-flux
discrepancy delivers that, a-priori on the attached matrix: train a
conditional normalizing flow on dq given (invariants, M_t) on low-Mach
cases, and score its conditional predictive distribution on held-out
high-Mach cases, against the honest distributional baseline (a Gaussian
conditional model of the same dq on identical inputs) and against the
committed global correction. It is a-priori by design: the separated-flow
study located its propagated failure in the anisotropy-only injection
channel, not in the conditional distribution family (a-priori with dense
data the flow beat the Gaussian on five of six held-out bands), so this
study interrogates the distribution and conditioning questions without an
injection confound. A second axis, folded into the same study, separates
scale from shape: a physically normalized conformal score applied to the
existing global-correction residuals, so that "was the scale the whole
problem" is answered by measurement, not assumption.

## Scope decisions fixed at study start (2026-07-07, before any training run)

1. Train/test split along Mach. Inherit the committed cross-Mach split of the
   attached-flow study on the Gerolymos-Vallet matrix: TRAIN on the eight
   M_CLx < 1.0 cases, HOLD OUT the sixteen M_CLx >= 1.47 cases (identical
   case lists to the committed calibration study; recorded in the numbers
   JSON). Within-matrix variant: leave-one-Mach-family-out, families pinned
   by nominal M_CLx at [0, 0.5), [0.5, 1.0), [1.0, 1.7), [1.7, 2.15) and
   [2.15, inf), giving 3, 5, 6, 8 and 2 cases; each family is held out in
   turn and the model trains on the other four, so interior families measure
   interpolation along the axis and the end families measure both
   extrapolation directions. The Coleman-Kim-Moser cases (bulk M 1.5 and
   3.0) never enter training: they are the independent-code spot check,
   scored with the low-Mach-trained models on the wall-normal component
   (all that record carries; its derived B_q and lower-half provenance are
   documented in DNS_data/README.md). The five Zhang-Duan-Choudhari flat
   plates never enter training: they are the pure held-out far-transfer leg,
   compounding flow type (channel to plate), baseline route (1-D solve to
   frozen-mean reconstruction), Mach (to 13.64) and wall cooling (Tw/Tr 0.18
   to 1.0), reported as the compound shift it is, on the derived wall-normal
   flux at its valid stations only. In-distribution controls: (a) train on
   all 24 matrix cases and score in sample (the machinery check); (b)
   leave-one-case-out within the eight low-Mach training cases (the
   in-distribution held-out control; the transfer clauses are attributable
   only if this control is healthy).

2. Target definition. The primary target is the wall-normal component
   dq_y = (q_hat - q_GDH)_y in the record's (u_tau, T_w) units, formed
   against the gradient-diffusion baseline at the fixed Pr_t = 0.9
   convention, exactly as the committed extraction module builds it
   (UQ.datasets.compressible_discrepancy); it is the one component every
   dataset defines (the channel matrix carries all three enthalpy-flux
   components; the supersonic-channel cross-check and the plates carry
   wall-normal only, the plates by the documented derived-flux inversion
   restricted to the valid mask). The family check trains the JOINT
   (dq_x, dq_y) on the channel matrix: the streamwise component is a pure
   structural miss of the gradient-diffusion baseline (grad T is wall-normal
   for these statistically 1-D profiles, so q_GDH_x = 0 and dq_x = q_hat_x,
   which no Pr_t value can represent), and the joint leg is where the two
   model families genuinely differ (degeneracy note below). The spanwise
   component is a symmetry null and is reported as a parse check only, not
   modeled.

   Degeneracy note, stated in advance as a structural fact: for a
   one-dimensional target the affine-coupling flow used here (UQ.generative,
   RealNVP with alternating masks) reduces exactly to a conditionally
   Gaussian law (every active coupling is an affine map of the latent with
   feature-dependent coefficients), so on the scalar primary leg the flow
   and the Gaussian baseline are the SAME distribution family in different
   parametrizations. The scalar leg therefore tests the conditioning (the
   Mach-aware location and scale), not the family shape; the family-shape
   question (skewness, tails, cross-component structure a diagonal Gaussian
   cannot represent) is carried by the joint leg, where coupling across
   components makes the flow genuinely non-Gaussian. Approximate scalar-leg
   parity between the two models is an expected shape, not a family finding,
   and is pre-registered as such.

3. Conditioning features. The M_t-extended invariant set exactly as the
   attached-flow study built and committed it: the five Galilean-invariant
   Pope invariants from the baseline strain and rotation with the baseline
   timescale, extended with the turbulent Mach number M_t computed from the
   record itself; six features per point, through
   UQ.datasets.compressible_discrepancy and UQ.discrepancy.feature_set. The
   semi-local coordinate y* stays a carried diagnostic, not a feature. The
   thermodynamic-fluctuation correlations in the matrix's TTS tables
   (present, so far unused) are NOT added: enriching the conditioning with
   thermodynamic-structure features changes the data requirements and the
   differentiation argument of every downstream consumer, and is a separate
   decision to take on its own evidence. Two facts stated, not hidden: on
   1-D profiles several Pope invariants are functionally dependent
   (pure-shear kinematics), so the effective conditioning is the shear
   invariant, M_t, and their case-to-case covariation; and the held-out
   cases exceed the training range along M_t, so the transfer clauses score
   an extrapolation in the conditioning variable, which is exactly the
   question, with the per-case M_t span recorded in the numbers JSON.

4. The normalized conformal score (the secondary axis). This axis re-scores
   the committed global-correction residuals with a physically normalized
   split-conformal score; it does NOT recalibrate. The cached calibration
   ensembles are keyed to the attached-flow study's pre-registered
   observables, which are unchanged here, so their reuse is valid by
   construction; any recalibration with different observables would
   invalidate them, and none is performed. Pinned score definitions, both
   applied to the same pooled tempered-posterior point predictions the
   committed study used, calibrated on the training cases' thermal
   residuals and evaluated on the held-out cases' thermal block through
   UQ.conformal.conformal_quantile:

   - Primary: s_i = |y_i - m_i| / max(|B_q_base(case)|, 1e-4), where
     B_q_base(case) is the case's own deterministic 1-D SST baseline
     wall-flux prediction at the fixed nominal coefficients (a1 = 0.31,
     betaStar = 0.09, Pr_t = 0.9). The scale is a model quantity available
     for every channel case and uses no held-out truth (the case's true B_q
     is itself a held-out thermal QoI, so normalizing by the truth would
     leak; the baseline's prediction does not). The floor is a guard against
     the thermally quiescent limit and never binds on this matrix (the
     smallest baseline wall flux is order 2e-3).
   - Sensitivity: the semi-local rescaling s_i = |y_i - m_i| sqrt(rho_plus_i)
     per point, the (u_tau, T_w)-to-semi-local flux-unit conversion, with
     rho_plus the case's own mean density profile (an operating-condition
     input throughout the attached-flow study); at the wall rho_plus = 1, so
     the wall-flux residual is unchanged. This tests whether a purely
     local-property rescaling carries the Mach growth, against the
     wall-flux-parameter scaling of the primary.

   Both scores run at both committed sigma levels (0.5 percent primary,
   1 percent sensitivity, cached ensembles at each), on the identical
   primary split, exactly parallel to the committed absolute-score transfer,
   whose measured held-out thermal coverage of 0.006 (gap 0.894; 0.011 and
   gap 0.889 at the sensitivity level) is the comparison line.

## Models and training protocol (pinned)

- Conditional flow: UQ.generative.GenerativeDiscrepancyModel with
  n_features = 6, n_targets = 1 (primary) or 2 (joint leg), n_layers = 8,
  hidden = 64; fit(epochs = 400, lr = 1e-3, batch = 256): the settings the
  separated-flow study pinned, unchanged.
- Gaussian conditional baseline: UQ.gaussian_modelform.GaussianDiscrepancyModel
  with the same feature and target dimensions, hidden = 64, fit on identical
  inputs with identical settings; the distribution family is the only
  difference, which is the property under test on the joint leg.
- Diagnostic reference, labeled as such and not a criterion baseline: an
  unconditional pooled Gaussian on the training dq (constant mean and
  variance, no features), to measure what conditioning itself buys.
- Training rows: all interior points of the training cases, interior =
  (y+ > 30) and (y/delta < 0.9), the committed convention of the
  attached-flow extraction; on the plates additionally the derived-flux
  valid mask. The held-out unit is the CASE (wall-normal profiles are
  internally correlated; point-level splits would leak, per the
  separated-flow protocol precedent). The training set is order a thousand
  rows or fewer across the eight low-Mach cases, the sparse side of the
  family precedent (the dense-field leg of the separated-flow study had
  13189 rows and the flow won there; its 1077-row sparse leg it did not),
  stated in advance as a known risk to the family clause; it licenses no
  post-hoc mask relaxation.
- Sampling: 128 samples per point per model, independent latents per row
  (the pointwise conditional law; coherent shared-latent field realizations
  are an injection-side construction and out of scope here). No post-hoc
  clipping or projection of dq samples (realizability note below).
- Seeds: model seeds {0, 1, 2} with deterministic data assembly; every
  metric is reported as the mean over the three seeds with the min-max
  range, and every criterion below is evaluated on the seed mean (the
  training-stochasticity guard, pinned in advance). Torch 2.10.0 on the
  pinned Python 3.11.
- Training-side adjustments (for example a learning-rate or epoch change on
  a fit whose training loss diverges) are permitted on training-loss
  diagnostics of the training rows only, never on any held-out metric, and
  every such adjustment is recorded in the memo.

## The question (the gate)

Does a conditional generative model of the heat-flux discrepancy, trained on
low-Mach attached cases and conditioned on (invariants, M_t), deliver
calibrated conditional coverage on held-out high-Mach cases, where the
committed global correction failed, and does it beat the Gaussian
conditional baseline on proper scores. And, on the folded-in secondary axis,
does a physically normalized conformal score alone restore the cross-Mach
transfer of the global correction, so that the scale-versus-shape question
is answered cleanly.

### Pre-registered positive shape

1. Conditional transfer (primary target; per-case coverage of dq_y from
   UQ.evaluation.coverage_from_samples at nominal 0.90, aggregated as the
   unweighted mean over cases): the in-distribution control is healthy
   (leave-one-case-out over the eight training cases lands the case-mean
   coverage in [0.80, 0.98]), AND the flow's held-out high-Mach coverage has
   case mean at least 0.80 with every case at least 0.60. The comparison
   lines are the committed global correction on the same axis (tempered
   thermal 0.176, conformal 0.006 at nominal 0.90) and the Gaussian
   conditional scored identically. A pre-named intermediate shape: case mean
   at or above 0.80 with isolated far-end cases below 0.60 reads as
   distance-graded partial transfer, characterized by the Mach-ordered
   coverage profile, and is NOT the full positive.
2. Family and proper scores (the joint leg): the flow's energy score
   (UQ.evaluation.energy_score, unweighted case mean over the sixteen
   held-out cases) is lower than the Gaussian's, and its wall-normal CRPS
   (UQ.evaluation.crps_ensemble) is no worse in aggregate. On the scalar leg
   approximate parity is the expected degenerate-family shape and carries no
   family claim in either direction.
3. Calibration in the middle of the distribution: at nominal 0.50 the
   held-out case-mean coverage lies in [0.35, 0.65] (a two-sided guard that
   transfer is not bought with vacuous width), and the reliability and PIT
   diagnostics (UQ.evaluation.reliability_error, pit_values, pit_histogram,
   pit_uniformity_pvalue) are reported per case and per split.
4. The far legs are characterization, not pass/fail clauses: the
   never-trained supersonic-channel cases and the five plates are scored
   with the same metrics against the same lines, with the compound nature of
   the plate shift stated. The committed global-correction precedent
   (thermal 0.82 at bulk M 1.5 inside the calibrated span, 0.00 at M 3.0
   outside it) fixes the reading of interest: whether the conditional model
   degrades with distance along the axis slower than the global correction
   did.

### Pre-registered null shapes (equally reportable, each a decision point)

- (a) Family null: the flow does not beat the Gaussian conditional baseline
  on the joint-leg proper scores, or its held-out coverage is nowhere
  better. Reading: the conditionally Gaussian family suffices for the
  attached heat-flux discrepancy at this data volume; the generative family
  earns nothing a-priori here, and the separated high-speed direction should
  default to the cheaper family unless its discrepancy shows non-Gaussian
  structure the attached matrix lacks. The sparse-training caveat above is
  part of this reading (the family advantage was measured to need dense
  data on the separated geometries).
- (b) Transfer null: the in-distribution control is healthy but the held-out
  high-Mach case-mean coverage falls below 0.80 (or failures are widespread
  rather than far-end). Reading: (invariants, M_t) conditioning as built
  does not carry the thermal discrepancy law across Mach; combined with the
  secondary-axis quadrant below, this decides whether scale normalization or
  feature enrichment is the motivated follow-on for the separated
  high-speed direction.
- Degenerate control: if the in-distribution control itself fails its band,
  no transfer claim of either sign is made; the finding is then about
  trainability of the conditional law at this data volume, reported as such.

### The scale-versus-shape reading grid (all four quadrants pre-assigned)

Define N+ as the primary normalized conformal score restoring held-out
thermal coverage into [0.80, 0.98] at nominal 0.90 (against the committed
absolute-score 0.006), N- otherwise; define F+ as the full positive of
clause 1, F- otherwise (the partial shape noted alongside).

- N+ and F+: the Mach growth is dominantly a scale effect AND the
  conditional model carries it. The cheap normalized score is the default
  correction for attached-matrix thermal transfer; the generative model's
  added value rests on the family clause: with clause 2 also positive the
  direction is confirmed with measured shape gains, without it the measured
  recommendation is the normalized score plus the Gaussian conditional.
- N+ and F-: the scale was the whole problem and feature conditioning as
  built does not otherwise transfer. The measured recommendation is the
  normalized-score correction; the premise of a learnable Mach-conditional
  shape beyond scale is unsupported on attached data, a real decision point
  for the separated high-speed direction, whose evaluation should then carry
  the normalized score as the honest first-line correction.
- N- and F+: normalization alone cannot restore transfer but the conditional
  model can: the discrepancy shape itself is Mach-dependent and the
  conditional model is doing irreducible work. The strongest support for the
  generative direction this study can produce.
- N- and F-: neither a physical rescaling nor (invariants, M_t) conditioning
  carries the thermal law across Mach. The conditioning features are the
  leading suspect (the separated-flow transfer measured the same signature:
  the invariant features did not separate the regimes), and feature
  enrichment (for example the thermodynamic-fluctuation correlations the
  matrix's TTS tables already carry) becomes the motivated follow-on, as its
  own scope decision, not an adjustment to this study.

The semi-local sensitivity score refines the N reading: if the
wall-flux-normalized primary restores transfer and the semi-local variant
does not, the carrier of the Mach growth is the wall-flux parameter rather
than local properties, and vice versa; disagreement between the two is
reported, not averaged.

### Thresholds

The bands above ([0.80, 0.98] for the control and for normalized-score
restoration; case mean at least 0.80 with per-case floor 0.60 held out;
[0.35, 0.65] at nominal 0.50; strict aggregate inequality on the joint-leg
energy score) are fixed here in advance. No model, feature, mask, score,
seed, or training choice is tuned toward any of them.

## Baselines the result is measured against

- The Gaussian conditional model on dq with identical inputs: the honest
  distributional baseline (the family axis).
- The committed global correction (generalized-Bayes tempering plus split
  conformal), whose measured cross-Mach thermal numbers (0.176 tempered and
  0.006 conformal at the primary sigma level; 0.148 and 0.011 at the
  sensitivity level) are the requirement line; cited from the committed
  numbers JSONs, not rerun.
- The unconditional pooled Gaussian on dq: a labeled diagnostic for what
  conditioning buys, not a criterion baseline.
- Structural exclusion, stated as a reportable fact about the method
  landscape: the eigenspace-perturbation framework, the dominant model-form
  UQ method elsewhere and a scored baseline of the separated-flow study,
  perturbs the Reynolds-stress anisotropy eigenvalues within the barycentric
  realizable set; the turbulent heat flux is outside that object, so an
  anisotropy perturbation cannot represent a heat-flux correction at any
  amplitude. It is therefore structurally excluded as a baseline here, and
  that exclusion carries into the memo: the discrepancy this study models
  lives in a quantity the dominant realizable method cannot reach.

## Metrics

UQ.evaluation.coverage_from_samples at nominal 0.90 and 0.50 (coverage and
sharpness), crps_ensemble per component, energy_score on the joint leg,
reliability_curve and reliability_error over the default nominal grid,
pit_values with pit_histogram and pit_uniformity_pvalue; per case, per
split, seed mean with min-max range; aggregation is the unweighted mean over
cases (station counts differ by a factor of four across the matrix;
point-pooled values are recorded as a diagnostic view only). Every number
from the fixed-seed reproduce script.

## Observation uncertainty

The generative leg adds NO modeled observation sigma: the models are fit on
the raw (feature, dq) pairs and their own predictive dispersion is what is
scored, exactly the separated-flow a-priori protocol. The modeled-sigma
convention of the attached-flow study enters only through the secondary
axis, which re-scores that study's residuals at its two committed levels
from the cached ensembles; the caches are valid for this reuse because the
observables are unchanged, and no recalibration is run.

## Realizability and invariance (separate checks, stated scope)

This study samples no Reynolds stress, so the barycentric realizability
projection has no role in its sampling path; and there is no committed
realizability-type admissibility set for the turbulent heat-flux vector in
this framework (a Cauchy-Schwarz-type bound would need thermal-variance
fields the datasets do not uniformly carry), so none is invented here: dq
samples are scored as drawn, with no clipping. The two standing checks still
run where they apply: the DNS stress record of every case passes the
barycentric realizability check (fraction 1.0 expected, as committed), and
the conditioning features are Galilean invariant by construction (they
depend on the record only through the velocity gradient and the baseline
timescale), both asserted in the assembly tests.

## Reproduce plan

A study module under UQ.datasets (assembly of (features, dq) across the
splits, the fit and scoring loops, mirroring the separated-flow a-priori
module), a fixed-seed driver script that writes one numbers JSON, unit tests
on the assembly and split logic plus a smoke-epoch training test, and the
evidence package in this directory (this file, the finding memo, figures,
the numbers JSON). Long runs go through the detached-run pattern with a done
marker per standing practice.

## Data attribution

All three datasets are third-party DNS, not produced by this project, cited
where used: Gerolymos and Vallet, J. Fluid Mech. 958 (2023) A19 (Mendeley
Data, doi:10.17632/wt8t5kxzbs.1, CC BY 4.0); Coleman, Kim and Moser, J.
Fluid Mech. 305 (1995) 159-183; Zhang, Duan and Choudhari, AIAA Journal
56(11) (2018) 4297-4311, doi:10.2514/1.J057296. The latter two are hosted by
the migrated NASA Turbulence Modeling Resource (tmbwg.github.io/turbmodels).
Raw fields stay local and gitignored.
