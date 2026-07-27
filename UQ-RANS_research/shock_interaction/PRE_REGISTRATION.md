# Shock-interaction model-form UQ: pre-registration

Fixed 2026-07-10, before any discrepancy was extracted from the interaction
data and before any model-form or coupled result was computed on it. This file
records the scope decisions and the success and null criteria in advance, so
they are timestamped ahead of every result and are not tuned toward afterwards
(the no-tuning-toward-the-number rule). It is committed first; nothing
downstream of these decisions is built before this file is merged.

## The work

The direct test, at a shock-boundary-layer interaction, of the requirement
three completed studies fixed. The compressible attached-flow study measured
standard Bayesian calibration severely overconfident on held-out thermal
targets (coverage 0.17 in distribution and 0.01 under the cross-Mach shift, at
nominal 0.90) and its global coverage correction unable to reach the thermal
block (0.42 in distribution, 0.18 under transfer). The separated-flow
model-form study located its propagated failure in the anisotropy-only
injection channel (magnitude-capped by the running turbulence energy), with
the conditional distribution family exonerated on the dense field. The
heat-flux model-form study found the conditional generative route
data-volume-limited before family-limited (434 training rows), and measured a
wall-flux-normalized conformal score recovering over half the cross-Mach
transfer gap at zero model cost. Together they fix one requirement: a usable
correction must reach the heat-flux quantities themselves, carry a Mach-aware
scale, and be tested where the misspecification is severe, which is the
interaction regime.

This study tests whether a stochastic, realizability-constrained,
feature-conditioned model-form correction improves shock-boundary-layer
interaction prediction over baseline RANS with quantified uncertainty. Two
legs are fixed here together:

- A-priori (the discrepancy-distribution leg, no coupled claim): train the
  conditional generative model and the honest Gaussian conditional baseline on
  attached compressible discrepancy data and score the held-out interaction
  discrepancy (stress anisotropy db and turbulent heat flux dq), conditioned
  on the M_t-extended invariant features; and, within the interaction, run the
  wall-thermal cross-validation at the data density the attached matrix
  lacked, which is where the family question is live.
- A-posteriori (the coupled-prediction leg, the primary result): propagate
  sampled realizable closures through the shock-capturing solve to the
  predicted wall quantities (Stanton number, wall pressure including the shock
  jump, skin friction, separation and reattachment, shock position), measure
  predictive coverage and error change against baseline SST, against the
  eigenspace-perturbation envelope and the Gaussian conditional propagated
  through the identical injection, and measure the attached-flow control.

## Data (verified at study start; full provenance blocks go in DNS_data/README.md)

Both interaction datasets are the same impinging oblique-shock configuration:
free-stream Mach 2.28, shock-generator incidence 8 degrees, gamma 1.4,
statistically two-dimensional (span-homogeneous) fields, lengths in incoming
boundary-layer thickness units. The shock strength is therefore fixed and the
interaction generalization axis is the wall thermal condition.

- Adiabatic interaction (Pirozzoli and Bernardini, AIAA J. 49(6), 2011,
  doi:10.2514/1.J050901). Re = 16750 on the inlet thickness. Verified layout:
  32 Tecplot block files for each averaging (favre_33 to favre_64, reynolds_33
  to reynolds_64), each zone 61 x 344 with shared edge columns, tiling the
  downstream half of the 80.58 x 12.89 domain (x in [40.29, 80.58]); Favre
  files carry 8 columns (x, y, u, v, u''v'', u''u'', v''v'', w''w''), Reynolds
  files 14 columns (x, y, rho, u, v, p, T, rho'rho', u'u', v'v', w'w', u'v',
  p'p', T'T'; the readme's read loop says 13, the files hold 14, so loaders
  count columns from the data). Wall series stat.dat: 13 columns (x, cf, pw,
  pwrms, utau, deltav, delta99, delta*, theta, delta*inc, thetainc, H, Hinc),
  3904 rows spanning the full domain (block-edge x values duplicated, deduped
  at load). Incoming-boundary-layer profile blinc.dat at x = 43.6 (Re_tau 466,
  Re_theta 2344, cf 2.56e-3, H 3.55), 153 rows, 12 columns. Turbulence
  kinetic-energy budget at three interaction stations x* = -1.93, -0.05, 2.10
  (7 columns; the pressure-dilatation term is lumped with mass diffusion).
  No temperature-velocity covariance and no wall heat flux (adiabatic): this
  dataset carries the stress leg, Cf, Cp, the inflow anchor and the budget
  anchor, not the heat-flux leg.
- Heated and cooled interaction (Bernardini, Asproulias, Larsson, Pirozzoli
  and Grasso, Phys. Rev. Fluids 1, 084403, 2016). The same interaction at five
  wall-to-recovery-temperature ratios s = 0.5, 0.75, 1.0, 1.4, 1.9 (strongly
  cooled through adiabatic to heated); incoming Re_theta about 2500 at the
  reference station; nominal impingement at 69.5 inlet thicknesses. Verified
  layout: one 13-column Tecplot Favre field per s (x*, y*, rho, u, v, p, T,
  u''u'', v''v'', w''w'', u''v'', u''T'', v''T''), zones 3001 x 284 (s = 0.5,
  0.75) and 1610 x 230 (s = 1.0, 1.4, 1.9), x* in [-13.5, 8.2] about the
  impingement point, y* in [0, 2.5]. This is the only source of the turbulent
  heat-flux vector, so the dq leg lives here. Wall series wallstat_s.dat per
  s: x*, cf, pw, pwrms, St, x* in [-40, 15.2]; verified quirks: the s = 1.0
  file has four columns (no Stanton column at all, adiabatic) and its pw
  normalization differs from the other four (order 132 versus order 1
  upstream), so wall pressure is reduced to Cp against each file's own
  upstream plateau rather than an assumed reference.
- Attached supersonic boundary layers (Pirozzoli and Bernardini, JFM 688,
  2011, and Phys. Fluids 25, 021704, 2013; reynolds.dma.uniroma1.it/dnsm2).
  Twelve flat-plate cases: M2 at eight friction Reynolds numbers (nominal
  Re_tau 200 to 1110), M3 and M4 at Re_tau 400 and 500. Verified format: a
  prose header (cf, friction Mach number, four Reynolds numbers, shape
  factors) and 20 profile columns (y/delta99, y+, u+, u_vd+, urms+, vrms+,
  wrms+, uv+, sqrt(rho/rho_w), prms+, trms+, rhorms+, the indicator function,
  skewness and flatness of u and T, vorticity intensities). Verified
  limitation, load-bearing: these files carry no turbulent heat-flux vector,
  no wall heat flux and no mean temperature column (adiabatic-wall cases;
  mean density is recovered from the sqrt(rho/rho_w) column, mean temperature
  by the constant-pressure boundary-layer relation, stated as a
  reconstruction). This set therefore serves the attached stress axis and the
  attached-flow control, and does not enter heat-flux training or heat-flux
  generalization claims (per the acceptance gate for compressible heat-flux
  generalization).

The attached heat-flux training side reuses the committed compressible
attached matrix: the Gerolymos-Vallet 24-case channel matrix (the only
attached source carrying the full turbulent heat-flux vector), with the
Coleman-Kim-Moser supersonic channel kept as the never-trained
independent-code spot check and the Zhang-Duan-Choudhari plates as a
pre-named sensitivity pool (derived wall-normal flux only), exactly as those
roles were fixed in the attached studies.

## Scope decisions fixed at study start

### 1. Out-of-distribution axis and splits

The data resolves the axis: shock strength is fixed (one Mach, one incidence)
and the interaction generalization axis is the wall-to-recovery-temperature
ratio, five conditions, meeting the at-least-two-conditions acceptance gate.
Splits, all pinned now:

- Within-interaction (the primary conditional-transfer axis):
  leave-one-wall-thermal-out over the five heated-set Favre fields. Train on
  four s conditions, score the held-out fifth; five folds; the held-out unit
  is the s condition (2-D fields are internally correlated; point-level
  splits would leak). The dq targets exist in all five fields; the db targets
  additionally use the adiabatic 32-block dataset as an independent-campaign
  test surface for the s = 1.0 fold.
- Attached-to-interaction (the far-transfer axis): train on the attached
  matrix, score the interaction fields. For dq the training pool is the
  24-case channel matrix through the committed extraction; for db the pool is
  the channel matrix plus the twelve attached boundary-layer cases (the
  flow-type match to the interaction boundary layer), with the plates'
  ready-made anisotropy as a labeled sensitivity variant. This axis compounds
  flow type, pressure-gradient state and turbulence regime at once; the
  compound nature is stated in advance (the analogous compound plate shift
  collapsed every conditional model in the attached heat-flux study), and the
  region-graded reading below applies.
- Attached in-distribution control for the far-transfer axis:
  leave-one-Mach-family-out over the channel matrix (the five committed
  families). If this control fails its band, the degenerate-control branch
  applies to the far-transfer clause exactly as in the heat-flux study.
- In-sample machinery check: train on all five interaction fields, score in
  sample (near-nominal coverage expected; a fitting-capacity check, not a
  claim).
- Region grading, pinned from each configuration's wall series: separation
  x_s and reattachment x_r are the Cf sign crossings (linear interpolation);
  the interaction onset x_onset is where the smoothed wall pressure first
  exceeds 1.05 times its upstream plateau (centered moving average of width
  0.5 reference lengths); regions are upstream attached (x* < x_onset),
  interaction (x_onset <= x* <= x_r + 2), and relaxation (x* > x_r + 2).
  Every coverage and error number is reported per region as well as pooled;
  the interaction heat-flux error is never quoted as a single flat figure.

### 2. Baseline route for forming db and dq (and the solver prerequisite)

Measured facts first. The one-dimensional fully-developed channel solve has no
analogue at an interaction. The frozen-mean algebraic plate reconstruction
breaks exactly through separation, shock foot and reattachment (sign-changing
shear, no transported k, no streamwise strain). The shock-capturing
density-based solver on the trunk is verified on its convective core (shock
tube, manufactured-solution order, oblique-shock reflection, wall observation
operator) but its committed test suite records that the explicit pseudo-time
march does not converge viscous-dominated steady states on near-wall meshes
(the viscous spectral radius scales as the inverse square of the wall-normal
spacing); an implicit steady driver is the documented fix.

Decision: the baseline for both legs is the two-dimensional density-based SST
solve of the interaction, one solve per wall-thermal configuration, with

- an implicit steady driver (LU-SGS) added to the density-based solver as a
  prerequisite build, validated on the two rungs the solver's own verification
  names as gated: the supersonic laminar flat plate against the self-similar
  solution, and the turbulent flat plate at Mach 2.28 against the incoming
  boundary-layer profile of the adiabatic dataset;
- inflow anchored to the measured incoming boundary layer (Re_tau 466,
  cf 2.56e-3, van-Driest profile), the incident shock imposed through the top
  boundary by the oblique-shock state at 8 degrees incidence, supersonic
  outflow, and the wall adiabatic or isothermal at the case's
  wall-to-recovery ratio;
- baseline fields (velocity, temperature, k, omega, eddy viscosity, the SST
  timescale) sampled at the DNS points; db = b_DNS - b_baseline with the
  limiter-consistent Boussinesq convention of the separated study;
  dq = q_hat_DNS - q_GDH with the gradient-diffusion baseline at fixed
  Pr_t = 0.9, DNS mean gradients and the baseline eddy viscosity, the
  committed convention of the attached extraction.

This route preserves the train-inject consistency the separated study pinned
(the a-priori training target and the a-posteriori injection add db to the
same baseline object) and the same solve is the a-posteriori engine, so one
validated baseline serves both legs.

Feasibility gates, fixed in advance and prerequisites rather than result
clauses: (gate A) the attached turbulent baseline at Mach 2.28 reproduces the
incoming layer, skin friction within 10 percent of the measured 2.56e-3 at
matched momentum-thickness Reynolds number and the van-Driest log region
within 5 percent rms; (gate B) the coupled interaction baseline converges
(solver-classified) for every wall-thermal configuration, and its impingement
position lands within one reference length of the DNS half-rise point, with
any offset reported. Pre-registered attribution diagnostic: the baseline
shock-position offset against the DNS and the fraction of interaction-region
discrepancy magnitude concentrated inside the offset band are reported, so
position error is not silently read as closure error.

Pre-named fallback: if gate B cannot be met, the a-priori leg switches to a
frozen-mean-field transport baseline (the DNS Favre mean held fixed, the k and
omega transport equations marched to steadiness with the same implicit driver,
stated as the constitutive-error-at-matched-mean convention), and the
a-posteriori clauses are reported as not adjudicable at this solver maturity,
which is the solver-capability null shape below, a real and reportable
finding.

### 3. Loaders (shared infrastructure)

Two new loader families, on the established loader-plus-tests-plus-provenance
template, consuming the verified formats above:

- A two-dimensional compressible interaction loader handling both interaction
  datasets: the 32-block tiling with shared edge columns dropped (adiabatic),
  the single-zone Tecplot fields per s (heated set), the wall series (with
  block-edge dedup, the four-column s = 1.0 quirk, and Cp formed against each
  file's own upstream plateau), the incoming-layer profile, and the
  three-station budget files. It serves either the dense 2-D field or
  wall-normal profiles at chosen x* stations. The stress-column convention
  (whether the tabulated double-prime covariances are density-weighted) is
  pinned at loader time by the adiabatic favre-versus-reynolds cross-check
  and the papers' definitions, and recorded in the loader tests.
- An attached boundary-layer profile loader for the twelve M2/M3/M4 cases on
  the compressible profile base, with the frozen-mean plate baseline route,
  the density column squared for mean density, the constant-pressure
  temperature reconstruction stated as such, and no heat-flux fields.

Units are converted once at the loaders into the committed wall-unit Favre
conventions; the interaction fields use the case's upstream reference
friction state (the friction velocity at the reference attached station and
the case wall temperature) as the inner scale, the interaction analogue of
the channel (u_tau, T_w) convention, so a friction velocity that vanishes at
separation never divides a target. Realizability of every DNS stress record
(barycentric check, fraction reported) and Galilean invariance of the feature
construction are asserted as separate loader-test checks.

### 4. The injection coupling in the density-based solver (the heat-flux reach)

The sampled correction enters the density-based residual as a
deferred-correction flux, mirroring the incompressible pattern and extending
it to the energy equation, which is the direct answer to the separated-flow
magnitude-cap diagnosis:

- Momentum: the divergence of the injected stress difference,
  2 <rho> k (b_target - b_Boussinesq), assembled as a conservative face flux;
  the eddy-viscosity diffusion stays implicit; a zero correction reproduces
  the baseline solve identically (asserted).
- Energy: two consistent reaches. The mean-flow work of the injected stress
  difference, and the turbulent-heat-flux correction, the divergence of
  <rho> cp dq_target against the gradient-diffusion term, which stays
  implicit. An anisotropy-only variant (energy reach disabled) is retained as
  a labeled diagnostic so the value of the heat-flux reach is itself
  measured.
- Every sampled b_target is projected into the barycentric realizable set
  before injection and re-checked every outer iteration; dq has no committed
  admissibility set and none is invented (the solver's
  Converged/Unconverged/Diverged classification handles pathological members,
  which are counted and excluded with the exclusion stated, per the separated
  protocol).
- Python surface: set and clear the target correction and read injection
  diagnostics on the density-based binding, matching the incompressible
  forward model's surface; C++ and Python tests cover the zero-correction
  identity, a prescribed-field flux check, and the realizability re-check.
- A single coupled converged solve at Mach 2.28 with the imposed shock and a
  nonzero injected correction is verified before any ensemble is run.

### 5. Target definition and the joint leg

- db: the independent Favre anisotropy components from the 2-D stress record
  (b11, b22, b12, with b33 fixed by the trace and b13 = b23 = 0 by span
  homogeneity), formed against the limiter-consistent baseline anisotropy.
- dq: the two live components (dq_x, dq_y) from the heated set's u''T'' and
  v''T'', in the upstream-referenced wall-unit convention; the heated set is
  the only dq source, stated as scope.
- The wall Stanton number and wall heat flux are a-posteriori quantities of
  interest, not a-priori targets.
- The joint multi-component legs carry the family question (the scalar
  affine-coupling flow is exactly a conditionally Gaussian law, the committed
  degeneracy note): the (dq_x, dq_y) joint leg and the db-vector leg are the
  family surfaces; scalar-leg parity is the expected degenerate shape and
  carries no family claim.
- Moat clause, stated as a structural fact and bounded by what this data
  measures: the eigenspace-perturbation framework perturbs the anisotropy
  eigenvalues within the barycentric set and cannot represent a heat-flux or
  wall-flux correction at any amplitude, so it is structurally excluded from
  the heat-flux legs and scored only on the stress and propagated-QoI legs.
  The dilatational moat modes are not separately measured in this data (the
  budget files lump pressure-dilatation with mass diffusion at three stations
  of the adiabatic case), so no dilatational-mode claim is made; the moat
  evidence here is heat-flux-centred.

### 6. Realizability at the shock (separate from invariance)

The barycentric realizability projection is applied to every sampled target
before injection and asserted every outer iteration of every coupled solve
(zero tolerated violations among converged members); the Galilean-invariant
feature construction is verified separately in the assembly tests. The DNS
stress records' own realizable fraction is reported per field. These two
checks never substitute for each other.

### 7. Observation uncertainty (modeled, anchored, labeled)

The interaction files carry no per-point statistical uncertainty. The
a-priori generative leg adds no modeled sigma (models fit raw pairs; their
own predictive dispersion is scored, the committed protocol). Where
observations enter a comparison against solves (the a-posteriori QoI scoring
and any residual-based conformal), observation sigma is modeled at 0.5
percent of the local scale (primary) and 1 percent (sensitivity), per
configuration floored at data-only physics anchors measured by the loaders
and recorded in the loader tests:

- the cross-campaign adiabatic residual: the adiabatic wall series of the
  2011 dataset against the s = 1.0 wall series of the 2016 dataset over their
  shared interaction range (Cf and Cp at matched x*), a data-only
  between-campaigns consistency measure;
- the momentum-integral residual of the upstream attached region of the
  adiabatic wall series (the compressible von Karman balance from the series'
  own cf and theta columns);
- the budget-closure residual of the three adiabatic budget stations (the
  tabulated terms summed against zero).

These are labeled modeled and anchored, never called a DNS uncertainty.

### 8. Optional history-feature probe (off by default)

The spatial non-local (strain-history) ansatz remains a hypothesis, not a
premise. If scoped in at confirmation time, it enters only as an a-priori
feature ablation: one added conditioning feature, the upstream mean-streamline
exponentially-weighted average of the shear invariant with the baseline
timescale as the relaxation scale, and the pinned question is whether the
held-out interaction-region misfit improves against the local feature set. No
a-posteriori use, no claim beyond the ablation. If not scoped in, nothing
here changes.

## Models and training protocol (pinned)

- Conditional flow: UQ.generative.GenerativeDiscrepancyModel, n_layers 8,
  hidden 64, fit(epochs 400, lr 1e-3, batch 256), the settings pinned by the
  separated study and reused unchanged since.
- Gaussian conditional baseline: UQ.gaussian_modelform.GaussianDiscrepancyModel,
  same features, targets and fit settings; the distribution family is the only
  difference.
- Pooled unconditional Gaussian on the training targets: the labeled
  diagnostic for what conditioning buys, not a criterion baseline.
- Conditioning features: the five Galilean-invariant Pope invariants from the
  baseline strain and rotation with the baseline timescale, extended with the
  turbulent Mach number from the record, six features per point, through the
  committed feature path (UQ.discrepancy.feature_set with the M_t extra). At
  the interaction the invariants are genuinely two-dimensional (no pure-shear
  degeneracy), stated because it changes the effective conditioning relative
  to the attached studies. The training-versus-test M_t spans are recorded in
  the numbers JSON.
- Interior mask for the 2-D interaction fields, pinned: y* in [0.05, 2.0]
  (the lower bound corresponds to y+ about 23 at the incoming Re_tau 466,
  mirroring the attached y+ > 30 convention without dividing by a separating
  friction velocity) and k above 1e-3 of the field maximum (excludes the
  laminar free stream while keeping the shock-amplified region). No post-hoc
  mask relaxation.
- Subsampling, pinned per grid family: test rows at stride (8, 4) on the
  3001 x 284 fields and the adiabatic tiled field, (4, 4) on the 1610 x 230
  fields; training rows at twice the test stride in each direction. This
  targets order 2e4 test rows per field and order 1.5e4 rows per four-field
  training pool, at or above the dense-field volume where the family
  advantage was measured (13189 rows), which is the point of the
  within-interaction leg. Exact counts are recorded, not tuned.
- Sampling: 128 samples per point, independent latents per row a-priori;
  coherent shared-latent field realizations a-posteriori (one latent per
  member), ensemble size 24 per probabilistic method, seed 0, the separated
  protocol unchanged.
- Seeds: model seeds {0, 1, 2}; every metric is the seed mean with the
  min-max range; every criterion is evaluated on the seed mean.
- Training-side adjustments are permitted on training-loss diagnostics of
  training rows only, never on any held-out metric, and recorded in the memo.

## The question (the gate)

Does a conditional, realizable, generative model-form correction of the
stress and heat-flux closures, with the heat-flux reach built into the
coupled solve, deliver calibrated uncertainty and measured error reduction at
a shock-boundary-layer interaction, out of distribution along the wall
thermal axis, without degrading the attached flow.

### A-priori leg: pre-registered positive shape

1. Within-interaction conditional transfer (primary): the
   leave-one-wall-thermal-out case-mean coverage of dq_y at nominal 0.90
   (UQ.evaluation.coverage_from_samples, unweighted mean over the five folds)
   is at least 0.80 with every fold at least 0.60, and the in-sample
   machinery check sits in [0.80, 0.98]. Comparison lines: the Gaussian
   conditional scored identically and the pooled unconditional diagnostic.
2. Family clauses at density (the ceiling-lift question): on the
   leave-one-wall-thermal-out folds the flow beats the Gaussian on the
   multivariate energy score with aggregate per-component CRPS no worse, on
   BOTH the joint (dq_x, dq_y) leg and the db-vector leg (each conjunction
   evaluated on the seed mean over folds). Scalar-leg parity carries no
   family content, as committed.
3. Mid-distribution guard: at nominal 0.50 the held-out case-mean coverage
   lies in [0.35, 0.65] (transfer not bought with vacuous width).
4. Far-transfer characterization (attached-to-interaction): scored with the
   same metrics against the same lines, gated by the attached
   leave-one-Mach-family-out control landing in [0.80, 0.98]; the positive
   shape is case-mean held-out interaction coverage at least 0.80 with the
   upstream-attached region at least 0.80 on every configuration; the
   pre-named intermediate shape is upstream-attached coverage restored while
   the interaction core fails (distance-graded partial transfer, read by
   region); the compound-shift null below is the equally likely outcome and
   is stated in advance.
5. The wall-flux-normalized conformal line (carried forward as the
   established first-line thermal correction): the far-transfer thermal
   residuals re-scored with the split-conformal score normalized by the
   baseline's own predicted local wall heat flux (floor 1e-5 in Stanton
   units; the adiabatic configuration is excluded where the floor binds),
   against the absolute-score control through the identical path. N-plus
   means restoration into [0.80, 0.98] at nominal 0.90; the four-quadrant
   scale-versus-shape grid of the heat-flux study carries over unchanged,
   with the interaction replacing the high-Mach cases.

### A-priori leg: pre-registered null shapes (equally reportable)

- Family null: the flow does not beat the Gaussian at interaction density on
  either joint leg. Reading: the conditionally Gaussian family suffices for
  this discrepancy even dense, and the generative family earns nothing
  a-priori in the regime chosen to favor it; the family question is then
  answered for this program.
- Transfer null (within-interaction): the machinery check passes but
  leave-one-wall-thermal-out fails its bands: (invariants, M_t) conditioning
  does not carry the discrepancy law across wall temperature even inside one
  interaction; the wall-thermal state is then not encoded by the chosen
  features, the concrete feature-enrichment decision point.
- Compound-shift null (far transfer): the attached control is healthy but
  attached-to-interaction coverage collapses: the compound shift exceeds the
  conditioning, as it did for the plates, and the honest reading is that
  attached-trained model-form does not transfer to interactions at this
  feature set, with the region-graded profile reported.
- Degenerate control: if the within-interaction machinery check or the
  attached family control fails its band, the corresponding clause is not
  adjudicated in either direction and the finding is about trainability at
  that data volume, per the committed precedent.

### A-posteriori leg: gates, positive shape, null shapes

Gates A and B of scope decision 2 are prerequisites; if either fails, the
solver-capability null applies. The propagated folds, pinned for cost: the
held-out s = 0.5 (cooling end), s = 1.0 (adiabatic middle, with the
independent-campaign wall series as a second truth surface), and s = 1.9
(heating end); each fold propagates the leave-that-s-out trained models; the
attached-trained model is additionally propagated into the adiabatic
configuration as the far-transfer characterization. 24 coherent members per
probabilistic method per fold, the eigenspace corner family (full projection
1.0 with 0.5 moderation) through the identical injection, non-converged
members counted and excluded with the exclusion stated; a method with fewer
than 18 of 24 converged members is labeled propagation-unstable and its
scores carry that label.

Positive shape, all clauses on the pinned wall-QoI set (Cf, Cp, and St where
defined, at stations x* = -12 to 12 in steps of 1 intersected with each
series' range, plus the scalars: separation, reattachment, shock position as
the half-rise point of the smoothed wall-pressure series, one rule for DNS
and every solve):

1. Coverage: the flow ensemble's 0.90 bands cover at least 0.80 of stations
   pooled per held-out configuration (per-region values reported), and
   contain the scalar QoIs; at nominal 0.50 the pooled station coverage lies
   in [0.35, 0.65].
2. Accuracy: the ensemble-median Stanton error in the interaction region is
   below the baseline SST's on the held-out heated folds, and the
   ensemble-median Cf and Cp errors are below baseline in at least the
   interaction region, reported per region and per configuration, never as
   one flat figure.
3. Sharpness guard: where a band covers, its half-width in the interaction
   region is smaller than the baseline's own point error there (a band wider
   than the error it must cover is vacuous, the overlay lesson of the
   separated study).
4. Baseline comparison: clauses 1 to 3 hold for the flow at least as well as
   for the Gaussian conditional through the identical injection, and the
   eigenspace envelope's containment and uniform-reading scores are reported
   alongside (its structural exclusion from the heat-flux QoIs stated: an
   anisotropy-only method carries no St correction other than through the
   flow-field response).
5. Realizability: the running barycentric check passes every outer iteration
   for every converged member of every method.
6. Wall-thermal generalization: coverage at the cooling and heating end
   folds degrades gracefully from the adiabatic fold (bands widen as
   coverage falls) rather than silently.
7. Preserve-attached control: each fold's closure, propagated through the
   attached Mach 2.28 boundary-layer solve (the gate-A configuration), keeps
   the ensemble-median Cf error within 1.5 times the baseline's and its 0.90
   Cf band half-width below 0.15 of the measured cf (non-vacuous), so the
   correction does not degrade the attached flow it was not asked to fix.

Null and negative shapes (equally reportable, each a decision point):

- Solver-capability null: gate A or B unattainable; the a-posteriori clauses
  are not adjudicable at this solver maturity, reported plainly, and the
  phase verdict rests on the a-priori leg.
- Magnitude-cap recurrence: the coupled response is directionally right but
  the bands cannot reach the truth even with the heat-flux reach; the
  separated diagnosis then survives its motivated fix, which is decisive
  against the deferred-correction injection route as built.
- Shock-foot propagation instability: realizability-projected members
  diverge at the shock foot at rates that gut the ensembles; reported with
  the convergence counts and the foot-region diagnostics.
- Vacuous coverage: clause 1 passes only where clause 3 fails; coverage
  bought with uninformative width, the overlay precedent repeated at the
  interaction.
- No accuracy gain: coverage and sharpness restored but clause 2 fails; the
  correction is a calibrated uncertainty on a baseline-quality prediction,
  reported as exactly that.
- Attached degradation: clause 7 fails; the correction is not deployable
  as-is regardless of interaction gains.

### Thresholds

The bands above ([0.80, 0.98] controls and machinery checks; case mean at
least 0.80 with per-fold floor 0.60; [0.35, 0.65] at nominal 0.50; strict
aggregate energy-score inequality with CRPS-no-worse conjunctions; station
coverage at least 0.80 pooled; the 1.5x and 0.15 attached-control bounds; 18
of 24 convergence labeling; the 1.05 onset factor, the x_r + 2 region edge,
the half-rise shock-position rule, the station grid, the strides and the
interior mask) are fixed here in advance. No model, feature, mask, score,
stride, seed, fold, station set or training choice is tuned toward any of
them.

## Baselines the result is measured against

- Deterministic baseline SST through the density-based solve (the point
  reference and the discrepancy baseline).
- The Gaussian conditional model-form on identical inputs through the
  identical projection and injection (the family axis, a-priori and
  propagated).
- The eigenspace-perturbation corner family at full and moderated projection
  through the same injection (the dominant realizable model-form method),
  scored by envelope containment and the labeled uniform reading; its
  structural inability to represent a heat-flux correction is carried as a
  reportable fact and bounds which legs it can enter.
- The pooled unconditional Gaussian diagnostic (what conditioning buys).
- The wall-flux-normalized conformal score on the far-transfer thermal
  residuals (the established cheap correction, the line any conditional
  model must beat to earn its cost).

## Metrics

UQ.evaluation.coverage_from_samples at nominal 0.90 and 0.50 with sharpness,
crps_ensemble per component, energy_score on the joint and db-vector legs,
reliability_error and PIT diagnostics per fold and per region,
UQ.conformal.conformal_quantile for the normalized-score line; a-posteriori
station coverage, region-resolved QoI errors, convergence counts and
realizability assertions; per fold, per region, per configuration, seed mean
with min-max range; aggregation is the unweighted mean over folds or
configurations, with point-pooled views recorded as diagnostics. Every number
from the fixed-seed reproduce scripts.

## Realizability and invariance (separate checks, stated scope)

Every sampled stress target is projected into the barycentric realizable set
before scoring (a-priori) and before injection with per-iteration re-checks
(a-posteriori). dq samples are scored and injected as drawn, no clipping, no
invented admissibility set, with solver classification as the safety net. The
DNS records' own realizable fractions and the numerically verified Galilean
invariance of the feature construction are separate assembly-test assertions.

## Reproduce plan

Study modules under UQ.datasets (the interaction extraction and splits, the
fit and scoring loops, the coupled ensemble driver), fixed-seed driver
scripts writing one numbers JSON per leg, unit tests on assembly, splits,
injection identity and realizability (training in tests capped at a smoke
epoch), and the evidence package in this directory (this file, the finding
memo, figures, the numbers JSONs). Long solves and ensembles run through the
detached-run pattern with done markers per standing practice.

## Data attribution

All datasets are third-party DNS, not produced by this project, cited where
used: Pirozzoli and Bernardini, AIAA Journal 49(6) (2011) 1307-1312,
doi:10.2514/1.J050901 (adiabatic interaction); Bernardini, Asproulias,
Larsson, Pirozzoli and Grasso, Physical Review Fluids 1 (2016) 084403
(wall-thermal interaction sweep); Pirozzoli and Bernardini, Journal of Fluid
Mechanics 688 (2011) 120-168 and Physics of Fluids 25 (2013) 021704 with the
distribution at reynolds.dma.uniroma1.it/dnsm2 (attached supersonic boundary
layers); Gerolymos and Vallet, Journal of Fluid Mechanics 958 (2023) A19,
Mendeley Data doi:10.17632/wt8t5kxzbs.1, CC BY 4.0 (attached channel matrix);
Coleman, Kim and Moser, Journal of Fluid Mechanics 305 (1995) 159-183; Zhang,
Duan and Choudhari, AIAA Journal 56(11) (2018) 4297-4311,
doi:10.2514/1.J057296. Raw fields stay local and gitignored; only manifest
entries and curated evidence are tracked.

## Dated addendum (2026-07-10): loader-verification amendments, fixed before any extraction

Confirmed 2026-07-10, after the loaders first read the raw files and before
any discrepancy extraction, model fit, or coupled run existed. Every pinned
constant below matches what the data supports; the criteria, splits,
thresholds and claims above are unchanged.

1. Upstream reference station x* = -7 (was -10). Measured with the loaders:
   the heated campaign grows its layer under an adiabatic wall and switches
   to the s-condition INSIDE the saved fields (wall at recovery temperature
   for x* < -10.5, half-switch at -8.97, established by -8, matching the
   paper's switch station converted to reference lengths), and the widest
   interaction onset is -5.16 (s = 1.9). The pre-registered -10 sits on the
   ramp; -7 is post-switch for every case and upstream of every onset.
   Measured reference states at -7: wall temperatures within 0.1 percent of
   s times the recovery value for every s; the adiabatic dataset's
   cf-and-wall-density friction velocity agrees with its tabulated u_tau
   column to 0.02 percent.
2. Landmark source. Onset and shock position keep their pinned rules but are
   read from the field's own wall-pressure row (validated against the
   healthy wall series to 0.02 reference lengths), available for every case
   and every solve. Reason: the s = 1.0 wall series' pressure column is
   unusable as pressure (integer-quantized, drifting 30 percent where the
   zero-pressure-gradient plateau belongs; its cf column is healthy), so
   that case's wall-pressure truth comes from its field wall row and from
   the adiabatic campaign's series, which the coupled leg already names as
   the adiabatic fold's second truth surface.
3. Region refinement for the heated fields. The upstream region splits at
   the measured thermal switch: the scored upstream-attached region is
   post-switch pre-onset; the pre-switch zone (the same adiabatic incoming
   layer in all five cases) is a labeled cross-case consistency view, not a
   scored region. The coupled baseline solves impose the same
   adiabatic-then-isothermal wall switch at the measured station, so
   upstream boundary-condition mismatch cannot masquerade as model-form
   discrepancy.
4. Anchor labeling. The per-dataset modeled-sigma floors use the
   self-consistency anchors (the momentum-integral residual, measured 1.0
   percent; the budget-closure residuals, measured 0.5 to 4.3 percent of
   peak production; the incoming-profile van-Driest residual, 0.4 percent).
   The cross-campaign adiabatic residual (median relative cf 9.3 percent,
   median absolute Cp 0.013 over the shared window) is the between-campaigns
   spread: it floors only comparisons that mix the two campaigns (the
   adiabatic fold's borrowed wall-pressure truth), not within-dataset
   scoring, since it compounds the campaigns' Reynolds-number difference
   with statistical convergence.
5. Scope decision 8 activated. The spatial strain-history feature ablation
   is scoped IN as a labeled secondary a-priori leg exactly as specified
   there: one added conditioning feature, the same splits, no a-posteriori
   use; a null is a finding.

Measured constants recorded for reuse (the loader tests assert them):
recovery wall temperature 1.9318 T_inf (both campaigns agree to 0.1
percent); interaction onsets -2.38 (adiabatic 2011) and -2.46, -2.93, -3.39,
-4.15, -5.16 (s = 0.5, 0.75, 1.0, 1.4, 1.9); the separation bubble growing
monotonically with wall heating (0.4 to 4.1 reference lengths between the cf
sign crossings); the adiabatic 2011 interaction origin at raw x = 51.25 (its
half-rise landmark).
## Dated addendum (2026-07-12): post-audit amendments, fixed before any corrected-baseline extraction

An external code audit (adjudicated 2026-07-12) found defects in the solver and
scoring stack this pre-registration depends on. The amendments below are fixed
BEFORE any baseline, target, or ensemble is computed on the corrected code.
No success band, null shape, fold definition, or clause threshold changes;
what changes is the computational substrate, the estimator definitions, one
baseline's name, one added baseline, and one representation declaration.

1. SST omega-production form pinned. The density-based baseline solver
   assembled the omega-equation production as alpha*rho*S^2, the documented
   misprint of the 2003 SST paper; the specification form
   alpha*rho*min(S^2, 10 betaStar rho k omega / mu_t) is now implemented
   (bounded above by the misprint pointwise; where the limiter actually
   activates on this data is MEASURED, not assumed: every corrected baseline
   reports its limiter activation fraction and a spatial activation map,
   committed with the gate A/B evidence, so the localization to shock feet
   and separated shear layers is a reported result rather than a premise).
   Every quantity this study derives from a baseline
   solve (gates A and B, db and dq targets, injection references, wall QoIs)
   is regenerated on the corrected solver. All interaction baselines, a-priori
   targets, trained models, and a-posteriori ensembles computed before this
   addendum are DISCARDED and never compared against the corrected runs.
   Gates A and B are re-adjudicated on the corrected solver before any
   downstream stage runs.

2. Score estimator conventions pinned. UQ.evaluation.crps_ensemble and
   energy_score now implement the fair (unbiased, M(M-1) off-diagonal)
   finite-ensemble estimators; the previous M^2 forms remain available as
   *_biased. The convention split is pinned by what the members are: every
   sampled predictive in this study (flow draws, Gaussian draws, posterior
   ensembles) is scored with the fair estimators, and every deterministic
   bounding family (the eigenspace variants) is scored by the M^2 plug-in,
   which for a finite discrete forecast is its EXACT CRPS/energy score, not
   a biased estimate; the fair reading of a bounding family is reported as
   sensitivity only, never in a clause. No claim is made that estimator
   choice preserves method orderings (at equal ensemble size the fair and
   plug-in scores differ by a dispersion-dependent term, so reordering is
   possible in principle); instead the convention above is fixed here,
   before any corrected-target score is computed, and every clause uses it.

3. Eigenspace baseline named precisely, and one baseline added. The
   pre-registered "eigenspace-perturbation corner family" is the three-corner
   EIGENVALUE-ONLY method (Emory, Larsson and Iaccarino 2013), and is scored
   exactly as pre-registered. The five-state extension (Iaccarino, Mishra and
   Ghili 2017: the 1C and 2C eigenvalue corners each paired with both
   production-extremal eigenvector alignments, plus the isotropic 3C corner,
   the set {(1C, vmax), (1C, vmin), (2C, vmax), (2C, vmin), 3C},
   UQ.eigenspace.five_state_set) is ADDED as a reported
   baseline on the same folds with the same injection and scoring; it does
   not replace the pre-registered corner family in any clause, and its
   envelope is reported alongside. The structural fact stands unchanged:
   neither variant can represent a heat-flux correction (anisotropy-only),
   so both are absent from the dq legs by construction.

4. Conformal language scoped to its validity. The wall-flux-normalized
   split-conformal line keeps its pre-registered role and thresholds. Its
   description is corrected from guarantee language to what the design
   delivers: the calibration units are whole held-out cases, the score
   quantile is the finite-sample split-conformal quantile, and the coverage
   statement holds under exchangeability of calibration and test cases; the
   cross-Mach and interaction transfers deliberately break exchangeability,
   so measured coverage there is an empirical result with the gap reported,
   not a distribution-free guarantee.

5. Objective representation of the db leg, declared before training. The db
   targets were pre-registered as raw Favre anisotropy components conditioned
   on invariant features, which is not rotation-equivariant. Because every
   target regenerates under amendment 1 and no corrected-target model has
   been trained, the db leg (flow AND Gaussian, identically) now predicts
   integrity-basis coefficients (UQ.discrepancy.basis_coefficients /
   basis_reconstruct) with the same invariant conditioning; scoring clauses
   evaluate the reconstructed tensor components exactly as pre-registered, so
   thresholds are untouched. The representation carries pre-registered
   FEASIBILITY GATES, checked on the corrected a-priori targets BEFORE any
   training and reported in the memo either way
   (UQ.discrepancy.basis_diagnostics): (i) the per-station basis rank and
   condition number over the interaction-region samples are reported (2-D
   mean flows admit at most three independent tensors, so a rank collapse is
   expected structure, not failure); (ii) the DNS db targets must be
   ACHIEVABLE in the basis, gated as median relative reconstruction residual
   at most 0.20 over interaction-region samples, with the 90th percentile
   reported as a diagnostic. If the gate fails, the db leg REVERTS to the
   raw-component parameterization for flow and Gaussian identically, the
   reversion is stated in the memo, and the residual numbers are published;
   the basis is then a documented negative on this data, not a silent swap. The dq legs are vectors under the wall-frame
   convention already fixed by the loaders and are unchanged. The coupling
   masks of the flow are the corrected complementary-alternation masks (the
   shift-and-shrink defect at five components is fixed); the two-component dq
   architecture is bit-identical to the pre-registered one.

6. Procedural integrity carried into this study: ensemble caches carry
   validated configuration fingerprints (a reduced-cost run can never
   masquerade as a production artifact); member records carry convergence
   status, non-converged members never contribute QoIs (the extraction is
   status-gated, not only the scoring); and diverged solves invalidate any
   held fields so no member can inherit its predecessor's state.


## Adjudication note (2026-07-18): gate rulings and corrections, recorded before any claim-bearing run

Recorded by the reviewer's ruling during the post-audit restart, before any
a-priori training or coupled propagation on the corrected implementation.

1. Gate B is adjudicated PER CASE. The document's prerequisite clause reads
   whole-gate; the reviewer ruled per-case on 2026-07-18, noting that the
   pinned propagated folds (s = 0.5, 1.0, 1.9) are unaffected by the one
   failing configuration. A case failing gate B is excluded from
   claim-bearing coupled legs, takes the registered frozen-mean fallback on
   its a-priori role, and its far-transfer-target role is reported as the
   solver-capability boundary. The gates_adjudication.json record in the
   results tree is the machine-readable form, enforced by the drivers.

2. The adiabatic-campaign injection probe is EXPLORATORY. It was run after
   the gate-B baseline was observed and cannot rehabilitate the gate; its
   role is the pre-registered position-error attribution: the converged
   sensitivity of the half-rise to the strongest moderated realizable
   perturbation measured 0.16 reference lengths against the 2.95 offset,
   with two of three bounding directions admitting no steady solution.

3. The adiabatic wall boundary condition is corrected to the solver's
   zero-heat-flux wall. The previous convention prescribed the DNS recovery
   temperature as an isothermal wall, which forces a nonzero modeled wall
   heat flux wherever the model's own recovery differs; that is a
   matched-thermal-state sensitivity, not an adiabatic wall. Every gate and
   baseline quantity for the adiabatic configuration regenerates under the
   honest condition before adjudication. The s = 1.0 campaign REMAINS
   isothermal at its measured recovery row: it is a controlled
   isothermal-wall experiment in its own dataset, adjudicated as such.

4. Gate A is measured in full: skin friction within 10 percent, the
   momentum-thickness Reynolds number reported against the record's own
   value at the same station, and the van-Driest log-region RMS within
   5 percent, with the density convention rho_hat = 1/T_hat and each side's
   own wall stress in the transform, one rule for both sides. Convergence is
   the completed all-equation criterion of the density-based solver
   (every live conservative equation's relative decay below tolerance plus
   a direct per-cell state validation), and non-converged solves never
   populate baseline caches.

5. Dated criterion refinement (2026-07-18, before any corrected gate or
   baseline result exists): a per-equation diagnostic on the attached
   adiabatic configuration showed the all-equation reduction pool
   degenerating through the omega equation, whose volume-scaled residual
   RMS is dominated by wall-adjacent cells (omega scales as nu over y
   squared and the wall rows are re-pinned every iteration), a reference
   scale many orders above the mean flow; pooling it floors every other
   equation into quiescence and reduces the criterion to an omega decay
   unreachable from a wall-slaved scale. The criterion is refined to the
   convention already adjudicated for the compressible SIMPLE solver in
   the post-audit remediation: the four mean-flow conservative equations
   gate on relative residual decay with the quiescent-direction floor
   pooled over their commensurate scales, and the turbulence equations
   gate on carried dimensionless field-change norms measured on the
   accepted post-clamp state, all below the same tolerance, plus the
   direct per-cell state validation. Every solved equation remains in the
   criterion under stated semantics; no gate or baseline artifact
   predating this note was produced under it.

## Dated addendum (2026-07-21): seed-resolved ensembles, lineage, and repair conventions, fixed before any corrected-lineage result

Recorded after the reviewer's ten-finding provenance review invalidated the
first corrected-era a-priori extractions, targets, and coupled rounds
(quarantined, never quoted), and BEFORE any extraction, trained model,
target, ensemble, or score exists on the repaired lineage. No success band,
null shape, fold definition, or clause threshold changes.

1. Seed-resolved ensemble protocol (the estimand made precise). Model
   seeds {0, 1, 2} train separately; the sampling seed stays at the
   registered 0 and is distinct from the model seed; both appear in every
   cache path and record. Each probabilistic method propagates 24 coherent
   members per fold PER MODEL SEED; each 24-member ensemble is scored
   separately; per-seed values are reported with the seed mean and the
   min-max range; every criterion is evaluated on the seed mean. The
   18-of-24 convergence rule applies independently per seed with no
   borrowing of converged members across seeds, and an aggregate is
   labeled propagation-unstable if ANY constituent seed falls below 18 of
   24, with all per-seed counts retained. The 72 members of one method and
   fold are never pooled into one forecast (pooling would change coverage,
   the proper scores, and the registered estimand). Deterministic
   zero-discrepancy controls, eigenspace corners, and five-state baselines
   run once per physical fold, not per seed.

2. Transitive content-hash lineage. Every cache carries a configuration
   fingerprint whose configuration embeds the exact content hash of every
   upstream cache it was built from: converged fields into extractions,
   extractions and conditioning fields into targets, the exact target file
   into each member, targets and members into the fold score. A mutated,
   regenerated, or absent parent therefore invalidates every descendant
   automatically (the stale-extraction failure mode of the review cannot
   recur silently). All cache writes are atomic (temp file and rename).
   This extends amendment 6; it changes no estimator or threshold.

3. Effective running realizability, diagnostic only. The per-iteration
   realizability re-check of the coupled solve is evaluated on the
   EFFECTIVE running anisotropy b_eff(W) = b_B(W) + db_stored, where b_B
   is the solver's own Boussinesq anisotropy at the current state and
   db_stored the injected discrepancy, replacing the frozen absolute-target
   check the review flagged (which measured the pre-projected target, not
   the anisotropy the momentum equation carries). The worst margin and the
   largest violation are recorded with their iteration and cell; b_eff is
   never projected or clipped during the run. Clause 5 of the a-posteriori
   leg reads this diagnostic.

4. Wall transport coefficients at resolved walls are molecular. The
   density-based solver's no-slip wall fluxes (momentum shear, isothermal
   heat flux, k and omega wall diffusion) and the wall observation
   operator previously used the owner-cell eddy viscosity; at the resolved
   first cell (y+ 0.05) the measured eddy-to-molecular ratio is order
   1e-15 to 1e-12, so the change is numerically immaterial and is made for
   correctness, with the physics schema token bumped and an equivalence
   audit (saved-state observation deltas, warm-reconverged fields and gate
   metrics against pinned records) run per configuration before any
   downstream stage.

5. The frozen-mean fallback made concrete. The registered gate-B fallback
   (the DNS Favre mean held fixed, k and omega transport marched to
   steadiness with the same implicit driver) is implemented as a solver
   mode and applied EXACTLY where the 2026-07-18 per-case ruling assigns
   it: the adiabatic 2011 campaign's a-priori surface. Its extraction
   parent is the frozen-mean march, never the gate-failing free-running
   solve; the s = 1.0 fold remains fully eligible; and the s = 1.0 fold's
   second-truth-surface baseline comparison line is the s = 1.0 baseline
   scored on the 2011 series (the excluded adiabatic solve supplies no
   claim-bearing reference). The attached-to-adiabatic far-transfer
   propagation lives in an exploratory results namespace that the formal
   assembler structurally cannot consume, and runs only under an explicit
   opt-in flag.

6. Far-transfer db pool verified and the conformal roles made disjoint.
   The registered db pool (the channel matrix plus the twelve attached
   boundary-layer cases, with the plates' ready-made anisotropy as a
   labeled sensitivity variant) is implemented as registered; the twelve
   staged boundary-layer cases were verified against the data section
   (M2 at eight friction Reynolds numbers, M3 and M4 at two each) with no
   discrepancy. The wall-flux-normalized conformal line's roles are fixed
   at WHOLE-CASE level before any result: the attached channel cases split
   into disjoint fit and calibration sets by a frozen within-Mach-family
   alternation, the conformal predictor trains on the fit cases only, the
   quantile calibrates on the held-out calibration cases, and no
   calibration-case row enters the predictor's training. The far
   deployment mask is saved and fingerprinted with the targets it masks.

7. Early abort retired for formal members. The 15000-iteration / 0.3
   fail-fast rule was calibrated on the confounded pre-repair round, so
   every formal member solve runs abort-off at the full budget. Any
   production re-enablement requires an independent full-budget diagnostic
   panel and a dated amendment fixing the thresholds, before any member it
   governs is scored.

## Dated addendum (2026-07-26): substrate corrections from the Phase L review, fixed before any corrected-lineage result

Recorded after the reviewer's blocking review of the first repair round and
BEFORE any extraction, trained model, target, ensemble, or score exists on
the corrected lineage. No success band, null shape, fold definition, or
clause threshold changes.

1. Frozen-mean sweep isolation. The implicit sweeps of the frozen-mean
   transport mode zero the mean-flow increment rows the moment they are
   computed, before any neighbor coupling consumes them: the pinned mean's
   residual never vanishes, so a phantom mean increment would otherwise
   contaminate the k and omega coupling even though the final mean is
   re-pinned. Pinned by a discriminating test (an artificial mean-row
   source leaves the turbulence march bit-identical); the frozen-mean
   surface regenerates under the corrected sweep.

2. Checkpoint-restart semantics and a quiescence convergence route. The
   converged baselines checkpoint their Venkatakrishnan reconstruction-
   limiter state with the fields, and every member warm-start restores it,
   so a reloaded state resumes the exact discrete operator it converged
   under and carries no limiter-refresh re-transient (measured about two
   orders larger than the remaining-decay drift without the restore). The
   convergence criterion gains a quiescence route: a state whose
   conservative components all stop moving at the full Courant number over
   an accepted step is a discrete fixed point and classifies Converged
   regardless of the relative-decay ratios, which can never classify a
   quiescent restart (their reference IS the tiny restart residual); the
   full-CFL guard keeps a rejection-throttled solve from masquerading as
   quiescent. Before any target generation, zero-correction checkpoint
   round-trip probes run on the gate-A configuration and the three
   propagated folds, and their measured drift is the recorded common-mode
   restart floor of the member ensembles.

   Scope correction, same day (2026-07-26, before any gate record was
   consumed): the quiescence route is RESTRICTED to restored-checkpoint
   restarts. On a cold solve a per-step field change below tolerance does
   not bound the accumulated remaining drift (slow creep), so the
   registered relative-decay criterion must govern cold classification;
   the first cold regeneration measured the unrestricted route stopping
   the attached gate at 12999 iterations versus the registered 47424 and
   the adiabatic case at 50051 versus 77379, while every heated fold hit
   the relative criterion first and reproduced its prior cold trajectory
   bit-exactly. The two affected cases were re-solved under the restricted
   route before any record was consumed.

   Second scope correction (2026-07-27, from the substrate review, before
   any A-prime artifact exists): (i) the quiescence route additionally
   requires an exactly zero injected correction. A nonzero stored
   discrepancy is a fresh force, so its restart is not at a fixed point of
   the injected operator and its slowly developing response can sit under
   the member tolerance per step; forced members therefore converge only
   by the registered relative decay of the injection response, and the
   solve report carries the classification route so a forced member can
   be asserted never to have ridden quiescence. (ii) Prescribed-field
   initializations (member restarts, warm reloads, the frozen-mean march)
   compute the discrete gradients before the first property evaluation:
   the first eddy viscosity of a restart was previously formed at zero
   strain (Bradshaw limiter unbound), a spurious operator transient in the
   first residual that also polluted the member criterion's
   injection-response reference. Uniform cold initializations skip the
   pre-march gradient pass BY CONSTRUCTION and keep their historical
   trajectories bit-identical (with prescribed boundary profiles even a
   uniform field has nonzero boundary-cell gradients, so an unconditional
   pass would have silently shifted the validated cold baselines); the
   frozen-mean surface and the zero-correction checkpoint probes
   regenerate under the corrected restart path.

3. One landmark rule everywhere. The gate and audit paths now read the
   impingement half-rise through the registered rule (the records'
   smoothing width, linearly interpolated level crossing), replacing a
   raw-series grid-index reading whose offsets were quantized at the
   streamwise cell width; gate records regenerate under it. The gate-A
   momentum-thickness integral stops at the layer's own edge (first
   u = 0.99 crossing; the full-height integral accumulated far-field
   noise); the metric remains report-only.

4. Case-level conformal calibration. Per amendment 4's whole-case
   calibration units, the conformal score of a calibration case is the
   0.90 empirical quantile of its normalized row residuals, and the band
   quantile is the finite-sample split-conformal quantile ACROSS the
   calibration-case scores (with about twelve cases at alpha 0.10 that is
   their maximum), never a quantile over pooled correlated rows. The
   conformal predictor runs at the pinned model seeds {0, 1, 2} with the
   criterion on the seed mean, like every other modeled line.

5. Far-transfer db surfaces completed. The far db leg scores the adiabatic
   2011 campaign's frozen-mean db surface as the labeled independent-
   campaign surface, alongside the s = 1.0 pairing and never inside the
   five-case primary mean.

6. Lineage completed and physics token bumped to v4. Wall records bind
   their exact fields cache; gate records, the adjudication, partial and
   assembled numbers files, and fold scores carry identity blocks
   validated at every consumer (a stale partial can never assemble);
   identity-current target files are never rewritten on a resume (a
   rewrite would invalidate completed members); the compiled-binding
   content hash is recorded in every cache as provenance (never identity,
   per the adjudicated code-revision convention; the physics token remains
   the identity lever). All seven baselines and the frozen-mean surface
   COLD-regenerate under sbli-dbns-v4; the warm-reconverged v3 states are
   superseded and never consumed.
