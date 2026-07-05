# Compressible attached-flow heat-flux and turbulent-Prandtl UQ: pre-registration

Fixed 2026-07-04, before any calibration result was computed on the real
compressible data. This file records the success and null criteria in advance,
so they are timestamped ahead of every result and are not tuned toward
afterwards (the no-tuning-toward-the-number rule). It is committed first.

## The work

The bridge into the compressible regime. The coverage-correction study
(channel), the cross-flow study (Couette), and the separated model-form study
(pre-registered null/negative verdict, with the injection-channel diagnosis)
are complete and merged. This work brings the temperature and heat-flux
machinery up on real compressible ATTACHED flows, calibrates the
turbulent-Prandtl-number and heat-flux-closure assumptions with honest UQ, and
characterizes cross-Mach and wall-cooling transfer. It is a-priori by design:
the in-tree compressible solver is low-Mach (ceiling about Ma 0.5), so no
coupled compressible claim is made; the solver serves only as the documented
low-Mach control.

Data: the Gerolymos-Vallet compressible channel matrix (24 cases, Re_tau* 97
to 985, M_CLx 0.32 to 2.49, isothermal walls; the workhorse), the
Coleman-Kim-Moser supersonic channel (bulk M 1.5 and 3.0; independent-code
cross-check), and the Zhang-Duan-Choudhari flat plates (M 2.5 to 13.64, Tw/Tr
0.18 to 1.0; the hypersonic extension, the wall-cooling axis, and the MEASURED
turbulent-Prandtl-number reference profiles). Provenance in DNS_data/README.md.

## Scope decisions taken at the start (confirmed 2026-07-04)

1. Baseline route (high-Mach, profile-based by necessity). A one-dimensional
   fully-developed compressible SST solve on the DNS mean thermodynamics: U,
   T, k, omega as a boundary-value problem with variable rho(T) and mu(T),
   isothermal walls, gas model consistent with the data (constant gamma and
   cp), and the turbulent Prandtl number in the energy equation, so Pr_t is
   identifiable through the predicted temperature profile. The flat plate has
   no fully-developed limit, so its baseline is a frozen-mean SST-consistent
   algebraic reconstruction at the analysis station, stated as such. The
   in-tree compressible SIMPLE solver (validated at Ma 0.1, ceiling about Ma
   0.5) runs the three GV M_CLx 0.32-0.35 cases as an independent low-Mach
   control on the 1-D baseline; its Mach ceiling is documented and nothing
   above it is claimed a-posteriori.
2. Calibration observables. The likelihood sees the mean velocity profile,
   the mean temperature profile, and the skin friction cf. The wall heat flux
   B_q and the turbulent heat-flux profile are NEVER in the likelihood: they
   are the held-out predictive-coverage targets. Inference parameters: Pr_t
   plus the SST coefficient set used by the incompressible studies (a1,
   betaStar as primary; the wider set as a robustness variant).
3. Averaging and normalization. Favre (density-weighted) moments in wall
   units are THE discrepancy convention, converted once at the loaders:
   stress R_ij = <rho u_i"u_j">/<rho>, turbulent heat flux q_hat_i =
   <rho h"u_i">/(<rho> cp_w u_tau T_w), temperature T/T_w, semi-local y*
   carried as a diagnostic and conditioning feature, Reynolds moments kept
   only as cross-check views where a file provides them.

## Observation uncertainty (modeled, pinned here)

No compressible file carries a per-point statistical uncertainty. Observation
sigma is MODELED: a relative value, 0.5 percent of the local scale as the
primary level and 1 percent as the sensitivity level, labeled modeled and
never called a DNS uncertainty. Per case the effective level is
max(level, anchor rms), where the anchor is the dataset's data-only physics
residual measured by its loader:

- GV channel: the variable-density total-stress balance under the
  per-unit-mass forcing form (the form the data itself pins: the outer-region
  balance closes two orders of magnitude better than under uniform forcing),
  evaluated outside the buffer layer (y+ > 40, where the neglected
  viscosity-fluctuation correlation does not enter). Measured across the 24
  cases: 0.02 to 0.56 percent (the Re_tau* 985 case is the 0.56; its
  effective primary level is therefore its own anchor).
- CKM channel: the same total-stress identity, with the forcing form pinned
  from the data at loader time and the level recorded in the loader tests.
- ZDC flat plate: the van-Driest-transform reconstruction residual
  (recomputing u_VD+ from the file's own u+ and density columns against its
  u_VD+ column; measured median 0.06 to 0.9 percent per case). The anisotropy
  trace identity (b11+b22+b33 = 0, measured 4e-10) is a parse guard, exact by
  construction, not the anchor.

## The question (the gate)

Can the heat-flux discrepancy and the turbulent-Prandtl-number and
heat-flux-closure assumptions be calibrated and given honest UQ on real
compressible attached data, and does the calibrated UQ transfer across Mach
number and across wall cooling.

### Pre-registered splits (fixed before any result)

- In-distribution: per-case calibration on the GV matrix at the primary
  sigma level; coverage evaluated on the held-out thermal QoIs of the same
  cases.
- Cross-Mach (primary axis): calibrate on the GV cases with M_CLx < 1.0
  (eight cases), predict the M_CLx >= 1.47 cases (sixteen); plus
  leave-one-Mach-family-out within the matrix (families at fixed nominal
  Re_tau*). Graceful-versus-silent characterized by whether intervals widen
  as coverage degrades.
- Independent-code check: the CKM cases are never calibrated on; the
  channel-matrix posterior predicts them (M 1.5 overlaps the GV span, M 3.0
  sits just outside it) as an external cross-check of loader, baseline and
  discrepancy conventions.
- Wall-cooling axis (bonus, strong OOD): the channel-calibrated posterior
  predicts the flat-plate derived wall-normal heat-flux profile and B_q
  across Tw/Tr 0.18 to 1.0 through the frozen-mean plate baseline; this
  compounds flow-type, Mach, and cooling shifts and is reported as such. The
  M2p5 case (Tw/Tr = 1.0, B_q about 0) is the thermally quiescent control.
- Pr_t reference: the marginal Pr_t posterior is compared against the five
  MEASURED flat-plate Pr_t profiles (their log-region bands read from the
  data at evaluation time), and against the fixed-0.9 convention.

### Pre-registered positive shape

1. Standard Bayes (eta = 1) is overconfident on the held-out thermal QoIs:
   empirical coverage of nominal 90 percent intervals on the wall heat flux
   and the heat-flux profile sits well below nominal, as the incompressible
   studies measured on their QoIs (about 0.05 to 0.33).
2. Generalized Bayes (learning rate moment-matched on a calibration split
   only, never on evaluated coverage) and split conformal restore held-out
   coverage to within a few points of nominal at the sharpness the
   misspecification requires, without inflating the observation floor.
3. The Pr_t posterior is a genuine measurement: its spread contracts
   meaningfully below the prior, and its mass is consistent with the
   flat-plate measured Pr_t log-region bands rather than pinned to 0.9.
4. Cross-Mach and wall-cooling coverage degrade gracefully and
   characterizably (intervals widen; the conformal gap is reported), not
   silently.

### Pre-registered null or negative shapes (equally reportable)

- Standard calibration is already calibrated on the thermal QoIs: a real
  finding and a decision point (it would push the coverage story to the
  shock-interaction regime, where misspecification is severe).
- Pr_t is unidentifiable from the U, T, cf likelihood (posterior spread at
  the prior spread): the identifiability finding, reported plainly; the
  fixed-0.9 convention is then unconstrained by attached mean observables.
- Silent cross-Mach or cross-cooling collapse (coverage falls without the
  intervals widening): negative for the transfer claim.

### Thresholds

Shape-based, consistent with the incompressible pre-registrations ("well
below nominal", "within a few points", "graceful, not silent"). No numeric
threshold is fixed in advance, and no learning rate, conformal score, feature
set, prior, or observable weighting is tuned toward any test number. The Pr_t
prior is fixed before any calibration run: uniform on [0.5, 1.5] (a
weakly-informative box around the 0.9 convention spanning the measured
attached-flow range).

## Baselines the result is measured against

- Deterministic compressible SST baseline (the 1-D solve) at fixed Pr_t =
  0.9: the point-accuracy and GDH reference.
- Standard Bayesian calibration (eta = 1): the overconfidence reference.
- Generalized Bayes and split conformal: the corrected UQ under test,
  composed exactly as the incompressible studies composed them.

## Metrics

Empirical coverage of nominal intervals (0.5 and 0.9), sharpness at matched
coverage, CRPS, reliability diagrams, PIT histograms; the Pr_t posterior
summary against the measured profiles; per-case and per-axis (Mach, Tw/Tr)
breakdowns; realizability of every DNS stress (fraction 1.0 expected) and the
Galilean-invariant feature construction as SEPARATE checks; every number from
fixed-seed reproduce scripts.

## Data attribution

All three datasets are third-party DNS, not produced by this project, cited
where used: Gerolymos and Vallet, J. Fluid Mech. 958 (2023) A19 (Mendeley
Data, CC BY 4.0); Coleman, Kim and Moser, J. Fluid Mech. 305 (1995) 159-183;
Zhang, Duan and Choudhari, AIAA Journal 56(11) (2018) 4297-4311. Hosted
copies via the migrated NASA Turbulence Modeling Resource where noted in
DNS_data/README.md. Raw fields stay local and gitignored.
