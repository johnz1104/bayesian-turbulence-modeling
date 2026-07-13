# Compressible attached-flow heat-flux and turbulent-Prandtl UQ finding

Evidence package for the compressible attached-flow study: can the heat-flux
discrepancy and the turbulent-Prandtl-number assumption be calibrated with
honest UQ on real compressible attached data, and does the calibrated UQ
transfer across Mach number and wall cooling.

This memo is measured against the pre-registered criterion
(PRE_REGISTRATION.md in this directory, committed before any calibration
result existed). The numbers come from `python/UQ/reproduce_compressible.py`
(fixed seed 0, configuration in that file, ensembles cached under the
gitignored `results/compressible*`); nothing was tuned toward the criterion.
The committed `finding_numbers.json` (primary sigma level) and
`finding_numbers_rel01.json` (sensitivity level) are the traceable source of
every quoted value.

## 1. Pre-registered criterion (fixed first)

Positive shape: (1) standard Bayes overconfident on the held-out thermal
targets (the wall-heat-flux parameter B_q and the heat-flux profile, never in
any likelihood); (2) generalized Bayes and conformal restore held-out
coverage to within a few points of nominal; (3) the Pr_t posterior is a
genuine measurement consistent with the measured flat-plate profiles; (4)
cross-Mach and wall-cooling coverage degrade gracefully (intervals widen),
not silently. Equally reportable null shapes: already-calibrated; Pr_t
unidentifiable from the U, T, cf likelihood; silent transfer collapse.

## 2. Data, baselines and conventions

Data: the Gerolymos-Vallet compressible channel matrix (24 cases, Re_tau* 97
to 985, M_CLx 0.32 to 2.49, isothermal walls; J. Fluid Mech. 958 (2023) A19,
Mendeley CC BY 4.0), the Coleman-Kim-Moser supersonic channel (bulk M 1.5 and
3.0; J. Fluid Mech. 305 (1995); never calibrated on), and the
Zhang-Duan-Choudhari flat plates (M 2.5 to 13.64, Tw/Tr 0.18 to 1.0; AIAA J.
56(11) 2018). All third-party and cited; provenance and the loader-verified
data quirks in DNS_data/README.md. No file carries a per-point statistical
uncertainty: observation sigma is MODELED (0.5 percent primary, 1 percent
sensitivity), per case floored at the loader-measured data-only physics
anchor (the per-unit-mass total-stress balance for the channels, 0.02 to
0.56 percent; the van-Driest reconstruction residual for the plates, 0.06 to
0.9 percent), never called a DNS uncertainty.

Baselines (the confirmed scope decisions): a one-dimensional fully-developed
compressible SST solve per channel case (wall units, per-unit-mass forcing as
the data pins, molecular properties from the case's own profiles, Pr_t in the
energy equation), converging on all 26 channel cases with a Mach-growing
misfit (centreline-velocity error 4.0 / 8.7 / 14.6 percent at M 0.8 / 1.5 /
2.0 at matched Re_tau*); the stated frozen-mean SST-consistent reconstruction
at the plate stations; the in-tree low-Mach solver as the control (its
developing skin friction approaches the 1-D fully-developed value from above
at M 0.32; envelope documented in the baseline tests). Convention: Favre
moments, wall units; theta = (a1, betaStar, Pr_t) with the Menter-centred
truncated-normal SST prior and the pre-registered exactly-uniform Pr_t on
[0.5, 1.5]. Realizability of every DNS stress (barycentric fraction 1.0) and
the numerically verified Galilean invariance of the feature set hold as
separate checks throughout.

## 3. In-distribution result: the correction restores the observables it is
matched on, and only partially the held-out thermal block

Per-case calibration on all 24 channel-matrix cases, nominal 0.90, primary
sigma level, coverage averaged over the matrix (per-case values in the JSON):

| block | standard Bayes | generalized Bayes |
|---|---|---|
| likelihood (U, T, cf)        | 0.479 [0.00, 0.79] | 0.955 [0.79, 1.00] |
| held-out thermal (B_q, q profile) | 0.174 [0.00, 0.73] | 0.417 [0.09, 1.00] |

Standard Bayes is severely overconfident on the held-out thermal block
(clause 1 confirmed: mean 0.17 at nominal 0.90). The moment-matched
generalized-Bayes correction (learning rate matched on the LIKELIHOOD
residuals only, median eta 7.7e-3, range 9.4e-5 to 4.1e-2) restores the
likelihood block to 0.955, but lifts the held-out thermal block only to
0.417. The two cases whose eta happened to be smallest (heaviest tempering)
reach thermal coverage 1.0, confirming the mechanism: a single global
inflation CAN cover the thermal block when large enough, but the
likelihood-matched value usually is not, because the thermal misfit is
structurally larger than the mean-flow misfit it is matched to. This is the
diffuse-inflation limitation the rotating-channel diagnostic exposed for the
stress components in the incompressible program, now measured on the
heat-flux leg: a scalar correction cannot localise to the block where the
structural error concentrates.

## 4. Cross-Mach result: the mechanism widens, the magnitude falls short

Pre-registered primary split (calibrate pooled on the eight M_CLx < 1.0
cases, predict the sixteen M_CLx >= 1.47 cases; pooled eta 2.0e-2), nominal
0.90, averaged over the held-out cases:

| block | standard | generalized Bayes | conformal |
|---|---|---|---|
| likelihood | 0.036 | 0.508 | 0.727 |
| held-out thermal | 0.011 | 0.176 | 0.006 |

The tempered thermal intervals widen by a median factor of 14 relative to
standard Bayes, so the transfer is NOT a silent collapse (the epistemic
mechanism responds); but coverage still falls far below nominal, so it is
not the pre-registered graceful shape either. The conformal interval,
calibrated on the training cases' thermal residuals, collapses to 0.006
under the shift: the thermal residual SCALE grows steeply along the Mach
axis (the wall-flux parameter B_q grows two orders of magnitude across the
matrix), so a constant half-width carried from low Mach cannot cover high
Mach; the conformal coverage gap of 0.894 is exactly the pre-registered
honest exchangeability-violation measure. Leave-one-Re-family-out inside the
matrix shows the same shape (standard thermal 0.068, tempered 0.303).

The never-calibrated external check separates distance from code: the CKM
bulk-M 1.5 case, which sits inside the calibrated matrix's span, is covered
well by the transferred correction (tempered thermal 0.82, likelihood 1.00);
the bulk-M 3.0 case, beyond the span, is not (0.00). The degradation tracks
distance along the Mach axis, mechanistically characterised, insufficient in
magnitude.

## 5. The Pr_t posterior is not a measurement: the pre-registered
identifiability null

Per case, the Pr_t marginal contracts (median posterior sd 0.199 against the
prior's 0.289) but onto CASE-DEPENDENT pseudo-true values that pile at both
prior edges (means spanning 0.52 to 1.48; median 40 percent of posterior
mass within 0.02 of an edge). The pooled posterior over the calibration
cases spans essentially the whole prior box (mean 0.672, sd 0.369, 5-95
range 0.50 to 1.50). Meanwhile the MEASURED flat-plate Pr_t profiles are
tight and mutually consistent (log-region medians 0.883 to 0.935,
interquartile bands about 0.83 to 0.99 across M 2.5 to 13.64 and Tw/Tr 0.18
to 1.0), and the fixed-0.9 convention sits inside every one of them.

The attached mean observables (U, T, cf) therefore do NOT measure Pr_t
through this baseline: the posterior is a misspecification artifact (the
sampler bends Pr_t to compensate the baseline's case-dependent temperature
misfit, hitting whichever box edge helps), not information about the
physical turbulent Prandtl number. This is the pre-registered null shape,
reported as the finding: the fixed 0.9 is unconstrained by attached
mean-flow calibration, while the direct measurement (available only because
the plate files carry Pr_t profiles) is tight and consistent with 0.9.

## 6. The wall-cooling axis (compound out-of-distribution, stated scope)

The channel-calibrated pooled posterior, pushed through the frozen-mean
plate baseline onto the derived wall-normal heat-flux profiles, under-covers
badly across the whole cooling range (standard 0.00 to 0.06; tempered 0.04
to 0.16 at Tw/Tr 0.18 to 1.0): the compound shift (flow type, Mach, and
cooling at once) is far outside what the global correction carries, the same
insufficient-magnitude story as the Mach axis. Pre-registration scope note,
stated not hidden: the plate wall-flux clause is not evaluable through the
approved frozen-mean plate baseline (it predicts no B_q), so the wall-flux
coverage claims rest on the channel matrix, where the 1-D solve predicts
B_q; the plate leg evaluates the heat-flux profile.

## 7. Sigma-level sensitivity

The full pipeline rerun at the 1 percent modeled-sigma level (same seed, same
splits, ensembles regenerated at this level with the same per-case anchor
flooring, so the CKM bulk-M 1.5 case runs at its 1.5 percent anchor; the
committed `finding_numbers_rel01.json` is the source of every value here).

In distribution (24 cases, nominal 0.90):

| block | standard Bayes | generalized Bayes |
|---|---|---|
| likelihood (U, T, cf)        | 0.505 [0.00, 0.91] | 0.944 [0.70, 1.00] |
| held-out thermal (B_q, q profile) | 0.170 [0.00, 0.73] | 0.436 [0.09, 1.00] |

The in-distribution clauses are level-independent: standard-Bayes thermal
overconfidence 0.170 (0.174 at the primary level), the correction restoring
the likelihood block to 0.944 (0.955) but the thermal block only to 0.436
(0.417). The matched learning rate is milder at the wider sigma (median eta
2.9e-2 against 7.7e-3), as expected. The Pr_t identifiability null is
unchanged: median per-case posterior sd 0.198 against the prior's 0.289,
means spanning 0.51 to 1.48, median edge fraction 0.41, while the measured
plate profiles are the same data-only medians 0.883 to 0.935.

Cross-Mach primary split (nominal 0.90, averaged over the sixteen held-out
cases):

| block | standard | generalized Bayes | conformal |
|---|---|---|---|
| likelihood | 0.178 | 0.451 | 0.758 |
| held-out thermal | 0.153 | 0.148 | 0.011 |

The transfer failure and the conformal scale collapse are level-independent
(thermal coverage far below nominal in every column; conformal gap 0.889
against 0.894), and leave-one-Re-family-out matches (standard thermal 0.053,
tempered 0.402). What does NOT recur is the primary level's loud widening:
the tempered thermal intervals change by a median factor of 0.94 (range 0.31
to 1.35), not 14, because the pooled tempered posterior degenerates onto the
upper prior edge (Pr_t 1.4964, sd 0.0036, against the diffuse 0.672, sd
0.369 at the primary level). This is the pooled expression of the
case-dependent edge-piling artifact of section 5, not a new behaviour (the
leave-one-family pools sit near-degenerate at BOTH levels, median sd 0.0006
and 0.002), and it propagates to every consumer of that posterior: the CKM
inside-span advantage shrinks (bulk-M 1.5 tempered thermal 0.27 against
0.82; bulk-M 3.0 stays low at 0.09), and the plate leg reshapes (standard
coverage 0.35 to 1.00 against 0.00 to 0.06, tempered 0.19 to 0.44 against
0.04 to 0.16, with the tempered bands about 12 times NARROWER than the
standard ones where they were 8 times wider at the primary level, the direct
imprint of the collapsed pooled posterior).

The sensitivity reading, stated plainly: every verdict clause of section 8
reads the same at both levels (thermal overconfidence confirmed; the
correction restores the likelihood block only; the Pr_t null; transfer
coverage far below nominal with an essentially unchanged conformal gap). The
one characterisation that does not survive the doubling is the mechanism of
clause 4: the widen-but-undercover shape at the primary level becomes an
edge-collapsed, non-widening shape at 1 percent, so the "loud, not silent"
qualifier is itself sigma-dependent and the sensitivity-level transfer sits
nearer the silent shape. The coverage conclusions and the decision-point
reading are unchanged.

## 8. Verdict against the criterion

Measured clause by clause against the pre-registration:

1. Held-out thermal overconfidence: POSITIVE (confirmed; 0.17 in
   distribution and 0.01 under the Mach shift, at nominal 0.90).
2. Correction restores held-out thermal coverage: NEGATIVE as pre-registered
   for the thermal block (0.42 in distribution, 0.18 under transfer), while
   the same correction restores the likelihood block (0.955 in distribution;
   conformal 0.73 under transfer). The failure is structural, not a missing
   knob: the global scalar inflation is matched to the mean-flow residuals
   and cannot localise to the thermal block where the misfit concentrates.
3. Pr_t a genuine measurement: NULL as pre-registered (case-dependent
   edge-piled pseudo-true values; the pooled posterior fills the prior box;
   the measured profiles are tight around 0.9 and the fixed convention is
   consistent with them).
4. Graceful transfer: CHARACTERISED, between the pre-registered shapes: the
   intervals widen strongly (median factor 14, so not silent), but coverage
   does not stay near nominal (so not graceful); the conformal gap of 0.89
   is the honest measure, driven by the Mach-growth of the thermal residual
   scale; the external CKM check confirms the degradation tracks distance
   along the axis.

The decision-point reading: the compressible attached study confirms the
overconfidence diagnosis and shows that the global coverage correction
established on the incompressible flows does not reach the thermal
observables, in distribution or across Mach. What the thermal block needs is
a correction that reaches the heat-flux quantities themselves, with a
Mach-aware scale, which is the same conclusion the separated-flow model-form
study reached from the injection side and the rotating-channel diagnostic
from the component side. That triangulation, on three independent legs,
fixes the requirement for the shock-interaction direction.

Honest caveats, none of which change the shape: coverage is quantised by the
11-QoI thermal block (steps of 0.09); the ensembles are 64 members per case
with GP surrogates (the same budget family as the incompressible studies;
the overconfidence and the block asymmetry are far outside surrogate error);
the learning rate is matched on the likelihood block BY PRE-REGISTRATION
(matching on the thermal block would leak the held-out targets into the
correction); the conformal transfer uses absolute residual widths, and a
scale-normalised score would be a different, post-hoc method; the baseline's
own Mach-growing misfit is part of the measured misspecification, documented
in the baseline tests, not subtracted.

## 9. Reproduce

`PYTHONPATH=build:python python3 python/UQ/reproduce_compressible.py`
(`--quick` for a smoke run, `--regen-ensembles` to re-run the forward
solves, `--rel 0.01 --results results/compressible_rel01` for the
sensitivity level). Fixed seed 0. The committed `finding_numbers.json` and
`finding_numbers_rel01.json` are byte-for-byte the production outputs; the
figures in `figures/` come from `make_compressible_figures.py`.

## Data attribution

All datasets are third-party DNS, not produced by this project, cited where
used: Gerolymos and Vallet, J. Fluid Mech. 958 (2023) A19 (Mendeley Data,
doi:10.17632/wt8t5kxzbs.1, CC BY 4.0); Coleman, Kim and Moser, J. Fluid
Mech. 305 (1995) 159-183; Zhang, Duan and Choudhari, AIAA Journal 56(11)
(2018) 4297-4311, doi:10.2514/1.J057296. The latter two are hosted by the
migrated NASA Turbulence Modeling Resource (tmbwg.github.io/turbmodels).
Raw fields stay local and gitignored.
## Post-audit revision (2026-07-12): labeling precision

An external audit (adjudicated 2026-07-12) reviewed what each experiment in this study
actually tests. No experiment is withdrawn and no number changes (the calibration matrix
runs on the one-dimensional profile baseline, which none of the solver-level audit fixes
touch); the following labels are sharpened, and the original text should be read through
them.

1. "Held-out thermal" means held-out HEAT FLUX. The likelihood fits the velocity
   profile, the TEMPERATURE profile, and the friction coefficient; the held-out block is
   the wall-heat-flux parameter B_q and the turbulent heat-flux profile, exactly as the
   code's lik_index/heldout_index define. Because temperature is fitted, Pr_t receives
   thermal information directly, and the phrase "held-out thermal" overstates the
   independence of the held-out block. Every "held-out thermal" in this memo should be
   read as "held-out heat-flux (B_q + q profile)". This RENAMING STRENGTHENS the
   identifiability finding within its scope: Pr_t is edge-piled and case-dependent EVEN
   WITH the temperature profile in the likelihood, so the failure to identify Pr_t is
   structural to THIS conditional one-dimensional model class (prescribed wall scale,
   fixed grid, uniform-Pr_t closure) with THIS observable set (U and T profiles plus the
   consistency cf), not a lack of thermal data. Whether a coupled two-dimensional model
   or a richer observable set (the heat-flux profile itself in the likelihood, or
   station-resolved wall fluxes) identifies Pr_t is untested here and is not claimed.

2. The baseline's friction coefficient is not an independent wall-stress prediction. The
   one-dimensional model takes the DNS friction Reynolds and Mach numbers as inputs, so
   the wall-stress SCALE is prescribed; the reported cf is a centreline-dynamic-head
   consistency quantity derived from the predicted profile shape. Statements about cf in
   the likelihood should be read as constraining profile shape, not as independent
   friction prediction.

3. The flat-plate heat-flux reference is a DERIVED quantity (already labeled as such at
   every use): it inverts the dataset's own turbulent-Prandtl definition using the
   measured Reynolds shear and temperature gradient. It is not an independent heat-flux
   measurement; the non-circularity in the comparison is that the truth uses the DNS
   Pr_t profile while the prediction uses the calibrated Pr_t with the model's nu_t.

4. PIT p-values were heuristic. The committed PIT histograms stand as diagnostics; the
   KS p-values quoted against them tested DISCRETE ensemble ranks against a continuous
   uniform and are not calibrated. The evaluation library now provides the randomized
   PIT (exactly uniform under calibration) for any formal test; no committed conclusion
   rested on a PIT p-value.

5. Scores: the library's CRPS/energy estimators are now the fair M(M-1) forms. The
   change from the plug-in is NOT a common shift (it subtracts each method's own
   internal-dispersion term divided by M(M-1), so orderings are not automatically
   preserved). The magnitude is recorded by the estimator identity: fair minus plug-in
   equals half the mean intra-ensemble absolute pair difference divided by (M - 1), so
   the relative correction is bounded by (spread/score)/(M - 1). This study's
   predictive ensembles are the full posterior chains (thousands of members per case),
   and every committed failure mode is UNDER-dispersion (spread at or below the score
   scale), so the correction is below a tenth of a percent of each score and far below
   every committed margin; the committed conclusions stand, and any regenerated leg is
   scored with the fair forms.
