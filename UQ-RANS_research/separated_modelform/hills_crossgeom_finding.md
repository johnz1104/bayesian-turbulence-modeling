# Periodic-hills and cross-geometry model-form evidence (second geometry)

Evidence package for the periodic-hills leg of the separated-flow model-form
study and for the cross-geometry generalization clause, measured against the
criterion of PRE_REGISTRATION.md as operationalized in
METHODS_OPERATIONALIZATION.md sections 8 and 9 (both pinned before the
corresponding results were computed). The backward-facing-step leg is recorded
in bfs_aposteriori_finding.md; this memo completes the second geometry and the
transfer clause, and closes the study's verdict across both geometries.

Every number traces to `hills_numbers.json` in this directory, curated from
`python/UQ/reproduce_separated_crossgeom.py` and
`python/UQ/reproduce_separated_hills_aposteriori.py` (fixed seed 0, production
grids, 400 training epochs, 128 samples per point a-priori, ensemble size 24
a-posteriori, nominal level 0.90). Nothing was tuned toward any criterion.

## 1. Setup

Baseline: the streamwise-periodic SST solve on the curved-bottom mesh, with the
hill surface extracted from the dataset's own blanking mask and the body-force
drive matched so the crest-column bulk velocity is one (achieved 0.999). The
production grid is 60x40, the documented mesh-quality ceiling of the
terrain-following pressure operator; the reattachment is grid-stable against
48x32 (7.62 against 7.66). The baseline converges to x_r/h = 7.617 against the
DNS reading 4.684, an over-prediction of 63 percent. Both geometries'
baselines over-predict the reattachment, but the hills error is an order
larger than the corrected step baseline's 5.3 percent, so this geometry
carries the model-form headroom the step lacked.

Discrepancy and features: 13189 (feature, db) pairs at the pinned stride-3
interior points, b_DNS gradient-free from the DNS stress, b_baseline and the
five invariant features from the converged RANS field, limiter-consistent
conventions identical to the first geometry. The training discrepancy
magnitude grows monotonically downstream (band means 0.164 to 0.245), largest
where the shear layer reattaches and recovers.

Injection: the same explicit deferred-correction force as the first geometry,
with every target projected into the barycentric realizable set before
injection and re-checked each outer iteration. The no-injection reference
through the injection wrapper reproduces the baseline solve exactly
(x_r 7.617).

## 2. A-priori checkpoint on the dense field

In distribution (train on all six bands, score all points):

| model | coverage at 0.90 | CRPS | energy score | realizable fraction |
|---|---|---|---|---|
| generative flow | 0.876 | 0.0138 | 0.0445 | 1.000 |
| Gaussian model-form | 0.907 | 0.0121 | 0.0408 | 1.000 |

Leave-one-band-out (the pinned held-out unit), mean over components:

| held-out band | flow coverage | gauss coverage | flow CRPS | gauss CRPS |
|---|---|---|---|---|
| 0 | 0.694 | 0.662 | 0.0245 | 0.0252 |
| 1 | 0.844 | 0.668 | 0.0171 | 0.0239 |
| 2 | 0.818 | 0.712 | 0.0139 | 0.0154 |
| 3 | 0.897 | 0.846 | 0.0114 | 0.0116 |
| 4 | 0.765 | 0.662 | 0.0159 | 0.0167 |
| 5 | 0.616 | 0.667 | 0.0244 | 0.0234 |
| mean | 0.772 | 0.703 | 0.0179 | 0.0194 |

On the dense field the ordering of the first geometry's in-distribution
checkpoint reverses under the held-out-band protocol: the flow covers better
than the Gaussian on five of six held-out bands and wins the aggregate CRPS
(0.0179 against 0.0194) and energy score (0.0567 against 0.0624). Both
families under-cover held-out bands (the same covariate-shift signature as
the held-out stations of the first geometry), worst at the domain ends where
the periodic wrap places the crest. With 13189 dense training pairs the
conditional flow is the better family; on the first geometry's 1077 sparse
station points it was not. Figure: `figures/hills_apriori_coverage.png`.

## 3. Cross-geometry a-priori transfer (the precursor of clause 4)

Train on all valid points of one geometry, score on all valid points of the
other, both directions, both families, identical fit / sample / project /
score path; w is the mean 90 percent interval width:

| direction | model | coverage | w | CRPS | energy score |
|---|---|---|---|---|---|
| within hills (anchor) | flow | 0.876 | 0.068 | 0.0138 | 0.0445 |
| within hills (anchor) | gauss | 0.907 | 0.067 | 0.0121 | 0.0408 |
| step to hills | flow | 0.365 | 0.105 | 0.0294 | 0.0962 |
| step to hills | gauss | 0.399 | 0.101 | 0.0255 | 0.0827 |
| within step (anchor) | flow | 0.948 | 0.149 | 0.0262 | 0.0868 |
| within step (anchor) | gauss | 0.920 | 0.133 | 0.0275 | 0.0937 |
| hills to step | flow | 0.690 | 0.077 | 0.0403 | 0.1375 |
| hills to step | gauss | 0.661 | 0.108 | 0.0399 | 0.1343 |

Every sampled prediction in every direction is realizable (fraction 1.0).
Three measured facts:

1. The transfer is not the pre-registered silent collapse, but it is not
   graceful either. Transferring the step-trained models to the hills widens
   the intervals by roughly 1.5x (0.068 to 0.105 for the flow) yet coverage
   still falls from 0.88 to 0.37: the widening points the right way and
   under-estimates the shift by a wide margin. The honest reading sits
   between the two pre-registered shapes and is recorded as such.
2. The transfer is strongly asymmetric. Dense-to-sparse (hills to step) holds
   0.66 to 0.69 coverage where sparse-to-dense reaches only 0.37 to 0.40. The
   hills-trained flow is in fact SHARPER on the step (w 0.077) than the
   step's own fit (w 0.149) while still covering 0.69 of it: the dense
   training set produces a tighter, better-placed conditional than the sparse
   geometry can produce for itself.
3. The two families degrade together. No family transfers coverage; the flow
   and the Gaussian land within a few points of each other in both
   directions, so the transfer gap is a property of the feature-conditioned
   discrepancy law itself (the five invariants do not separate the two
   geometries' discrepancy regimes), not of the distribution family.

Figure: `figures/hills_crossgeom_transfer.png`.

## 4. A-posteriori result on the hills (24 coherent members per method)

Reattachment (truth 4.684; baseline point error 2.93):

| method | converged | 90 percent band / envelope | contains truth | ensemble-mean error | CRPS |
|---|---|---|---|---|---|
| generative flow | 24/24 | [6.22, 7.66] | NO | 2.11 | 1.81 |
| Gaussian model-form | 23/24 | [5.52, 7.82] | NO | 1.67 | 1.27 |
| eigenspace, Delta_B = 0.5 | 2/3 | [5.30, 7.89] | NO | (envelope) | 1.27 (uniform reading) |
| eigenspace, Delta_B = 1.0 | 0/3 | none | none | none | none |

Mean streamwise velocity at the pinned probes (22 admissible points):

| method | probe coverage | sharpness | CRPS | energy score |
|---|---|---|---|---|
| generative flow | 0.182 | 0.018 | 0.0377 | 0.233 |
| Gaussian model-form | 0.273 | 0.032 | 0.0366 | 0.225 |
| eigenspace, Delta_B = 0.5 | 0.455 | (2 members) | 0.0412 | 0.242 |

Realizability holds for every converged member of every method (the
barycentric check passes every outer iteration, zero violations); the
exclusions are one Gaussian member and, at the corner projections, one member
of the moderated family and all three of the full family. The white-latent
diagnostic under-disperses as pinned (flow band [6.15, 7.55], Gaussian
[5.96, 6.42]) and is not a usable band.

Four measured facts on this geometry:

- Both probabilistic ensembles move the reattachment TOWARD the truth
  (baseline 7.62, ensemble means 6.79 and 6.36 against 4.68) and both reduce
  the baseline point error (2.11 and 1.67 against 2.93). The direction of the
  coupled response is opposite to the first geometry, where the same
  construction displaced the prediction away from an already-accurate
  baseline. The correction direction is therefore not a fixed property of the
  method; it follows the training discrepancy of the geometry.
- The correction magnitude is systematically too small. No method's band or
  envelope reaches a truth 63 percent below the baseline; the closest
  approach anywhere is the moderated two-component corner at 5.30. The
  injected force is div(2 k db) with the RUNNING k, and the baseline
  under-resolves the separated-shear-layer turbulence energy, so the
  achievable mean-flow displacement is capped by the same closure deficit
  the correction is meant to represent. This is the anisotropy-only
  partial-substitution limit already flagged on the first geometry, now
  measured at large error headroom.
- The dominant model-form method fails outright at its standard setting: the
  full corner projection admits NO steady solution on the production grid
  (all three corners diverge), so the eigenspace envelope exists only at the
  moderated half projection.
- The separation point is injection-insensitive (every method's band sits
  within [0.34, 0.37] against the truth 0.27): the anisotropy correction
  cannot move a separation fixed by the crest geometry and pressure
  gradient, so the bubble-length error is carried almost entirely by
  reattachment.

The probe coverage row is the sharpest overconfidence signal of the study so
far: at nominal 0.90 the best method covers 0.27 of the mean-velocity truth
(the standing coverage-correction layer was pinned for the first geometry's
measured wall-friction stations and has no pinned form for this geometry, so
none is applied here; composing it is follow-on scope, and it would repair
coverage only, not the proper scores). Figure:
`figures/hills_reattachment_intervals.png`.

## 5. Cross-geometry propagation (the propagated clause 4)

Models trained on all valid points of one geometry, sampled at the other
geometry's solver cells through the five invariant features, injected through
the other geometry's production forward model, 24 members, seed 0:

Into the hills (truth 4.684):

| trained on the step | converged | band | contains truth | mean | CRPS |
|---|---|---|---|---|---|
| flow | 22/24 | [0.94, 7.53] | YES | 3.05 | 1.97 |
| Gaussian | 24/24 | [5.23, 7.57] | NO | 6.85 | 1.71 |

Into the step (truth 6.28; step baseline 6.611, its point error 0.33):

| trained on the hills | converged | band | contains truth | mean | CRPS |
|---|---|---|---|---|---|
| flow | 24/24 | [6.31, 7.04] | NO | 6.73 | 0.316 |
| Gaussian | 24/24 | [6.40, 7.03] | NO | 6.73 | 0.331 |

The propagated transfer is asymmetric in the same direction as the a-priori
transfer, and each direction fails differently:

- Sparse to dense: the step-trained flow is the ONLY configuration in the
  study whose band contains the hills truth, and the containment is bought
  with near-vacuous dispersion (member spread 0.94 to 7.53, standard
  deviation 3.0, the worst reattachment CRPS of any method on this
  geometry). Two members lose almost the whole bubble. This is the
  a-priori under-coverage of section 3 surfacing a-posteriori as wild
  member-to-member variation, not as a usable predictive band.
- Dense to sparse: tight, stable ensembles (24/24 converged) that move the
  step's prediction the wrong way (up from 6.61 against a truth of 6.28),
  the same directional failure as the step's own within-geometry ensembles.
  The notable measured fact: the hills-trained flow scores CRPS 0.316 on the
  step reattachment, better than the step's OWN within-geometry flow (0.651)
  and level with the step's own Gaussian (0.318). A dense-field training set
  transfers a better-behaved conditional to the sparse geometry than the
  sparse geometry's own six stations support. Station-Cf coverage stays at
  the step's own low level (0.20).

Every sampled and injected member in both directions is realizable in the
running solve. Figure: `figures/hills_crossgeom_transfer.png` (a-priori) and
`figures/hills_reattachment_intervals.png` (propagated bands).

## 6. Verdict against the pre-registered criterion

The positive shape required all four clauses; on this geometry the generative
flow meets clause 3 and fails clauses 1 and 2, and clause 4 is now
characterized in full:

1. Coverage and accuracy: FAILED for the flow. Its within-geometry band
   [6.22, 7.66] does not contain 4.684. The ensemble mean does reduce the
   baseline point error (2.11 against 2.93), as does the Gaussian's (1.67),
   but no method delivers containment, so the conjunction fails for every
   method on this geometry.
2. Proper scoring rules: FAILED for the flow. On reattachment CRPS it is
   worse than both baselines (1.81 against 1.27 for the Gaussian and 1.27
   for the moderated eigenspace uniform reading); on the probe field it
   beats the eigenspace but not the Gaussian.
3. Realizability in the running solve: PASSED for every method, every
   converged member, both within-geometry and transferred.
4. Cross-geometry generalization: characterized honestly in both directions,
   a-priori (section 3) and propagated (section 5). It is not the
   pre-registered silent collapse (the intervals widen under transfer, and
   the transferred models remain realizable and mostly convergent), and it
   is not graceful either: sparse-to-dense under-covers a-priori and turns
   near-vacuous a-posteriori, while dense-to-sparse under-covers with
   wrong-directed tight bands yet improves the sparse geometry's own
   proper scores.

Combined with the first geometry (bfs_aposteriori_finding.md: clauses 1 and 2
failed, clause 3 passed), the study's overall verdict matches the
pre-registered null or negative shape on both geometries: the conditional
generative model-form, propagated through the solver as an anisotropy-only
deferred correction, does not beat the eigenspace-perturbation envelope or
the Gaussian model-form on the proper scores, and does not restore coverage.
Per the pre-registration this is a real result, reported plainly.

What the second geometry adds beyond confirming the shape: the failure is not
a small-headroom artifact. With 63 percent of baseline error available, every
method's coupled response is directionally right but magnitude-capped
(section 4), the dominant envelope method cannot produce its standard-setting
envelope at all, and the mean-velocity field is severely over-confident for
every method. The measured pattern points at the shared injection channel
(anisotropy-only correction against baseline turbulence transport, the
partial-substitution limit of Wu et al. 2019) rather than at the conditional
distribution family; the a-priori leg (section 2), where the flow beats the
Gaussian under the held-out-band protocol, supports the same reading. A
correction that also reaches the turbulence-transport quantities (the
turbulence energy in the injected stress, or the production term itself) is
the concrete follow-on this evidence motivates, and it is a scope decision to
take before any such training runs, not an adjustment to this study.

## 7. Reproduce

```
PYTHONPATH=build:python python3 python/UQ/reproduce_separated_crossgeom.py
PYTHONPATH=build:python python3 python/UQ/reproduce_separated_hills_aposteriori.py --sensitivity
python3 UQ-RANS_research/separated_modelform/make_hills_figures.py
```

Fixed seed 0 throughout; raw outputs cache under the gitignored
`results/separated/`; the curated numbers here are `hills_numbers.json`.
## Post-audit revision (2026-07-12)

An external code audit (adjudicated 2026-07-12; see the root post-audit report) touched
four aspects of this study. The original text above is preserved unchanged; this section
supersedes it where stated. The pre-registered verdict is NOT withdrawn: the coverage
clauses still fail for the flow on both geometries, and the diagnosis (the anisotropy-only
injection is magnitude-capped by the running k) is unchanged by every item below.

1. Erratum: two contaminated hills member records. The incompressible forward model left
   its cached fields untouched when a solve ended in DivergenceDetected, and the hills
   wrapper extracted bubble geometry without gating on status, so two diverged members in
   hills_numbers.json carry wall QoIs bit-identical to the immediately preceding converged
   member (the flow ensemble's diverged member with iterations 3757 following the
   converged member with iterations 8536, and the corresponding pair in the gauss
   ensemble). The scored statistics are NOT affected: every coverage, band, CRPS, and
   envelope quantity filtered members to status Converged, so the stale values never
   entered a reported number. The committed JSON is preserved as the record of what ran;
   the defect is fixed at the source (fields cleared on divergence, extraction
   status-gated, regression-tested), and any regenerated ensemble uses the fixed path.

2. Score conventions stated precisely, and the comparison recomputed under both. The
   committed CRPS/energy values were computed with the M^2 (diagonal-included) plug-in
   estimator under a "fair" docstring. The correction splits by what the M members ARE.
   For the flow and Gaussian ensembles the members are an iid sample of an underlying
   predictive, so the fair M(M-1) estimator (Ferro 2014) is the right convention and is
   now the library default; the recomputation from the committed member records moves
   those scores by under three percent (BFS reattachment CRPS, truth 6.28: flow 0.651 to
   0.634 at M = 23, Gaussian 0.318 to 0.306 at M = 24; hills: flow 1.808 to 1.795,
   Gaussian 1.271 to 1.253) and no conclusion moves. For the eigenspace corner families
   the members are not a sample of anything: the family IS the forecast, a finite
   discrete distribution, and for a discrete forecast the M^2 plug-in is the EXACT CRPS,
   not a biased estimate. The committed corner values therefore stand as exact
   discrete-forecast scores (BFS delta 1.0: 0.839; delta 0.5: 0.380; hills delta 0.5:
   1.265). The "fair" numbers for the corner families (0.671, 0.092, 0.619) answer a
   hypothetical (the CRPS of an imagined predictive the corners sample) and are kept in
   fair_scores_recompute.json as a sensitivity column only; the biased columns reproduce
   the committed values to the last digit, validating the read.

   The primary pre-registered reading for the deterministic families remains envelope
   containment (METHODS_OPERATIONALIZATION.md section 9), with CRPS on them secondary.
   Under the corrected conventions the committed comparative statement is unchanged: the
   flow does not beat the moderated corner family on reattachment CRPS on either
   geometry (BFS 0.634 fair vs 0.380 exact; hills 1.795 fair vs 1.265 exact). One
   ranking is convention-dependent and is reported as such: on the hills the Gaussian
   fair score (1.253) and the corner family's exact score (1.265) are numerically close
   (no Monte Carlo uncertainty analysis backs a stronger word), so "which non-flow
   method scores best there" has no convention-robust answer; on the BFS the corner
   family leads under either convention.

3. Precise naming of the eigenspace baseline. What this study ran is the THREE-CORNER
   EIGENVALUE-ONLY perturbation of Emory, Larsson and Iaccarino (2013): eigenvalues moved
   to the barycentric corners, eigenvectors preserved. Sentences reading "the dominant
   model-form method" should be read as naming that 2013 variant, not the five-state
   extension of Iaccarino, Mishra and Ghili (2017), which additionally permutes
   eigenvectors to production-extremal alignment and is reported to improve bounds. The
   2017 five-state family is now implemented (UQ.eigenspace.five_state_set) and enters
   the corrected-solver BFS probe below; the full-projection non-existence result on the
   hills grid (no steady solution at delta_B = 1.0) is a statement about the eigenvalue
   corners and is unaffected by eigenvector permutation at the same corners' amplitude.

4. Solver corrections and the corrected-solver probe. Three solver-level audit fixes move
   separated baselines specifically: the SST omega production now uses the limited
   specification form (production-reducing exactly in separated shear layers), the omega
   cross-diffusion source is no longer clipped, and the baseline momentum operator now
   assembles the full Boussinesq deviatoric stress (the variable-viscosity transpose
   term, identically zero in the attached calibrations, is not zero here). Two distinct
   exactness statements are kept apart. The DELTA-SOURCE algebra was verified exact and
   stands: the injected force is -div(2 k db) pointwise, it telescopes against the
   baseline stress operator whatever that operator contains, and a zero correction
   recovers the baseline solve bit for bit (the audit's contrary reading of a
   cancellation defect is incorrect). The TOTAL modeled stress is a different matter:
   before the operator completion the equation the injected solve satisfied was the
   baseline's incomplete stress divergence plus the exact delta source, so the total
   target stress -div(2 nu_t S - 2/3 k I + 2 k db) is represented exactly only on the
   corrected solver. The magnitude-cap diagnosis rests on the delta-source algebra and
   the running-k scale, not on total-stress exactness, so it stands as written; the
   corrected operator is one of the reasons the probe below is run at all. A BFS probe
   on the corrected solver (baseline + flow/Gaussian injection ensembles + three-corner
   AND five-state families, fair scoring for the sampled ensembles and exact discrete-
   forecast scoring for the bounding families, status-gated members) re-tests the
   verdict. As of this revision (2026-07-13) THE PROBE HAS NOT BEEN RUN: this section
   records the protocol before its outcome is observed, the probe result will be
   reported in a separate follow-up pull request, and the hills regeneration decision
   (regenerate only if the probe moves a conclusion rather than a number) is likewise
   pre-committed here.
