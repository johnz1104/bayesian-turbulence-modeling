# Compressible heat-flux model-form finding

Evidence package for the heat-flux model-form study: does a conditional
generative model of the heat-flux discrepancy, trained on low-Mach attached
cases and conditioned on (invariants, M_t), deliver calibrated conditional
coverage on held-out high-Mach cases, where the committed global correction
failed; does it beat the Gaussian conditional baseline on proper scores; and
does a physically normalized conformal score alone restore the cross-Mach
transfer of the global correction (the scale-versus-shape question).

This memo is measured against the pre-registered criterion
(PRE_REGISTRATION.md in this directory, committed before any training run).
The numbers come from `python/UQ/reproduce_heatflux_modelform.py` (driver
seed 0, model seeds {0, 1, 2}, every criterion read on the seed mean as
pinned; outputs cached under the gitignored `results/heatflux/`). The
committed `finding_numbers.json` here is byte-for-byte the production
output and the traceable source of every quoted value. Nothing was tuned
toward the criterion; no training-side adjustment was needed or made.

## 1. Pre-registered criterion (fixed first)

Positive shape: (1) the leave-one-case-out control over the eight training
cases lands in [0.80, 0.98] at nominal 0.90 AND the flow's held-out
high-Mach case-mean coverage is at least 0.80 with every case at least 0.60;
(2) on the joint (dq_x, dq_y) leg the flow beats the Gaussian on the energy
score with wall-normal CRPS no worse; (3) at nominal 0.50 the held-out
case mean lies in [0.35, 0.65]; (4) the never-trained supersonic-channel
cases and the plates are characterization legs. Equally reportable null
shapes: the family null, the transfer null, and the degenerate control (the
control itself fails its band, in which case no transfer claim of either
sign is made and the finding is about trainability at this data volume).
Secondary axis: N+ means the B_q-normalized conformal score restores
held-out thermal coverage into [0.80, 0.98]; the four-quadrant grid was
assigned readings in advance.

## 2. Data and protocol as executed

All 31 cases assembled, none skipped: the 24-case channel matrix, the two
independent-code supersonic-channel cases, the five flat plates
(third-party DNS, cited below; provenance and data quirks in
DNS_data/README.md). Committed interior mask (y+ > 30, y/delta < 0.9, plus
the derived-flux valid mask on the plates); the DNS stress record of every
case passes the barycentric realizability check (fraction 1.0), and the
feature construction is Galilean invariant by construction, as separate
checks. The training volume is the pre-registered known risk realized: the
eight low-Mach cases supply 434 interior rows (25 to 174 per case), against
13189 on the dense separated-flow field where the flow family won and 1077
on the sparse one where it did not. The conditioning extrapolation is deep,
as pre-registered: the training M_t tops out at 0.119, the held-out matrix
cases reach 0.396, the plates 0.66.

## 3. In-sample machinery check: passes

Trained on the full matrix and scored in sample (nominal 0.90, case means):
flow 0.945, Gaussian 0.949 on dq_y; on the joint leg flow 0.947 with energy
score 0.0049 against the Gaussian's 0.893 and 0.0219, and in-sample
wall-normal CRPS 0.0005 against 0.0014. The machinery reproduces
near-nominal coverage in sample, and in sample the flow represents the
joint discrepancy structure much better than the diagonal Gaussian (a
factor 4.5 on the energy score). Whatever follows is not a fitting-capacity
failure in distribution.

## 4. The in-distribution control fails its band: the degenerate-control
branch governs the primary clause

Leave-one-case-out over the eight training cases, nominal 0.90, case means
of seed means:

| model | case mean | per-case span | seed envelope |
|---|---|---|---|
| conditional flow      | 0.722 | [0.45, 0.95] | [0.34, 0.96] |
| conditional Gaussian  | 0.793 | [0.14, 1.00] | [0.09, 1.00] |
| pooled unconditional (diagnostic) | 0.920 | [0.61, 1.00] | [0.58, 1.00] |

The flow's control (0.722) falls below the pre-registered [0.80, 0.98]
band (the Gaussian's 0.793 falls marginally below it too). Per the
pre-registration, the degenerate-control branch applies: NO transfer claim
of either sign is made for the conditional models, and the primary finding
is about trainability. At 434 rows the feature-conditioned families
under-cover held-out cases even inside the training span, with a seed
spread wide enough to move a case by half the interval, while the
unconditional pooled Gaussian, which has no conditioning to get wrong,
covers 0.920 by paying width. At nominal 0.50 the flow's control is 0.430
(inside its two-sided band), so the mid-distribution shape is roughly
right in distribution; the deficit is in the tails and in case-to-case
variability. Together with the in-sample check, this is an
overfitting-versus-generalization gap at this data volume, the sparse-leg
precedent reproduced on the thermal quantities.

## 5. Cross-Mach behaviour, reported as characterization

Fit on the eight low-Mach cases, scored on the sixteen held-out high-Mach
cases (nominal 0.90, case means of seed means, per-case span in brackets):

| model | held-out matrix | independent code (M 1.5 / 3.0) | plates |
|---|---|---|---|
| flow    | 0.128 [0.02, 0.29] | 0.19 / 0.04 | 0.02 [0.00, 0.10] |
| Gaussian | 0.664 [0.28, 0.96] | 0.88 / 0.39 | 0.12 [0.00, 0.56] |
| pooled unconditional | 0.249 [0.01, 0.68] | 0.43 / 0.07 | 0.03 [0.00, 0.14] |

Figure: `figures/heatflux_crossmach_coverage.png`. Three measured facts:

- The two conditional families fail in OPPOSITE modes under the M_t
  extrapolation, measured by interval width relative to the pooled
  diagnostic on the same cases: the flow's held-out intervals are
  NARROWER than the unconditional training-scale bands (median ratio 0.72,
  range 0.44 to 0.84), a silent overconfident extrapolation; the
  Gaussian's are much WIDER (median 7.9, range 2.9 to 10.2), the loud
  widen-but-undercover shape the attached-flow study measured for its
  tempered transfer. Neither is the pre-registered graceful shape. A
  mechanism reading consistent with both (saturating tanh networks
  extrapolate their scale heads differently) is stated as untested.
- The independent-code cases track the matrix at matched conditions (the
  Gaussian covers 0.88 at bulk M 1.5, inside the family of matrix M 1.5
  cases at low Re, and 0.39 at M 3.0 beyond the matrix span), so the
  degradation is condition-driven, not loader- or code-driven.
- The plates (flow type, baseline route, Mach and wall cooling shifted at
  once) collapse for every model (0.00 to 0.12; the thermally quiescent
  M 2.5 plate is the only nonzero entry). The compound shift is far
  outside what any of these conditionals carries, the same reading as the
  attached-flow study's wall-cooling leg.
- Within the matrix, leave-one-Mach-family-out (training then brackets the
  held-out family except at the ends) recovers substantially more coverage
  than the low-to-high extrapolation, but noisily and non-monotonically:
  flow 0.40 to 0.80, Gaussian 0.19 to 0.89 across the five families, with
  per-family seed envelopes as wide as [0.00, 0.63]. At this volume the
  family ordering flips from family to family; no clean
  interpolation-versus-extrapolation profile survives the seed spread.

At nominal 0.50 the held-out case means are flow 0.054, Gaussian 0.421,
pooled 0.114: the Gaussian's mid-distribution transfer sits inside the
two-sided band while its 0.90-level coverage does not, another expression
of the tail deficit.

## 6. The family clause on the joint leg: negative as pre-registered, with
a measured split

On the sixteen held-out cases (seed means; per-case wins in brackets;
figure `figures/heatflux_family_scores.png`):

| metric | flow | Gaussian | flow better |
|---|---|---|---|
| energy score (joint) | 0.0707 | 0.0798 | 10/16 |
| CRPS, streamwise component | 0.0705 | 0.0797 | 10/16 |
| CRPS, wall-normal component | 0.00431 | 0.00398 | 0/16 |

The pre-registered clause required the energy-score win AND wall-normal
CRPS no worse; the conjunction FAILS (the flow is about eight percent worse
on the wall-normal marginal on every held-out case). The measured split:
the flow carries the streamwise structural component of the discrepancy
(the part the gradient-diffusion baseline cannot represent at any Pr_t)
better than the diagonal Gaussian, held out as well as in sample, while the
Gaussian holds the calibrated wall-normal marginal better under transfer.
Per the pre-registered degeneracy note, the scalar leg is a
parametrization-robustness comparison (both families are conditionally
Gaussian on a scalar target), so the wall-normal loss is a fitting
difference within one family, not a family gap; the family-level content of
this clause is the multivariate structure, where the flow wins but not by
the pre-registered conjunction.

## 7. The secondary axis: the scale carries most of the transfer failure,
the carrier is the wall-flux parameter, and the residual is shape

Normalized split-conformal re-scoring of the committed global-correction
residuals on the held-out thermal block (pooled tempered posterior exactly
as committed: matched learning rate 0.0212 at the primary sigma level,
0.0267 at the sensitivity level; both cached ensemble sets complete;
figure `figures/heatflux_conformal_scores.png`). Held-out thermal coverage
at nominal 0.90 over the sixteen cases:

| score | 0.5 percent level | 1 percent level |
|---|---|---|
| absolute (identical-path control) | 0.114 (gap 0.786) | 0.108 (gap 0.792) |
| B_q-normalized (primary)          | 0.608 (gap 0.292) | 0.648 (gap 0.252) |
| semi-local sqrt(rho+) (sensitivity) | 0.125 (gap 0.775) | 0.114 (gap 0.786) |

Four measured facts:

- Normalizing the score by the case's own BASELINE-predicted wall flux
  recovers over half the conformal transfer gap (0.11 to 0.61 and 0.65)
  with no recalibration and no learned model, but does NOT reach the
  pre-registered [0.80, 0.98] band: N MINUS as pinned, by a margin that is
  itself informative.
- The residual failure after scale normalization grows along the axis: the
  per-case B_q-normalized coverage decays from 0.82 at M 1.47 to 0.18 at
  M 2.49 (primary level). What survives the wall-flux scaling is a
  Mach-dependent misfit SHAPE, not a residual scale error.
- The semi-local rescaling does nothing (median per-case change against the
  absolute control +0.000): the carrier of the Mach growth is the wall-flux
  parameter, not local mean properties, the pre-registered refinement
  answered cleanly.
- At nominal 0.50 the B_q-normalized coverage is 0.131 and 0.188: even
  where the 0.90-level interval half-works, the mid-distribution shape is
  wrong (the same reliability bow the incompressible studies measured for
  global corrections over non-Gaussian discrepancies).

Protocol note, stated not hidden: the absolute-score control here (0.114)
differs from the committed transfer value (0.006) because this re-scoring
pools the calibration residuals over all eight training cases where the
committed protocol used the first training case's stations; the pooling
alone buys about a tenth of coverage, and the normalization comparison runs
through the identical pooled path, so the 0.11-to-0.61 movement isolates
the score itself. Both sigma levels agree on every reading in this section.

## 8. Verdict against the criterion

Measured clause by clause:

1. Conditional transfer: NOT ADJUDICATED, by the pre-registered
   degenerate-control branch (the flow's control 0.722 is below the
   [0.80, 0.98] band). The finding this clause yields is trainability: at
   434 training rows the conditional families do not deliver calibrated
   held-out tails even in distribution. The cross-Mach collapse (flow
   0.128) is reported as characterization, not as a transfer verdict of
   either sign.
2. Family and proper scores: NEGATIVE as pre-registered (the conjunction
   fails on the wall-normal CRPS), with the measured split: the flow wins
   the multivariate structure (energy score and the streamwise component,
   10 of 16 cases, and a factor 4.5 in sample), the Gaussian the scalar
   calibrated marginal.
3. Mid-distribution calibration: in distribution inside the band (0.430);
   under transfer far outside it (0.054). Mixed, reported.
4. Far legs: characterized (independent code tracks the matrix; the
   compound plate shift collapses every model), consistent with
   distance-driven degradation throughout.

Scale-versus-shape grid: the study lands in N MINUS and F MINUS, and the
pre-registered reading of that quadrant applies with two measured
refinements. As pre-registered: neither a physical rescaling nor
(invariants, M_t) conditioning, as built, carries the thermal discrepancy
law across Mach, and the conditioning features are the leading suspect for
the conditional route (the separated-flow transfer measured the same
signature). The refinements the numbers force: first, the failure is
scale-DOMINATED even though not scale-only (the B_q-normalized score alone
recovers over half the gap at zero model cost, and what remains is a
Mach-dependent shape); second, the F MINUS is confounded by the failed
in-distribution control, so it reads as insufficient training data for the
conditional law on this matrix, not as evidence that conditioning cannot
work at adequate volume.

Decision-point reading for the high-speed direction: (a) a
predicted-wall-flux-normalized conformal score is the cheap, measured,
first-line correction for cross-Mach thermal transfer and belongs in the
baseline set of any subsequent high-speed UQ study; (b) local-property
(semi-local) rescaling is ruled out as the carrier; (c) the conditional
generative route on attached data is data-volume-limited before it is
family-limited: the flow's multivariate advantage is real and reproduces
in sample and held out, but neither conditional family delivers calibrated
tails from a few hundred rows, so feature enrichment (the unused
thermodynamic-fluctuation tables) or denser training matrices are the
motivated follow-ons, each a scope decision of its own; (d) the structural
exclusion stands as a reportable fact about the method landscape: the
eigenspace-perturbation framework perturbs only the Reynolds-stress
anisotropy and cannot represent a heat-flux correction at any amplitude,
so the relevant baselines here were the Gaussian conditional on dq and the
committed global correction, and both are now characterized on this axis.

Honest caveats, none of which change the shape: the criterion is read on
seed means as pinned, and the seed envelopes are wide at this volume
(quoted throughout; a single seed can move a case by half the interval);
per-case coverage is quantized by station count (28 to 241 test points per
case); the conditional models' hyperparameters were pinned from the
separated-flow study and deliberately not re-tuned; the streamwise
component is identically zero on the plates and the independent-code cases
(their records carry the wall-normal flux only), so the family clause
lives entirely on the channel matrix; the extrapolation-mechanism reading
in section 5 is stated as consistent-with, untested.

## 9. Reproduce

```
export QBTM_DNS_DATA=<repo>/DNS_data
PYTHONPATH=build:python python3 python/UQ/reproduce_heatflux_modelform.py
python3 UQ-RANS_research/heatflux_modelform/make_heatflux_figures.py
```

Driver seed 0, model seeds {0, 1, 2}; `--stage apriori` and
`--stage conformal` compose into the same numbers JSON; `--quick` is the
smoke path. The conformal stage consumes the calibration ensemble caches
under `results/compressible*` (regenerable via
`python/UQ/reproduce_compressible.py`). The committed
`finding_numbers.json` is byte-for-byte the production output; the figures
come from `make_heatflux_figures.py`.

## Data attribution

All datasets are third-party DNS, not produced by this project, cited where
used: Gerolymos and Vallet, J. Fluid Mech. 958 (2023) A19 (Mendeley Data,
doi:10.17632/wt8t5kxzbs.1, CC BY 4.0); Coleman, Kim and Moser, J. Fluid
Mech. 305 (1995) 159-183; Zhang, Duan and Choudhari, AIAA Journal 56(11)
(2018) 4297-4311, doi:10.2514/1.J057296. The latter two are hosted by the
migrated NASA Turbulence Modeling Resource (tmbwg.github.io/turbmodels).
Raw fields stay local and gitignored.
