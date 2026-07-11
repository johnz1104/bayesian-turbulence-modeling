# Shock-interaction model-form UQ: finding memo

Status: DRAFT SKELETON. Every number below is TBD until filled from the
committed numbers JSONs (`apriori_numbers.json`, `aposteriori_numbers.json`);
no value is quoted from anywhere else. The criteria are quoted from the
merged pre-registration and are not restated with different thresholds.

## 1. What was run

- A-priori leg: leave-one-wall-thermal-out transfer of the conditional flow
  and Gaussian discrepancy models on the impinging-shock DNS (five heated-
  campaign conditions plus the adiabatic campaign), the in-sample machinery
  check, the attached-to-interaction far transfer with its attached
  leave-one-Mach-family-out control, and the wall-flux-normalized conformal
  line. Settings as pinned: epochs 400, seeds {0, 1, 2}, 128 draws per
  point, pre-registered strides and interior mask, criterion on the seed
  mean.
- A-posteriori leg: gates A and B, then the propagated folds s = 0.5, 1.0,
  1.9 (leave-that-s-out models, 24 coherent members per method), the
  eigenspace corner family (Delta_B 1.0 and 0.5) through the identical
  injection, the preserve-attached control for every fold closure, and the
  attached-trained far-transfer propagation into the adiabatic interaction
  configuration.
- Reproduction: `python/UQ/reproduce_sbli_apriori.py` and
  `python/UQ/reproduce_sbli_aposteriori.py` (fixed seeds; solve-once caches
  under the gitignored results directory).

## 2. Gates

| Gate | Criterion (pre-registered) | Measured | Verdict |
|---|---|---|---|
| A (attached baseline) | Cf at x* = -7.65 within 10 percent of the measured 2.56e-3 | Converged, 12401 sweeps, Cf 2.354e-3, 8.1 percent | TBD (reviewer) |
| B (interaction baselines) | all six interaction configurations converge; impingement offset vs the DNS half-rise reported per case | TBD per case | TBD |

Gate-B offsets (solve half-rise minus DNS half-rise, per case): TBD table.

## 3. A-priori verdict against the pre-registered clauses

| Clause | Criterion | Measured (seed mean) | Verdict |
|---|---|---|---|
| 1 within-interaction transfer | LOso case-mean dq_y coverage at 0.90 >= 0.80, every fold >= 0.60; in-sample in [0.80, 0.98] | TBD | TBD |
| 2 family at density | flow beats Gaussian on energy score with CRPS no worse, on BOTH joint legs | TBD | TBD |
| 3 mid-distribution guard | held-out case-mean coverage at 0.50 in [0.35, 0.65] | TBD | TBD |
| 4 far transfer | attached control in [0.80, 0.98]; case-mean interaction coverage >= 0.80 with upstream region >= 0.80 | TBD | TBD |
| 5 conformal line | St-normalized score restores [0.80, 0.98] at 0.90 vs absolute control | TBD | TBD |

Pre-registered null shapes touched (if any): TBD (family null / transfer
null / compound-shift null / degenerate control).

Region-resolved reading (upstream, interaction, downstream): TBD.

History-feature ablation (dq_y leg): TBD (in vs out, seed mean).

## 4. A-posteriori verdict against the pre-registered clauses

Convergence counts (per fold and method; the 18-of-24 label): TBD.

| Clause | Criterion | Measured | Verdict |
|---|---|---|---|
| 1 coverage | flow 0.90 bands cover >= 0.80 of stations pooled per fold; scalars contained; 0.50 in [0.35, 0.65] | TBD | TBD |
| 2 accuracy | ensemble-median St error below baseline in the interaction region on heated folds; Cf and Cp below baseline in at least the interaction region | TBD | TBD |
| 3 sharpness guard | covering bands narrower than the baseline's own interaction-region point error | TBD | TBD |
| 4 baseline comparison | clauses 1-3 for the flow at least as good as the Gaussian; corner envelope containment reported alongside | TBD | TBD |
| 5 realizability | running barycentric check passes every outer iteration for every converged member | TBD | TBD |
| 6 wall-thermal generalization | cooling and heating folds degrade gracefully from the adiabatic fold | TBD | TBD |
| 7 preserve-attached | ensemble-median Cf error within 1.5x baseline; 0.90 band half-width below 0.15 of measured cf | TBD | TBD |

Pre-registered null shapes touched (if any): TBD (solver-capability /
magnitude-cap recurrence / shock-foot instability / vacuous coverage / no
accuracy gain / attached degradation).

Far-transfer propagation (attached-trained, adiabatic interaction): TBD
characterization.

s = 1.0 second truth surface (independent campaign): TBD.

## 5. Diagnosis carried by the evidence

TBD after the numbers: what the coverage, sharpness, landmark and
realizability blocks jointly say about the injection route, the heat-flux
reach, the family question at interaction density, and the wall-thermal
conditioning.

## 6. Scope notes and deviations

- The s = 1.0 wall-pressure column is integer-quantized (loader marks it
  invalid); its Cp clauses run on the field-row landmarks per the
  pre-registration addendum.
- Observation-uncertainty convention: modeled, anchored, labeled as in the
  addendum; never the DNS's own statistics.
- Any deviation from the pinned protocol discovered while filling this memo
  is listed here, dated, with its effect stated: TBD (none known at
  skeleton time).

## 7. Decision points recorded for the program

TBD: each verdict's motivated follow-on, in the pre-registration's own
decision-point language.
