# UQ-RANS_research: the findings and results archive

The single, version-controlled home for every finding the UQ-for-RANS program
produces, so results accumulate in one trackable place on the trunk. One
subfolder per study, each holding the curated evidence package for
that step.

## Data attribution

Every dataset analysed here is third-party DNS, not produced by this project, and
is cited where it is used. Full provenance (source, reference, license) is in
`DNS_data/README.md`. The raw fields are kept local and are not redistributed in
this repository.

- Plane channel (channel coverage-correction): Lee and Moser (2015), J. Fluid Mech. 774, 395-415 (UT
  Austin Oden Institute, https://turbulence.oden.utexas.edu), research-use with
  citation.
- Plane Couette (cross-flow generalization): Pirozzoli, Bernardini and Orlandi (2014), J. Fluid Mech.
  742, 171-191 (Roma / Sapienza University of Rome), research-use with citation.
- Turbulent pipe (cross-flow companion): Pirozzoli (2024), J. Fluid Mech. 989, A5
  (Roma / Sapienza University of Rome), research-use with citation.
- Streamwise-rotating channel (cross-flow companion): Yang and Wang (2018), J. Fluid
  Mech. 838, 658-689 (University of Manitoba). All rights reserved by the
  University of Manitoba; the data may be used with reference, so it is cited
  wherever used.
- Periodic hills, parametric slope family (separated-flow model-form): Xiao, Wu,
  Laizet and Duan (2020), Comput. Fluids 200, 104431
  (github.com/xiaoh/para-database-for-PIML). No explicit license is stated in the
  source repository, so it is treated as research-use with citation.
- Backward-facing step (separated-flow model-form): Le, Moin and Kim (1997), J.
  Fluid Mech. 330, 349-374, research-use with citation.
- Compressible plane-channel matrix (compressible attached): Gerolymos and
  Vallet (2023), J. Fluid Mech. 958, A19 (Mendeley Data,
  doi:10.17632/wt8t5kxzbs.1), CC BY 4.0.
- Isothermal-wall supersonic channel (compressible attached, external check):
  Coleman, Kim and Moser (1995), J. Fluid Mech. 305, 159-183 (hosted by the
  migrated NASA Turbulence Modeling Resource), research-use with citation.
- Supersonic and hypersonic flat plates (compressible attached, wall-cooling
  axis and measured Pr_t reference): Zhang, Duan and Choudhari (2018), AIAA
  Journal 56(11), 4297-4311 (hosted by the migrated NASA Turbulence Modeling
  Resource), research-use with citation.

## What goes here (tracked)

- the per-step finding memo (the writeup measured against the pre-registered
  criterion),
- the key figures (reliability diagrams, PIT histograms, coverage plots),
- the numbers JSON that every quoted figure traces to.

These are small and curated, so they are committed and travel with the code when
a step's branch merges into the trunk.

## What does NOT go here (stays gitignored, regenerable)

- the raw DNS fields (`DNS_data/`, local only, manifest tracked separately),
- the large solver-run caches (ensembles, baseline fields) under `results/`,
- any artifact a reproduce script can regenerate from a fixed seed.

The reproduce scripts live with the code (for the channel: `python/UQ/
reproduce_channel.py`); rerunning them regenerates the gitignored caches and the
figures, and the figures here are the committed snapshot.

## Layout

```
UQ-RANS_research/
  step1_plane_channel/        plane channel, real-data coverage correction
    channel_finding.md        the memo (verdict against the pre-registered criterion)
    finding_numbers.json      a1-betaStar in-distribution + cross-Re + evaluation
    nw4_in_distribution.json  four-coefficient robustness check
    figures/                  reliability, coverage, PIT
  step2_couette/              cross-flow generalization (Couette a-posteriori)
  separated_modelform/        separated-flow generative model-form
  compressible_attached/      compressible heat-flux and Prandtl UQ
  ...
```

## Index of findings

| Step | Case | Finding | Verdict |
|---|---|---|---|
| 1 | Plane channel, 5 Re_tau | standard Bayesian calibration is overconfident on real channel DNS; generalized Bayes and conformal restore coverage in distribution and across held-out Reynolds numbers | positive (coverage 0.20 -> 0.87/0.91 in distribution; graceful cross-Re degradation, conformal gap <= 0.10) |
| 2 | Plane Couette (a-posteriori) + pipe / rotating companions | the channel calibration plus calibrated UQ transfers to the held-out Couette flow type through a real moving-wall solve; standard Bayes is overconfident, generalized Bayes and conformal restore coverage gracefully; the rotating channel motivates the separated-flow model-form | positive, graceful (cross-flow coverage 0.05-0.26 -> restored fully at low Couette Re, partially at high Re; conformal gap up to +0.22 at the 0.5% band, within-Couette correction reaches nominal; rotating <uw> is structurally unrepresentable, global correction is diffuse) |
| - | Compressible attached matrix (24-case channel family, CKM external check, hypersonic plates) | can the heat-flux discrepancy and the turbulent Prandtl number be calibrated with honest UQ on real compressible attached data, and does the calibrated UQ transfer across Mach and wall cooling | mixed, per the pre-registered shapes: overconfidence confirmed (held-out HEAT-FLUX coverage, the B_q plus q-profile block never fitted, 0.17 in distribution, 0.01 cross-Mach at nominal 0.90; the temperature profile IS a likelihood observable, so the block name is precise post-audit); the global correction restores the likelihood block (0.96) but NOT the held-out heat-flux block (0.42 in distribution, 0.18 cross-Mach; intervals widen x14 so not silent); Pr_t is unidentifiable from attached mean observables (edge-piled pseudo-true posteriors; the measured plate profiles are tight around 0.9); the held-out heat-flux block needs a correction reaching the heat-flux quantities, triangulating with the separated-flow and rotating-channel diagnoses |
