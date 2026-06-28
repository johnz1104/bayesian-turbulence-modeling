# UQ-RANS_research: the findings and results archive

The single, version-controlled home for every finding the UQ-for-RANS program
produces, so results accumulate in one trackable place on the trunk. One
subfolder per step (DNS_plan.md), each holding the curated evidence package for
that step.

## Data attribution

Every dataset analysed here is third-party DNS, not produced by this project, and
is cited where it is used. Full provenance (source, reference, license) is in
`DNS_data/README.md`. The raw fields are kept local and are not redistributed in
this repository.

- Plane channel (Step 1): Lee and Moser (2015), J. Fluid Mech. 774, 395-415 (UT
  Austin Oden Institute, https://turbulence.oden.utexas.edu), research-use with
  citation.
- Plane Couette (Step 2): Pirozzoli, Bernardini and Orlandi (2014), J. Fluid Mech.
  742, 171-191 (Roma / Sapienza University of Rome), research-use with citation.
- Turbulent pipe (Step 2 companion): Pirozzoli (2024), J. Fluid Mech. 989, A5
  (Roma / Sapienza University of Rome), research-use with citation.
- Streamwise-rotating channel (Step 2 companion): Yang and Wang (2018), J. Fluid
  Mech. 838, 658-689 (University of Manitoba). All rights reserved by the
  University of Manitoba; the data may be used with reference, so it is cited
  wherever used.

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
  step1_plane_channel/        plane channel, real-data coverage correction (research.md)
    channel_finding.md        the memo (verdict against the pre-registered criterion)
    finding_numbers.json      a1-betaStar in-distribution + cross-Re + evaluation
    nw4_in_distribution.json  four-coefficient robustness check
    figures/                  reliability, coverage, PIT
  step2_couette/              (cross-flow OOD, when run)
  step3_separated/            (generative model-form, when run)
  step4_compressible_attached/
  step5_sbli_apriori/         (the compressible flagship)
  ...
```

## Index of findings

| Step | Case | Finding | Verdict |
|---|---|---|---|
| 1 | Plane channel, 5 Re_tau | standard Bayesian calibration is overconfident on real channel DNS; generalized Bayes and conformal restore coverage in distribution and across held-out Reynolds numbers | positive (coverage 0.20 -> 0.87/0.91 in distribution; graceful cross-Re degradation, conformal gap <= 0.10) |
| 2 | Plane Couette (a-posteriori) + pipe / rotating companions | the channel calibration plus calibrated UQ transfers to the held-out Couette flow type through a real moving-wall solve; standard Bayes is overconfident, generalized Bayes and conformal restore coverage gracefully; the rotating channel motivates Step 3 | positive, graceful (cross-flow coverage 0.05-0.26 -> restored fully at low Couette Re, partially at high Re; conformal gap up to +0.22 at the 0.5% band, within-Couette correction reaches nominal; rotating <uw> is structurally unrepresentable, global correction is diffuse) |
