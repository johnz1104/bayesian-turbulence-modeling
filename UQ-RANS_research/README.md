# UQ-RANS_research: the findings and results archive

The single, version-controlled home for every finding the UQ-for-RANS program
produces, so results accumulate in one trackable place on the trunk. One
subfolder per step (DNS_plan.md), each holding the curated evidence package for
that step.

## Data attribution

Every dataset analysed here is third-party DNS, not produced by this project, and
is cited where it is used. Full provenance (source, reference, license) is in
`DNS_data/README.md`. The plane-channel data are from Lee and Moser (2015),
J. Fluid Mech. 774, 395-415 (UT Austin Oden Institute,
https://turbulence.oden.utexas.edu), used under their research-use-with-citation
terms. The raw fields are kept local and are not redistributed in this repository.

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
