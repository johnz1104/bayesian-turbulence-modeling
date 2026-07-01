# Data acquisition track (Thread B)

Scaffold and acceptance bar for the hypersonic shock-boundary-layer interaction
(SBLI) data the program needs. The actual downloading and access requests are a
separate acquisition task; this file is the candidate list, the required fields, and
the gate that must pass before the out-of-distribution (OOD)
protocol.

Every dataset that lands gets a full entry in `data/README.md` (the provenance
manifest). The candidate names below are starting points to confirm, NOT
confirmed holdings: their fidelity, exact conditions, and especially their
Reynolds-stress content must be verified against the primary source before use.
There is no real hypersonic data in the repo today.

## What we need and why

- A-priori calibration and OOD on FROZEN high-fidelity fields (the solver is
  low-Mach, Ma <= 0.5; hypersonic a-posteriori is a later stretch). So the
  hypersonic datasets are used a-priori: frozen mean fields plus the Reynolds
  stresses and anisotropy needed to test the closure ansatz, and surface
  Cf / Cp / St for the model-form-discrepancy target.
- Reynolds stresses (full tensor + anisotropy) are required for the a-priori test
  of the spatial non-local ansatz and the tensor-basis construction. Surface-only
  experimental data cannot supply this; it needs DNS or wall-resolved LES.
- At least two regimes (two Mach numbers OR two ramp/shock angles from comparable
  setups) are required so the study can hold one out and measure OOD generalization.

## Candidate hypersonic / SBLI databases (to confirm)

Fidelity tag: DNS, WRLES (wall-resolved LES), or EXP (experiment).

### High-fidelity (carry Reynolds stresses; primary sources)

- Compression-ramp STBLI DNS, Martin group (e.g. Priebe and Martin, JFM 2012;
  later hypersonic extensions). Geometry: compression ramp. Mach ~2.9 baseline
  with hypersonic (Mach ~7) cases in follow-on work. Carries full Reynolds-stress
  tensor. Confirm hypersonic conditions and ramp-angle variants.
- Compression-ramp / impinging-shock STBLI DNS, Pirozzoli and Bernardini and
  collaborators. Geometry: ramp and impinging oblique shock. Supersonic to low
  hypersonic. Full Reynolds stresses. Confirm which cases reach the hypersonic
  regime and whether multiple deflection angles exist.
- Hypersonic turbulent boundary layer + STBLI DNS, Helm and Martin (Mach ~7).
  Geometry: flat-plate TBL and STBLI. Carries Reynolds stresses and wall heat
  flux. Strong candidate for the hypersonic anchor; confirm angle/Mach sweep.

### Surface-resolved experiments (Cf / Cp / St; angle and Mach sweeps)

- Schulein, DLR, Mach ~5 impinging oblique-shock SWBLI (e.g. AIAA J 2006). Skin
  friction and wall heat flux across SEVERAL shock-generator angles. Excellent for
  the "two angles" criterion and for the St discrepancy target; Reynolds stresses
  are not generally available (experiment), so pair it with a DNS source.
- CUBRC / Holden hypersonic SWBLI (LENS): double cone and hollow-cylinder-flare,
  Mach ~9-16, surface pressure and heat flux across conditions. Good for OOD Mach
  spread and the St target; no Reynolds stresses; geometry is canonical-validation,
  not a ramp.

## Control cases (solver-runnable; a-posteriori + attached-flow preservation)

These the current low-Mach solver CAN run, so they carry the a-posteriori coupling
and the "preserve attached-flow accuracy" checks (root CLAUDE.md, Option 3):

- Attached flat plate / channel: the simple attached baseline (already in
  `python/case_library.py`, Dean and Schoenherr Cf anchors).
- Backward-facing step, Driver and Seegmiller 1985: incompressible separated
  control (already wired as `DS1985`).
- Periodic hills (e.g. Breuer et al.): incompressible separated WRLES/DNS with
  full Reynolds stresses; a separated control the solver can run a-posteriori and
  that also supports the a-priori machinery.
- Darmstadt attached-flow control: the attached-flow a-posteriori control.

## Required fields per dataset

Each accepted dataset must fill every applicable field of the `data/README.md`
template: source and citation; fidelity; geometry; Mach; ramp or wedge angle;
Reynolds number and its definition; normalization and reference state (freestream
vs edge vs wall units); separation and reattachment definitions; coordinate origin
(shock-relative placement and sign convention); available quantities (mean fields,
Reynolds stresses and anisotropy, surface Cf / Cp / St); license and access status.

## Acceptance check (gate before the OOD protocol is designed)

The OOD protocol design does not start until BOTH hold, recorded in
`data/README.md`:

1. Reynolds-stress availability CONFIRMED for at least one hypersonic (or, as an
   interim, supersonic-to-hypersonic) high-fidelity dataset: the full Reynolds
   stress tensor and anisotropy are present, not just surface quantities.
2. At least TWO regimes from comparable setups are in hand: two Mach numbers OR
   two ramp/shock angles. This is the minimum needed to hold one regime out and
   measure OOD generalization.

Until both are checked off, Thread B stays on the a-priori machinery and the
solver-runnable controls.
