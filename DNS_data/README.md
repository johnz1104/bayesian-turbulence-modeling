# DNS data to compile

The high-fidelity datasets the UQ-for-RANS program needs, organized by the role
each plays in research.md (incompressible) and compressible_research.md
(compressible). The list is meant to be comprehensive, so it includes more than any
single paper requires. Priority order is in the last section. The actual data files
stay local and gitignored. Only this manifest and the per-dataset provenance entries
are tracked, following the same rule as data/README.md.

Reference values for Reynolds number, Mach number, and ramp angle are as published.
Confirm the exact values, the wall thermal condition, and the available fields on
download, because they decide whether a case is usable (Section "Acceptance
criteria").

## What each dataset must provide

A dataset is only useful to this program if the discrepancy against the RANS
baseline can be formed from it. The required and desired fields:

- **Mandatory.** Geometry and flow parameters (Re, and for compressible Ma and the
  wall thermal condition), the mean velocity field, and the full Reynolds-stress
  tensor (the four nonzero components in 2D plus the spanwise normal stress). Without
  the Reynolds stresses the model-form discrepancy and the anisotropy cannot be
  computed, and the case is not usable.
- **Mandatory for compressible.** Mean temperature and density, the turbulent heat
  flux vector, and wall heat flux or Stanton number. These carry the heat-flux
  target and the compressible model-form modes.
- **Mandatory for separated and SBLI cases.** Wall pressure (Cp), wall skin friction
  (Cf), and the separation and reattachment locations, plus the shock position for
  SBLI.
- **Highly desired.** Turbulent kinetic energy and Reynolds-stress budgets (for
  closure-term analysis), the turbulent Mach number field, and where available the
  dilatational quantities (pressure-dilatation, dilatational dissipation, turbulent
  mass flux), which are exactly the compressible-moat modes.
- **Always.** The grid, the numerics, the boundary conditions, the averaging
  procedure, and the citation, recorded in the provenance entry.

## Acceptance criteria (the gate)

- Reynolds-stress availability is non-negotiable. A case without it is excluded.
- For any out-of-distribution claim the axis must be populated by at least two
  conditions: two Reynolds numbers, or two Mach numbers, or two ramp angles or shock
  strengths. A single operating point cannot support a generalization test.
- For compressible heat-flux claims the turbulent heat flux and the wall heat flux
  must both be present.
- Reproducibility metadata (grid, scheme, BCs) must be documented.

## Tier 1. Incompressible attached flows (calibration and cross-flow or cross-Re generalization)

Real DNS to replace the Dean correlation as the calibration target, and to provide
clean in-distribution and cross-flow generalization.

| Case | Role | Re (as published) | Key fields | Source and references |
|---|---|---|---|---|
| Plane channel | primary calibration, cross-Re OOD | Re_tau 180, 395, 590; 2003; up to 5186 | U, full Reynolds stress, budgets | Moser, Kim and Mansour 1999; Hoyas and Jimenez 2006; Lee and Moser 2015; TU Darmstadt FDY (Oberlack) |
| Plane Couette | cross-flow OOD (calibrate on channel, predict Couette) | Re_tau up to 2000 | U, Reynolds stress | TU Darmstadt FDY: Avsarkisov et al. 2014, Kraheberger et al. 2018, Hoyas and Oberlack 2024; Pirozzoli, Bernardini and Orlandi 2014 |
| Pipe flow | cross-geometry OOD | Re_tau up to ~1000-6000 | U, Reynolds stress | El Khoury et al. 2013; Pirozzoli et al. 2021 |
| Rotating channel | model-stress-test (SST fails without rotation correction) | spanwise and streamwise rotation | U, Reynolds stress | TU Darmstadt FDY: Oberlack et al. 2006, Mehdizadeh and Oberlack 2010; Brethouwer 2017 |
| Zero-pressure-gradient TBL | attached boundary-layer calibration and control | Re_theta up to ~6500 | U, Reynolds stress, Cf | Schlatter and Orlu 2010; Sillero, Jimenez and Moser 2013 |
| Adverse-pressure-gradient TBL | non-equilibrium attached, pre-separation | various beta | U, Reynolds stress, Cf, Cp | Bobke et al. 2017; Pozuelo et al. 2022 |

## Tier 2. Incompressible separated flows (the high-impact incompressible out-of-distribution test)

Where baseline SST is most wrong incompressibly, so the largest improvement and the
clearest coverage claim live here.

| Case | Role | Re (as published) | Key fields | Source and references |
|---|---|---|---|---|
| Backward-facing step | canonical separation OOD; geometry already scaffolded in the repo | Re_h 5100 | U, Reynolds stress, Cf, Cp, reattachment length | Le, Moin and Kim 1997 |
| Periodic hills | canonical separated RANS benchmark; parametric geometry for a clean OOD sweep | Re_b 700 to 10595; slope-varied geometries | U, Reynolds stress, Cf, Cp, separation and reattachment | Breuer et al. 2009; Xiao et al. 2020 (parametric geometry) |
| Converging-diverging channel | adverse-pressure-gradient separation | Re_tau ~600 | U, Reynolds stress, Cf, Cp | Marquillie, Ehrenstein and Laval 2011 |
| Curved backward-facing step | smooth-body separation | Re_h ~13700 | U, Reynolds stress, Cf, Cp | Bentaleb, Lardeau and Leschziner 2012 |
| Square duct | secondary-flow model-form (linear eddy viscosity fails) | Re_tau ~150-1000 | U, full Reynolds stress (the secondary motion) | Pinelli et al. 2010 |
| NASA wall-mounted hump and 2D bump | separation with reference data and CFD comparison | as in the database | U, Cp, Cf, reattachment | NASA Turbulence Modeling Resource |

## Tier 3. Compressible attached flows (compressible calibration and cross-Mach generalization)

The compressible analogue of Tier 1. Supplies the momentum and heat-flux closure
calibration and the attached-flow control, and exercises the temperature and
heat-flux machinery before any SBLI claim.

| Case | Role | Ma and Re (as published) | Key fields | Source and references |
|---|---|---|---|---|
| Compressible plane channel | compressible calibration, cross-Mach OOD | bulk Ma 1.5 and 3.0; higher Re variants | U, T, rho, Reynolds stress, turbulent heat flux, wall heat flux | Coleman, Kim and Moser 1995; Modesti and Pirozzoli 2016 |
| Supersonic turbulent boundary layer | attached compressible, calibration and control | Ma ~2 to 2.25 | U, T, rho, Reynolds stress, turbulent heat flux, Cf, q_w | Pirozzoli, Grasso and Gatski 2004; Pirozzoli and Bernardini 2011 |
| Hypersonic turbulent boundary layer | high-Mach attached, the moat regime begins here; wall-temperature effects | Ma 5 to 12, cold and hot walls | U, T, rho, Reynolds stress, turbulent heat flux, q_w, turbulent Mach number | Duan, Beekman and Martin 2010, 2011; Zhang, Duan and Choudhari 2018 |

## Tier 4. Compressible shock-boundary-layer interaction (the flagship compressible out-of-distribution test)

The headline target. The Mach number and the ramp angle or shock strength are the
out-of-distribution axes, so collect at least two operating points along whichever
axis is used.

| Case | Role | Ma and angle (as published) | Key fields | Source and references |
|---|---|---|---|---|
| Compression-ramp SBLI | canonical SBLI; ramp-angle OOD axis | Ma 2.9, 24 degree; hypersonic ramps | U, T, Reynolds stress, turbulent heat flux, Cp, Cf, q_w, separation, reattachment, shock position | Wu and Martin 2007; Priebe and Martin 2012; Fang et al. 2020 (hypersonic) |
| Impinging oblique-shock SBLI | canonical SBLI; shock-strength OOD axis | Ma 2.25 and up | as above | Pirozzoli and Grasso 2006; Bernardini, Pirozzoli and co-workers |
| Transonic SBLI and shock buffet | transonic shock-boundary-layer, a distinct regime | Ma ~0.7 to 0.9 | U, Cp, Cf, q_w, shock position | Bachalo and Johnson axisymmetric bump and transonic-bump DNS and LES |
| TFAST and UFAST SBLI databases | curated SBLI cases across conditions, with experiment | multiple Ma and configurations | mean and second-moment fields, wall data | TFAST and UFAST European project databases |

## Tier 5. Canonical shock-turbulence interaction (physics for the compressible moat)

Not a RANS validation case, but the cleanest evidence for the compressible-moat
argument, that the model-form uncertainty across a shock lives in dilatational and
thermodynamic modes the anisotropy tensor cannot hold.

| Case | Role | Parameters | Key fields | Source and references |
|---|---|---|---|---|
| Isotropic turbulence through a normal shock | quantify the compressible model-form modes directly | turbulent Mach number and shock Mach swept | Reynolds stress amplification, dilatational dissipation, pressure-dilatation | Larsson, Bermejo-Moreno and Lele 2013; Ryu and Livescu 2014 |

## Databases and access portals

- **UT Austin Oden Institute (turbulence.oden.utexas.edu).** Plane turbulent
  channel DNS up to Re_tau 5200 (Lee and Moser 2015). The source of the compiled
  channel data (dataset 1 below) and the real-DNS calibration source actually used
  for Step 1. https://turbulence.oden.utexas.edu
- **TU Darmstadt FDY (Oberlack group).** Attached canonical flows: channel,
  Couette, rotating channel, plane jets. A candidate source for the attached
  calibration and cross-flow cases (still a strong option for the Step 2 Couette
  data); it is not the source of the compiled channel data.
  https://www.fdy.tu-darmstadt.de/fdyresearch/dns/direkte_numerische_simulation.en.jsp
- **Johns Hopkins Turbulence Databases (JHTDB).** Channel, boundary layer, isotropic,
  and other space-time-resolved DNS with a query interface.
- **NASA Turbulence Modeling Resource (TMR).** Separated and SBLI verification and
  validation cases with reference CFD and experiment, including the wall-mounted hump
  and 2D bump.
- **ERCOFTAC Classic Database and SIG15.** Periodic hills and other separated
  benchmarks.
- **Polimi group (Pirozzoli, Bernardini).** Compressible channel, supersonic and
  hypersonic boundary layers, SBLI.
- **Maryland and group of Martin.** Hypersonic boundary layers and compression-ramp
  SBLI.
- **TFAST and UFAST.** European SBLI project databases.

## Provenance logging

Record one entry per compiled dataset, mirroring data/README.md, so any consumer can
fetch or regenerate it:

```
## <dataset-name>
- Source or URL:
- Reference (paper):
- Version or date:
- Regime: incompressible | compressible
- Parameters: Re, Ma, ramp angle, wall thermal condition
- Fields provided: U, Reynolds stress, T, rho, turbulent heat flux, q_w, Cp, Cf, budgets
- OOD axis it populates: Reynolds number | Mach number | ramp angle | none (calibration)
- Checksum (sha256):
- Used by: research.md | compressible_research.md (which spine and milestone)
- License and notes:
```

## Priority order

Collect in this sequence so each acquisition unblocks a concrete milestone.

1. **Plane channel DNS** (Tier 1). Replaces the Dean correlation with real data and
   unblocks every incompressible result.
2. **One Couette and one additional Reynolds number** (Tier 1). Gives the first
   cross-flow and cross-Re generalization tests.
3. **One separated case, backward-facing step or periodic hills** (Tier 2). The
   incompressible out-of-distribution headline.
4. **Compressible plane channel and one supersonic boundary layer** (Tier 3). Brings
   up the temperature and heat-flux machinery and the compressible calibration.
5. **One SBLI case at two ramp angles or two Mach numbers** (Tier 4). The flagship
   compressible out-of-distribution claim. This is the highest-value compressible
   acquisition.
6. **Hypersonic boundary layer and isotropic-turbulence-through-a-shock** (Tiers 3
   and 5). Extends to the high-Mach regime and supplies the compressible-moat
   evidence.

## Data hygiene

The downloaded fields are large and stay local and gitignored. Only this manifest
and the provenance entries are tracked. Add `DNS_data/` data files to the ignore
rules (keeping this README) before any are placed here, the same way data/* is
handled.

## Compiled dataset 1: plane channel (Lee and Moser 2015)

Status: downloaded from the source database (below) and verified 2026-06-25;
third-party data, not produced by this project. First dataset, Step 1 of
DNS_plan.md. Coverage is complete and the labeling error in the original upload
(the Re_tau = 5200 case sat in a folder named "5000") has been corrected.

### Provenance

- Source or URL: https://turbulence.oden.utexas.edu (UT Austin, Oden Institute)
- Reference: Myoungkyu Lee and Robert D. Moser, Direct numerical simulation of
  turbulent channel flow up to Re_tau = 5200, J. Fluid Mech. 774 (2015) 395-415
- Regime: incompressible
- Cases and per-case parameters (from the file headers):

  | folder | nominal Re_tau | actual Re_tau | nu | u_tau | wall-normal points |
  |---|---|---|---|---|---|
  | LM_Channel_Re_tau=180  | 180  | 182.088  | 3.50e-04 | 0.0637309 | 96  |
  | LM_Channel_Re_tau=550  | 550  | 543.496  | 1.00e-04 | 0.0543496 | 192 |
  | LM_Channel_Re_tau=1000 | 1000 | 1000.512 | 5.00e-05 | 0.0500256 | 256 |
  | LM_Channel_Re_tau=2000 | 2000 | 1994.756 | 2.30e-05 | 0.0458794 | 384 |
  | LM_Channel_Re_tau=5200 | 5200 | 5185.897 | 8.00e-06 | 0.0414872 | 768 |

- Fields provided: mean velocity profile (U, dU/dy, W, P), the full Reynolds-stress
  tensor (u'u', v'v', w'w', u'v', u'w', v'w'), turbulent kinetic energy k,
  Reynolds-stress-transport-equation budgets, vorticity and pressure variances,
  velocity-pressure correlations, and a per-quantity standard-deviation profile
  (the statistical uncertainty).
- OOD axis it populates: Reynolds number (five conditions, a rich cross-Re axis)
- Used by: research.md and DNS_plan.md Step 1 (the calibration anchor that
  replaces the Dean correlation, and the cross-Re generalization test)
- License and notes: free for research use with the citation above. All
  quantities are normalized in wall units (signified by ^+) by u_tau and nu.

### Raw layout and file format

Per Reynolds number, under channel_flow/LM_Channel_Re_tau=<N>/:

- mean_velocity_pressure/   columns: y/delta, y^+, U^+, dU/dy^+, W^+, P^+   (6)
- covariances_velocity/     columns: y/delta, y^+, u'u'^+, v'v'^+, w'w'^+,
                            u'v'^+, u'w'^+, v'w'^+, k^+   (9)
- terms_RSTE/{k,uu,uv,vv,ww}/   Reynolds-stress-transport budgets (highly
                            desired, not required for the discrepancy)
- variances_vorticity_pressure/, velocity_pressure_correlation/   further moments

Each quantity has a _prof.dat.txt (the profile) and a _stdev.dat.txt (its
statistical uncertainty). Files are ASCII, header lines begin with %, columns are
whitespace separated, and all quantities are in wall units. The reader is
numpy.loadtxt(..., comments='%').

### Standardized processing format (the contract every loader targets)

To keep processing uniform across this and every future dataset, a loader converts
each raw case into one canonical record (the dns_field schema in python/UQ). For an
incompressible channel case the record holds, per wall-normal station: y and y^+,
the friction quantities (u_tau, nu, Re_tau, half-height), the mean velocity and its
wall-normal gradient, the full Reynolds-stress tensor and k, the per-quantity
standard deviation carried as observation uncertainty, and metadata (regime, case,
Reynolds number, provenance). Later datasets (separated, compressible, SBLI) add
their own raw loaders but emit the SAME canonical record, extended with temperature,
heat flux, and two-dimensional fields as the case requires, so the discrepancy
extraction, the UQ, and the evaluation layers always see one uniform representation.
