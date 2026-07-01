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

## Compiled dataset 2: plane Couette (Pirozzoli, Bernardini and Orlandi 2014)

Status: downloaded and verified 2026-06-27; third-party data, not produced by this
project. Step 2 of DNS_plan.md, the cross-flow generalization primary (calibrate on
channel, predict Couette). Loader: UQ.datasets.CouetteDNS.

### Provenance

- Source or URL: Roma (Sapienza University of Rome) DNS database, group of
  S. Pirozzoli; distributed with the reference below.
- Reference: S. Pirozzoli, M. Bernardini and P. Orlandi, "Turbulence statistics in
  Couette flow at high Reynolds number", J. Fluid Mech. 742 (2014) 171-191.
- Regime: incompressible
- Cases and per-case parameters (from the RETAU title line of each file):

  | file | Re_tau | wall-normal points (half gap) | sha256 |
  |---|---|---|---|
  | roma_couette_171.txt | 171 | 128 | d2190dd4...28de10d |
  | roma_couette_260.txt | 260 | 128 | 49416bd3...142dd6 |
  | roma_couette_507.txt | 507 | 192 | 7f2d9522...7ca0ab |
  | roma_couette_986.txt | 986 | 256 | 6ada1862...8384ca |

- Fields provided: per wall-normal station (wall units), y, y^+, mean velocity U^+,
  the rms velocity fluctuations u'^+, v'^+, w'^+ (the normal stresses are their
  squares), and the shear covariance u'v'^+. No mean-gradient column (dU^+/dy^+ is
  finite-differenced), no k column (k^+ = 0.5(uu+vv+ww)), and no per-point _stdev.
- OOD axis it populates: flow type (cross-flow, channel -> Couette), and a bonus
  cross-Re axis within Couette (four friction Reynolds numbers).
- Observation uncertainty: MODELED relative value (UQ.datasets.observation_sigma),
  not a DNS _stdev (the file carries none). Anchored by the data-only constant-
  total-stress identity dU^+/dy^+ - u'v'^+ = 1, whose rms deviation is 0.03 to
  0.15 percent across the four cases.
- Used by: research.md and DNS_plan.md Step 2 (cross-flow generalization).
- License and notes: free for research use with the citation above. All quantities
  are in wall units (^+); the profile spans one wall to the centerline (the half
  gap), U^+ from 0 to the centerline value (~21.5 at Re_tau 986).

### Raw layout and file format

Per case, DNS_data/couette_flow/roma_couette_<Re_tau>.txt: a header of '*' blocks
and bare prose lines (NOT '%'), the friction Reynolds number only in the title
line "... AT RETAU = <N>", then seven whitespace-separated wall-unit columns:
y, y^+, U^+, u'^+, v'^+, w'^+, u'v'^+. paper.pdf is included alongside for reference.

## Compiled dataset 3: turbulent pipe flow (Pirozzoli 2024)

Status: downloaded and verified 2026-06-27; third-party data, not produced by this
project. Step 2 cross-geometry companion. Loader: UQ.datasets.PipeDNS.

### Provenance

- Source or URL: Roma (Sapienza University of Rome) DNS database, group of
  S. Pirozzoli; distributed with the reference below.
- Reference: S. Pirozzoli, "On the streamwise velocity variance in the near-wall
  region of turbulent flows", J. Fluid Mech. 989 (2024) A5.
- Regime: incompressible
- Cases and per-case parameters (from the '%' header block of each file):

  | file | Re_tau (actual) | Re_bulk | points | sha256 |
  |---|---|---|---|---|
  | Pipe_Re_tau500.txt   | 495.6   | 1.70e4 | 95   | 96381405...4c5b330e |
  | Pipe_Re_tau1140.txt  | 1132.2  | 4.40e4 | 163  | 20025e3b...f677d26c |
  | Pipe_Re_tau2000.txt  | 1972.0  | 8.25e4 | 242  | c001aacb...94ab09c1 |
  | Pipe_Re_tau3000.txt  | 3027.3  | 1.33e5 | 326  | 25c9971c...c6d7d320 |
  | Pipe_Re_tau6000.txt  | 6007.9  | 2.85e5 | 545  | 4c5839cb...26ee13d2 |
  | Pipe_Re_tau12000.txt | 12054.6 | 6.12e5 | 1023 | 959fc065...69be44ef |

- Fields provided: per station (wall units), y^+, mean U_z^+, mean pressure P^+,
  the velocity variances u_z^2, u_r^2, u_t^2 (streamwise, radial, azimuthal), the
  shear covariance u_z u_r, and the pressure variance p^2. No mean-gradient column
  (FD), no k column (computed), no _stdev. P^+ and p^2 are ignored by the loader.
- Cylindrical-to-Cartesian mapping: u_z^2 -> R_xx, u_r^2 -> R_yy (radial =
  wall-normal), u_t^2 -> R_zz (azimuthal = spanwise), and u_z u_r -> R_xy with a
  SIGN FLIP (R_xy = -u_z u_r), because the wall-normal direction into the fluid is
  -r. The flip is fixed, not chosen, by the linear-total-stress identity (below).
- OOD axis it populates: geometry (cross-geometry, channel -> pipe), plus cross-Re.
- Observation uncertainty: MODELED relative value, anchored by the linear-total-
  stress identity dU^+/dy^+ - R_xy = 1 - y^+/Re_tau (rms deviation 0.05 to 0.28
  percent across the six cases; order one if the shear sign flip is omitted).
- Used by: research.md and DNS_plan.md Step 2 (cross-geometry companion).
- License and notes: free for research use with the citation above. Only y^+ is
  given; the outer coordinate is y/R = y^+/Re_tau. The lowest-Reynolds file omits
  the friction-factor header line; it is derived from f = 8(2 Re_tau/Re_bulk)^2.

## Compiled dataset 4: streamwise-rotating channel (University of Manitoba)

Status: downloaded and verified 2026-06-27; third-party data, not produced by this
project. Step 2 model-stress-test companion (standard SST has no rotation
correction). Loader: UQ.datasets.RotatingChannelDNS.

### Provenance

- Source or URL: University of Manitoba streamwise-rotating channel DNS database
  (Department of Mechanical Engineering, Winnipeg).
- Reference: Z. Yang and B.-C. Wang, "Capturing Taylor-Gortler vortices in a
  streamwise-rotating channel at very high rotation numbers", J. Fluid Mech. 838
  (2018) 658-689, doi:10.1017/jfm.2017.892.
- Regime: incompressible
- Cases and per-case parameters (Re_tau = 180 for all; from the FLOW CONDITIONS
  header block):

  | file | Ro_tau | stations (full channel) | sha256 |
  |---|---|---|---|
  | tu-darmstadt_re180_ro7.5  | 7.5 | 129 | 7596fa5e...96bc157c4 |
  | tu-darmstadt_re180_ro15.txt  | 15  | 129 | 89c32d47...725f02cce |
  | tu-darmstadt_re180_ro30.txt  | 30  | 129 | fb455093...4f0e6f2563 |
  | tu-darmstadt_re180_ro75.txt  | 75  | 129 | b4965a23...f96327b80 |
  | tu-darmstadt_re180_ro150.txt | 150 | 129 | 8db787ba...0729352ab |

- Fields provided: per station (wall units), y/h, mean U^+ and W^+, the full
  Reynolds-stress tensor <uu>,<vv>,<ww>,<uv>,<uw>,<vw>^+, TKE^+, and the
  Reynolds-stress-transport budget terms (production, rotation, pressure-strain,
  dissipation, diffusions, residual) for each stress. Loaded columns: the first ten
  plus the six budget residual columns res_*^+. No _stdev.
- OOD axis it populates: system rotation (a case standard SST fails without a
  rotation correction); the rotation number Ro_tau is swept across five conditions.
- Streamwise rotation makes W^+ and <uw>,<vw>^+ nonzero, so the mean velocity
  gradient carries dU^+/dy^+ and dW^+/dy^+ and the full anisotropy is active. The
  profile spans the full channel (y/h from +1 to -1, 129 stations); rotation breaks
  the top/bottom symmetry (U^+ even, W^+ odd about the centerline).
- Observation uncertainty: MODELED relative value, anchored by the file's own RSTE
  budget-closure residual columns res_*^+ (rms 0.06 to 0.44 percent, growing with
  Ro_tau). Files use CRLF line endings, and the Ro_tau 7.5 file has no .txt suffix.
- Used by: research.md and DNS_plan.md Step 2 (rotation model-stress-test).
- License and notes: ALL RIGHTS RESERVED by the University of Manitoba; the data may
  be used WITH REFERENCE (cite Yang and Wang 2018). Honor this citation requirement
  wherever the data is used.

## Compiled dataset 5: periodic hills, parametric slope family (Xiao et al. 2020)

Status: downloaded and verified 2026-07-01; third-party data, not produced by this
project. The dense, two-dimensional separated case for the separated-flow model-form
study, and the cross-geometry generalization axis (hill steepness varied at fixed
Reynolds number). Loader: UQ.datasets.PeriodicHillsDNS.

### Provenance

- Source or URL: https://github.com/xiaoh/para-database-for-PIML (blobless sparse
  checkout of pehill-5-cases-DNS and pehill-5-cases-OpenFOAM; the 7.35 GB
  pehill-29-cases-DNS set was not pulled).
- Reference: H. Xiao, J.-L. Wu, S. Laizet and L. Duan, "Flows over periodic hills of
  parameterized geometries: a dataset for data-driven turbulence modeling from direct
  simulations", Comput. Fluids 200 (2020) 104431.
- Regime: incompressible.
- Cases and per-case parameters: a slope family parameterized by the hill-steepness
  alpha, spanning incipient (0.5) to massive (1.5) separation, at a fixed bulk
  Reynolds number Re_b = 5600 (crest bulk velocity and hill height h = 1;
  cross-checked against the companion OpenFOAM drive Ubar = 0.020188 volume-averaged,
  0.020188 / 0.7210 = 0.028 crest bulk, nu = 5e-6). Two on-disk formats:

  | case | alpha | format | grid (nX x nY) | mean-file sha256 (16) |
  |---|---|---|---|---|
  | case_0p5         | 0.5 | VTK .vtr   | 736 x 385 | 3895d4cb26d623e7 |
  | case_0p8         | 0.8 | ASCII .dat | 704 x 385 | 041e62e5142fbb17 |
  | case_1p0         | 1.0 | ASCII .dat | 512 x 257 | aa81c64e28bb3056 |
  | case_1p0_refined | 1.0 | ASCII .dat | finer mesh | c4635582e1f4a9f7 |
  | case_1p2         | 1.2 | ASCII .dat | 832 x 385 | e2972b8ebd487237 |
  | case_1p5         | 1.5 | VTK .vtr   | 934 x 385 | 8b97dc28a9abf668 |

- Fields provided: mean velocity (U, V, W), mean pressure, and the full
  Reynolds-stress tensor (UU, VV, WW, UV, UW, VW), interpolated onto a rectilinear
  bounding grid with the solid hill interior blanked to exact zeros. Reynolds number
  is fixed across the family, so the varied geometry is the out-of-distribution axis.
- OOD axis it populates: geometry (hill steepness, a cross-geometry axis at constant
  Reynolds number), the dense-field separated case.
- Observation uncertainty: MODELED relative value, anchored by two data-only physics
  residuals computed by the loader: the interpolated mean satisfies continuity
  (du/dx + dv/dy has an RMS of about 1.0 to 1.6 percent of the RMS strain rate on
  interior points), and the DNS Reynolds stress is realizable at every fluid point
  (barycentric check passes at fraction 1.0).
- Used by: the separated-flow model-form study (cross-geometry generalization),
  the incompressible precursor to the compressible shock-boundary-layer study.
- License and notes: no explicit license is stated in the source repository; cite
  Xiao et al. 2020 and treat as research-use-with-citation. All quantities are
  normalized by the crest bulk velocity and the hill height (h = 1). The companion
  pehill-5-cases-OpenFOAM set carries the DNS interpolated onto a coarse RANS-like
  mesh (UDNS, TauDNS) plus the hill mesh, used as a convenience cross-check.

### Standardized processing format

The loader parses both the VTK RectilinearGrid (.vtr) and the ASCII columnar (.dat)
format into the same canonical dns_field record: N flattened field points (x-fastest
tensor order), the mean velocity, the full Reynolds-stress tensor and k, and the mean
velocity-gradient tensor formed by differencing on the grid. A fluid mask excludes
the blanked solid interior and an interior mask marks points whose grid neighbours
are all fluid (a clean central-difference stencil), so the discrepancy, UQ, and
evaluation layers see the same uniform record the wall-bounded loaders emit.

## Compiled dataset 6: backward-facing step (Le, Moin and Kim 1997)

Status: downloaded and verified 2026-07-01; third-party data, not produced by this
project. The sparser cross-geometry companion to the periodic hills for the
separated-flow model-form study. Loader: UQ.datasets.BackwardFacingStepDNS.

### Provenance

- Source or URL: the Le and Moin backward-facing-step DNS (distributed with the
  reference below).
- Reference: H. Le, P. Moin and J. Kim, "Direct numerical simulation of turbulent flow
  over a backward-facing step", J. Fluid Mech. 330 (1997) 349-374 (and the Stanford
  report, Le and Moin 1994).
- Regime: incompressible.
- Parameters: step height h, inlet free-stream U0, Re_h = U0 h / nu = 5100, expansion
  ratio 1.2, published mean reattachment length x_r/h = 6.28.
- Format: wall-normal profiles at six streamwise stations (NOT a dense field). The
  nodal index nnn maps to x/h (readme.txt): 181 to -3, 360 to 4, 411 to 6, 513 to 10,
  641 to 15, 744 to 19 (the six bracket the reattachment).

  | file | contents | columns / notes | sha256 (16) |
  |---|---|---|---|
  | x-nnn.dat    | mean and stresses | y/h, U/U0, V/U0, u'/U0, v'/U0, w'/U0, u'v'/U0^2 | x-411: 95e7a3deb8b39ecf |
  | stat-inf.dat | per-station wall data | x/h, U_e, U_tau, Cf, Cp and BL thicknesses | f739a1dc1c1e9f06 |
  | rs*-nnn.dat  | RSTE budgets | per-component (not required for the discrepancy) | - |

- Fields provided: the two-dimensional mean velocity (U and V) and the full in-plane
  Reynolds-stress tensor (normal stresses are the squares of the rms columns, R_xy is
  the signed u'v' column, spanwise off-diagonals vanish for the spanwise-homogeneous
  mean), plus per-station Cf, Cp, U_e and U_tau. Normalized by the inlet free-stream
  U0. NOTE: readme.txt refers to "stat-info.dat"; the actual file is "stat-inf.dat".
- OOD axis it populates: geometry (separated), the sparser cross-geometry companion.
- Observation uncertainty: MODELED relative value, anchored by two data-only physics
  facts checked by the loader: the DNS Reynolds stress is realizable at every resolved
  station point (fraction 1.0), and the wall Cf changes sign across the reattachment
  (negative inside the recirculation at x/h = 4, positive in recovery), bracketing the
  published x_r/h = 6.28.
- Used by: the separated-flow model-form study (the cross-geometry transfer test
  paired with the periodic hills).
- License and notes: research-use with citation (cite Le, Moin and Kim 1997). This is
  sparse in x, so the model-form target b_DNS = R/(2k) - I/3 comes from the DNS
  Reynolds stress at the profile points (gradient-free), while the Boussinesq baseline
  anisotropy and the conditioning features come from the dense RANS baseline field
  interpolated to these locations; the reattachment-length truth is the published
  x_r/h = 6.28, since six Cf stations are too sparse to locate it directly.

### Standardized processing format

The loader concatenates the six station profiles into the same canonical dns_field
record (N flattened points across the stations, with a station index): the mean
velocity, the full Reynolds-stress tensor and k, the per-station wall quantities, and
a wall-normal-resolved velocity gradient (the streamwise gradient is left zero because
the stations are too sparse to difference in x). Downstream layers consume it exactly
as they consume the dense and wall-bounded records.
