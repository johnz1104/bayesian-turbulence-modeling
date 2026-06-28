# Step 2 plan: cross-flow generalization (Couette), with pipe and rotating-channel companions

Planning stub written before the Step 2 agent runs, to capture the data formats and the
methodological decisions in the tracked archive. The operative launch brief is the Step 2 CLI
prompt; this file is the durable record. Finding-relevance: Step 2 is the cross-flow
generalization that de-risks transfer of the Step 1 calibrated closure plus calibrated UQ to a
held-out flow type (research.md, the supporting result before the separated and compressible
cases).

Status: planning stub. The finding is not yet run. Numbers below are data characterizations
and the Step 1 result, not Step 2 outcomes.

## Goal and finding (the gate)

Calibrate on channel (reuse the Step 1 calibration), predict Couette, and test whether the
calibrated closure plus calibrated UQ transfers to a held-out flow type with reliable
coverage. The reviewer pre-registers the success or null criterion. Pre-registered shape
(DNS_plan.md Step 2): coverage stays near nominal or degrades gracefully and characterizably
under the cross-flow shift, with the conformal coverage gap reported honestly; the
rotating-channel companion is expected to expose where SST breaks and where calibrated
uncertainty should widen. A silent coverage collapse is a negative finding; an already
calibrated transfer (no correction needed) is a null finding and a decision point.

## Data (third-party, cited, not produced here; full provenance in DNS_data/README.md)

| dataset | source | files | format notes |
|---|---|---|---|
| Couette (primary) | Pirozzoli, Bernardini, Orlandi 2014, JFM 742 | couette_flow/roma_couette_{171,260,507,986}.txt | `*`/text header; 7 cols y, y+, U+, u'+, v'+, w'+ (RMS), u'v'+; Re_tau in title only; no nu/u_tau; half-gap |
| Pipe (cross-geometry companion) | Pirozzoli 2024, JFM 989 A5 | pipe_flow/Pipe_Re_tau{500..12000}.txt | `%` header with param block; 8 cols, cylindrical variances u_z^2,u_r^2,u_t^2,u_z u_r; only y+ |
| Rotating channel (SST-failure companion) | University of Manitoba (cite per its license block) | rotating_channel_flow/tu-darmstadt_re180_ro{7.5..150}.txt | `%` + license; 59 cols (use first 10); W+ and uw,vw nonzero; y/h; one file missing `.txt` + CRLF |

Common adjustments versus the channel loader, for all three: single flat file per case (no
subfolders); no `dU/dy` column (finite-difference it); no per-point `_stdev` (see below);
square the RMS columns for Couette; map cylindrical to Cartesian for pipe (u_z->x streamwise,
u_r->y wall-normal, u_t->z spanwise); populate the full anisotropy (both `dU/dy` and `dW/dy`)
for the rotating channel. Each loader emits the same canonical dns_field record.

## Observation-uncertainty model (decision: modeled, physics-anchored)

These files carry no `_stdev`, and the true sampling error is unrecoverable from flat profiles
(it needs the averaging time / integral scale). Use a MODELED relative observation uncertainty
(about 0.5 to 1 percent of the local scale), ANCHORED per case by a data-only physics residual:
Couette constant total stress `dU+/dy+ - u'v'+ = 1` (rms deviation ~0.11 percent at Re_tau 986),
channel and pipe linear total stress, and the rotating-channel budget `res_*` columns. Label it
a modeled observation uncertainty, never as the DNS statistical `_stdev`. This matches Step 1,
which effectively ran on a 0.5 percent floor. Open item for the reviewer: confirm the relative
level.

## Couette baseline and the solver gate (open decision)

The incompressible solver has no Couette moving-wall forward model (`wall_velocity` exists only
on the compressible dbns BoundarySpec). Default per DNS_plan.md: a-priori. Form the Couette
Boussinesq discrepancy and features and test transfer with a defensible profile or analytic
baseline (Couette's exactly-constant total stress makes a minimal baseline well-posed), reusing
the Step 1 channel coefficient posterior rather than recalibrating. A moving-wall 2-D Couette
RANS for a-posteriori coverage on predicted profiles is a gated `core/` stretch, proposed only
if the a-priori result motivates it.

## Reuse and sequencing (zero-setup PR per piece)

Reuse the Step 1 calibration harness (`ChannelCalibration` and the cross-condition protocol),
the UQ modules (discrepancy, realizability, conformal, generalized_bayes, evaluation), and the
evaluation harness. Build order: (1) Couette loader + tests (against the per-case Re_tau and the
constant-total-stress identity); (2) the physics-anchored `observation_sigma` helper; (3) the
a-priori Couette baseline; (4) the cross-flow calibration and coverage (reuse the channel
posterior; evaluate coverage and the cross-flow gap with generalized Bayes and conformal;
Couette at four Re also populates a within-flow cross-Re axis); (5) evaluation on the Couette
QoIs; (6) pipe and rotating-channel a-priori discrepancy/feature companions. Curated findings
land in this folder (`UQ-RANS_research/step2_couette/`).
