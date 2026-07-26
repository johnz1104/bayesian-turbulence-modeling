"""Fixed-seed reproduce driver for the a-priori shock-interaction study.

Stages (composable into one numbers JSON, so long runs chunk cleanly):

  baselines   solve the density-based SST baselines (the attached gate-A
              configuration and the interaction configuration per record),
              record the gate metrics (convergence, skin friction at the
              incoming-profile station, impingement offset), and build the
              strided extraction caches
  loso        leave-one-wall-thermal-out over the interaction records
              (dq_y, joint, db legs; the history ablation on the dq_y leg)
  insample    the train-on-all machinery check (all legs)
  far         attached-to-interaction transfer (dq legs) plus the attached
              leave-one-Mach-family-out control and the Stanton-normalized
              conformal line against its absolute-score control
  all         everything above

Everything runs at the pre-registered settings (strides, epochs 400, seeds
{0, 1, 2}, 128 draws per point); --quick is a smoke path (coarse grids, two
epochs, one seed, quadrupled strides) whose outputs land in *_quick files and
never overwrite production numbers. Baseline solves and extractions cache
under the gitignored results directory keyed by case and stride; delete the
cache or pass --regen to re-run them.

    export QBTM_DNS_DATA=<repo>/DNS_data
    PYTHONPATH=build:python python3 python/UQ/reproduce_sbli_apriori.py \
        --stage all --results results/sbli
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "build"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from UQ import conformal
from UQ import cache_fingerprint as cfp
from UQ.datasets.sbli_interaction import SBLIInteractionDNS, SBLI_S_CASES
from UQ.datasets.sbli_baseline import SBLIBaseline
from UQ.datasets import sbli_apriori
from UQ.datasets.sbli_apriori import (SBLIAPriori, TEST_STRIDE, SBLI_PHYSICS,
                                      sbli_ident, extraction_ident,
                                      extraction_fields_path,
                                      conformal_case_split)

DRIVER_SEED = 0


GATE_A_CF = 2.56e-3        # measured incoming-layer skin friction (dataset)
GATE_A_STATION = -7.65     # the incoming-profile station in x*
ST_FLOOR = 1e-5            # pinned Stanton floor of the normalized score


def _all_records(root=None):
    records = {"adiabatic": SBLIInteractionDNS.adiabatic(root)}
    for s in SBLI_S_CASES:
        records[f"s{s}"] = SBLIInteractionDNS.wall_thermal(s, root)
    return records


def _configure(record, quick, with_shock=True, max_iterations=None,
               convergence_tol=None, early_abort_iter=0,
               early_abort_rel_max=0.0, injection_ramp_iters=0,
               injection_frozen_k=False, frozen_mean=False):
    if quick:
        return SBLIBaseline.configure(record, with_shock=with_shock,
                                      nx=160, ny=112, x_hi=6.0, height=6.0,
                                      cfl=100.0,
                                      max_iterations=max_iterations or 30000,
                                      convergence_tol=convergence_tol
                                      or 3e-6,
                                      early_abort_iter=early_abort_iter,
                                      early_abort_rel_max=early_abort_rel_max,
                                      injection_ramp_iters=injection_ramp_iters,
                                      injection_frozen_k=injection_frozen_k,
                                      frozen_mean=frozen_mean,
                                      yplus_target=0.05)
    # the resolved first cell is load-bearing: the omega wall anchor scales
    # as 1/y1^2 and at y1+ near one it is too weak to select the log-law
    # branch against the spurious near-wall equilibrium (the gate bring-up
    # measured the flip at matched conditions); growth stays near 1.04
    verbose = os.environ.get("QBTM_SBLI_VERBOSE", "") == "1"
    return SBLIBaseline.configure(record, with_shock=with_shock,
                                  verbose=verbose, report_interval=2000,
                                  nx=480, ny=224, x_hi=14.0, height=8.0,
                                  cfl=300.0,
                                  max_iterations=max_iterations or 250000,
                                  convergence_tol=convergence_tol or 1e-6,
                                  early_abort_iter=early_abort_iter,
                                  early_abort_rel_max=early_abort_rel_max,
                                  injection_ramp_iters=injection_ramp_iters,
                                  injection_frozen_k=injection_frozen_k,
                                  frozen_mean=frozen_mean,
                                  yplus_target=0.05)


def _wall_path(results_dir, case):
    return os.path.join(results_dir, f"wall_{case}.npz")


def _gate_ident(results_dir, case):
    """Gate-record identity with the transitive lineage to the exact
    caches the gate measured (a regenerated fields or wall cache
    invalidates the gate record)."""
    return sbli_ident("gate-record", case, gate={
        "lineage": {"fields": cfp.file_sha(_fields_path(results_dir, case)),
                    "wall": cfp.file_sha(_wall_path(results_dir, case))}})


def _json_ident_ok(path, ident):
    if not os.path.isfile(path):
        return False
    status, _ = cfp.check_json(json.load(open(path)), ident)
    return status == "match"


def _partial_ident(results_dir, stage, leg, seeds, epochs):
    """Per-stage or per-leg partial-result identity: the settings plus the
    exact extraction caches the stage consumed (every case, both strides,
    history files), so a partial computed on superseded extractions can
    never assemble into the numbers (existence-based assembly was the
    review's stale-partial finding)."""
    lin = {}
    for c in sorted(TEST_STRIDE):
        for st in (TEST_STRIDE[c], sbli_apriori._train_stride(c)):
            pth = SBLIAPriori._cache_path(results_dir, c, st, True)
            lin[os.path.basename(pth)] = cfp.file_sha(pth)
    return sbli_ident("apriori-partial", f"{stage}:{leg or 'all'}",
                      part={"seeds": list(seeds), "epochs": int(epochs),
                            "lineage": lin})


def _save_wall(results_dir, case, w):
    """The wall record binds the exact fields cache of the same solve (its
    lineage parent), so the fields MUST be saved first at every call site;
    a mutated or regenerated fields file invalidates the wall record."""
    arrays = {"x_star": w["x_star"], "Cf": w["Cf"], "Cp": w["Cp"],
              "qw": w["qw"], "St": w["St"]}
    ident = sbli_ident("sbli_wall", case, wall={
        "lineage": {"fields": cfp.file_sha(_fields_path(results_dir,
                                                        case))}})
    cfp.savez_atomic(_wall_path(results_dir, case),
                     cfp.attach(arrays, ident))


def _fields_path(results_dir, case):
    return os.path.join(results_dir, f"fields_{case}.npz")


def _save_fields(results_dir, case, solver):
    """Persist the converged primitive state (n_cells, 6): the a-posteriori
    members warm-start from it (init_field) so each is a perturbation solve,
    and the fold targets condition on the same converged baseline. `case`
    is the cache tag (the frozen-mean adiabatic march saves under
    adiabatic_frozenmean, its own identity)."""
    f = solver.fields()
    prim = np.stack([np.asarray(f[k], dtype=float)
                     for k in ("rho", "u", "v", "p", "k", "omega")], axis=1)
    arrays = {"primitive": prim}
    if hasattr(solver, "reconstruction_limiter"):
        # the Venkatakrishnan limiter state of the converged solve: the
        # member warm-starts restore it so a reloaded state resumes the
        # exact discrete operator it converged under (checkpoint-restart
        # semantics; distinct from the SST limiter_active branch record)
        arrays["limiter_recon"] = np.asarray(solver.reconstruction_limiter(),
                                             dtype=np.float64)
    if hasattr(solver, "limiter_active"):
        # the solver's own omega-limiter branch record from the last residual
        # evaluation of the converged solve: the EXACT activation map (the
        # python recomputation stays as the approximate fallback for caches
        # that predate this export)
        arrays["limiter_active"] = np.asarray(solver.limiter_active(),
                                              dtype=bool)
    cfp.savez_atomic(_fields_path(results_dir, case),
                     cfp.attach(arrays, sbli_ident("sbli_fields", case)))


def _gate_a_profile_metrics(base, record, cf_station):
    """The pre-registered gate-A incoming-layer clauses beyond skin friction:
    the momentum-thickness Reynolds number against the record's own value at
    the same station, and the van-Driest transformed log-region RMS.

    Conventions (one rule for both sides): density from the boundary-layer
    constant-pressure relation rho_hat = 1/T_hat; theta from the standard
    momentum-thickness integral in delta0 units, Re_theta = theta_hat times
    the dataset inlet Reynolds number; van-Driest u_vd = int sqrt(rho/rho_w)
    du with u_tau from each side's OWN wall stress (the solve's station Cf,
    the record's measured gate value), and the log-region band
    y+ in [30, 0.3 delta+]. The RMS is of the relative U+ difference over
    that band, interpolated in log y+.
    """
    ys = np.asarray(record.y, float)
    i_st = int(np.argmin(np.abs(np.asarray(record.x, float) - GATE_A_STATION)))
    u_dns = np.asarray(record.U, float)[i_st]
    T_dns = np.maximum(np.asarray(record.T, float)[i_st], 1e-6)

    s = base.sample_fields(np.full(ys.size, GATE_A_STATION), ys)
    u_sol = np.asarray(s["u"], float)
    T_sol = np.maximum(np.asarray(s["T"], float), 1e-6)

    def theta_hat(u, T):
        # integrate to the layer's own edge (the first u = 0.99 crossing):
        # the full-height integral accumulates far-field noise where 1 - u
        # hovers at the free-stream residual, inflating theta by multiples
        # (the review's Re_theta finding); the metric stays report-only
        rho = 1.0 / T
        above = np.nonzero(u >= 0.99)[0]
        i_edge = int(above[0]) + 1 if above.size else ys.size
        integrand = rho[:i_edge] * np.clip(u[:i_edge], 0.0, None) \
            * (1.0 - np.clip(u[:i_edge], 0.0, 1.0))
        return float(np.trapz(integrand, ys[:i_edge]))

    re_inlet = 16750.0
    re_theta_sol = theta_hat(u_sol, T_sol) * re_inlet
    re_theta_dns = theta_hat(u_dns, T_dns) * re_inlet

    def vd_profile(u, T, cf):
        rho = 1.0 / T
        rho_w = rho[0]
        u_tau = np.sqrt(max(cf, 1e-12) / 2.0 / rho_w)
        u_vd = np.concatenate([[0.0], np.cumsum(
            np.sqrt(rho[1:] / rho_w) * np.diff(u))])
        y_plus = ys * rho_w * u_tau * re_inlet
        return y_plus, u_vd / u_tau

    yp_s, up_s = vd_profile(u_sol, T_sol, cf_station)
    yp_d, up_d = vd_profile(u_dns, T_dns, GATE_A_CF)
    delta_plus = float(np.interp(0.99, np.clip(u_dns, 0, 1), yp_d))
    lo, hi = 30.0, 0.3 * delta_plus
    band = np.geomspace(max(lo, yp_d[1]), max(hi, lo * 1.5), 24)
    up_s_b = np.interp(np.log(band), np.log(np.maximum(yp_s, 1e-12)), up_s)
    up_d_b = np.interp(np.log(band), np.log(np.maximum(yp_d, 1e-12)), up_d)
    vd_rms = float(np.sqrt(np.mean(((up_s_b - up_d_b) / up_d_b) ** 2)))
    return {
        "re_theta_solve": re_theta_sol, "re_theta_dns": re_theta_dns,
        "re_theta_ratio": float(re_theta_sol / max(re_theta_dns, 1e-12)),
        "vd_log_rms": vd_rms,
        "vd_band_yplus": [float(band[0]), float(band[-1])],
    }


def _impingement_offset(w, record):
    """The solve's wall-pressure half-rise against the record's own
    landmark, through the REGISTERED landmark rule on the solve side (the
    records' smoothing width and the linearly interpolated level crossing,
    landmarks_from_wall, one rule for every solve). The earlier raw-series
    grid-index reading quantized the offset at the streamwise cell width,
    the review's audit-metric finding."""
    from UQ.datasets.sbli_aposteriori import landmarks_from_wall
    x_half = landmarks_from_wall(w)["shock"]
    dns_half = record.shock_position()
    if x_half is None or dns_half is None:
        return None, x_half, dns_half
    return float(x_half - dns_half), x_half, dns_half


def _extraction_current(results_dir, case, history=True):
    """True when every extraction cache for the case matches its full
    identity, including the transitive lineage edge to the current
    converged-fields cache; a stale cache regenerates from the CACHED
    converged fields through a warm-loaded baseline, never a re-solve."""
    strides = (TEST_STRIDE[case], sbli_apriori._train_stride(case))
    for st in strides:
        path = SBLIAPriori._cache_path(results_dir, case, st, history)
        if not os.path.isfile(path):
            return False
        z = np.load(path, allow_pickle=True)
        status, _ = cfp.check({k: z[k] for k in z.files},
                              extraction_ident(case, st, history,
                                               results_dir))
        if status != "match":
            return False
    return True


def _warm_baseline(records, case, results_dir, quick, with_shock=True):
    """Rebuild the case and warm it with the cached converged primitive
    state; prepare_properties populates the derived fields the extraction
    samples WITHOUT advancing the state (the one-iteration solve this once
    used drifted the cached state by the order of the converged residual,
    the review's extraction-advance finding; the primitive state stays
    byte-identical, pinned by test). The adiabatic case warms its
    FROZEN-MEAN state (the registered gate-B fallback route of its
    a-priori role)."""
    frozen = case == "adiabatic"
    base = _configure(records[case], quick, with_shock=with_shock,
                      frozen_mean=frozen)
    tag = case if with_shock else "gate_a_attached"
    if frozen:
        tag = "adiabatic_frozenmean"
    prim = np.load(_fields_path(results_dir, tag))["primitive"]
    base.solver.init_field(prim)
    base.solver.prepare_properties()
    return base


def _npz_ident_ok(path, ident):
    """True when the cache exists AND carries the matching identity; a
    pre-fingerprint or mismatched cache is stale (path existence alone
    authorized reuse of artifacts whose producing solver could not be
    proven, the review's dependency-identity finding)."""
    if not os.path.isfile(path):
        return False
    z = np.load(path, allow_pickle=True)
    status, _ = cfp.check({k: z[k] for k in z.files}, ident)
    return status == "match"


def _case_cached(results_dir, case, history=True):
    strides = (TEST_STRIDE[case], sbli_apriori._train_stride(case))
    have = all(os.path.isfile(SBLIAPriori._cache_path(
        results_dir, case, st, history)) for st in strides)
    wall_ident = sbli_ident("sbli_wall", case, wall={
        "lineage": {"fields": cfp.file_sha(_fields_path(results_dir,
                                                        case))}})
    return (have
            and _npz_ident_ok(_wall_path(results_dir, case), wall_ident)
            and _npz_ident_ok(_fields_path(results_dir, case),
                              sbli_ident("sbli_fields", case)))


def ensure_frozen_mean_adiabatic(records, results_dir, quick, regen=False):
    """March the frozen-mean adiabatic transport baseline (the registered
    gate-B fallback of the adiabatic a-priori surface) if its fields cache
    is not identity-current: the primitive mean pinned to the record's own
    Favre mean, k and omega marched to steadiness. A non-converged march is
    a stop-and-report state (the registered fallback route itself would be
    unavailable)."""
    fm_path = _fields_path(results_dir, "adiabatic_frozenmean")
    fm_ident = sbli_ident("sbli_fields", "adiabatic_frozenmean")
    if not regen and _npz_ident_ok(fm_path, fm_ident):
        return
    t0 = time.time()
    fm = _configure(records["adiabatic"], quick, with_shock=True,
                    frozen_mean=True)
    fm.init_from_record_mean()
    rep = fm.solve()
    entry = {"status": str(rep.status),
             "iterations": rep.iterations,
             "final_residual": float(rep.final_residual),
             "wall_time_s": round(time.time() - t0, 1)}
    os.makedirs(results_dir, exist_ok=True)
    print(f"[frozen-mean adiabatic] {rep.status} iters {rep.iterations}")
    if str(rep.status).endswith("Converged"):
        _save_fields(results_dir, "adiabatic_frozenmean", fm.solver)
        cfp.json_atomic(
            os.path.join(results_dir, "frozen_mean_adiabatic.json"),
            cfp.attach_json(entry, sbli_ident(
                "frozen-mean-march", "adiabatic_frozenmean",
                march={"lineage": {"fields": cfp.file_sha(_fields_path(
                    results_dir, "adiabatic_frozenmean"))}})))
    else:
        cfp.json_atomic(
            os.path.join(results_dir, "frozen_mean_adiabatic.json"),
            cfp.attach_json(entry, sbli_ident(
                "frozen-mean-march", "adiabatic_frozenmean")))
        print("[frozen-mean adiabatic] NOT converged: cache withheld; the "
              "registered fallback route is unavailable, stopping (report "
              "before any downstream stage)")
        sys.exit(1)


def stage_baselines(records, results_dir, quick, regen,
                    skip_extract=False, only=None):
    """Solve the baselines where their caches are missing, measure the
    gates, and build the extraction caches. Cached cases never re-solve
    unless --regen. skip_extract stops after the gate adjudication (the
    baseline-only cold-regeneration boundary: extraction regeneration
    belongs to the released a-priori phase, not the substrate rebuild).
    only (a set of {"gatea", "frozenmean", "gateb"}) restricts which solve
    blocks run, so independent long solves parallelize across processes; a
    restricted pass never writes the adjudication (the unrestricted or
    refresh pass assembles it from the per-case records)."""
    only = set(only) if only else None

    def _runs(block):
        return only is None or block in only

    out = {"gates": {"B": {}}, "solves": {}}
    baselines = {}

    # gate A: the attached configuration on the adiabatic record
    gate_a_path = os.path.join(results_dir, "gate_a.json")
    if not _runs("gatea"):
        if os.path.isfile(gate_a_path):
            out["gates"]["A"] = json.load(open(gate_a_path))
        print("[gate A] outside this pass restriction")
    elif not regen and _json_ident_ok(gate_a_path,
                                    _gate_ident(results_dir,
                                                "gate_a_attached")):
        out["gates"]["A"] = json.load(open(gate_a_path))
        print("[gate A] cached:", out["gates"]["A"]["status"])
    elif "adiabatic" not in records:
        print("[gate A] skipped: adiabatic record not in this partition")
    else:
        t0 = time.time()
        gate_a = _configure(records["adiabatic"], quick, with_shock=False)
        rep = gate_a.solve()
        w = gate_a.wall()
        cf_station = float(np.interp(GATE_A_STATION, w["x_star"], w["Cf"]))
        out["gates"]["A"] = {
            "status": str(rep.status), "iterations": rep.iterations,
            "final_residual": float(rep.final_residual),
            "cf_at_station": cf_station, "cf_data": GATE_A_CF,
            "cf_rel_error": float(abs(cf_station / GATE_A_CF - 1.0)),
            "wall_time_s": round(time.time() - t0, 1),
        }
        out["gates"]["A"].update(
            _gate_a_profile_metrics(gate_a, records["adiabatic"], cf_station))
        out["gates"]["A"]["pass"] = bool(
            str(rep.status).endswith("Converged")
            and out["gates"]["A"]["cf_rel_error"] <= 0.10
            and out["gates"]["A"]["vd_log_rms"] <= 0.05)
        os.makedirs(results_dir, exist_ok=True)
        if str(rep.status).endswith("Converged"):
            _save_fields(results_dir, "gate_a_attached", gate_a.solver)
            _save_wall(results_dir, "gate_a_attached", w)
        else:
            print("[gate A] NOT converged: caches withheld")
        cfp.json_atomic(gate_a_path, cfp.attach_json(
            out["gates"]["A"], _gate_ident(results_dir, "gate_a_attached")))
        print(f"[gate A] {rep.status} iters {rep.iterations} "
              f"cf {cf_station:.3e} vs {GATE_A_CF:.3e} "
              f"({out['gates']['A']['cf_rel_error']*100:.1f} percent)")

    # the adiabatic 2011 campaign's a-priori role takes the registered
    # frozen-mean fallback (the reviewer's per-case gate-B ruling): its
    # extraction parent is the frozen-mean transport march at the record's
    # own Favre mean, never the gate-failing free-running interaction solve
    # (which stays cached only for the gate record and the labeled
    # exploratory legs). Prepared before the gate-B loop so any
    # extraction-regeneration warm load finds the cache in place.
    if "adiabatic" in records and _runs("frozenmean"):
        ensure_frozen_mean_adiabatic(records, results_dir, quick, regen)

    # gate B: the interaction configuration per record, solve-once cached
    for case, record in (records.items() if _runs("gateb") else ()):
        gate_path = os.path.join(results_dir, f"gate_b_{case}.json")
        if _case_cached(results_dir, case) and not regen \
                and _json_ident_ok(gate_path, _gate_ident(results_dir,
                                                          case)):
            out["gates"]["B"][case] = json.load(open(gate_path))
            if _extraction_current(results_dir, case):
                baselines[case] = None  # extraction reads the caches
            else:
                # stale-format extraction: regenerate from the cached
                # converged fields (a warm reload, never a re-solve)
                baselines[case] = _warm_baseline(records, case, results_dir,
                                                 quick)
            print(f"[gate B {case}] cached: "
                  f"{out['gates']['B'][case]['status']}")
            continue
        t0 = time.time()
        base = _configure(record, quick, with_shock=True)
        rep = base.solve()
        w = base.wall()
        offset, x_half, dns_half = _impingement_offset(w, record)
        entry = {
            "status": str(rep.status), "iterations": rep.iterations,
            "final_residual": float(rep.final_residual),
            "impingement_solve": x_half, "impingement_dns": dns_half,
            "offset": offset,
            "wall_time_s": round(time.time() - t0, 1),
        }
        entry["pass"] = bool(str(rep.status).endswith("Converged")
                             and offset is not None and abs(offset) <= 1.0)
        out["gates"]["B"][case] = entry
        os.makedirs(results_dir, exist_ok=True)
        if str(rep.status).endswith("Converged"):
            _save_fields(results_dir, case, base.solver)
            _save_wall(results_dir, case, w)
        else:
            print(f"[gate B {case}] NOT converged: caches withheld")
        cfp.json_atomic(gate_path, cfp.attach_json(
            entry, _gate_ident(results_dir, case)))
        print(f"[gate B {case}] {rep.status} iters {rep.iterations} "
              f"offset {offset}")
        baselines[case] = base

    # the hard gate adjudication record, written before ANY downstream stage:
    # gate A must pass outright; gate B is adjudicated PER CASE under the
    # reviewer's recorded 2026-07-18 ruling (the pinned propagated folds all
    # pass; a failing case is excluded from claim-bearing coupled legs, takes
    # the registered frozen-mean fallback on its a-priori role, and any
    # injection probe on it is EXPLORATORY, never gate rehabilitation)
    if only is not None:
        print(f"[baselines] restricted pass {sorted(only)} complete; "
              f"adjudication deferred to the unrestricted or refresh pass")
        return out, None
    adjud = {
        "ruling": "per-case (reviewer adjudication 2026-07-18); gate "
                  "failures take the registered fallback and are excluded "
                  "from claim-bearing coupled legs",
        "gate_a_pass": bool(out["gates"].get("A", {}).get("pass", False)),
        "gate_b_pass_cases": sorted(c for c, e in out["gates"]["B"].items()
                                    if e.get("pass")),
        "gate_b_fail_cases": sorted(c for c, e in out["gates"]["B"].items()
                                    if not e.get("pass")),
    }
    lin = {"gate_a.json": cfp.file_sha(os.path.join(results_dir,
                                                    "gate_a.json"))}
    for c in sorted(out["gates"]["B"]):
        lin[f"gate_b_{c}.json"] = cfp.file_sha(
            os.path.join(results_dir, f"gate_b_{c}.json"))
    cfp.json_atomic(os.path.join(results_dir, "gates_adjudication.json"),
                    cfp.attach_json(adjud, sbli_ident(
                        "gates-adjudication", "all-cases",
                        adjud={"lineage": lin})))
    if not adjud["gate_a_pass"]:
        print("[gates] GATE A FAILED: stopping before any downstream stage "
              "(solver-capability null; see the pre-registration)")
        sys.exit(1)
    if adjud["gate_b_fail_cases"]:
        print(f"[gates] gate B failing cases {adjud['gate_b_fail_cases']}: "
              f"excluded from claim-bearing coupled legs per the recorded "
              f"ruling; passing cases {adjud['gate_b_pass_cases']}")
    if skip_extract:
        print("[baselines] extraction build skipped (baseline-only pass)")
        return out, None

    # the adiabatic extraction baseline is the frozen-mean state regardless
    # of what the gate-B loop assigned (its free-running solve serves the
    # gate record only)
    if "adiabatic" in records:
        baselines["adiabatic"] = (
            None if _extraction_current(results_dir, "adiabatic")
            else _warm_baseline(records, "adiabatic", results_dir, quick))

    # extraction caches at both strides (with the history feature); cached
    # cases pass baseline None and are never touched
    study = SBLIAPriori.build(records, baselines, results_dir, history=True)
    for case, ext in study.test_sets.items():
        out["solves"][case] = {
            "n_test_rows": int(ext["meta"]["n_points"]),
            "realizable_fraction": float(ext["realizable_fraction"]),
        }

    # the deployed-path basis-feasibility record (the standalone divergent
    # config is retired): the same per-fold gate the training legs consult,
    # persisted as the evidence artifact
    heated = {c: e for c, e in study.test_sets.items()
              if e["dq"] is not None}
    if heated:
        cfp.json_atomic(os.path.join(results_dir, "basis_feasibility.json"),
                        study.db_gate())
    return out, study


def stage_loso(study, seeds, epochs):
    out = {}
    for leg in ("dq_y", "dq_joint", "db"):
        out[leg] = study.loso(leg, history=False, seeds=seeds, epochs=epochs)
    out["dq_y_history"] = study.loso("dq_y", history=True, seeds=seeds,
                                     epochs=epochs)
    return out


def stage_insample(study, seeds, epochs):
    return {leg: study.insample(leg, history=False, seeds=seeds,
                                epochs=epochs)
            for leg in ("dq_y", "dq_joint", "db")}


RECOVERY_T_HAT = 1.9318    # measured recovery wall temperature, T/T_inf


def _st_scale(study, case, record):
    """The baseline's predicted local wall flux as the normalized-conformal
    scale, CONVERTED into the same (u_tau_ref, T_w) flux units the residuals
    carry (a raw Stanton number lives in free-stream units and would shrink
    the scale by an order of magnitude for the cooled cases):

        q_hat_scale(x) = |St(x)| (T_aw - T_w) / (rho_w u_tau T_w)

    in free-stream nondimensional quantities from the record's own reference
    state, with the measured turbulent recovery temperature. Floored at the
    pinned Stanton floor AFTER conversion of the floor through the same
    factor, so the adiabatic exclusion semantics survive the unit change."""
    ext = study.test_sets[case]
    wall_path = os.path.join(study.results_dir, f"wall_{case}.npz")
    if not os.path.isfile(wall_path):
        return None
    w = np.load(wall_path)
    st = np.interp(ext["x"], w["x_star"], w["St"])
    ref = record.reference
    conv = abs(RECOVERY_T_HAT - ref["T_w"]) / (
        ref["rho_w"] * ref["u_tau_over_uinf"] * ref["T_w"])
    return np.maximum(np.abs(st) * conv, ST_FLOOR * conv)


def stage_far(study, seeds, epochs, records):
    out = {"transfer": {}, "control": {}, "conformal": {}}
    for leg in ("dq_y", "dq_joint", "db"):
        out["transfer"][leg] = study.far_transfer(leg, seeds=seeds,
                                                  epochs=epochs)
    # the labeled ZDC-anisotropy sensitivity variant of the registered db
    # pool (the pre-registration's plates clause), never a primary row
    out["transfer"]["db_plates_sensitivity"] = study.far_transfer(
        "db", seeds=seeds, epochs=epochs, plates_sensitivity=True)
    out["control"]["dq_y"] = study.attached_control("dq_y", seeds=seeds,
                                                    epochs=epochs)

    # the Stanton-normalized conformal line (the established first-line
    # thermal correction) against the absolute-score control. REDESIGNED
    # roles (the review's fit-equals-calibration finding): the attached
    # channel cases split at WHOLE-CASE level into disjoint fit and
    # calibration sets (the frozen within-Mach-family alternation), the
    # conformal predictor trains on the fit cases only, and no
    # calibration-case row ever enters the predictor's training.
    # CASE-LEVEL calibration (amendment 4 plus the dated 2026-07-26 note):
    # the exchangeable units are whole cases, so the conformal score of a
    # calibration case is the 0.90 empirical quantile of its normalized row
    # residuals, and the band quantile is the finite-sample split-conformal
    # quantile ACROSS the calibration-case scores (with about twelve cases
    # at alpha 0.10 that is their maximum), never a quantile over pooled
    # correlated rows. Per pinned model seed, criterion on the seed mean.
    import torch
    fit_tags, cal_tags = conformal_case_split(study.attached.gv)
    X_fit, dq_fit = study._attached_dq_pool(fit_tags)
    Y_fit = dq_fit[:, 1:2]
    out["conformal"]["roles"] = {
        "fit_cases": list(fit_tags),
        "calibration_cases": list(cal_tags),
        "disjoint": not (set(fit_tags) & set(cal_tags)),
        "rule": "whole-case within-Mach-family alternation, frozen before "
                "any corrected-lineage result",
        "score": "per-case 0.90 empirical quantile of the (normalized) "
                 "row residuals; band quantile = finite-sample conformal "
                 "quantile across calibration-case scores",
        "n_calibration_cases": len(cal_tags),
    }
    per_seed = {}
    for seed in seeds:
        model = study._make("gauss", X_fit.shape[1], 1, seed)
        model.fit(X_fit, Y_fit, epochs=epochs, lr=1e-3, batch=256)
        case_abs, case_norm = [], []
        for tag in cal_tags:
            rec = study.attached.cases[tag]
            torch.manual_seed(seed)
            S = np.asarray(model.sample(rec["features"], n_per=128))
            m = np.median(S[:, :, 0], axis=1)
            resid = np.abs(rec["dq"][:, 1] - m)
            scale = max(abs(rec["b_q_base"]), 1e-4)
            case_abs.append(float(np.quantile(resid, 0.90)))
            case_norm.append(float(np.quantile(resid / scale, 0.90)))
        q_abs = float(conformal.conformal_quantile(
            np.asarray(case_abs), alpha=0.10))
        q_norm = float(conformal.conformal_quantile(
            np.asarray(case_norm), alpha=0.10))
        entry = {"q_abs": q_abs, "q_norm": q_norm,
                 "calibration_case_scores": {"absolute": case_abs,
                                             "normalized": case_norm},
                 "cases": {}}
        for case, ext in study.test_sets.items():
            if ext["dq"] is None:
                continue
            X_te = study._features(ext, history=False)
            Y_te = ext["dq"][:, 1]
            torch.manual_seed(seed)
            S_te = np.asarray(model.sample(X_te, n_per=128))
            m_te = np.median(S_te[:, :, 0], axis=1)
            resid = np.abs(Y_te - m_te)
            scale = _st_scale(study, case, records[case])
            row = {"absolute": float(np.mean(resid <= q_abs)),
                   "n": int(Y_te.size)}
            if scale is not None:
                row["normalized"] = float(np.mean(resid <= q_norm * scale))
            entry["cases"][case] = row
        per_seed[str(seed)] = entry
    out["conformal"]["per_seed"] = per_seed
    mean_cases = {}
    for case in next(iter(per_seed.values()))["cases"]:
        vals_a = [per_seed[str(sd)]["cases"][case]["absolute"]
                  for sd in seeds]
        mean_cases[case] = {"absolute": float(np.mean(vals_a))}
        if "normalized" in per_seed[str(seeds[0])]["cases"][case]:
            mean_cases[case]["normalized"] = float(np.mean(
                [per_seed[str(sd)]["cases"][case]["normalized"]
                 for sd in seeds]))
    out["conformal"]["seed_mean"] = mean_cases
    return out


RECONVERGE_CASES = ("gate_a_attached", "adiabatic", "s0.5", "s0.75",
                    "s1.0", "s1.4", "s1.9")


def stage_reconverge(records, results_dir, quick, cases):
    """The wall-transport equivalence audit and schema migration. Per cached
    converged baseline: (1) measure the direct physics delta of the
    molecular-wall convention on the SAVED state (the eddy-to-molecular
    viscosity ratio at the wall-adjacent row, which bounds the relative
    wall-flux change); (2) regenerate the wall observables from the saved
    fields under the corrected observation operator and compare; (3)
    warm-reconverge the solve under the corrected solver to the production
    criterion; (4) compare the reconverged state and gate metrics against
    the saved ones; (5) rewrite the fields/wall/gate caches under the
    current physics schema. The pre-migration caches are the migration
    INPUT, loaded explicitly rather than identity-reused. Re-running the
    baselines stage afterwards refreshes gates_adjudication.json from the
    rewritten per-case gate records."""
    for case in cases:
        with_shock = case != "gate_a_attached"
        record = records["adiabatic" if case == "gate_a_attached" else case]
        old_path = _fields_path(results_dir, case)
        if not os.path.isfile(old_path):
            print(f"[reconverge {case}] no fields cache; skipped")
            continue
        old = np.load(old_path)
        prim_old = np.asarray(old["primitive"], dtype=float)
        audit = {"case": case}
        # preserve the pre-migration gate record inside the audit before
        # the rewrite (the cold-solve history stays quotable)
        gate_file = ("gate_a.json" if case == "gate_a_attached"
                     else f"gate_b_{case}.json")
        gp = os.path.join(results_dir, gate_file)
        if os.path.isfile(gp):
            audit["gate_pre_migration"] = json.load(open(gp))

        # (1) the direct physics delta on the saved state
        base = _configure(record, quick, with_shock=with_shock)
        base.solver.init_field(prim_old)
        base.solver.prepare_properties()
        u = base.units
        cc = np.asarray(base.mesh.cell_centers())
        xs_wall = np.unique(np.round(cc[:, 0], 12))
        y1_star = float(cc[:, 1].min()) / u.delta0
        s = base.sample_fields(xs_wall / u.delta0 + base.meta["x_lo"],
                               np.full(xs_wall.size, y1_star))
        T_dim = np.asarray(s["T"], float) * u.T_inf
        mu_lam = np.array([u.eos.viscosity(t) for t in T_dim])
        nu_lam_hat = mu_lam / (np.asarray(s["rho"], float) * u.rho_inf) \
            / (u.U_inf * u.delta0)
        ratio = np.asarray(s["nu_t"], float) / np.maximum(nu_lam_hat, 1e-300)
        audit["wall_row_mut_over_mu"] = {
            "median": float(np.median(ratio)), "max": float(np.max(ratio))}

        # (2) wall observables from the saved state under the corrected
        # observation operator, against the stored wall cache
        w_frozen = base.wall()
        wall_path = _wall_path(results_dir, case)
        if os.path.isfile(wall_path):
            wz = np.load(wall_path)
            deltas = {}
            for q in ("Cf", "Cp", "qw", "St"):
                a = np.asarray(w_frozen[q], float)
                b = np.asarray(wz[q], float)
                scale = np.max(np.abs(b)) or 1.0
                deltas[q] = float(np.max(np.abs(a - b)) / scale)
            audit["wall_obs_max_rel_delta_saved_state"] = deltas

        # (3) warm-reconverge under the corrected solver
        t0 = time.time()
        solve = _configure(record, quick, with_shock=with_shock)
        solve.solver.init_field(prim_old)
        rep = solve.solve()
        audit["reconverge"] = {
            "status": str(rep.status), "iterations": rep.iterations,
            "final_residual": float(rep.final_residual),
            "wall_time_s": round(time.time() - t0, 1)}
        print(f"[reconverge {case}] {rep.status} iters {rep.iterations} "
              f"({audit['reconverge']['wall_time_s']}s)")
        if not str(rep.status).endswith("Converged"):
            cfp.json_atomic(os.path.join(results_dir,
                                         f"reconverge_{case}.json"), audit)
            print(f"[reconverge {case}] NOT converged: caches NOT "
                  f"rewritten; adjudicate before proceeding")
            continue

        # (4) state drift and gate metrics against the saved record
        f = solve.solver.fields()
        prim_new = np.stack([np.asarray(f[k], dtype=float)
                             for k in ("rho", "u", "v", "p", "k", "omega")],
                            axis=1)
        drift = {}
        for j, name in enumerate(("rho", "u", "v", "p", "k", "omega")):
            scale = np.max(np.abs(prim_old[:, j])) or 1.0
            drift[name] = float(np.max(np.abs(prim_new[:, j]
                                              - prim_old[:, j])) / scale)
        audit["field_drift_max_rel"] = drift
        w = solve.wall()
        if case == "gate_a_attached":
            cf_station = float(np.interp(GATE_A_STATION, w["x_star"],
                                         w["Cf"]))
            entry = {
                "status": str(rep.status), "iterations": rep.iterations,
                "final_residual": float(rep.final_residual),
                "cf_at_station": cf_station, "cf_data": GATE_A_CF,
                "cf_rel_error": float(abs(cf_station / GATE_A_CF - 1.0)),
                "warm_reconverge": True,
            }
            entry.update(_gate_a_profile_metrics(solve, record, cf_station))
            entry["pass"] = bool(entry["cf_rel_error"] <= 0.10
                                 and entry["vd_log_rms"] <= 0.05)
            audit["gate"] = entry
        else:
            offset, x_half, dns_half = _impingement_offset(w, record)
            entry = {
                "status": str(rep.status), "iterations": rep.iterations,
                "final_residual": float(rep.final_residual),
                "impingement_solve": x_half, "impingement_dns": dns_half,
                "offset": offset, "warm_reconverge": True,
            }
            entry["pass"] = bool(offset is not None and abs(offset) <= 1.0)
            audit["gate"] = entry

        # (5) rewrite the caches under the current schema, then the gate
        # record bound to them (its lineage needs the final files on disk)
        _save_fields(results_dir, case, solve.solver)
        _save_wall(results_dir, case, w)
        gate_file = ("gate_a.json" if case == "gate_a_attached"
                     else f"gate_b_{case}.json")
        cfp.json_atomic(os.path.join(results_dir, gate_file),
                        cfp.attach_json(audit["gate"],
                                        _gate_ident(results_dir, case)))
        cfp.json_atomic(os.path.join(results_dir,
                                     f"reconverge_{case}.json"),
                        cfp.attach_json(audit, sbli_ident(
                            "reconverge-audit", case)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("baselines", "loso", "insample", "far",
                             "assemble", "reconverge", "all"))
    ap.add_argument("--results", default="results/sbli")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--only", default="",
                    help="baselines stage only: comma set of "
                         "{gatea,frozenmean,gateb} restricting which solve "
                         "blocks run (parallel substrate rebuilds); implies "
                         "no adjudication write")
    ap.add_argument("--skip-extract", action="store_true",
                    dest="skip_extract",
                    help="baselines stage only: stop after the gate "
                         "adjudication (the substrate cold-regeneration "
                         "boundary; extraction regeneration belongs to the "
                         "released a-priori phase)")
    ap.add_argument("--cases", default="",
                    help="comma list restricting the baseline solves (e.g. "
                         "adiabatic,s0.5); partitions run in parallel and a "
                         "final unrestricted pass assembles the cached gates")
    ap.add_argument("--legs", default="",
                    help="comma list restricting a loso/insample stage to "
                         "named target legs (dq_y, dq_joint, db, "
                         "dq_y_history), one per-leg partial file each; the "
                         "memory-bounded path on small machines is one leg "
                         "per process, merged by the assemble stage")
    args = ap.parse_args()

    np.random.seed(DRIVER_SEED)
    seeds = (0,) if args.quick else sbli_apriori.SEEDS
    epochs = 2 if args.quick else sbli_apriori.EPOCHS
    suffix = "_quick" if args.quick else ""
    if args.quick and os.path.basename(
            os.path.normpath(args.results)) != "quick":
        # the smoke path gets its own cache universe so a quick run can
        # never clobber production baselines, fields or gate records
        args.results = os.path.join(args.results, "quick")
    os.makedirs(args.results, exist_ok=True)
    numbers_path = os.path.join(args.results,
                                f"apriori_numbers{suffix}.json")
    numbers = {}
    if os.path.isfile(numbers_path):
        numbers = json.load(open(numbers_path))

    records = _all_records()
    if args.stage == "reconverge":
        # --cases here names the RECONVERGE list (fields-cache tags, e.g.
        # gate_a_attached), not a record partition; default is every case
        cases = ([c.strip() for c in args.cases.split(",") if c.strip()]
                 or list(RECONVERGE_CASES))
        stage_reconverge(records, args.results, args.quick, cases)
        return
    if args.cases:
        keep = [c.strip() for c in args.cases.split(",") if c.strip()]
        records = {k: v for k, v in records.items() if k in keep}
    if args.quick:
        # the smoke path quadruples the strides so two-epoch fits stay light
        for k in sbli_apriori.TEST_STRIDE:
            sx, sy = sbli_apriori.TEST_STRIDE[k]
            sbli_apriori.TEST_STRIDE[k] = (4 * sx, 4 * sy)

    only = ([t.strip() for t in args.only.split(",") if t.strip()]
            if (args.stage == "baselines" and args.only) else None)
    gates, study = stage_baselines(
        records, args.results, args.quick, args.regen,
        skip_extract=(args.stage == "baselines" and args.skip_extract),
        only=only)

    # standalone stages write their own partial file so concurrent stage
    # processes never race on the combined numbers; assemble merges them
    def _part_path(stage):
        return os.path.join(args.results, f"apriori_{stage}{suffix}.json")

    if args.legs and args.stage in ("loso", "insample"):
        for leg in [l.strip() for l in args.legs.split(",") if l.strip()]:
            history = leg.endswith("_history")
            base = leg[:-8] if history else leg
            if args.stage == "loso":
                res = study.loso(base, history=history, seeds=seeds,
                                 epochs=epochs)
            else:
                res = study.insample(base, history=history, seeds=seeds,
                                     epochs=epochs)
            path = os.path.join(
                args.results, f"apriori_{args.stage}_{leg}{suffix}.json")
            cfp.json_atomic(path, cfp.attach_json(
                {leg: res}, _partial_ident(args.results, args.stage, leg,
                                           seeds, epochs)))
            print("wrote", path)
        return
    if args.stage in ("loso", "insample", "far"):
        if args.stage == "loso":
            result = stage_loso(study, seeds, epochs)
        elif args.stage == "insample":
            result = stage_insample(study, seeds, epochs)
        else:
            result = stage_far(study, seeds, epochs, records)
        cfp.json_atomic(_part_path(args.stage), cfp.attach_json(
            {args.stage: result}, _partial_ident(args.results, args.stage,
                                                 None, seeds, epochs)))
        print("wrote", _part_path(args.stage))
        return
    if args.stage == "baselines":
        # the per-case gate JSONs and caches are the record; the combined
        # file is assembled later so parallel partitions never race
        print("baselines stage complete")
        return

    numbers["gates"] = gates["gates"]
    numbers["solves"] = gates["solves"]
    if args.stage == "all":
        numbers["loso"] = stage_loso(study, seeds, epochs)
        json.dump(numbers, open(numbers_path, "w"), indent=1)
        numbers["insample"] = stage_insample(study, seeds, epochs)
        json.dump(numbers, open(numbers_path, "w"), indent=1)
        numbers["far"] = stage_far(study, seeds, epochs, records)
        json.dump(numbers, open(numbers_path, "w"), indent=1)
    if args.stage == "assemble":
        def _valid_partial(path, stage, leg):
            # identity-validated consumption: a partial whose recorded
            # extraction lineage no longer matches the current caches is
            # withheld, never silently assembled
            if not os.path.isfile(path):
                return None
            loaded = json.load(open(path))
            status, _ = cfp.check_json(
                loaded, _partial_ident(args.results, stage, leg, seeds,
                                       epochs))
            if status != "match":
                print(f"[assemble] {os.path.basename(path)} stale identity "
                      f"({status}); withheld")
                return None
            return loaded

        for stage in ("loso", "insample", "far"):
            loaded = _valid_partial(_part_path(stage), stage, None)
            if loaded is not None:
                numbers[stage] = loaded[stage]
                continue
            # merge per-leg partials from the memory-bounded path; the
            # stage enters the numbers only when every expected leg is
            # present, so a partial sweep can never masquerade as complete
            expected = (("dq_y", "dq_joint", "db", "dq_y_history")
                        if stage == "loso"
                        else ("dq_y", "dq_joint", "db"))
            legs = {}
            for leg in expected:
                lp = os.path.join(
                    args.results, f"apriori_{stage}_{leg}{suffix}.json")
                lv = _valid_partial(lp, stage, leg)
                if lv is not None:
                    legs[leg] = lv[leg]
            if len(legs) == len(expected):
                numbers[stage] = legs
            elif legs:
                print(f"[assemble] {stage}: {len(legs)}/{len(expected)} "
                      f"leg partials present; stage withheld")

    numbers["config"] = {
        "driver_seed": DRIVER_SEED, "seeds": list(seeds), "epochs": epochs,
        "samples_per_point": sbli_apriori.SAMPLES_PER_POINT,
        "strides_test": {k: list(v)
                         for k, v in sbli_apriori.TEST_STRIDE.items()},
        "quick": bool(args.quick),
    }
    cfp.json_atomic(numbers_path, cfp.attach_json(
        numbers, sbli_ident("apriori-numbers", "all-cases", numbers={
            "seeds": list(seeds), "epochs": int(epochs),
            "quick": bool(args.quick)})))
    print("wrote", numbers_path)


if __name__ == "__main__":
    main()
