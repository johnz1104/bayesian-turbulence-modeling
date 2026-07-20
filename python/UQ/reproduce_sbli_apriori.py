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
from UQ.datasets.sbli_apriori import SBLIAPriori, TEST_STRIDE

DRIVER_SEED = 0
# Physics schema token for every SBLI cache identity (the fingerprint
# machinery of UQ.cache_fingerprint): bump exactly when the producing model
# changes. v2 marks the honest adiabatic wall boundary condition and the
# completed all-equation implicit convergence criterion; v1 (implicit, never
# stamped) was the isothermal-convention, density-only-criterion round.
SBLI_PHYSICS = "sbli-dbns-v2"


def sbli_ident(kind, case, **extra):
    """The cache identity every SBLI artifact carries: physics token, case,
    and the production configuration that shaped the solve."""
    ident = {"kind": kind, "physics": SBLI_PHYSICS, "case": case,
             "config": {"nx": 480, "ny": 224, "x_hi": 14.0, "height": 8.0,
                        "cfl": 300.0, "convergence_tol": 1e-6,
                        "yplus_target": 0.05}}
    ident.update(extra)
    return ident


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
               early_abort_rel_max=0.0):
    if quick:
        return SBLIBaseline.configure(record, with_shock=with_shock,
                                      nx=160, ny=112, x_hi=6.0, height=6.0,
                                      cfl=100.0,
                                      max_iterations=max_iterations or 30000,
                                      convergence_tol=convergence_tol
                                      or 3e-6,
                                      early_abort_iter=early_abort_iter,
                                      early_abort_rel_max=early_abort_rel_max,
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
                                  yplus_target=0.05)


def _wall_path(results_dir, case):
    return os.path.join(results_dir, f"wall_{case}.npz")


def _save_wall(results_dir, case, w):
    arrays = {"x_star": w["x_star"], "Cf": w["Cf"], "Cp": w["Cp"],
              "qw": w["qw"], "St": w["St"]}
    np.savez_compressed(_wall_path(results_dir, case),
                        **cfp.attach(arrays, sbli_ident("sbli_wall", case)))


def _fields_path(results_dir, case):
    return os.path.join(results_dir, f"fields_{case}.npz")


def _save_fields(results_dir, case, solver):
    """Persist the converged primitive state (n_cells, 6): the a-posteriori
    members warm-start from it (init_field) so each is a perturbation solve,
    and the fold targets condition on the same converged baseline."""
    f = solver.fields()
    prim = np.stack([np.asarray(f[k], dtype=float)
                     for k in ("rho", "u", "v", "p", "k", "omega")], axis=1)
    arrays = {"primitive": prim}
    if hasattr(solver, "limiter_active"):
        # the solver's own omega-limiter branch record from the last residual
        # evaluation of the converged solve: the EXACT activation map (the
        # python recomputation stays as the approximate fallback for caches
        # that predate this export)
        arrays["limiter_active"] = np.asarray(solver.limiter_active(),
                                              dtype=bool)
    np.savez_compressed(_fields_path(results_dir, case),
                        **cfp.attach(arrays, sbli_ident("sbli_fields", case)))


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
        rho = 1.0 / T
        integrand = rho * np.clip(u, 0.0, None) * (1.0 - np.clip(u, 0.0, 1.0))
        return float(np.trapz(integrand, ys))

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
    """The solve's wall-pressure half-rise against the record's own landmark
    (the pinned rule on both sides)."""
    sm = w["Cp"]
    plat_up = float(np.median(sm[w["x_star"] < w["x_star"][0] + 3.0]))
    plat_dn = float(np.median(sm[w["x_star"] > w["x_star"][-1] - 2.0]))
    level = 0.5 * (plat_up + plat_dn)
    above = sm >= level
    idx = int(np.argmax(above)) if above.any() else -1
    x_half = float(w["x_star"][idx]) if idx > 0 else None
    dns_half = record.shock_position()
    if x_half is None or dns_half is None:
        return None, x_half, dns_half
    return float(x_half - dns_half), x_half, dns_half


def _extraction_current(results_dir, case, history=True):
    """True when every extraction cache for the case carries the
    objective-basis keys (the amendment's representation); a stale-format
    cache regenerates from the CACHED converged fields through a warm-loaded
    baseline, never a re-solve."""
    strides = (TEST_STRIDE[case], sbli_apriori._train_stride(case))
    for st in strides:
        path = SBLIAPriori._cache_path(results_dir, case, st, history)
        if not os.path.isfile(path):
            return False
        if "g_basis" not in np.load(path, allow_pickle=True).files:
            return False
    return True


def _warm_baseline(records, case, results_dir, quick, with_shock=True):
    """Rebuild the case and warm it with the cached converged primitive
    state; one sweep populates the derived fields the extraction samples."""
    base = _configure(records[case], quick, with_shock=with_shock,
                      max_iterations=1, convergence_tol=1e-30)
    prim = np.load(_fields_path(
        results_dir, case if with_shock else "gate_a_attached"))["primitive"]
    base.solver.init_field(prim)
    base.solver.solve()
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
    return (have
            and _npz_ident_ok(_wall_path(results_dir, case),
                              sbli_ident("sbli_wall", case))
            and _npz_ident_ok(_fields_path(results_dir, case),
                              sbli_ident("sbli_fields", case)))


def stage_baselines(records, results_dir, quick, regen):
    """Solve the baselines where their caches are missing, measure the
    gates, and build the extraction caches. Cached cases never re-solve
    unless --regen."""
    out = {"gates": {"B": {}}, "solves": {}}
    baselines = {}

    # gate A: the attached configuration on the adiabatic record
    gate_a_path = os.path.join(results_dir, "gate_a.json")
    if os.path.isfile(gate_a_path) and not regen:
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
            _save_wall(results_dir, "gate_a_attached", w)
            _save_fields(results_dir, "gate_a_attached", gate_a.solver)
        else:
            print("[gate A] NOT converged: caches withheld")
        json.dump(out["gates"]["A"], open(gate_a_path, "w"), indent=1)
        print(f"[gate A] {rep.status} iters {rep.iterations} "
              f"cf {cf_station:.3e} vs {GATE_A_CF:.3e} "
              f"({out['gates']['A']['cf_rel_error']*100:.1f} percent)")

    # gate B: the interaction configuration per record, solve-once cached
    for case, record in records.items():
        gate_path = os.path.join(results_dir, f"gate_b_{case}.json")
        if _case_cached(results_dir, case) and os.path.isfile(gate_path) \
                and not regen:
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
            _save_wall(results_dir, case, w)
            _save_fields(results_dir, case, base.solver)
        else:
            print(f"[gate B {case}] NOT converged: caches withheld")
        json.dump(entry, open(gate_path, "w"), indent=1)
        print(f"[gate B {case}] {rep.status} iters {rep.iterations} "
              f"offset {offset}")
        baselines[case] = base

    # the hard gate adjudication record, written before ANY downstream stage:
    # gate A must pass outright; gate B is adjudicated PER CASE under the
    # reviewer's recorded 2026-07-18 ruling (the pinned propagated folds all
    # pass; a failing case is excluded from claim-bearing coupled legs, takes
    # the registered frozen-mean fallback on its a-priori role, and any
    # injection probe on it is EXPLORATORY, never gate rehabilitation)
    adjud = {
        "ruling": "per-case (reviewer adjudication 2026-07-18); gate "
                  "failures take the registered fallback and are excluded "
                  "from claim-bearing coupled legs",
        "gate_a_pass": bool(out["gates"]["A"].get("pass", False)),
        "gate_b_pass_cases": sorted(c for c, e in out["gates"]["B"].items()
                                    if e.get("pass")),
        "gate_b_fail_cases": sorted(c for c, e in out["gates"]["B"].items()
                                    if not e.get("pass")),
    }
    json.dump(adjud, open(os.path.join(results_dir,
                                       "gates_adjudication.json"), "w"),
              indent=1)
    if not adjud["gate_a_pass"]:
        print("[gates] GATE A FAILED: stopping before any downstream stage "
              "(solver-capability null; see the pre-registration)")
        sys.exit(1)
    if adjud["gate_b_fail_cases"]:
        print(f"[gates] gate B failing cases {adjud['gate_b_fail_cases']}: "
              f"excluded from claim-bearing coupled legs per the recorded "
              f"ruling; passing cases {adjud['gate_b_pass_cases']}")

    # extraction caches at both strides (with the history feature); cached
    # cases pass baseline None and are never touched
    study = SBLIAPriori.build(records, baselines, results_dir, history=True)
    for case, ext in study.test_sets.items():
        out["solves"][case] = {
            "n_test_rows": int(ext["meta"]["n_points"]),
            "realizable_fraction": float(ext["realizable_fraction"]),
        }
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
    for leg in ("dq_y", "dq_joint"):
        out["transfer"][leg] = study.far_transfer(leg, seeds=seeds,
                                                  epochs=epochs)
    out["control"]["dq_y"] = study.attached_control("dq_y", seeds=seeds,
                                                    epochs=epochs)

    # the Stanton-normalized conformal line (the established first-line
    # thermal correction) against the absolute-score control, identical path:
    # the Gaussian conditional's far-transfer median prediction, calibrated
    # on its own training-pool residuals, evaluated on each record's dq_y
    X_tr, dq_tr = study._attached_dq_pool()
    Y_tr = dq_tr[:, 1:2]
    model = study._make("gauss", X_tr.shape[1], 1, DRIVER_SEED)
    model.fit(X_tr, Y_tr, epochs=epochs, lr=1e-3, batch=256)
    import torch
    torch.manual_seed(DRIVER_SEED)
    S_cal = np.asarray(model.sample(X_tr, n_per=128))
    m_cal = np.median(S_cal[:, :, 0], axis=1)
    resid_cal = np.abs(Y_tr[:, 0] - m_cal)
    # attached calibration scales: the case's own baseline wall flux
    scale_cal = []
    for tag in study.attached.gv:
        rec = study.attached.cases[tag]
        scale_cal.append(np.full(rec["n"],
                                 max(abs(rec["b_q_base"]), 1e-4)))
    scale_cal = np.concatenate(scale_cal)
    q_abs = conformal.conformal_quantile(resid_cal, alpha=0.10)
    q_norm = conformal.conformal_quantile(resid_cal / scale_cal, alpha=0.10)

    for case, ext in study.test_sets.items():
        if ext["dq"] is None:
            continue
        X_te = study._features(ext, history=False)
        Y_te = ext["dq"][:, 1]
        torch.manual_seed(DRIVER_SEED)
        S_te = np.asarray(model.sample(X_te, n_per=128))
        m_te = np.median(S_te[:, :, 0], axis=1)
        resid = np.abs(Y_te - m_te)
        scale = _st_scale(study, case, records[case])
        cover_abs = float(np.mean(resid <= q_abs))
        entry = {"absolute": cover_abs, "n": int(Y_te.size)}
        if scale is not None:
            entry["normalized"] = float(np.mean(resid <= q_norm * scale))
        out["conformal"][case] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("baselines", "loso", "insample", "far",
                             "assemble", "all"))
    ap.add_argument("--results", default="results/sbli")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--regen", action="store_true")
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
    if args.cases:
        keep = [c.strip() for c in args.cases.split(",") if c.strip()]
        records = {k: v for k, v in records.items() if k in keep}
    if args.quick:
        # the smoke path quadruples the strides so two-epoch fits stay light
        for k in sbli_apriori.TEST_STRIDE:
            sx, sy = sbli_apriori.TEST_STRIDE[k]
            sbli_apriori.TEST_STRIDE[k] = (4 * sx, 4 * sy)

    gates, study = stage_baselines(records, args.results, args.quick,
                                   args.regen)

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
            json.dump({leg: res}, open(path, "w"), indent=1)
            print("wrote", path)
        return
    if args.stage in ("loso", "insample", "far"):
        if args.stage == "loso":
            result = stage_loso(study, seeds, epochs)
        elif args.stage == "insample":
            result = stage_insample(study, seeds, epochs)
        else:
            result = stage_far(study, seeds, epochs, records)
        json.dump({args.stage: result}, open(_part_path(args.stage), "w"),
                  indent=1)
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
        for stage in ("loso", "insample", "far"):
            if os.path.isfile(_part_path(stage)):
                numbers[stage] = json.load(open(_part_path(stage)))[stage]
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
                if os.path.isfile(lp):
                    legs[leg] = json.load(open(lp))[leg]
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
    json.dump(numbers, open(numbers_path, "w"), indent=1)
    print("wrote", numbers_path)


if __name__ == "__main__":
    main()
