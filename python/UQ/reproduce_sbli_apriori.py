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
from UQ.datasets.sbli_interaction import SBLIInteractionDNS, SBLI_S_CASES
from UQ.datasets.sbli_baseline import SBLIBaseline
from UQ.datasets import sbli_apriori
from UQ.datasets.sbli_apriori import SBLIAPriori, TEST_STRIDE

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
               convergence_tol=None, cfl=None):
    if quick:
        return SBLIBaseline.configure(record, with_shock=with_shock,
                                      nx=160, ny=112, x_hi=6.0, height=6.0,
                                      cfl=cfl or 100.0,
                                      max_iterations=max_iterations or 30000,
                                      convergence_tol=convergence_tol
                                      or 3e-6, yplus_target=0.05)
    # the resolved first cell is load-bearing: the omega wall anchor scales
    # as 1/y1^2 and at y1+ near one it is too weak to select the log-law
    # branch against the spurious near-wall equilibrium (the gate bring-up
    # measured the flip at matched conditions); growth stays near 1.04
    return SBLIBaseline.configure(record, with_shock=with_shock,
                                  nx=480, ny=224, x_hi=14.0, height=8.0,
                                  cfl=cfl or 300.0,
                                  max_iterations=max_iterations or 250000,
                                  convergence_tol=convergence_tol or 1e-6,
                                  yplus_target=0.05)


def _wall_path(results_dir, case):
    return os.path.join(results_dir, f"wall_{case}.npz")


def _save_wall(results_dir, case, w):
    np.savez_compressed(_wall_path(results_dir, case), x_star=w["x_star"],
                        Cf=w["Cf"], Cp=w["Cp"], qw=w["qw"], St=w["St"])


def _fields_path(results_dir, case):
    return os.path.join(results_dir, f"fields_{case}.npz")


def _save_fields(results_dir, case, solver):
    """Persist the converged primitive state (n_cells, 6): the a-posteriori
    members warm-start from it (init_field) so each is a perturbation solve,
    and the fold targets condition on the same converged baseline."""
    f = solver.fields()
    prim = np.stack([np.asarray(f[k], dtype=float)
                     for k in ("rho", "u", "v", "p", "k", "omega")], axis=1)
    np.savez_compressed(_fields_path(results_dir, case), primitive=prim)


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


def _case_cached(results_dir, case, history=True):
    strides = (TEST_STRIDE[case], sbli_apriori._train_stride(case))
    have = all(os.path.isfile(SBLIAPriori._cache_path(
        results_dir, case, st, history)) for st in strides)
    return (have and os.path.isfile(_wall_path(results_dir, case))
            and os.path.isfile(_fields_path(results_dir, case)))


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
        os.makedirs(results_dir, exist_ok=True)
        _save_wall(results_dir, "gate_a_attached", w)
        _save_fields(results_dir, "gate_a_attached", gate_a.solver)
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
            baselines[case] = None      # extraction reads the caches
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
        out["gates"]["B"][case] = entry
        os.makedirs(results_dir, exist_ok=True)
        _save_wall(results_dir, case, w)
        _save_fields(results_dir, case, base.solver)
        json.dump(entry, open(gate_path, "w"), indent=1)
        print(f"[gate B {case}] {rep.status} iters {rep.iterations} "
              f"offset {offset}")
        baselines[case] = base

    # extraction caches at both strides (with the history feature); cached
    # cases pass baseline None and are never touched
    study = SBLIAPriori.build(records, baselines, results_dir, history=True)
    for case, ext in study.test_sets.items():
        out["solves"][case] = {
            "n_test_rows": int(ext["meta"]["n_points"]),
            "realizable_fraction": float(ext["realizable_fraction"]),
        }
    return out, study


def stage_loso(study, seeds, epochs, partial_path=None):
    """Runs the pre-registered loso legs; each completed fold prints a
    line and rewrites the partial file, so a long production run reports
    incrementally, a morning cut has every finished fold, and a restart
    resumes past the folds the partial file already holds (the fits are
    seed-deterministic, so resumed and rerun results are identical)."""
    out = {}
    if partial_path is not None and os.path.isfile(partial_path):
        out = json.load(open(partial_path))
        for leg, folds in out.items():
            print(f"[loso {leg}] resuming past folds {sorted(folds)}",
                  flush=True)

    def _tracked(label, acc):
        def cb(leg, held, fold_result):
            acc[held] = fold_result
            print(f"[loso {label}] fold {held} done "
                  f"(n_train {fold_result['n_train']})", flush=True)
            if partial_path is not None:
                json.dump(out, open(partial_path, "w"), indent=1)
        return cb

    for leg in ("dq_y", "dq_joint", "db"):
        acc = out.setdefault(leg, {})
        study.loso(leg, history=False, seeds=seeds, epochs=epochs,
                   progress=_tracked(leg, acc), skip=set(acc))
    acc = out.setdefault("dq_y_history", {})
    study.loso("dq_y", history=True, seeds=seeds, epochs=epochs,
               progress=_tracked("dq_y+history", acc), skip=set(acc))
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

    if args.stage in ("loso", "insample", "far"):
        if args.stage == "loso":
            result = stage_loso(study, seeds, epochs,
                                partial_path=os.path.join(
                                    args.results,
                                    f"apriori_loso_partial{suffix}.json"))
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
