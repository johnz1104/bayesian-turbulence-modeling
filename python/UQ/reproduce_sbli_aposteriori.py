"""Fixed-seed reproduce driver for the a-posteriori shock-interaction study.

Stages (composable; workers are separate processes so an overnight
orchestration throttles the concurrent coupled solves):

  targets      refit the leave-fold-out flow and Gaussian models from the
               a-priori extraction caches, condition them on the converged
               baseline fields at the solver cells, and write the member
               target files (coherent shared-latent draws, realizability-
               projected anisotropy, masked outside the training-like
               region) plus the eigenspace corner family and the attached-
               control targets
  member       one injected coupled solve: load a target row, warm-start
               from the converged baseline primitive state, inject, solve,
               record the wall quantities and landmarks
  score        gather a fold's members into station coverage, band widths,
               landmark containment, the corner envelope, and the attached-
               control skin-friction shifts
  orchestrate  the full overnight pipeline over the propagated folds with a
               concurrency throttle

The propagated folds are the pre-registered s = 0.5, 1.0, 1.9. Everything
inherits the a-priori conventions and caches (run its baselines stage
first); --quick runs the coarse configuration into a separate quick results
subdirectory and never touches production caches.

    export QBTM_DNS_DATA=<repo>/DNS_data
    python3 python/UQ/reproduce_sbli_aposteriori.py --stage orchestrate \
        --results results/sbli --throttle 4
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "build"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from UQ.datasets.sbli_interaction import SBLIInteractionDNS
from UQ.datasets.sbli_baseline import SBLIBaseline, SBLIUnits
from UQ.datasets import sbli_apriori, sbli_aposteriori
from UQ import cache_fingerprint as cfp
from UQ import evaluation
from UQ.datasets.sbli_apriori import SBLIAPriori
from UQ.datasets.sbli_aposteriori import (
    fold_models, cell_conditioning, member_targets, corner_targets,
    five_state_targets,
    dq_to_solver_units, landmarks_from_wall, score_ensemble,
    N_MEMBERS, MEMBER_SEED, STATIONS)
from UQ.reproduce_sbli_apriori import (
    _all_records, _configure, _fields_path, sbli_ident, GATE_A_CF,
    GATE_A_STATION)

FOLDS = ("s0.5", "s1.0", "s1.9")
KINDS = ("flow", "gauss")
# members are warm-started perturbation solves: convergence is a thousand
# fold decay OF THE INJECTION RESPONSE (the first-iteration residual). The
# budget is sized to the MEASURED corrected-solver cost (the adiabatic
# corner probe's settling member converged at 43631 iterations; the old
# 15000 cap was calibrated on the pre-correction smoke and recorded every
# corrected-era member Unconverged), and the solver-side early abort fails
# genuinely non-settling members fast instead of burning the budget: the
# measured separation at 15000 iterations is settling members at order
# 1e-2 and below versus stuck members at 0.8 and above, so the 0.3
# threshold sits inside the empty gap
MEMBER_MAX_ITER = 45000
MEMBER_TOL = 1e-3
MEMBER_ABORT_ITER = 15000
MEMBER_ABORT_RELMAX = 0.3


def _apo_dir(results_dir):
    d = os.path.join(results_dir, "aposteriori")
    os.makedirs(d, exist_ok=True)
    return d


def _targets_path(results_dir, fold, kind, attached=False):
    tag = "attached_" if attached else ""
    return os.path.join(_apo_dir(results_dir),
                        f"targets_{tag}{fold}_{kind}.npz")


def _member_path(results_dir, fold, kind, index, attached=False):
    tag = "attached_" if attached else ""
    return os.path.join(_apo_dir(results_dir),
                        f"member_{tag}{fold}_{kind}_{index}.npz")


def _member_config():
    """The member-cache identity: physics token, production configuration
    and the member solve budget. An existence-only reuse check once admitted
    pre-audit member records into a corrected-era fold; identity-mismatched
    or unfingerprinted member files are treated as absent and re-solved."""
    return sbli_ident("apo-member", "all-cases", member={
        "max_iter": MEMBER_MAX_ITER, "tol": MEMBER_TOL,
        "abort_iter": MEMBER_ABORT_ITER,
        "abort_rel_max": MEMBER_ABORT_RELMAX,
        "n_members": N_MEMBERS, "seed": MEMBER_SEED})


def _member_current(path):
    if not os.path.isfile(path):
        return False
    status, _reason = cfp.check(np.load(path), _member_config())
    return status == "match"


def _job_output(results_dir, fold, argv):
    """The output file a member worker argv will write (resume support:
    the orchestrator skips jobs whose output already exists)."""
    if "--corner" in argv:
        lab = argv[argv.index("--corner") + 1]
        return os.path.join(_apo_dir(results_dir),
                            f"member_{fold}_corner_{lab}.npz")
    kind = argv[argv.index("--kind") + 1]
    index = int(argv[argv.index("--index") + 1])
    return _member_path(results_dir, fold, kind, index,
                        attached="--attached" in argv)


def _load_baseline(records, case, results_dir, quick, with_shock=True,
                   member_caps=False, derived_probe=False):
    """Configure the case and warm the solver with the cached converged
    primitive state. member_caps swaps in the member iteration budget (a
    warm-started perturbation solve, not a cold start). derived_probe runs
    a single sweep so the solver populates its derived fields: init_field
    sets only the primitive state, and the turbulent viscosity the target
    conditioning reads through sample_fields is computed during solving
    (the quick smoke caught nu_t identically zero without this, which
    zeroed the baseline anisotropy under every target). One sweep from the
    converged state drifts it by the order of the converged residual."""
    kw = {}
    if member_caps:
        kw = {"max_iterations": MEMBER_MAX_ITER,
              "convergence_tol": MEMBER_TOL,
              "early_abort_iter": MEMBER_ABORT_ITER,
              "early_abort_rel_max": MEMBER_ABORT_RELMAX}
    if derived_probe:
        kw = {"max_iterations": 1, "convergence_tol": 1e-30}
    base = _configure(records[case], quick, with_shock=with_shock, **kw)
    prim = np.load(_fields_path(
        results_dir, case if with_shock else "gate_a_attached"))["primitive"]
    base.solver.init_field(prim)
    if derived_probe:
        # property preparation (gradients + derived fields) WITHOUT a solver
        # iteration: the loaded converged state is not advanced (the one-sweep
        # probe drifted it by the order of the converged residual)
        base.solver.prepare_properties()
    return base, prim


def stage_targets(records, results_dir, fold, quick, n_members, epochs):
    """Refit the fold models and write every member target file the fold's
    workers will consume."""
    t0 = time.time()
    study = SBLIAPriori.build(records, {c: None for c in records},
                              results_dir, history=True)
    models = fold_models(study, fold, history=False,
                         seed=MEMBER_SEED, epochs=epochs)

    meta = {"fold": fold, "n_members": n_members, "epochs": epochs,
            "seed": MEMBER_SEED}

    # interaction targets, conditioned on the fold's own converged baseline
    base, _ = _load_baseline(records, fold, results_dir, quick,
                             derived_probe=True)
    feats, b_base, mask, basis_M, strain = cell_conditioning(base, records[fold],
                                                     m_t_from_fields=True)
    meta["n_cells"] = int(len(feats))
    meta["mask_fraction"] = float(np.mean(mask))
    meta["b_base_abs_med_masked"] = float(np.median(np.abs(b_base[mask])))
    for kind in KINDS:
        b_t, dq = member_targets(models[kind], feats, b_base, mask, basis_M,
                                 db_raw=not study.db_gate()["all_pass"],
                                 n_members=n_members, seed=MEMBER_SEED)
        np.savez_compressed(
            _targets_path(results_dir, fold, kind),
            b=b_t.astype(np.float32), dq=dq.astype(np.float32),
            mask=mask)
        meta[f"{kind}_db_fro_med_masked"] = float(np.median(
            np.linalg.norm((b_t - b_base[None])[:, mask], axis=(2, 3))))
        meta[f"{kind}_dq_abs_med"] = float(np.median(np.abs(dq)))
    corners = corner_targets(b_base, mask)
    corners.update(five_state_targets(b_base, mask, strain))
    np.savez_compressed(
        _targets_path(results_dir, fold, "corners"),
        mask=mask,
        **{lab: b.astype(np.float32) for lab, b in corners.items()})

    # attached-control targets: the same fold models conditioned on the
    # attached configuration's own converged state (M_t from the fields;
    # no DNS record exists for this configuration)
    abase, _ = _load_baseline(records, "adiabatic", results_dir, quick,
                              with_shock=False, derived_probe=True)
    afeats, ab_base, amask, abasis_M, _astrain = cell_conditioning(
        abase, records["adiabatic"], m_t_from_fields=True)
    meta["attached_mask_fraction"] = float(np.mean(amask))
    for kind in KINDS:
        b_t, dq = member_targets(models[kind], afeats, ab_base, amask, abasis_M,
                                 db_raw=not study.db_gate()["all_pass"],
                                 n_members=n_members, seed=MEMBER_SEED)
        np.savez_compressed(
            _targets_path(results_dir, fold, kind, attached=True),
            b=b_t.astype(np.float32), dq=dq.astype(np.float32),
            mask=amask)

    meta["wall_time_s"] = round(time.time() - t0, 1)
    json.dump(meta, open(os.path.join(
        _apo_dir(results_dir), f"targets_{fold}_meta.json"), "w"), indent=1)
    print(f"[targets {fold}] {meta['n_cells']} cells, "
          f"mask {meta['mask_fraction']:.2f}, "
          f"{meta['wall_time_s']}s")


def stage_targets_far(records, results_dir, quick, n_members, epochs):
    """The far-transfer propagation targets: flow and Gaussian conditionals
    trained on the attached matrix's joint heat-flux pool, sampled at the
    adiabatic interaction baseline's cells; the anisotropy target stays the
    baseline itself (the attached pool carries no interaction anisotropy
    law), so this set characterizes the propagated thermal correction."""
    import torch
    t0 = time.time()
    study = SBLIAPriori.build(records, {c: None for c in records},
                              results_dir, history=True)
    X_tr, dq_tr = study._attached_dq_pool()
    Y_tr = dq_tr[:, 0:2]
    base, _ = _load_baseline(records, "adiabatic", results_dir, quick,
                             derived_probe=True)
    feats, b_base, mask, basis_M, strain = cell_conditioning(base, records["adiabatic"],
                                                     m_t_from_fields=True)
    meta = {"fold": "faradiab", "n_members": n_members, "epochs": epochs,
            "n_train": int(len(X_tr)), "n_cells": int(len(feats)),
            "mask_fraction": float(np.mean(mask))}
    for kind in KINDS:
        model = SBLIAPriori._make(kind, X_tr.shape[1], Y_tr.shape[1],
                                  MEMBER_SEED)
        model.fit(X_tr, Y_tr, epochs=epochs, lr=1e-3, batch=256)
        torch.manual_seed(MEMBER_SEED + 1)
        dq_draws = np.asarray(model.sample(feats, n_per=n_members,
                                           shared_latent=True))
        dq = dq_draws[:, :, 0:2].copy()
        dq[~mask] = 0.0
        np.savez_compressed(
            _targets_path(results_dir, "faradiab", kind),
            b=b_base[None].astype(np.float32),
            dq=dq.transpose(1, 0, 2).astype(np.float32))
    meta["wall_time_s"] = round(time.time() - t0, 1)
    json.dump(meta, open(os.path.join(
        _apo_dir(results_dir), "targets_faradiab_meta.json"), "w"),
        indent=1)
    print(f"[targets faradiab] {meta['n_cells']} cells, "
          f"{meta['wall_time_s']}s")


def stage_member(records, results_dir, fold, kind, index, quick,
                 attached=False, corner=None):
    """One injected coupled solve, warm-started from the baseline. The
    fold "faradiab" is the attached-trained far-transfer propagation into
    the adiabatic interaction configuration (dq-only targets)."""
    case = "adiabatic" if (attached or fold == "faradiab") else fold
    record = records[case]
    base, _ = _load_baseline(records, case, results_dir, quick,
                             with_shock=not attached, member_caps=True)

    target_kind = kind[:-4] if (kind or "").endswith("_noq") else kind
    energy_reach = not (kind or "").endswith("_noq")
    if corner is not None:
        tg = np.load(_targets_path(results_dir, fold, "corners"))
        b_t = np.asarray(tg[corner], dtype=float)
        dq_dim = np.zeros((b_t.shape[0], 2))
        inj_mask = (np.asarray(tg["mask"], bool) if "mask" in tg.files
                    else np.array([], dtype=bool))
        out_path = os.path.join(_apo_dir(results_dir),
                                f"member_{fold}_corner_{corner}.npz")
    else:
        tg = np.load(_targets_path(results_dir, fold, target_kind,
                                   attached=attached))
        # dq-only target sets store a single shared anisotropy row (the
        # baseline itself); member rows index the heat-flux draws
        b_all = tg["b"]
        b_t = np.asarray(b_all[index if b_all.shape[0] > 1 else 0],
                         dtype=float)
        dq = np.asarray(tg["dq"][index], dtype=float)
        dq_dim = dq_to_solver_units(dq[None], record, base.units)[0]
        inj_mask = (np.asarray(tg["mask"], bool) if "mask" in tg.files
                    else np.array([], dtype=bool))
        out_path = _member_path(results_dir, fold, kind, index,
                                attached=attached)

    t0 = time.time()
    if not energy_reach:
        # the registered anisotropy-only diagnostic: identical targets, the
        # energy-equation reach disabled, so the dq contribution is isolated
        dq_dim = np.zeros_like(dq_dim)
    base.solver.set_target_correction(b_t, dq_dim, energy_reach,
                                      mask=inj_mask)
    rep = base.solver.solve()
    w = base.wall()
    lm = landmarks_from_wall(w)
    diag = base.solver.injection_diagnostics()
    np.savez_compressed(
        out_path, **cfp.attach(dict(
        x_star=w["x_star"], Cf=w["Cf"], Cp=w["Cp"],
        qw=w["qw"], St=w["St"],
        status=np.bytes_(str(rep.status).encode()),
        iterations=np.int64(rep.iterations),
        final_residual=np.float64(rep.final_residual),
        x_s=np.float64(np.nan if lm["x_s"] is None else lm["x_s"]),
        x_r=np.float64(np.nan if lm["x_r"] is None else lm["x_r"]),
        shock=np.float64(np.nan if lm["shock"] is None else lm["shock"]),
        all_realizable=np.bool_(bool(diag["all_realizable"])),
        max_violation=np.float64(diag["max_violation"]),
        max_db=np.float64(diag["max_db"]),
        max_dq=np.float64(diag["max_dq"]),
        wall_time_s=np.float64(round(time.time() - t0, 1))),
        _member_config()))
    print(f"[member] {os.path.basename(out_path)} {rep.status} "
          f"iters {rep.iterations} {round(time.time() - t0, 1)}s")


def _load_member(path):
    z = np.load(path)
    lm = {k: (None if np.isnan(float(z[k])) else float(z[k]))
          for k in ("x_s", "x_r", "shock")}
    return {"status": bytes(z["status"]).decode(),
            "iterations": int(z["iterations"]),
            "final_residual": float(z["final_residual"]),
            "landmarks": lm,
            "all_realizable": bool(z["all_realizable"]),
            "max_violation": float(z["max_violation"]),
            "wall": {k: np.asarray(z[k])
                     for k in ("x_star", "Cf", "Cp", "qw", "St")}}


def _baseline_wall(results_dir, case):
    p = os.path.join(results_dir, f"wall_{case}.npz")
    if not os.path.isfile(p):
        return None
    z = np.load(p)
    return {k: np.asarray(z[k]) for k in ("x_star", "Cf", "Cp", "qw", "St")}


def stage_score(records, results_dir, fold, n_members):
    """Assemble one fold's ensembles into the scored record."""
    case = "adiabatic" if fold == "faradiab" else fold
    record = records[case]
    bwall = _baseline_wall(results_dir, case)
    out = {"fold": fold, "n_members": n_members}
    for kind in KINDS:
        paths = [_member_path(results_dir, fold, kind, i)
                 for i in range(n_members)]
        # a missing worker output is a NON-CONVERGED member for every
        # denominator: the 18-of-24 instability rule reads against the
        # REQUESTED count, never the found-file count
        members = [(_load_member(p) if _member_current(p)
                    else {"status": "Missing(worker output absent or "
                                    "stale identity)",
                          "landmarks": {"x_s": None, "x_r": None,
                                        "shock": None}})
                   for p in paths]
        n_found = sum(1 for p in paths if _member_current(p))
        out[kind] = score_ensemble(members, record, baseline_wall=bwall)
        out[kind]["n_requested"] = n_members
        out[kind]["n_found"] = n_found
        conv_m = [m for m in members if "Converged" in m["status"]]
        if conv_m:
            # the pre-registered realizability-in-the-running-solve clause,
            # aggregated over CONVERGED members (a diverged member's
            # diagnostics describe no admitted state)
            out[kind]["all_realizable_fraction"] = float(np.mean(
                [m["all_realizable"] for m in conv_m]))
            out[kind]["max_violation"] = float(np.max(
                [m["max_violation"] for m in conv_m]))
        # the adiabatic-middle fold carries the independent campaign's wall
        # series as a second truth surface (the pre-registered pairing)
        if fold == "s1.0" and "adiabatic" in records and n_found:
            out[kind]["second_surface"] = score_ensemble(
                members, records["adiabatic"],
                baseline_wall=_baseline_wall(results_dir, "adiabatic"))

    # the registered anisotropy-only diagnostic: the flow targets propagated
    # with the energy reach disabled, scored identically and labeled as the
    # diagnostic it is (never a primary row)
    noq_paths = [_member_path(results_dir, fold, "flow_noq", i)
                 for i in range(n_members)]
    noq = [(_load_member(p) if _member_current(p)
            else {"status": "Missing(worker output absent or "
                            "stale identity)",
                  "landmarks": {"x_s": None, "x_r": None, "shock": None}})
           for p in noq_paths]
    if any(_member_current(p) for p in noq_paths):
        out["flow_energy_reach_disabled_diagnostic"] = score_ensemble(
            noq, record, baseline_wall=bwall)

    # eigenspace families: PER-AMPLITUDE envelopes over the converged
    # members (never merged across amplitudes), the three-corner family and
    # the documented five-state family separately, each with the EXACT
    # discrete-forecast CRPS (the members ARE the forecast; the fair
    # estimator applies to sampled predictives only)
    series = record.series
    xs = STATIONS[(STATIONS >= series.x[0]) & (STATIONS <= series.x[-1])]
    truths = {"Cf": np.interp(xs, series.x, series.cf)}
    if series.cp is not None:
        truths["Cp"] = np.interp(xs, series.x, series.cp)
    if series.St is not None and not np.all(np.isnan(series.St)):
        truths["St"] = np.interp(xs, series.x, series.St)

    def _family_scores(labels):
        fam = {}
        for lab in labels:
            p = os.path.join(_apo_dir(results_dir),
                             f"member_{fold}_corner_{lab}.npz")
            if _member_current(p):
                fam[lab] = _load_member(p)
        conv = {lab: m for lab, m in fam.items()
                if "Converged" in m["status"]}
        rec_out = {"n_members": len(labels), "n_found": len(fam),
                   "n_converged": len(conv),
                   "status": {lab: m["status"] for lab, m in fam.items()}}
        if len(conv) >= 2:
            for q, truth in truths.items():
                ens = np.stack([np.interp(xs, m["wall"]["x_star"],
                                          m["wall"][q])
                                for m in conv.values()])
                lo, hi = ens.min(axis=0), ens.max(axis=0)
                rec_out[f"{q}_envelope_containment"] = float(np.mean(
                    (truth >= lo) & (truth <= hi)))
                rec_out[f"{q}_envelope_halfwidth_median"] = float(
                    np.median(0.5 * (hi - lo)))
                rec_out[f"{q}_crps_exact_discrete"] = float(
                    evaluation.crps_ensemble_biased(truth, ens.T))
        return rec_out

    out["corners"] = {
        "d1.0": _family_scores(("1C_d1", "2C_d1", "3C_d1")),
        "d0.5": _family_scores(("1C_d0.5", "2C_d0.5", "3C_d0.5")),
    }
    out["five_state"] = {
        "d1.0": _family_scores(("1C_vmax_d1", "1C_vmin_d1", "2C_vmax_d1",
                                "2C_vmin_d1", "3C_d1")),
        "d0.5": _family_scores(("1C_vmax_d0.5", "1C_vmin_d0.5",
                                "2C_vmax_d0.5", "2C_vmin_d0.5", "3C_d0.5")),
    }

    # attached control: skin friction at the incoming-profile station per
    # member against the baseline's own value and the measured level
    gate_a = json.load(open(os.path.join(results_dir, "gate_a.json")))
    cf_base = float(gate_a["cf_at_station"])
    att = {"cf_baseline": cf_base, "cf_data": GATE_A_CF}
    for kind in KINDS:
        cfs = []
        n_conv = 0
        for i in range(n_members):
            p = _member_path(results_dir, fold, kind, i, attached=True)
            if not _member_current(p):
                continue
            m = _load_member(p)
            if "Converged" not in m["status"]:
                continue
            n_conv += 1
            cfs.append(float(np.interp(GATE_A_STATION,
                                       m["wall"]["x_star"],
                                       m["wall"]["Cf"])))
        base_err = abs(cf_base - GATE_A_CF)
        att[kind] = {
            "n_converged": n_conv,
            "cf_members": cfs,
            # the REGISTERED criteria: member error to the MEASURED value,
            # relative to the baseline's own error to it; and the 5-95 band
            # HALF-width over the measured skin friction
            "cf_error_over_baseline_error_median": (float(np.median(
                np.abs(np.asarray(cfs) - GATE_A_CF)) / max(base_err, 1e-12))
                if cfs else None),
            "cf_band_halfwidth_over_data": (float(
                0.5 * (np.quantile(cfs, 0.95) - np.quantile(cfs, 0.05))
                / GATE_A_CF) if len(cfs) >= 2 else None),
        }
    out["attached_control"] = att

    path = os.path.join(_apo_dir(results_dir), f"fold_scores_{fold}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[score {fold}] wrote {path}")
    return out


def _spawn(script_args, log_path):
    log = open(log_path, "ab")
    return subprocess.Popen(
        [sys.executable, os.path.abspath(__file__)] + script_args,
        stdout=log, stderr=log)


def _run_pool(jobs, throttle, log_path):
    """jobs: list of argv lists for member workers; runs them at most
    `throttle` at a time, returns the count of nonzero exits."""
    active, failed, queued = [], 0, list(jobs)
    while queued or active:
        while queued and len(active) < throttle:
            active.append(_spawn(queued.pop(0), log_path))
        time.sleep(10)
        still = []
        for p in active:
            if p.poll() is None:
                still.append(p)
            elif p.returncode != 0:
                failed += 1
        active = still
    return failed


def stage_orchestrate(records, results_dir, quick, throttle, n_members,
                      epochs, folds):
    """The full overnight pipeline: per fold, targets then the throttled
    member fan-out (interaction members, corners, attached control), then
    the fold score; a combined numbers file at the end."""
    common = ["--results", results_dir] + (["--quick"] if quick else [])
    log_path = os.path.join(_apo_dir(results_dir), "workers.log")

    def _member_jobs(fold, kinds=KINDS, corners=False, attached=False):
        jobs = []
        for kind in kinds:
            for i in range(n_members):
                j = ["--stage", "member", "--fold", fold,
                     "--kind", kind, "--index", str(i)] + common
                if attached:
                    j.append("--attached")
                jobs.append(j)
        if corners:
            for lab in ("1C_d1", "2C_d1", "3C_d1",
                        "1C_d0.5", "2C_d0.5", "3C_d0.5",
                        "1C_vmax_d1", "1C_vmin_d1", "2C_vmax_d1",
                        "2C_vmin_d1",
                        "1C_vmax_d0.5", "1C_vmin_d0.5", "2C_vmax_d0.5",
                        "2C_vmin_d0.5"):
                jobs.append(["--stage", "member", "--fold", fold,
                             "--corner", lab] + common)
        # cached members never re-solve: drop jobs whose output exists
        return [j for j in jobs if not _member_current(_job_output(
            results_dir, fold, j))]

    summary = {}
    # phase 1: the interaction ensembles, the eigenspace families (three-
    # corner and five-state) and the registered anisotropy-only diagnostic
    # (the flow targets with the energy reach disabled) per fold
    for fold in folds:
        stage_targets(records, results_dir, fold, quick, n_members, epochs)
        failed = _run_pool(_member_jobs(fold, kinds=KINDS + ("flow_noq",),
                                        corners=True), throttle,
                           log_path)
        print(f"[orchestrate {fold}] interaction members done, "
              f"{failed} worker failures")
        summary[fold] = stage_score(records, results_dir, fold, n_members)
        summary[fold]["worker_failures"] = int(failed)

    # phase 2: the preserve-attached control for every fold closure
    for fold in folds:
        failed = _run_pool(_member_jobs(fold, attached=True), throttle,
                           log_path)
        print(f"[orchestrate {fold}] attached control done, "
              f"{failed} worker failures")
        prior = summary[fold]["worker_failures"]
        summary[fold] = stage_score(records, results_dir, fold, n_members)
        summary[fold]["worker_failures"] = prior + int(failed)

    # phase 3: the attached-trained far-transfer propagation into the
    # adiabatic interaction (dq-only characterization)
    stage_targets_far(records, results_dir, quick, n_members, epochs)
    failed = _run_pool(_member_jobs("faradiab"), throttle, log_path)
    print(f"[orchestrate faradiab] members done, {failed} worker failures")
    summary["faradiab"] = stage_score(records, results_dir, "faradiab",
                                      n_members)
    summary["faradiab"]["worker_failures"] = int(failed)

    suffix = "_quick" if quick else ""
    numbers = {
        "folds": summary,
        "config": {"n_members": n_members, "member_seed": MEMBER_SEED,
                   "epochs": epochs, "kinds": list(KINDS),
                   "member_max_iterations": MEMBER_MAX_ITER,
                   "member_convergence_tol": MEMBER_TOL,
                   "corner_deltas": list(sbli_aposteriori.CORNER_DELTAS),
                   "stations": [float(v) for v in STATIONS],
                   "mask": {"y_max": sbli_aposteriori.MASK_Y_MAX,
                            "k_floor": sbli_aposteriori.MASK_K_FLOOR},
                   "quick": bool(quick)},
    }
    path = os.path.join(results_dir, f"aposteriori_numbers{suffix}.json")
    json.dump(numbers, open(path, "w"), indent=1)
    print("wrote", path)


def _require_gates(results_dir, fold):
    """Claim-bearing coupled stages run only on gate-passing configurations
    (the recorded per-case ruling); the far-transfer target is exempt only
    when explicitly labeled exploratory via QBTM_SBLI_EXPLORATORY=1."""
    path = os.path.join(results_dir, "gates_adjudication.json")
    if not os.path.isfile(path):
        print("[gates] no adjudication record; run the a-priori baselines "
              "stage first")
        sys.exit(1)
    adjud = json.load(open(path))
    if not adjud.get("gate_a_pass"):
        print("[gates] gate A failed; coupled stages are not adjudicable")
        sys.exit(1)
    case = "adiabatic" if fold == "faradiab" else fold
    if case in adjud.get("gate_b_fail_cases", []):
        if os.environ.get("QBTM_SBLI_EXPLORATORY") == "1":
            print(f"[gates] {case} failed gate B: proceeding EXPLORATORY "
                  f"(labeled, never claim-bearing)")
        else:
            print(f"[gates] {case} failed gate B: coupled stage refused "
                  f"(set QBTM_SBLI_EXPLORATORY=1 for a labeled exploratory "
                  f"run)")
            sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("targets", "member", "score", "orchestrate"))
    ap.add_argument("--results", default="results/sbli")
    ap.add_argument("--fold", default=None)
    ap.add_argument("--kind", default="flow",
                    choices=("flow", "gauss", "flow_noq"))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--attached", action="store_true")
    ap.add_argument("--corner", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--throttle", type=int, default=4)
    ap.add_argument("--folds", default=",".join(FOLDS))
    args = ap.parse_args()
    if getattr(args, "fold", None):
        _require_gates(args.results, args.fold)

    np.random.seed(MEMBER_SEED)
    n_members = 3 if args.quick else N_MEMBERS
    epochs = 2 if args.quick else sbli_apriori.EPOCHS
    if args.quick:
        for k in sbli_apriori.TEST_STRIDE:
            sx, sy = sbli_apriori.TEST_STRIDE[k]
            sbli_apriori.TEST_STRIDE[k] = (4 * sx, 4 * sy)
        # the a-priori smoke caches live in their own quick universe; the
        # guard keeps a worker relaunch from redirecting twice
        if os.path.basename(os.path.normpath(args.results)) != "quick":
            args.results = os.path.join(args.results, "quick")
    records = _all_records()

    if args.stage == "targets":
        stage_targets(records, args.results, args.fold, args.quick,
                      n_members, epochs)
    if args.stage == "member":
        stage_member(records, args.results, args.fold, args.kind,
                     args.index, args.quick, attached=args.attached,
                     corner=args.corner)
    if args.stage == "score":
        stage_score(records, args.results, args.fold, n_members)
    if args.stage == "orchestrate":
        folds = [f.strip() for f in args.folds.split(",") if f.strip()]
        stage_orchestrate(records, args.results, args.quick, args.throttle,
                          n_members, epochs, folds)


if __name__ == "__main__":
    main()
