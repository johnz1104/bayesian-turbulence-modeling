"""Fixed-seed reproduce driver for the a-posteriori shock-interaction study.

Stages (composable; workers are separate processes so an overnight
orchestration throttles the concurrent coupled solves):

  targets      refit the leave-fold-out flow and Gaussian models from the
               a-priori extraction caches at ONE model seed, condition them
               on the converged baseline fields at the solver cells, and
               write that seed's member target files (coherent shared-latent
               draws at the registered sampling seed, realizability-
               projected anisotropy, masked outside the training-like
               region); the deterministic families (eigenspace corners,
               five-state set, the zero-discrepancy control) are written
               once per fold
  member       one injected coupled solve: load a target row, warm-start
               from the converged baseline primitive state, inject, solve,
               record the wall quantities, landmarks and the effective-
               running realizability diagnostics
  score        gather a fold's per-seed ensembles into station coverage,
               band widths, landmark containment, the corner envelope, and
               the attached-control skin-friction shifts; per model seed
               separately, never pooled, with the seed mean and min-max
               spread and the any-seed 18-of-24 instability label
  orchestrate  the full pipeline over the propagated folds and model seeds
               with a concurrency throttle

Seed protocol (the dated pre-registration addendum): model seeds {0, 1, 2}
train separately; the sampling seed stays at the registered 0, distinct
from the model seed; both appear in every cache path and record. Every
cache carries a configuration fingerprint whose lineage embeds the content
hash of its exact upstream files (fields -> extraction -> targets ->
member -> score), so a mutated or regenerated parent invalidates every
descendant. The faradiab far-transfer propagation targets a gate-B failing
configuration and lives in the aposteriori/exploratory/ namespace, which
the formal assembler structurally cannot consume (opt-in via
QBTM_SBLI_EXPLORATORY=1).

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
from UQ.datasets.sbli_apriori import SBLIAPriori, TEST_STRIDE, _train_stride
from UQ.datasets.sbli_aposteriori import (
    fold_models, cell_conditioning, member_targets, corner_targets,
    five_state_targets,
    dq_to_solver_units, landmarks_from_wall, score_ensemble,
    N_MEMBERS, MEMBER_SEED, STATIONS)
from UQ.reproduce_sbli_apriori import (
    _all_records, _configure, _fields_path, _wall_path, sbli_ident,
    GATE_A_CF, GATE_A_STATION)
from UQ.datasets import dns_manifest

FOLDS = ("s0.5", "s1.0", "s1.9")
HEATED = ("s0.5", "s0.75", "s1.0", "s1.4", "s1.9")
KINDS = ("flow", "gauss")
# the seed-resolved protocol (the dated pre-registration addendum): model
# seeds train separately, one 24-member ensemble per (method, fold, model
# seed), scored separately, criteria on the seed mean, the 18-of-24 rule
# applied independently per seed, the 72 members never pooled into one
# forecast. The sampling seed stays at the registered 0, distinct from the
# model seed; both appear in every cache path and record.
MODEL_SEEDS = (0, 1, 2)
SAMPLE_SEED = MEMBER_SEED
# members are warm-started perturbation solves: convergence is a thousand
# fold decay OF THE INJECTION RESPONSE (the first-iteration residual). The
# budget is sized to the MEASURED corrected-solver cost (the adiabatic
# corner probe's settling member converged at 43631 iterations; the old
# 15000 cap was calibrated on the pre-correction smoke and recorded every
# corrected-era member Unconverged).
MEMBER_MAX_ITER = 45000
MEMBER_TOL = 1e-3
# the 15000-iteration / 0.3 early-abort rule is RETIRED for formal members:
# its thresholds were calibrated on the confounded pre-repair round, so
# every formal member runs abort-off pending an independent full-budget
# diagnostic panel and a dated amendment (the Phase P-prime gate); the
# settings stay in the cache identity at their off values
MEMBER_ABORT_ITER = 0
MEMBER_ABORT_RELMAX = 0.0
# the registered coupled leg runs the running-k stored-discrepancy form;
# a frozen-k or ramped variant is a dated amendment that flips these
# versioned constants (they are part of the member cache identity)
MEMBER_FROZEN_K = False
MEMBER_RAMP_ITERS = 0
# the corner family converges more slowly than the sampled members (two
# independent measurements at 43631 and past 45000 iterations), so its
# budget is sized separately
CORNER_MAX_ITER = 60000

# deterministic per-fold families (run once per fold, never per seed): the
# eigenspace corner and five-state labels plus the exact zero-discrepancy
# control member
CORNER_LABELS = ("1C_d1", "2C_d1", "3C_d1",
                 "1C_d0.5", "2C_d0.5", "3C_d0.5",
                 "1C_vmax_d1", "1C_vmin_d1", "2C_vmax_d1", "2C_vmin_d1",
                 "1C_vmax_d0.5", "1C_vmin_d0.5", "2C_vmax_d0.5",
                 "2C_vmin_d0.5", "zero_control")


def _is_exploratory(fold):
    """The attached-to-adiabatic far-transfer propagation targets a gate-B
    failing configuration: its artifacts live in the exploratory namespace,
    which the formal assembler structurally cannot consume."""
    return fold == "faradiab"


def _apo_dir(results_dir, fold=None, exploratory=None):
    if exploratory is None:
        exploratory = fold is not None and _is_exploratory(fold)
    parts = [results_dir, "aposteriori"] + (
        ["exploratory"] if exploratory else [])
    d = os.path.join(*parts)
    os.makedirs(d, exist_ok=True)
    return d


def _seed_tag(model_seed):
    return "" if model_seed is None else f"_ms{model_seed}"


def _targets_path(results_dir, fold, kind, attached=False, model_seed=None):
    tag = "attached_" if attached else ""
    return os.path.join(_apo_dir(results_dir, fold),
                        f"targets_{tag}{fold}_{kind}"
                        f"{_seed_tag(model_seed)}.npz")


def _member_path(results_dir, fold, kind, index, attached=False,
                 model_seed=None):
    tag = "attached_" if attached else ""
    return os.path.join(_apo_dir(results_dir, fold),
                        f"member_{tag}{fold}_{kind}"
                        f"{_seed_tag(model_seed)}_{index}.npz")


def _corner_member_path(results_dir, fold, lab):
    return os.path.join(_apo_dir(results_dir, fold),
                        f"member_{fold}_corner_{lab}.npz")


def _targets_lineage(results_dir, fold, attached=False):
    """Content hashes of every upstream cache a target file is built from:
    the heated extraction caches at both strides (the fold models and the
    feasibility gate consume them) and the conditioning-case converged
    fields. A mutated or regenerated parent invalidates the targets, which
    invalidates the members, which invalidates the scores (the transitive
    chain)."""
    lin = {}
    for c in HEATED:
        for st in (TEST_STRIDE[c], _train_stride(c)):
            p = SBLIAPriori._cache_path(results_dir, c, st, True)
            lin[os.path.basename(p)] = cfp.file_sha(p)
    cond = "gate_a_attached" if attached else \
        ("adiabatic" if _is_exploratory(fold) else fold)
    lin[f"fields_{cond}.npz"] = cfp.file_sha(_fields_path(results_dir, cond))
    # the raw-input edge: member targets consume the records directly
    # (reference scales, mask spans, the record Mach), so the target
    # identity binds the interaction dataset digests
    lin["dns"] = dns_manifest.digests(dns_manifest.INTERACTION_SET)
    return lin


def _targets_config(results_dir, fold, kind, model_seed, n_members, epochs,
                    representation, attached=False):
    """Target-file identity: fold, method, model seed, sampling seed,
    training cases, epochs, the mask constants, the baseline source, the
    db representation, and the exact upstream content hashes."""
    train = ([c for c in HEATED if c != fold]
             if not _is_exploratory(fold) else ["attached_pool"])
    return sbli_ident("apo-targets", fold, targets={
        "kind": kind, "model_seed": model_seed, "sample_seed": SAMPLE_SEED,
        "n_members": n_members, "epochs": epochs,
        "representation": representation,
        "training_cases": train,
        "baseline_source": ("gate_a_attached" if attached else
                            ("adiabatic" if _is_exploratory(fold)
                             else fold)),
        "mask": {"y_min": 0.05, "y_max": sbli_aposteriori.MASK_Y_MAX,
                 "k_floor": sbli_aposteriori.MASK_K_FLOOR},
        "lineage": _targets_lineage(results_dir, fold, attached=attached)})


def _corners_config(results_dir, fold):
    """Deterministic-family target identity (corners, five-state, the zero
    control): per fold, no seed dimension; the conditioning fields are the
    only upstream."""
    return sbli_ident("apo-targets", fold, targets={
        "kind": "corners", "labels": list(CORNER_LABELS),
        "deltas": list(sbli_aposteriori.CORNER_DELTAS),
        "mask": {"y_min": 0.05, "y_max": sbli_aposteriori.MASK_Y_MAX,
                 "k_floor": sbli_aposteriori.MASK_K_FLOOR},
        "lineage": {f"fields_{fold}.npz": cfp.file_sha(
            _fields_path(results_dir, fold)),
            "dns": dns_manifest.digests(dns_manifest.INTERACTION_SET)}})


def _member_config(results_dir, fold, kind=None, corner=None, index=None,
                   attached=False, model_seed=None):
    """The member-cache identity: physics token, production configuration,
    the member solve budget, the member's ROLE (fold, method or corner
    label, index, model and sampling seed), and the transitive lineage edge
    to the exact target file it propagated. An existence-only reuse check
    once admitted pre-audit member records into a corrected-era fold;
    identity-mismatched or unfingerprinted member files are treated as
    absent and re-solved."""
    if corner is not None:
        tpath = _targets_path(results_dir, fold, "corners")
    else:
        target_kind = kind[:-4] if (kind or "").endswith("_noq") else kind
        tpath = _targets_path(results_dir, fold, target_kind,
                              attached=attached, model_seed=model_seed)
    # the warm-start fields are bound DIRECTLY (not only through the
    # targets' own lineage): a member must never record a valid identity
    # after warm-starting from fields newer than the targets it propagated
    warm_case = ("gate_a_attached" if attached
                 else ("adiabatic" if _is_exploratory(fold) else fold))
    return sbli_ident("apo-member", fold, member={
        "injection_form": "stored-discrepancy",
        "limiter_checkpoint": True,
        "frozen_k": MEMBER_FROZEN_K, "ramp_iters": MEMBER_RAMP_ITERS,
        "max_iter": MEMBER_MAX_ITER, "tol": MEMBER_TOL,
        "abort_iter": MEMBER_ABORT_ITER,
        "abort_rel_max": MEMBER_ABORT_RELMAX,
        "corner_max_iter": CORNER_MAX_ITER,
        "n_members": N_MEMBERS,
        "role": {"fold": fold, "kind": kind, "corner": corner,
                 "index": index, "attached": bool(attached),
                 "model_seed": model_seed,
                 "sample_seed": None if corner is not None else SAMPLE_SEED},
        "lineage": {"targets": cfp.file_sha(tpath),
                    "fields": cfp.file_sha(
                        _fields_path(results_dir, warm_case))}})


def _member_current(path, config):
    if not os.path.isfile(path):
        return False
    status, _reason = cfp.check(np.load(path), config)
    return status == "match"


def _job_output(results_dir, fold, argv):
    """The output file a member worker argv will write (resume support:
    the orchestrator skips jobs whose output is identity-current)."""
    if "--corner" in argv:
        lab = argv[argv.index("--corner") + 1]
        return _corner_member_path(results_dir, fold, lab)
    kind = argv[argv.index("--kind") + 1]
    index = int(argv[argv.index("--index") + 1])
    ms = (int(argv[argv.index("--model-seed") + 1])
          if "--model-seed" in argv else None)
    return _member_path(results_dir, fold, kind, index,
                        attached="--attached" in argv, model_seed=ms)


def _job_config(results_dir, fold, argv):
    """The member-cache identity a worker argv will stamp (mirrors
    _job_output for the orchestrator's currency check)."""
    if "--corner" in argv:
        lab = argv[argv.index("--corner") + 1]
        return _member_config(results_dir, fold, corner=lab)
    kind = argv[argv.index("--kind") + 1]
    index = int(argv[argv.index("--index") + 1])
    ms = (int(argv[argv.index("--model-seed") + 1])
          if "--model-seed" in argv else None)
    return _member_config(results_dir, fold, kind=kind, index=index,
                          attached="--attached" in argv, model_seed=ms)


def _load_baseline(records, case, results_dir, quick, with_shock=True,
                   member_caps=False, corner_caps=False, derived_probe=False):
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
        kw = {"max_iterations": CORNER_MAX_ITER if corner_caps
              else MEMBER_MAX_ITER,
              "convergence_tol": MEMBER_TOL,
              "early_abort_iter": MEMBER_ABORT_ITER,
              "early_abort_rel_max": MEMBER_ABORT_RELMAX,
              "injection_frozen_k": MEMBER_FROZEN_K,
              "injection_ramp_iters": MEMBER_RAMP_ITERS}
    if derived_probe:
        kw = {"max_iterations": 1, "convergence_tol": 1e-30}
    base = _configure(records[case], quick, with_shock=with_shock, **kw)
    z = np.load(_fields_path(
        results_dir, case if with_shock else "gate_a_attached"))
    prim = z["primitive"]
    base.solver.init_field(prim)
    if member_caps and "limiter_recon" in z.files:
        # checkpoint-restart semantics (the dated addendum note): restore
        # the baseline's converged reconstruction-limiter state so the
        # member resumes the exact discrete operator it converged under;
        # without it the fresh limiters re-open the reconstruction and the
        # restart carries an operator re-transient two orders larger than
        # the remaining-decay drift (measured by the round-trip test)
        base.solver.set_frozen_reconstruction_limiter(
            np.asarray(z["limiter_recon"], dtype=float))
    if derived_probe:
        # property preparation (gradients + derived fields) WITHOUT a solver
        # iteration: the loaded converged state is not advanced (the one-sweep
        # probe drifted it by the order of the converged residual)
        base.solver.prepare_properties()
    return base, prim


def _targets_current(results_dir, fold, kind, model_seed, n_members,
                     epochs, attached=False):
    """True when the stored target file matches its full identity computed
    with the STORED representation and the CURRENT upstream lineage. A
    current file is never rewritten (a rewrite changes its content hash and
    would invalidate every completed member downstream, the review's
    restart-invalidation finding); the representation is safe to take from
    the stored config because the feasibility gate is a pure function of
    the extractions, whose hashes the identity already pins."""
    path = _targets_path(results_dir, fold, kind, attached=attached,
                         model_seed=model_seed)
    if not os.path.isfile(path):
        return False
    z = np.load(path, allow_pickle=True)
    if cfp.CONFIG_KEY not in z.files:
        return False
    stored = json.loads(str(np.asarray(z[cfp.CONFIG_KEY])[()]))
    rep = stored.get("targets", {}).get("representation")
    if rep is None:
        return False
    status, _ = cfp.check({k: z[k] for k in z.files},
                          _targets_config(results_dir, fold, kind,
                                          model_seed, n_members, epochs,
                                          rep, attached=attached))
    return status == "match"


def stage_targets(records, results_dir, fold, quick, n_members, epochs,
                  model_seed):
    """Refit the fold models at ONE model seed and write that seed's member
    target files (interaction and attached control); the deterministic
    families are written once per fold by stage_targets_deterministic.
    Identity-current target files are never rewritten and the refits are
    skipped with them (resume without member invalidation)."""
    if all(_targets_current(results_dir, fold, kind, model_seed, n_members,
                            epochs, attached=att)
           for kind in KINDS for att in (False, True)):
        print(f"[targets {fold} ms{model_seed}] current; not rewritten")
        return
    t0 = time.time()
    study = SBLIAPriori.build(records, {c: None for c in records},
                              results_dir, history=True)
    models = fold_models(study, fold, history=False,
                         seed=model_seed, epochs=epochs)
    db_raw = not study.db_gate()["all_pass"]
    representation = ("raw-components (registered feasibility reversion)"
                      if db_raw else "objective-basis")

    meta = {"fold": fold, "n_members": n_members, "epochs": epochs,
            "model_seed": model_seed, "sample_seed": SAMPLE_SEED,
            "representation": representation}

    # interaction targets, conditioned on the fold's own converged baseline
    base, _ = _load_baseline(records, fold, results_dir, quick,
                             derived_probe=True)
    feats, b_base, mask, basis_M, strain = cell_conditioning(
        base, records[fold], m_t_from_fields=True)
    meta["n_cells"] = int(len(feats))
    meta["mask_fraction"] = float(np.mean(mask))
    meta["b_base_abs_med_masked"] = float(np.median(np.abs(b_base[mask])))
    for kind in KINDS:
        b_t, dq = member_targets(models[kind], feats, b_base, mask, basis_M,
                                 db_raw=db_raw,
                                 n_members=n_members, seed=SAMPLE_SEED)
        # per-file resume: an identity-current sibling is never rewritten,
        # so its completed members survive a partial regeneration
        if _targets_current(results_dir, fold, kind, model_seed, n_members,
                            epochs):
            print(f"[targets {fold} ms{model_seed}] {kind} current; kept")
        else:
            cfp.savez_atomic(
                _targets_path(results_dir, fold, kind,
                              model_seed=model_seed),
                cfp.attach(dict(
                    b=b_t.astype(np.float32), dq=dq.astype(np.float32),
                    b_base=b_base.astype(np.float32), mask=mask),
                    _targets_config(results_dir, fold, kind, model_seed,
                                    n_members, epochs, representation)))
        meta[f"{kind}_db_fro_med_masked"] = float(np.median(
            np.linalg.norm((b_t - b_base[None])[:, mask], axis=(2, 3))))
        meta[f"{kind}_dq_abs_med"] = float(np.median(np.abs(dq)))

    # attached-control targets: the same fold models conditioned on the
    # attached configuration's own converged state (M_t from the fields;
    # no DNS record exists for this configuration)
    abase, _ = _load_baseline(records, "adiabatic", results_dir, quick,
                              with_shock=False, derived_probe=True)
    afeats, ab_base, amask, abasis_M, _astrain = cell_conditioning(
        abase, records["adiabatic"], m_t_from_fields=True)
    meta["attached_mask_fraction"] = float(np.mean(amask))
    for kind in KINDS:
        if _targets_current(results_dir, fold, kind, model_seed, n_members,
                            epochs, attached=True):
            print(f"[targets {fold} ms{model_seed}] attached {kind} "
                  f"current; kept")
            continue
        b_t, dq = member_targets(models[kind], afeats, ab_base, amask,
                                 abasis_M, db_raw=db_raw,
                                 n_members=n_members, seed=SAMPLE_SEED)
        cfp.savez_atomic(
            _targets_path(results_dir, fold, kind, attached=True,
                          model_seed=model_seed),
            cfp.attach(dict(
                b=b_t.astype(np.float32), dq=dq.astype(np.float32),
                b_base=ab_base.astype(np.float32), mask=amask),
                _targets_config(results_dir, fold, kind, model_seed,
                                n_members, epochs, representation,
                                attached=True)))

    meta["wall_time_s"] = round(time.time() - t0, 1)
    cfp.json_atomic(os.path.join(
        _apo_dir(results_dir, fold),
        f"targets_{fold}_ms{model_seed}_meta.json"), meta)
    print(f"[targets {fold} ms{model_seed}] {meta['n_cells']} cells, "
          f"mask {meta['mask_fraction']:.2f}, "
          f"{meta['wall_time_s']}s")


def stage_targets_deterministic(records, results_dir, fold, quick):
    """The deterministic per-fold family targets (eigenspace corners, the
    five-state set, the zero-control placeholder), written once per fold
    with no seed dimension; skipped when the cache is identity-current."""
    path = _targets_path(results_dir, fold, "corners")
    cfg = _corners_config(results_dir, fold)
    if _member_current(path, cfg):
        print(f"[targets {fold}] deterministic family current")
        return
    base, _ = _load_baseline(records, fold, results_dir, quick,
                             derived_probe=True)
    _feats, b_base, mask, _basis_M, strain = cell_conditioning(
        base, records[fold], m_t_from_fields=True)
    corners = corner_targets(b_base, mask)
    corners.update(five_state_targets(b_base, mask, strain))
    cfp.savez_atomic(path, cfp.attach(dict(
        mask=mask, b_base=b_base.astype(np.float32),
        **{lab: b.astype(np.float32) for lab, b in corners.items()}), cfg))
    print(f"[targets {fold}] deterministic family written")


def _save_far_targets(path, b_base, dq_draws, mask, cfg):
    """The far-target save contract: the deployment mask is enforced (rows
    outside it carry an exactly zero correction), SAVED with the targets,
    and fingerprinted inside the cache identity, so the deployed mask is a
    recorded artifact rather than an unstated in-memory choice (the
    review's far-mask finding). dq_draws is (n_cells, n_members, 2)."""
    dq = np.asarray(dq_draws, dtype=float).copy()
    dq[~mask] = 0.0
    cfp.savez_atomic(path, cfp.attach(dict(
        b=b_base[None].astype(np.float32),
        b_base=b_base.astype(np.float32),
        dq=dq.transpose(1, 0, 2).astype(np.float32),
        mask=np.asarray(mask, dtype=bool)), cfg))


def stage_targets_far(records, results_dir, quick, n_members, epochs):
    """The far-transfer propagation targets: flow and Gaussian conditionals
    trained on the attached matrix's joint heat-flux pool, sampled at the
    adiabatic interaction baseline's cells; the anisotropy target stays the
    baseline itself (the attached pool carries no interaction anisotropy
    law), so this set characterizes the propagated thermal correction.
    EXPLORATORY namespace only (the adiabatic configuration fails gate B):
    the deployment mask is saved and fingerprinted with the targets."""
    import torch
    t0 = time.time()
    study = SBLIAPriori.build(records, {c: None for c in records},
                              results_dir, history=True)
    X_tr, dq_tr = study._attached_dq_pool()
    Y_tr = dq_tr[:, 0:2]
    base, _ = _load_baseline(records, "adiabatic", results_dir, quick,
                             derived_probe=True)
    feats, b_base, mask, basis_M, strain = cell_conditioning(
        base, records["adiabatic"], m_t_from_fields=True)
    meta = {"fold": "faradiab", "n_members": n_members, "epochs": epochs,
            "model_seed": MODEL_SEEDS[0], "sample_seed": SAMPLE_SEED,
            "n_train": int(len(X_tr)), "n_cells": int(len(feats)),
            "mask_fraction": float(np.mean(mask)),
            "namespace": "exploratory"}
    for kind in KINDS:
        model = SBLIAPriori._make(kind, X_tr.shape[1], Y_tr.shape[1],
                                  MODEL_SEEDS[0])
        model.fit(X_tr, Y_tr, epochs=epochs, lr=1e-3, batch=256)
        torch.manual_seed(SAMPLE_SEED + 1)
        dq_draws = np.asarray(model.sample(feats, n_per=n_members,
                                           shared_latent=True))
        _save_far_targets(
            _targets_path(results_dir, "faradiab", kind,
                          model_seed=MODEL_SEEDS[0]),
            b_base, dq_draws[:, :, 0:2], mask,
            _targets_config(results_dir, "faradiab", kind,
                            MODEL_SEEDS[0], n_members, epochs,
                            "baseline-anisotropy (dq only)"))
    meta["wall_time_s"] = round(time.time() - t0, 1)
    cfp.json_atomic(os.path.join(
        _apo_dir(results_dir, "faradiab"), "targets_faradiab_meta.json"),
        meta)
    print(f"[targets faradiab] {meta['n_cells']} cells, "
          f"{meta['wall_time_s']}s (exploratory)")


def stage_member(records, results_dir, fold, kind, index, quick,
                 attached=False, corner=None, model_seed=None):
    """One injected coupled solve, warm-started from the baseline. The
    fold "faradiab" is the attached-trained far-transfer propagation into
    the adiabatic interaction configuration (dq-only targets, exploratory
    namespace). corner="zero_control" is the exact zero-discrepancy control
    of the fold (db = 0, dq = 0: bit-exact baseline reproduction by the
    discrete contract, run once per fold)."""
    case = "adiabatic" if (attached or fold == "faradiab") else fold
    record = records[case]
    base, _ = _load_baseline(records, case, results_dir, quick,
                             with_shock=not attached, member_caps=True,
                             corner_caps=corner is not None)

    target_kind = kind[:-4] if (kind or "").endswith("_noq") else kind
    energy_reach = not (kind or "").endswith("_noq")
    if corner is not None:
        tg = np.load(_targets_path(results_dir, fold, "corners"))
        b_base_t = np.asarray(tg["b_base"], dtype=float)
        if corner == "zero_control":
            b_t = b_base_t
            db_t = np.zeros_like(b_base_t)
        else:
            b_t = np.asarray(tg[corner], dtype=float)
            db_t = b_t - b_base_t
        dq_dim = np.zeros((b_t.shape[0], 2))
        inj_mask = (np.asarray(tg["mask"], bool) if "mask" in tg.files
                    else np.array([], dtype=bool))
        out_path = _corner_member_path(results_dir, fold, corner)
        out_cfg = _member_config(results_dir, fold, corner=corner)
    else:
        tg = np.load(_targets_path(results_dir, fold, target_kind,
                                   attached=attached,
                                   model_seed=model_seed))
        # dq-only target sets store a single shared anisotropy row (the
        # baseline itself); member rows index the heat-flux draws
        b_all = tg["b"]
        b_t = np.asarray(b_all[index if b_all.shape[0] > 1 else 0],
                         dtype=float)
        db_t = b_t - np.asarray(tg["b_base"], dtype=float)
        dq = np.asarray(tg["dq"][index], dtype=float)
        dq_dim = dq_to_solver_units(dq[None], record, base.units)[0]
        inj_mask = (np.asarray(tg["mask"], bool) if "mask" in tg.files
                    else np.array([], dtype=bool))
        out_path = _member_path(results_dir, fold, kind, index,
                                attached=attached, model_seed=model_seed)
        out_cfg = _member_config(results_dir, fold, kind=kind, index=index,
                                 attached=attached, model_seed=model_seed)

    t0 = time.time()
    if not energy_reach:
        # the registered anisotropy-only diagnostic: identical targets, the
        # energy-equation reach disabled, so the dq contribution is isolated
        dq_dim = np.zeros_like(dq_dim)
    # the STORED discrepancy is the operative injection input (the discrete
    # zero-correction contract); the absolute target rides along as a
    # provenance record, and the running realizability diagnostic checks
    # the EFFECTIVE anisotropy b_eff(W) = b_B(W) + db every residual
    # evaluation (recorded, never projected)
    base.solver.set_target_correction(db_t, b_t, dq_dim, energy_reach,
                                      mask=inj_mask)
    rep = base.solver.solve()
    w = base.wall()
    lm = landmarks_from_wall(w)
    diag = base.solver.injection_diagnostics()
    cfp.savez_atomic(
        out_path, cfp.attach(dict(
            x_star=w["x_star"], Cf=w["Cf"], Cp=w["Cp"],
            qw=w["qw"], St=w["St"],
            status=np.bytes_(str(rep.status).encode()),
            iterations=np.int64(rep.iterations),
            final_residual=np.float64(rep.final_residual),
            x_s=np.float64(np.nan if lm["x_s"] is None else lm["x_s"]),
            x_r=np.float64(np.nan if lm["x_r"] is None else lm["x_r"]),
            shock=np.float64(np.nan if lm["shock"] is None else lm["shock"]),
            all_realizable=np.bool_(bool(diag["all_realizable"])),
            min_margin=np.float64(diag["min_margin"]),
            min_margin_iter=np.int64(diag["min_margin_iter"]),
            min_margin_cell=np.int64(diag["min_margin_cell"]),
            max_violation=np.float64(diag["max_violation"]),
            max_violation_iter=np.int64(diag["max_violation_iter"]),
            max_violation_cell=np.int64(diag["max_violation_cell"]),
            max_db=np.float64(diag["max_db"]),
            max_dq=np.float64(diag["max_dq"]),
            wall_time_s=np.float64(round(time.time() - t0, 1))),
            out_cfg))
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
            "min_margin": float(z["min_margin"]),
            "max_violation": float(z["max_violation"]),
            "wall": {k: np.asarray(z[k])
                     for k in ("x_star", "Cf", "Cp", "qw", "St")}}


_MISSING = {"status": "Missing(worker output absent or stale identity)",
            "landmarks": {"x_s": None, "x_r": None, "shock": None}}


def _seed_reduce(per_seed, fn):
    """Elementwise reduction over the per-seed score dicts: numeric scalar
    leaves only (bools contribute as 0/1), lists and strings stay per-seed;
    keys reduce where present in every seed. The criteria read the mean;
    min and max give the reported seed spread."""
    def red(vs):
        if all(isinstance(v, dict) for v in vs):
            keys = sorted(set.intersection(*[set(v) for v in vs]))
            out = {}
            for k in keys:
                r = red([v[k] for v in vs])
                if r is not None:
                    out[k] = r
            return out or None
        if all(isinstance(v, (bool, int, float)) for v in vs):
            return fn([float(v) for v in vs])
        return None
    return red(list(per_seed.values())) or {}


def _score_one_ensemble(results_dir, fold, kind, model_seed, n_members,
                        record, bwall, lineage):
    """One (method, model seed) 24-member ensemble, scored separately (the
    seed-resolved protocol: per-seed ensembles are never pooled)."""
    paths = [_member_path(results_dir, fold, kind, i, model_seed=model_seed)
             for i in range(n_members)]
    cfgs = [_member_config(results_dir, fold, kind=kind, index=i,
                           model_seed=model_seed)
            for i in range(n_members)]
    # a missing worker output is a NON-CONVERGED member for every
    # denominator: the 18-of-24 instability rule reads against the
    # REQUESTED count, never the found-file count
    members = [(_load_member(p) if _member_current(p, c) else dict(_MISSING))
               for p, c in zip(paths, cfgs)]
    for p, c in zip(paths, cfgs):
        if _member_current(p, c):
            lineage[os.path.basename(p)] = cfp.file_sha(p)
    out = score_ensemble(members, record, baseline_wall=bwall)
    out["n_requested"] = n_members
    out["n_found"] = sum(1 for p, c in zip(paths, cfgs)
                         if _member_current(p, c))
    out["model_seed"] = model_seed
    conv_m = [m for m in members if "Converged" in m["status"]]
    if conv_m:
        # the pre-registered realizability-in-the-running-solve clause on
        # the EFFECTIVE anisotropy, aggregated over CONVERGED members (a
        # diverged member's diagnostics describe no admitted state)
        out["all_realizable_fraction"] = float(np.mean(
            [m["all_realizable"] for m in conv_m]))
        out["max_violation"] = float(np.max(
            [m["max_violation"] for m in conv_m]))
        out["min_effective_margin"] = float(np.min(
            [m["min_margin"] for m in conv_m]))
    return out, members


def _baseline_wall(results_dir, case):
    p = os.path.join(results_dir, f"wall_{case}.npz")
    if not os.path.isfile(p):
        return None
    z = np.load(p)
    return {k: np.asarray(z[k]) for k in ("x_star", "Cf", "Cp", "qw", "St")}


def stage_score(records, results_dir, fold, n_members):
    """Assemble one fold's ensembles into the scored record: per model seed
    separately (never pooled), with the seed mean and min-max spread, the
    any-seed 18-of-24 instability label, and the content-hash lineage of
    every consumed target and member file. The formal path refuses the
    exploratory fold outright (structural non-consumption)."""
    if _is_exploratory(fold):
        print(f"[score] {fold} is exploratory-namespace only; the formal "
              f"assembler does not consume it (stage_score_exploratory)")
        sys.exit(1)
    record = records[fold]
    bwall = _baseline_wall(results_dir, fold)
    out = {"fold": fold, "n_members": n_members,
           "model_seeds": list(MODEL_SEEDS), "sample_seed": SAMPLE_SEED}
    lineage = {}
    for kind in KINDS + ("flow_noq",):
        per_seed = {}
        for ms in MODEL_SEEDS:
            sc, members = _score_one_ensemble(
                results_dir, fold, kind, ms, n_members, record, bwall,
                lineage)
            # the adiabatic-middle fold carries the independent campaign's
            # wall series as a second truth surface (the pre-registered
            # pairing); the baseline comparison line is the fold's OWN
            # baseline scored on that surface (the excluded adiabatic solve
            # never supplies a claim-bearing reference)
            if fold == "s1.0" and "adiabatic" in records and sc["n_found"]:
                sc["second_surface"] = score_ensemble(
                    members, records["adiabatic"],
                    baseline_wall=_baseline_wall(results_dir, "s1.0"))
            per_seed[str(ms)] = sc
        entry = {
            "per_seed": per_seed,
            "seed_mean": _seed_reduce(per_seed, np.mean),
            "seed_min": _seed_reduce(per_seed, np.min),
            "seed_max": _seed_reduce(per_seed, np.max),
            "per_seed_n_converged": {s: per_seed[s]["n_converged"]
                                     for s in per_seed},
            # the aggregate carries the label when ANY constituent seed
            # falls below 18 of 24 (no borrowing across seeds)
            "propagation_unstable_any_seed": bool(any(
                per_seed[s]["propagation_unstable"] for s in per_seed)),
        }
        key = ("flow_energy_reach_disabled_diagnostic"
               if kind == "flow_noq" else kind)
        out[key] = entry
    # target-file lineage for every seed and kind, plus the baseline wall
    # and gate records the comparisons consume
    for kind in KINDS:
        for ms in MODEL_SEEDS:
            p = _targets_path(results_dir, fold, kind, model_seed=ms)
            lineage[os.path.basename(p)] = cfp.file_sha(p)
    p = _targets_path(results_dir, fold, "corners")
    lineage[os.path.basename(p)] = cfp.file_sha(p)
    for name in (f"wall_{fold}.npz", "gate_a.json",
                 "gates_adjudication.json"):
        lineage[name] = cfp.file_sha(os.path.join(results_dir, name))

    # the exact zero-discrepancy control (once per fold): status and the
    # wall drift against the fold baseline (the contract's field check)
    zc_path = _corner_member_path(results_dir, fold, "zero_control")
    zc_cfg = _member_config(results_dir, fold, corner="zero_control")
    if _member_current(zc_path, zc_cfg):
        zc = _load_member(zc_path)
        entry = {"status": zc["status"], "iterations": zc["iterations"]}
        if bwall is not None:
            drift = {}
            for q in ("Cf", "Cp", "qw", "St"):
                a = np.interp(bwall["x_star"], zc["wall"]["x_star"],
                              zc["wall"][q])
                scale = float(np.max(np.abs(bwall[q]))) or 1.0
                drift[q] = float(np.max(np.abs(a - bwall[q])) / scale)
            entry["max_rel_wall_drift"] = drift
        out["zero_control"] = entry
        lineage[os.path.basename(zc_path)] = cfp.file_sha(zc_path)

    # eigenspace families: PER-AMPLITUDE envelopes over the converged
    # members (never merged across amplitudes), the three-corner family and
    # the documented five-state family separately, each with the EXACT
    # discrete-forecast CRPS (the members ARE the forecast; the fair
    # estimator applies to sampled predictives only)
    series = record.series
    xs = STATIONS[(STATIONS >= series.x[0]) & (STATIONS <= series.x[-1])]
    # the SHARED truths construction (carries the s = 1.0 field-row Cp
    # fallback, so the eigenspace families score Cp on that fold too)
    truths = sbli_aposteriori.station_truths(record, xs)

    def _family_scores(labels):
        fam = {}
        for lab in labels:
            p = _corner_member_path(results_dir, fold, lab)
            if _member_current(p, _member_config(results_dir, fold,
                                                 corner=lab)):
                fam[lab] = _load_member(p)
                lineage[os.path.basename(p)] = cfp.file_sha(p)
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
    # member against the baseline's own value and the measured level; per
    # model seed separately (the tripled control of the seed protocol)
    from UQ.reproduce_sbli_apriori import _gate_ident, _json_ident_ok
    gate_a_path = os.path.join(results_dir, "gate_a.json")
    if not _json_ident_ok(gate_a_path,
                          _gate_ident(results_dir, "gate_a_attached")):
        print("[score] gate_a.json stale identity; regenerate the gates "
              "before scoring")
        sys.exit(1)
    gate_a = json.load(open(gate_a_path))
    cf_base = float(gate_a["cf_at_station"])
    att = {"cf_baseline": cf_base, "cf_data": GATE_A_CF}
    for kind in KINDS:
        per_seed = {}
        for ms in MODEL_SEEDS:
            cfs = []
            n_conv = 0
            for i in range(n_members):
                p = _member_path(results_dir, fold, kind, i, attached=True,
                                 model_seed=ms)
                c = _member_config(results_dir, fold, kind=kind, index=i,
                                   attached=True, model_seed=ms)
                if not _member_current(p, c):
                    continue
                lineage[os.path.basename(p)] = cfp.file_sha(p)
                m = _load_member(p)
                if "Converged" not in m["status"]:
                    continue
                n_conv += 1
                cfs.append(float(np.interp(GATE_A_STATION,
                                           m["wall"]["x_star"],
                                           m["wall"]["Cf"])))
            base_err = abs(cf_base - GATE_A_CF)
            per_seed[str(ms)] = {
                "n_converged": n_conv,
                "cf_members": cfs,
                # the REGISTERED criteria: member error to the MEASURED
                # value, relative to the baseline's own error to it; and
                # the 5-95 band HALF-width over the measured skin friction
                "cf_error_over_baseline_error_median": (float(np.median(
                    np.abs(np.asarray(cfs) - GATE_A_CF))
                    / max(base_err, 1e-12)) if cfs else None),
                "cf_band_halfwidth_over_data": (float(
                    0.5 * (np.quantile(cfs, 0.95) - np.quantile(cfs, 0.05))
                    / GATE_A_CF) if len(cfs) >= 2 else None),
            }
        att[kind] = {"per_seed": per_seed,
                     "seed_mean": _seed_reduce(per_seed, np.mean),
                     "seed_min": _seed_reduce(per_seed, np.min),
                     "seed_max": _seed_reduce(per_seed, np.max)}
    out["attached_control"] = att

    out["lineage"] = lineage
    path = os.path.join(_apo_dir(results_dir, fold),
                        f"fold_scores_{fold}.json")
    cfp.json_atomic(path, cfp.attach_json(out, sbli_ident(
        "fold-score", fold, score={"n_members": n_members,
                                   "model_seeds": list(MODEL_SEEDS),
                                   "sample_seed": SAMPLE_SEED})))
    print(f"[score {fold}] wrote {path}")
    return out


def fold_score_lineage_ok(results_dir, fold):
    """Transitive validation of a fold score for downstream consumers
    (figures, the memo): every file hash the score recorded must still
    match the file on disk. Names resolve in the fold's aposteriori
    directory first, then the results root."""
    path = os.path.join(_apo_dir(results_dir, fold),
                        f"fold_scores_{fold}.json")
    if not os.path.isfile(path):
        return False
    rec = json.load(open(path))
    for name, sha in rec.get("lineage", {}).items():
        cand = os.path.join(_apo_dir(results_dir, fold), name)
        if not os.path.isfile(cand):
            cand = os.path.join(results_dir, name)
        if cfp.file_sha(cand) != sha:
            return False
    return True


def stage_score_exploratory(records, results_dir, n_members):
    """The faradiab characterization score, exploratory namespace only:
    the attached-trained thermal correction propagated into the gate-B
    failing adiabatic configuration, labeled, single model seed, never
    consumed by the formal assembler."""
    fold = "faradiab"
    record = records["adiabatic"]
    bwall = _baseline_wall(results_dir, "adiabatic")
    out = {"fold": fold, "n_members": n_members, "namespace": "exploratory",
           "model_seed": MODEL_SEEDS[0], "sample_seed": SAMPLE_SEED,
           "label": "gate-B failing configuration; solver-capability "
                    "boundary characterization, never claim-bearing"}
    lineage = {}
    for kind in KINDS:
        sc, _members = _score_one_ensemble(
            results_dir, fold, kind, MODEL_SEEDS[0], n_members, record,
            bwall, lineage)
        out[kind] = sc
    out["lineage"] = lineage
    path = os.path.join(_apo_dir(results_dir, fold),
                        f"fold_scores_{fold}.json")
    cfp.json_atomic(path, out)
    print(f"[score {fold}] wrote {path} (exploratory)")
    return out


def stage_zerocheck(records, results_dir, quick, cases):
    """The pre-matrix checkpoint-round-trip probes (the review's restart
    requirement): reload each converged baseline through the member path
    (restored reconstruction limiter, member budget), inject the exact
    zero correction, solve, and record the classification and the wall
    drift against the baseline's own wall record. The measured drift is
    the common-mode restart floor every member of that fold shares; a
    material drift is a stop-and-adjudicate state before any target is
    generated."""
    for case in cases:
        attached = case == "gate_a_attached"
        rec_case = "adiabatic" if attached else case
        base, _ = _load_baseline(records, rec_case, results_dir, quick,
                                 with_shock=not attached, member_caps=True)
        nc = base.solver.n_cells()
        zero3 = np.zeros((nc, 3, 3))
        t0 = time.time()
        base.solver.set_target_correction(zero3, zero3, np.zeros((nc, 2)),
                                          True)
        rep = base.solver.solve()
        w = base.wall()
        out = {"case": case, "status": str(rep.status),
               "iterations": int(rep.iterations),
               "final_residual": float(rep.final_residual),
               "quiescent": bool(rep.quiescent),
               "wall_time_s": round(time.time() - t0, 1),
               "limiter_checkpoint": True,
               "member_budget": {"max_iter": MEMBER_MAX_ITER,
                                 "tol": MEMBER_TOL}}
        bwall = _baseline_wall(results_dir, case)
        if bwall is not None:
            drift = {}
            for q in ("Cf", "Cp", "qw", "St"):
                a2 = np.interp(bwall["x_star"], w["x_star"], w[q])
                scale = float(np.max(np.abs(bwall[q]))) or 1.0
                drift[q] = float(np.max(np.abs(a2 - bwall[q])) / scale)
            out["max_rel_wall_drift"] = drift
        lm = landmarks_from_wall(w)
        out["landmarks"] = {k: lm[k] for k in ("x_s", "x_r", "shock")}
        cfp.json_atomic(
            os.path.join(results_dir, f"zerocheck_{case}.json"),
            cfp.attach_json(out, sbli_ident("zerocheck", case, zc={
                "lineage": {
                    "fields": cfp.file_sha(_fields_path(results_dir, case)),
                    "wall": cfp.file_sha(_wall_path(results_dir, case))}})))
        print(f"[zerocheck {case}] {rep.status} iters {rep.iterations} "
              f"drift {out.get('max_rel_wall_drift')}")


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
                      epochs, folds, model_seeds=MODEL_SEEDS):
    """The full overnight pipeline: per fold, the deterministic family
    targets once, then per model seed the probabilistic targets, then the
    throttled member fan-out (interaction members per seed, deterministic
    corners and the zero control once, attached control per seed), then the
    per-seed fold score; a combined numbers file at the end. The faradiab
    exploratory phase runs ONLY under QBTM_SBLI_EXPLORATORY=1 and its
    outputs never enter the formal numbers."""
    common = ["--results", results_dir] + (["--quick"] if quick else [])
    log_path = os.path.join(_apo_dir(results_dir), "workers.log")

    def _member_jobs(fold, kinds=KINDS, corners=False, attached=False,
                     seeds=model_seeds):
        jobs = []
        for kind in kinds:
            for ms in seeds:
                for i in range(n_members):
                    j = ["--stage", "member", "--fold", fold,
                         "--kind", kind, "--index", str(i),
                         "--model-seed", str(ms)] + common
                    if attached:
                        j.append("--attached")
                    jobs.append(j)
        if corners:
            for lab in CORNER_LABELS:
                jobs.append(["--stage", "member", "--fold", fold,
                             "--corner", lab] + common)
        # identity-current members never re-solve: drop jobs whose output
        # matches its full lineage-bearing identity
        return [j for j in jobs if not _member_current(
            _job_output(results_dir, fold, j),
            _job_config(results_dir, fold, j))]

    summary = {}
    # phase 1: the interaction ensembles per model seed, the deterministic
    # families (three-corner, five-state, the zero control) once per fold,
    # and the registered anisotropy-only diagnostic (the flow targets with
    # the energy reach disabled) per seed
    for fold in folds:
        stage_targets_deterministic(records, results_dir, fold, quick)
        for ms in model_seeds:
            stage_targets(records, results_dir, fold, quick, n_members,
                          epochs, ms)
        failed = _run_pool(_member_jobs(fold, kinds=KINDS + ("flow_noq",),
                                        corners=True), throttle,
                           log_path)
        print(f"[orchestrate {fold}] interaction members done, "
              f"{failed} worker failures")
        summary[fold] = stage_score(records, results_dir, fold, n_members)
        summary[fold]["worker_failures"] = int(failed)

    # phase 2: the preserve-attached control for every fold closure and
    # every model seed (the tripled control of the seed protocol)
    for fold in folds:
        failed = _run_pool(_member_jobs(fold, attached=True), throttle,
                           log_path)
        print(f"[orchestrate {fold}] attached control done, "
              f"{failed} worker failures")
        prior = summary[fold]["worker_failures"]
        summary[fold] = stage_score(records, results_dir, fold, n_members)
        summary[fold]["worker_failures"] = prior + int(failed)

    # phase 3 (exploratory namespace only, explicit opt-in): the
    # attached-trained far-transfer propagation into the gate-B failing
    # adiabatic configuration; structurally outside the formal numbers
    if os.environ.get("QBTM_SBLI_EXPLORATORY") == "1":
        stage_targets_far(records, results_dir, quick, n_members, epochs)
        failed = _run_pool(_member_jobs("faradiab", seeds=(MODEL_SEEDS[0],)),
                           throttle, log_path)
        print(f"[orchestrate faradiab] members done, {failed} worker "
              f"failures (exploratory)")
        stage_score_exploratory(records, results_dir, n_members)
    else:
        print("[orchestrate] faradiab exploratory phase skipped "
              "(QBTM_SBLI_EXPLORATORY unset)")

    suffix = "_quick" if quick else ""
    numbers = {
        "folds": summary,
        "config": {"n_members": n_members,
                   "model_seeds": list(model_seeds),
                   "sample_seed": SAMPLE_SEED,
                   "epochs": epochs, "kinds": list(KINDS),
                   "member_max_iterations": MEMBER_MAX_ITER,
                   "member_convergence_tol": MEMBER_TOL,
                   "member_early_abort": {"iter": MEMBER_ABORT_ITER,
                                          "rel_max": MEMBER_ABORT_RELMAX,
                                          "note": "retired pending the "
                                                  "abort panel amendment"},
                   "corner_max_iterations": CORNER_MAX_ITER,
                   "corner_deltas": list(sbli_aposteriori.CORNER_DELTAS),
                   "stations": [float(v) for v in STATIONS],
                   "mask": {"y_max": sbli_aposteriori.MASK_Y_MAX,
                            "k_floor": sbli_aposteriori.MASK_K_FLOOR},
                   "quick": bool(quick)},
    }
    path = os.path.join(results_dir, f"aposteriori_numbers{suffix}.json")
    cfp.json_atomic(path, cfp.attach_json(numbers, sbli_ident(
        "aposteriori-numbers", "all-cases", numbers={
            "n_members": n_members, "model_seeds": list(model_seeds),
            "quick": bool(quick)})))
    print("wrote", path)


def _require_gates(results_dir, fold):
    """Claim-bearing coupled stages run only on gate-passing configurations
    (the recorded per-case ruling); the far-transfer target is exempt only
    when explicitly labeled exploratory via QBTM_SBLI_EXPLORATORY=1."""
    ok, why = dns_manifest.verify_manifest(
        os.path.join(results_dir, "dns_manifest.json"))
    if not ok:
        print(f"[gates] DNS manifest gate failed ({why}); run the a-priori "
              f"driver to adjudicate before any coupled stage")
        sys.exit(1)
    path = os.path.join(results_dir, "gates_adjudication.json")
    if not os.path.isfile(path):
        print("[gates] no adjudication record; run the a-priori baselines "
              "stage first")
        sys.exit(1)
    adjud = json.load(open(path))
    # transitive validation: the adjudication's recorded gate-record hashes
    # must match the records on disk (a superseded gate can never authorize
    # a coupled stage)
    ident = adjud.get(cfp.IDENTITY_JSON_KEY)
    if not isinstance(ident, dict) or "config_json" not in ident:
        print("[gates] adjudication record carries no identity block; "
              "regenerate it (baselines stage) before coupled stages")
        sys.exit(1)
    lin = json.loads(ident["config_json"]).get("adjud", {}).get("lineage",
                                                                {})
    for name, sha in lin.items():
        if cfp.file_sha(os.path.join(results_dir, name)) != sha:
            print(f"[gates] adjudication lineage stale for {name}; "
                  f"regenerate the adjudication before coupled stages")
            sys.exit(1)
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
                    choices=("targets", "member", "score", "orchestrate",
                             "zerocheck"))
    ap.add_argument("--results", default="results/sbli")
    ap.add_argument("--fold", default=None)
    ap.add_argument("--kind", default="flow",
                    choices=("flow", "gauss", "flow_noq"))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--model-seed", type=int, default=MODEL_SEEDS[0],
                    dest="model_seed")
    ap.add_argument("--attached", action="store_true")
    ap.add_argument("--corner", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--throttle", type=int, default=4)
    ap.add_argument("--folds", default=",".join(FOLDS))
    args = ap.parse_args()
    if args.stage != "zerocheck" and getattr(args, "fold", None):
        _require_gates(args.results, args.fold)

    np.random.seed(SAMPLE_SEED)
    n_members = 3 if args.quick else N_MEMBERS
    epochs = 2 if args.quick else sbli_apriori.EPOCHS
    model_seeds = (MODEL_SEEDS[0],) if args.quick else MODEL_SEEDS
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
        if _is_exploratory(args.fold):
            stage_targets_far(records, args.results, args.quick, n_members,
                              epochs)
        else:
            stage_targets_deterministic(records, args.results, args.fold,
                                        args.quick)
            stage_targets(records, args.results, args.fold, args.quick,
                          n_members, epochs, args.model_seed)
    if args.stage == "member":
        stage_member(records, args.results, args.fold, args.kind,
                     args.index, args.quick, attached=args.attached,
                     corner=args.corner, model_seed=args.model_seed)
    if args.stage == "score":
        if _is_exploratory(args.fold):
            stage_score_exploratory(records, args.results, n_members)
        else:
            stage_score(records, args.results, args.fold, n_members)
    if args.stage == "orchestrate":
        folds = [f.strip() for f in args.folds.split(",") if f.strip()]
        stage_orchestrate(records, args.results, args.quick, args.throttle,
                          n_members, epochs, folds,
                          model_seeds=model_seeds)
    if args.stage == "zerocheck":
        cases = ([f.strip() for f in args.folds.split(",") if f.strip()]
                 if args.folds != ",".join(FOLDS)
                 else ["gate_a_attached"] + list(FOLDS))
        stage_zerocheck(records, args.results, args.quick, cases)


if __name__ == "__main__":
    main()
