"""The discriminating lineage and role tests gating the corrected SBLI
regeneration (the seed-resolved plan's Phase L gate): transitive
content-hash invalidation (fields -> extraction, targets -> member),
model-seed cache separation, far-mask persistence, conformal role
disjointness, the frozen-mean fallback enforcement, and the exploratory
namespace the formal assembler cannot consume. Hermetic: synthetic caches
in a temp tree, no DNS data, no solver. The solver-side gates (effective
running realizability recording, injection conservation, sign and energy
work, the zero-discrepancy bit identity, the frozen-mean transport mode)
live in tests/cpp/test_dbns_injection.cpp."""
import json
import multiprocessing
import os
import shutil
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "python"))

from UQ import cache_fingerprint as cfp
from UQ.datasets import dns_manifest as dm
from UQ.datasets.sbli_apriori import (extraction_ident,
                                      extraction_fields_path,
                                      conformal_case_split, sbli_ident)
from UQ.datasets.heatflux_apriori import mach_family
from UQ.datasets.gv_channel import GV_CASES
from UQ import reproduce_sbli_aposteriori as apo


def _write_npz(path, cfg, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfp.savez_atomic(path, cfp.attach(arrays, cfg))


def _synthetic_dns_root(base, content=b"v1"):
    """A source-data root carrying every REGISTERED dataset with one file
    each: the hermetic stand-in for QBTM_DNS_DATA in the manifest and
    adoption gates (no real DNS, no solver)."""
    for name, sub in dm.DATASETS.items():
        p = os.path.join(base, sub)
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "f.dat"), "wb") as f:
            f.write(content + name.encode())
    return base


def _dataset_file(root, name):
    return os.path.join(root, dm.DATASETS[name], "f.dat")


def test_field_mutation_invalidates_extraction(tmp_path):
    d = str(tmp_path)
    fpath = extraction_fields_path(d, "s1.0")
    cfp.savez_atomic(fpath, {"primitive": np.zeros((4, 6))})
    ident = extraction_ident("s1.0", (4, 4), True, d)
    epath = os.path.join(d, "extract_s1.0_s4x4_hist.npz")
    _write_npz(epath, ident, features=np.zeros((3, 6)))
    z = np.load(epath)
    status, _ = cfp.check({k: z[k] for k in z.files},
                          extraction_ident("s1.0", (4, 4), True, d))
    assert status == "match"
    # mutate the parent fields cache: the extraction identity no longer
    # matches (the transitive edge), so reuse is refused
    cfp.savez_atomic(fpath, {"primitive": np.ones((4, 6))})
    status, _ = cfp.check({k: z[k] for k in z.files},
                          extraction_ident("s1.0", (4, 4), True, d))
    assert status == "mismatch"
    # a missing parent is embedded as None and also refuses
    os.remove(fpath)
    status, _ = cfp.check({k: z[k] for k in z.files},
                          extraction_ident("s1.0", (4, 4), True, d))
    assert status == "mismatch"


def test_target_mutation_invalidates_members(tmp_path):
    d = str(tmp_path)
    tpath = apo._targets_path(d, "s1.0", "flow", model_seed=0)
    cfp.savez_atomic(tpath, {"b": np.zeros((2, 3, 3, 3))})
    cfg = apo._member_config(d, "s1.0", kind="flow", index=0, model_seed=0)
    mpath = apo._member_path(d, "s1.0", "flow", 0, model_seed=0)
    _write_npz(mpath, cfg, Cf=np.zeros(5))
    assert apo._member_current(
        mpath, apo._member_config(d, "s1.0", kind="flow", index=0,
                                  model_seed=0))
    # regenerating the target file changes its content hash: the member is
    # treated as absent and re-solved
    cfp.savez_atomic(tpath, {"b": np.ones((2, 3, 3, 3))})
    assert not apo._member_current(
        mpath, apo._member_config(d, "s1.0", kind="flow", index=0,
                                  model_seed=0))


def test_model_seed_cache_separation(tmp_path):
    d = str(tmp_path)
    # distinct paths per model seed for targets and members
    assert apo._targets_path(d, "s1.0", "flow", model_seed=0) \
        != apo._targets_path(d, "s1.0", "flow", model_seed=1)
    assert apo._member_path(d, "s1.0", "flow", 0, model_seed=0) \
        != apo._member_path(d, "s1.0", "flow", 0, model_seed=1)
    # distinct identities: a member written under one seed's identity never
    # satisfies another seed's check, even at a forced shared path
    for ms in (0, 1):
        cfp.savez_atomic(apo._targets_path(d, "s1.0", "flow", model_seed=ms),
                         {"b": np.full((1, 2), float(ms))})
    cfg0 = apo._member_config(d, "s1.0", kind="flow", index=0, model_seed=0)
    cfg1 = apo._member_config(d, "s1.0", kind="flow", index=0, model_seed=1)
    assert cfp.fingerprint(cfg0) != cfp.fingerprint(cfg1)
    shared = os.path.join(d, "forced_shared.npz")
    _write_npz(shared, cfg0, Cf=np.zeros(3))
    assert apo._member_current(shared, cfg0)
    assert not apo._member_current(shared, cfg1)
    # the sampling seed is recorded distinct from the model seed
    role = cfg1["member"]["role"]
    assert role["model_seed"] == 1
    assert role["sample_seed"] == apo.SAMPLE_SEED


def test_far_mask_persistence_and_boundary(tmp_path):
    d = str(tmp_path)
    rng = np.random.default_rng(0)
    n, m = 40, 6
    b_base = rng.normal(0.0, 0.1, size=(n, 3, 3))
    dq_draws = rng.normal(0.0, 1.0, size=(n, m, 2))
    mask = np.zeros(n, dtype=bool)
    mask[10:30] = True
    path = apo._targets_path(d, "faradiab", "flow", model_seed=0)
    cfg = {"kind": "far-test", "mask": True}
    apo._save_far_targets(path, b_base, dq_draws, mask, cfg)
    z = np.load(path)
    # the deployment mask is persisted and fingerprinted with the targets
    assert "mask" in z.files
    assert np.array_equal(np.asarray(z["mask"], bool), mask)
    status, _ = cfp.check({k: z[k] for k in z.files}, cfg)
    assert status == "match"
    # boundary behavior: outside the mask the correction is exactly zero,
    # inside it is untouched
    dq = np.asarray(z["dq"], float)          # (m, n, 2)
    assert np.all(dq[:, ~mask, :] == 0.0)
    assert np.allclose(dq[:, mask, :],
                       dq_draws.transpose(1, 0, 2)[:, mask, :].astype(
                           np.float32))


def test_conformal_role_disjointness():
    fit, cal = conformal_case_split(GV_CASES)
    assert not (set(fit) & set(cal))
    assert sorted(fit + cal) == sorted(GV_CASES)
    assert len(cal) >= 5
    # both roles span every Mach family (the frozen alternation rule)
    fams = {mach_family(t) for t in GV_CASES}
    assert {mach_family(t) for t in fit} == fams
    assert {mach_family(t) for t in cal} == fams
    # deterministic: the split is frozen, not sampled
    assert conformal_case_split(GV_CASES) == (fit, cal)
    assert conformal_case_split(list(reversed(GV_CASES))) == (fit, cal)


def test_frozen_mean_fallback_enforcement(tmp_path):
    # the adiabatic 2011 campaign's extraction parent is the frozen-mean
    # march, never the free-running gate-failing solve
    d = str(tmp_path)
    assert extraction_fields_path(d, "adiabatic").endswith(
        "fields_adiabatic_frozenmean.npz")
    assert extraction_fields_path(d, "s1.0").endswith("fields_s1.0.npz")
    ident = extraction_ident("adiabatic", (8, 4), True, d)
    assert ident["extract"]["route"] == "frozen-mean"
    assert extraction_ident("s1.0", (4, 4), True, d)["extract"]["route"] \
        == "free-running"


def test_exploratory_namespace_structural(tmp_path):
    d = str(tmp_path)
    # faradiab artifacts land under aposteriori/exploratory/
    assert apo._is_exploratory("faradiab")
    assert not apo._is_exploratory("s1.0")
    p = apo._targets_path(d, "faradiab", "flow", model_seed=0)
    assert os.sep + "exploratory" + os.sep in p
    assert os.sep + "exploratory" + os.sep not in apo._targets_path(
        d, "s1.0", "flow", model_seed=0)
    # the formal scorer refuses the exploratory fold outright
    with pytest.raises(SystemExit):
        apo.stage_score({}, d, "faradiab", 3)


def test_seed_reduce_semantics():
    per_seed = {
        "0": {"cov": 0.8, "flag": True, "nested": {"a": 1.0},
              "arr": [1, 2], "name": "x"},
        "1": {"cov": 0.6, "flag": False, "nested": {"a": 3.0},
              "arr": [3, 4], "name": "y"},
    }
    mean = apo._seed_reduce(per_seed, np.mean)
    assert mean["cov"] == pytest.approx(0.7)
    assert mean["flag"] == pytest.approx(0.5)      # bools as 0/1 fractions
    assert mean["nested"]["a"] == pytest.approx(2.0)
    assert "arr" not in mean and "name" not in mean
    lo = apo._seed_reduce(per_seed, np.min)
    hi = apo._seed_reduce(per_seed, np.max)
    assert lo["cov"] == pytest.approx(0.6)
    assert hi["cov"] == pytest.approx(0.8)


def test_json_identity_roundtrip(tmp_path):
    cfg = {"kind": "gate-record", "case": "x", "gate": {"lineage": {"a": "1"}}}
    rec = cfp.attach_json({"pass": True}, cfg)
    assert cfp.check_json(rec, cfg)[0] == "match"
    cfg2 = {"kind": "gate-record", "case": "x", "gate": {"lineage": {"a": "2"}}}
    assert cfp.check_json(rec, cfg2)[0] == "mismatch"
    assert cfp.check_json({"pass": True}, cfg)[0] == "legacy"


def test_case_level_conformal_quantile():
    # the case-level convention: with twelve calibration cases at alpha 0.10
    # the finite-sample split-conformal quantile is their maximum (the
    # ceil((n+1)(1-alpha))/n level exceeds 11/12), never a row-pooled value
    from UQ import conformal
    scores = np.arange(12, dtype=float)
    assert conformal.conformal_quantile(scores, alpha=0.10) == 11.0


def test_station_truths_field_row_fallback():
    from UQ.datasets.sbli_aposteriori import station_truths

    class _Series:
        x = np.linspace(-10.0, 10.0, 41)
        cf = np.full(41, 2.5e-3)
        cp = None
        St = None

    class _Rec:
        series = _Series()

        @staticmethod
        def cp_from_field():
            xf = np.linspace(-10.0, 10.0, 21)
            return xf, 0.1 * xf

    xs = np.arange(-5.0, 6.0, 1.0)
    truths = station_truths(_Rec(), xs)
    # the s = 1.0 quantized-series case: Cp comes from the field wall row,
    # so the eigenspace family scorer sees Cp on that fold too
    assert "Cp" in truths
    assert np.allclose(truths["Cp"], 0.1 * xs)
    assert "St" not in truths


def test_targets_resume_without_rewrite(tmp_path):
    d = str(tmp_path)
    # synthetic upstream universe: heated extractions at both strides plus
    # the conditioning fields
    from UQ.datasets.sbli_apriori import TEST_STRIDE, _train_stride
    for c in apo.HEATED:
        for st in (TEST_STRIDE[c], _train_stride(c)):
            pth = os.path.join(d, f"extract_{c}_s{st[0]}x{st[1]}_hist.npz")
            cfp.savez_atomic(pth, {"features": np.zeros((2, 6))})
    cfp.savez_atomic(os.path.join(d, "fields_s1.0.npz"),
                     {"primitive": np.zeros((3, 6))})
    cfg = apo._targets_config(d, "s1.0", "flow", 0, 3, 2, "objective-basis")
    tpath = apo._targets_path(d, "s1.0", "flow", model_seed=0)
    cfp.savez_atomic(tpath, cfp.attach({"b": np.zeros((1, 2))}, cfg))
    sha_before = cfp.file_sha(tpath)
    assert apo._targets_current(d, "s1.0", "flow", 0, 3, 2)
    # the currency check never touches the file, so member identities
    # (which bind the target hash) stay intact across a resume
    assert cfp.file_sha(tpath) == sha_before
    mcfg1 = apo._member_config(d, "s1.0", kind="flow", index=0, model_seed=0)
    assert apo._targets_current(d, "s1.0", "flow", 0, 3, 2)
    mcfg2 = apo._member_config(d, "s1.0", kind="flow", index=0, model_seed=0)
    assert cfp.fingerprint(mcfg1) == cfp.fingerprint(mcfg2)
    # a mutated upstream extraction invalidates the target (regeneration is
    # then required, which is the moment members legitimately invalidate)
    pth = os.path.join(d, "extract_s0.75_s8x4_hist.npz")
    cfp.savez_atomic(pth, {"features": np.ones((2, 6))})
    assert not apo._targets_current(d, "s1.0", "flow", 0, 3, 2)


def test_prepare_properties_does_not_advance():
    rans = pytest.importorskip("rans_sst_py")
    mesh = rans.Mesh.make_plate_2d(8, 6, 0.02, 0.01, 1.0e4, 1.0)
    eos = rans.IdealGasEOS()
    solver = rans.DBNSSolver(mesh, eos, rans.SSTCoefficients(),
                             rans.DBNSBoundaryConditions(),
                             rans.DBNSSettings())
    n = solver.n_cells()
    rng = np.random.default_rng(3)
    prim = np.stack([
        np.full(n, 1.2) + 0.01 * rng.standard_normal(n),
        50.0 + 5.0 * rng.standard_normal(n),
        1.0 * rng.standard_normal(n),
        np.full(n, 1.0e5) + 100.0 * rng.standard_normal(n),
        np.abs(rng.standard_normal(n)) + 0.1,
        np.abs(rng.standard_normal(n)) * 10.0 + 50.0,
    ], axis=1)
    solver.init_field(prim)
    f0 = {k: np.array(solver.fields()[k])
          for k in ("rho", "u", "v", "p", "k", "omega")}
    solver.prepare_properties()
    f1 = solver.fields()
    # the extraction warm path: property preparation must leave the state
    # byte-identical (the one-iteration solve it replaced advanced it)
    for k in f0:
        assert np.array_equal(f0[k], np.asarray(f1[k])), k


def test_baseline_currency_decoupled_from_extractions(tmp_path):
    # the review's conflation finding: quarantined (absent) extractions
    # must never make a fully valid baseline look uncached (which would
    # trigger a multi-hour cold re-solve)
    from UQ.reproduce_sbli_apriori import _baseline_cached
    from UQ.datasets.sbli_apriori import sbli_ident
    d = str(tmp_path)
    fpath = os.path.join(d, "fields_s1.0.npz")
    cfp.savez_atomic(fpath, cfp.attach({"primitive": np.zeros((3, 6))},
                                       sbli_ident("sbli_fields", "s1.0")))
    wident = sbli_ident("sbli_wall", "s1.0", wall={
        "lineage": {"fields": cfp.file_sha(fpath)}})
    cfp.savez_atomic(os.path.join(d, "wall_s1.0.npz"),
                     cfp.attach({"Cf": np.zeros(4)}, wident))
    # no extraction files exist anywhere in d
    assert _baseline_cached(d, "s1.0")
    # a mutated fields cache still invalidates through the wall binding
    cfp.savez_atomic(fpath, cfp.attach({"primitive": np.ones((3, 6))},
                                       sbli_ident("sbli_fields", "s1.0")))
    assert not _baseline_cached(d, "s1.0")


def test_assemble_drops_stale_stages_and_records_lineage(tmp_path):
    from UQ.reproduce_sbli_apriori import (assemble_numbers, _partial_ident,
                                           numbers_ident,
                                           validate_apriori_numbers)
    d = str(tmp_path)
    seeds, epochs = (0,), 2
    # upstream extraction universe for the partial identities
    from UQ.datasets.sbli_apriori import TEST_STRIDE, _train_stride
    for c in sorted(TEST_STRIDE):
        for st in (TEST_STRIDE[c], _train_stride(c)):
            cfp.savez_atomic(
                os.path.join(d, f"extract_{c}_s{st[0]}x{st[1]}_hist.npz"),
                {"features": np.zeros((2, 6))})
    # one valid per-stage partial (insample); loso left with NO source
    cfp.json_atomic(os.path.join(d, "apriori_insample.json"),
                    cfp.attach_json({"insample": {"n": 1}},
                                    _partial_ident(d, "insample", None,
                                                   seeds, epochs)))
    numbers = {"loso": {"stale": "carried-from-an-earlier-era"},
               "config": {}}
    consumed = assemble_numbers(d, numbers, seeds, epochs, "")
    assert "loso" not in numbers          # stale block DROPPED
    assert numbers["insample"] == {"n": 1}
    assert "apriori_insample.json" in consumed
    # published numbers validate transitively through the recorded lineage
    npath = os.path.join(d, "apriori_numbers.json")
    cfp.json_atomic(npath, cfp.attach_json(
        numbers, numbers_ident(d, seeds, epochs, False, consumed)))
    ok, why = validate_apriori_numbers(d)
    assert ok, why
    # mutating a consumed partial breaks the published numbers' validation
    cfp.json_atomic(os.path.join(d, "apriori_insample.json"),
                    {"insample": {"n": 2}})
    ok, why = validate_apriori_numbers(d)
    assert not ok and "apriori_insample.json" in why


def test_dns_dataset_digest_and_manifest(tmp_path):
    r1 = str(tmp_path / "rootA")
    r2 = str(tmp_path / "rootB")
    for r, content in ((r1, b"favre-data-v1"), (r2, b"favre-data-v2")):
        os.makedirs(os.path.join(r, "shock_wave_BLI"))
        with open(os.path.join(r, "shock_wave_BLI", "block.dat"), "wb") as f:
            f.write(content)
    d1 = dm.dataset_digest("interaction_adiabatic", root=r1)
    d2 = dm.dataset_digest("interaction_adiabatic", root=r2)
    assert d1 and d2 and d1 != d2          # content drives the digest
    assert dm.dataset_digest("interaction_adiabatic", root=r1) == d1
    assert dm.dataset_digest("interaction_heated", root=r1) is None
    # manifest roundtrip and change detection over a narrowed required set
    # (only a hermetic test narrows it; a driver always rules the full
    # registered set, see test_manifest_verification_is_closed)
    mpath = str(tmp_path / "dns_manifest.json")
    one = ("interaction_adiabatic",)
    dm.write_manifest(mpath, names=one, root=r1)
    ok, why = dm.verify_manifest(mpath, root=r1, required=one)
    assert ok, why
    ok, why = dm.verify_manifest(mpath, root=r2, required=one)
    assert not ok and "interaction_adiabatic" in why


def test_manifest_verification_is_closed_over_the_registered_set(tmp_path):
    # the review's fail-open finding, reproduced as a regression: verifying
    # only the entries a record happens to carry passed a manifest written
    # against absent data and a record reduced to an empty dataset block
    root = _synthetic_dns_root(str(tmp_path / "root"))
    mpath = str(tmp_path / "dns_manifest.json")
    dm.write_manifest(mpath, root=root)
    ok, why = dm.verify_manifest(mpath, root=root)
    assert ok, why

    # (a) every dataset absent: the digests record None and can never verify
    empty_root = str(tmp_path / "absent")
    os.makedirs(empty_root)
    npath = str(tmp_path / "none_manifest.json")
    dm.write_manifest(npath, root=empty_root)
    ok, why = dm.verify_manifest(npath, root=empty_root)
    assert not ok and "no digest" in why

    # (b) the record hand-reduced to an empty dataset block: its identity no
    # longer matches the entries it carries
    rec = json.load(open(mpath))
    rec["datasets"] = {}
    rpath = str(tmp_path / "reduced.json")
    cfp.json_atomic(rpath, rec)
    ok, why = dm.verify_manifest(rpath, root=root)
    assert not ok and "identity" in why

    # (c) an entry dropped AND the identity block reforged to match: only
    # the closed-set rule catches this one
    rec = json.load(open(mpath))
    rec["datasets"].pop("zdc_plate")
    fpath = str(tmp_path / "forged.json")
    cfp.json_atomic(fpath, cfp.attach_json(
        {"datasets": rec["datasets"]}, dm.manifest_ident(rec["datasets"])))
    ok, why = dm.verify_manifest(fpath, root=root)
    assert not ok and "registered datasets" in why and "zdc_plate" in why

    # (d) a dataset directory removed after manifesting
    shutil.rmtree(os.path.join(root, dm.DATASETS["zdc_plate"]))
    ok, why = dm.verify_manifest(mpath, root=root)
    assert not ok and "zdc_plate" in why


def test_manifest_verification_recomputes_and_requires_content(tmp_path):
    root = _synthetic_dns_root(str(tmp_path / "root"))
    mpath = str(tmp_path / "dns_manifest.json")
    dm.write_manifest(mpath, root=root)          # memoizes every digest
    # same-root mutation: the ruling must be a FRESH pass, not the digest
    # memoized while writing the record moments earlier
    with open(_dataset_file(root, "interaction_heated"), "wb") as f:
        f.write(b"edited in place, different content and length")
    ok, why = dm.verify_manifest(mpath, root=root)
    assert not ok and "interaction_heated changed" in why
    # a dataset emptied of files is refused even though its directory (and
    # therefore a well-formed digest) still exists
    dm.write_manifest(mpath, root=root)
    os.remove(_dataset_file(root, "interaction_heated"))
    ok, why = dm.verify_manifest(mpath, root=root)
    assert not ok and "holds no files" in why


def test_extraction_identity_binds_dns_digest(tmp_path, monkeypatch):
    d = str(tmp_path / "results")
    os.makedirs(d)
    for tag, content in (("dataA", b"heated-v1"), ("dataB", b"heated-v2")):
        r = str(tmp_path / tag)
        os.makedirs(os.path.join(r, "heat_transfer_SBLI"))
        with open(os.path.join(r, "heat_transfer_SBLI", "f.dat"), "wb") as f:
            f.write(content)
    cfp.savez_atomic(os.path.join(d, "fields_s1.0.npz"),
                     {"primitive": np.zeros((3, 6))})
    monkeypatch.setenv("QBTM_DNS_DATA", str(tmp_path / "dataA"))
    identA = extraction_ident("s1.0", (4, 4), True, d)
    monkeypatch.setenv("QBTM_DNS_DATA", str(tmp_path / "dataB"))
    identB = extraction_ident("s1.0", (4, 4), True, d)
    # replacing the source dataset invalidates the extraction identity even
    # with an unchanged baseline fields cache (the raw-input lineage edge)
    assert cfp.fingerprint(identA) != cfp.fingerprint(identB)


def _complete_numbers(seeds):
    """The smallest record satisfying every recomputed completeness clause:
    each leg carries every held fold, each fold every scored model, each
    model one row per pinned seed."""
    from UQ.reproduce_sbli_apriori import (LOSO_LEGS, INSAMPLE_LEGS,
                                           FAR_TRANSFER_LEGS,
                                           SCORED_MODELS, HELD_FOLDS)

    def rows():
        return {k: [{"coverage_0.9": 0.9} for _ in seeds]
                for k in SCORED_MODELS}

    def folds():
        return {f: {"models": rows()} for f in HELD_FOLDS}

    # the attached health gate runs over the real channel matrix, so the
    # fixture carries real case tags: the roles declare the expected set and
    # the control must cover every one of them per family
    tags = list(GV_CASES[:4])
    control = {}
    for t in tags:
        control.setdefault(f"family_{mach_family(t)}", {})[t] = rows()
    return {
        "gates": {"A": {"pass": True}, "B": {"s1.0": {"pass": True}}},
        "loso": {leg: folds() for leg in LOSO_LEGS},
        "insample": {leg: {"models": rows()} for leg in INSAMPLE_LEGS},
        "far": {"transfer": {leg: folds() for leg in FAR_TRANSFER_LEGS},
                "control": {"dq_y": control},
                "conformal": {"roles": {"disjoint": True,
                                        "fit_cases": tags[0::2],
                                        "calibration_cases": tags[1::2]},
                              "per_seed": {str(s): {"q_abs": 1.0,
                                                    "cases": {"s1.0": {}}}
                                           for s in seeds},
                              "seed_mean": {"s1.0": {}},
                              "seed_min_max": {"s1.0": {}}}},
        "complete": True, "missing": [],
    }


def _adjudication_universe(d):
    """A current gate adjudication with identity lineage; returns the
    numbers lineage that binds it."""
    from UQ.reproduce_sbli_apriori import _gate_ident
    cfp.savez_atomic(os.path.join(d, "fields_s1.0.npz"),
                     {"primitive": np.zeros((2, 6))})
    gpath = os.path.join(d, "gate_b_s1.0.json")
    cfp.json_atomic(gpath, cfp.attach_json({"pass": True},
                                           _gate_ident(d, "s1.0")))
    lin = {"gate_b_s1.0.json": cfp.file_sha(gpath)}
    apath = os.path.join(d, "gates_adjudication.json")
    cfp.json_atomic(apath, cfp.attach_json(
        {"gate_a_pass": True}, sbli_ident("gates-adjudication", "all-cases",
                                          adjud={"lineage": lin})))
    return {"gates_adjudication.json": cfp.file_sha(apath)}


def test_strict_numbers_validation(tmp_path):
    from UQ.reproduce_sbli_apriori import (numbers_ident,
                                           validate_apriori_numbers)
    d = str(tmp_path)
    consumed = _adjudication_universe(d)
    full = _complete_numbers((0,))
    npath = os.path.join(d, "apriori_numbers.json")

    def _publish(rec, lineage=None, seeds=(0,), epochs=2, quick=False):
        cfp.json_atomic(npath, cfp.attach_json(
            rec, numbers_ident(d, seeds, epochs, quick,
                               consumed if lineage is None else lineage)))

    # importing the solver binding registers the binding provenance
    # process-wide; clear it to exercise the no-provenance refusal, then
    # restore the real value at the end
    saved_binding = cfp._BINDING_SHA
    cfp.set_binding_provenance(None)
    _publish(full)
    # non-strict passes on lineage alone; strict needs binding provenance
    ok, _ = validate_apriori_numbers(d)
    assert ok
    ok, why = validate_apriori_numbers(d, strict=True, expected_seeds=(0,),
                                       expected_epochs=2)
    assert not ok and "binding" in why
    cfp.set_binding_provenance("abc123")
    _publish(full)
    ok, why = validate_apriori_numbers(d, strict=True, expected_seeds=(0,),
                                       expected_epochs=2)
    assert ok, why
    # numbers produced by a DIFFERENT build refuse: a published number is
    # claim-bearing only against the binary that produced it
    cfp.set_binding_provenance("def456")
    ok, why = validate_apriori_numbers(d, strict=True, expected_seeds=(0,),
                                       expected_epochs=2)
    assert not ok and "abc123" in why and "def456" in why
    os.environ["QBTM_SBLI_ACCEPT_BINDING"] = "1"
    ok, why = validate_apriori_numbers(d, strict=True, expected_seeds=(0,),
                                       expected_epochs=2)
    assert ok, why
    os.environ.pop("QBTM_SBLI_ACCEPT_BINDING")
    cfp.set_binding_provenance("abc123")
    # wrong protocol settings refuse
    ok, why = validate_apriori_numbers(d, strict=True,
                                       expected_seeds=(0, 1, 2),
                                       expected_epochs=2)
    assert not ok and "seeds" in why
    # a record whose identity was built for the quick universe refuses even
    # though every field-by-field check passes (the fingerprint carries the
    # physics token and the quick flag)
    _publish(full, quick=True)
    ok, why = validate_apriori_numbers(d, strict=True, expected_seeds=(0,),
                                       expected_epochs=2)
    assert not ok and "protocol configuration" in why
    # numbers that do not name the authorizing gate adjudication refuse
    _publish(full, lineage={})
    ok, why = validate_apriori_numbers(d, strict=True, expected_seeds=(0,),
                                       expected_epochs=2)
    assert not ok and "gate adjudication" in why
    # an incomplete result refuses under strict
    partial = dict(full)
    partial["far"] = {}
    partial["complete"] = False
    partial["missing"] = ["far"]
    _publish(partial)
    ok, why = validate_apriori_numbers(d, strict=True, expected_seeds=(0,),
                                       expected_epochs=2)
    assert not ok and "incomplete" in why
    cfp.set_binding_provenance(saved_binding)


def test_numbers_completeness_is_recomputed_not_labeled(tmp_path):
    # the review's finding: presence-only completeness declared a record
    # whose every leg was an empty dictionary complete and valid
    from UQ.reproduce_sbli_apriori import (numbers_ident, numbers_missing,
                                           validate_apriori_numbers,
                                           LOSO_LEGS, INSAMPLE_LEGS)
    d = str(tmp_path)
    consumed = _adjudication_universe(d)
    hollow = {"gates": {"A": {}, "B": {}},
              "loso": {k: {} for k in LOSO_LEGS},
              "insample": {k: {} for k in INSAMPLE_LEGS},
              "far": {k: {} for k in ("transfer", "control", "conformal")},
              "complete": True, "missing": []}
    missing = numbers_missing(hollow, (0,))
    assert "gates" in missing and "loso:dq_y" in missing
    assert "far:transfer:dq_y" in missing
    saved_binding = cfp._BINDING_SHA
    cfp.set_binding_provenance("abc123")
    cfp.json_atomic(os.path.join(d, "apriori_numbers.json"),
                    cfp.attach_json(hollow, numbers_ident(d, (0,), 2, False,
                                                          consumed)))
    ok, why = validate_apriori_numbers(d, strict=True, expected_seeds=(0,),
                                       expected_epochs=2)
    assert not ok and "incomplete" in why
    # a short seed sweep is caught at the model rows, not just at the legs
    short = _complete_numbers((0,))
    assert not numbers_missing(short, (0,))
    assert any("seeds" in m for m in numbers_missing(short, (0, 1, 2)))
    cfp.set_binding_provenance(saved_binding)


def test_completeness_reaches_the_attached_control_and_conformal_seeds():
    # the attached leave-one-Mach-family-out health gate passed on a
    # non-empty dictionary, so completeness did not in fact reach every
    # per-seed model row; the conformal per-seed keys were presence-only
    from UQ.reproduce_sbli_apriori import numbers_missing
    seeds = (0, 1)
    rec = _complete_numbers(seeds)
    assert not numbers_missing(rec, seeds)
    # a control block that merely exists is now refused
    rec["far"]["control"]["dq_y"] = {"family_2": {}}
    assert any(m.startswith("far:control:dq_y") for m in
               numbers_missing(rec, seeds))
    # a case the roles declare but the control never scored
    rec = _complete_numbers(seeds)
    fam = sorted(rec["far"]["control"]["dq_y"])[0]
    dropped = sorted(rec["far"]["control"]["dq_y"][fam])[0]
    del rec["far"]["control"]["dq_y"][fam][dropped]
    assert f"far:control:dq_y:{fam}:{dropped}" in numbers_missing(rec, seeds)
    # a scored case missing a model, and one short a seed
    rec = _complete_numbers(seeds)
    fam = sorted(rec["far"]["control"]["dq_y"])[0]
    tag = sorted(rec["far"]["control"]["dq_y"][fam])[0]
    del rec["far"]["control"]["dq_y"][fam][tag]["pooled"]
    assert f"far:control:dq_y:{fam}:{tag}:pooled" in numbers_missing(rec,
                                                                     seeds)
    rec = _complete_numbers(seeds)
    rec["far"]["control"]["dq_y"][fam][tag]["flow"] = [{"coverage_0.9": 0.9}]
    assert any("seeds" in m and "far:control" in m
               for m in numbers_missing(rec, seeds))
    # an empty conformal per-seed entry no longer counts as present
    rec = _complete_numbers(seeds)
    rec["far"]["conformal"]["per_seed"]["1"] = {"q_abs": 1.0}
    assert "far:conformal:per_seed:1" in numbers_missing(rec, seeds)


def test_baseline_dns_adoption_gates_reuse(tmp_path, monkeypatch):
    # the review's mixed-era finding: accepting changed source data rewrote
    # the manifest and invalidated the extractions, but left the baselines
    # those extractions are measured against reusable
    from UQ.reproduce_sbli_apriori import (_adoption_status, _audit_adoption,
                                           _fields_path, _baseline_campaign,
                                           stage_adopt_baselines)
    root = _synthetic_dns_root(str(tmp_path / "root"))
    monkeypatch.setenv("QBTM_DNS_DATA", root)
    dm.forget(tuple(dm.DATASETS))
    d = str(tmp_path / "results")
    os.makedirs(d)
    for tag in ("s1.0", "gate_a_attached"):
        cfp.savez_atomic(_fields_path(d, tag),
                         cfp.attach({"primitive": np.zeros((3, 6))},
                                    sbli_ident("sbli_fields", tag)))
    # the migration allowlist of this hermetic universe (in production it is
    # the tracked audited record, which no test hash could ever match)
    allow = {t: {"campaign": _baseline_campaign(t),
                 "dns": dm.dataset_digest(_baseline_campaign(t)),
                 "fields": cfp.file_sha(_fields_path(d, t))}
             for t in ("s1.0", "gate_a_attached")}
    # an identity-current baseline with no adoption record is NOT current
    assert _adoption_status(d, "s1.0")[0] == "absent"
    with pytest.raises(SystemExit):
        _audit_adoption(d, ("s1.0",), False)
    stage_adopt_baselines(d, allowlist=allow)
    assert _adoption_status(d, "s1.0")[0] == "current"
    assert _adoption_status(d, "gate_a_attached")[0] == "current"
    # a heated-campaign correction strands the heated baseline; rewriting
    # the manifest cannot authorize its reuse, only a cold regeneration can
    with open(_dataset_file(root, "interaction_heated"), "wb") as f:
        f.write(b"corrected campaign")
    dm.forget(tuple(dm.DATASETS))
    status, why = _adoption_status(d, "s1.0")
    assert status == "stale" and "interaction_heated" in why
    with pytest.raises(SystemExit):
        _audit_adoption(d, ("s1.0",), False)
    # and the adiabatic-campaign baselines are untouched by that change
    # (per-case campaigns, so a correction never over-invalidates)
    assert _adoption_status(d, "gate_a_attached")[0] == "current"
    # the record is bound to the exact fields cache as well
    cfp.savez_atomic(_fields_path(d, "gate_a_attached"),
                     cfp.attach({"primitive": np.ones((3, 6))},
                                sbli_ident("sbli_fields", "gate_a_attached")))
    status, why = _adoption_status(d, "gate_a_attached")
    assert status == "stale" and "fields cache changed" in why


def test_stale_adoption_is_never_restored_by_re_adopting(tmp_path,
                                                         monkeypatch):
    # the migration stage rewrote ANY non-current record, so
    # adopt -> change DNS -> adopt returned the baseline to current with no
    # solve, contradicting the rule that only cold regeneration restores it
    from UQ.reproduce_sbli_apriori import (_adoption_status, _fields_path,
                                           _adoption_path, _baseline_campaign,
                                           stage_adopt_baselines)
    root = _synthetic_dns_root(str(tmp_path / "root"))
    monkeypatch.setenv("QBTM_DNS_DATA", root)
    dm.forget(tuple(dm.DATASETS))
    d = str(tmp_path / "results")
    os.makedirs(d)
    cfp.savez_atomic(_fields_path(d, "s1.0"),
                     cfp.attach({"primitive": np.zeros((3, 6))},
                                sbli_ident("sbli_fields", "s1.0")))
    allow = {"s1.0": {"campaign": _baseline_campaign("s1.0"),
                      "dns": dm.dataset_digest("interaction_heated"),
                      "fields": cfp.file_sha(_fields_path(d, "s1.0"))}}
    stage_adopt_baselines(d, allowlist=allow)
    assert _adoption_status(d, "s1.0")[0] == "current"
    with open(_dataset_file(root, "interaction_heated"), "wb") as f:
        f.write(b"corrected campaign")
    dm.forget(tuple(dm.DATASETS))
    assert _adoption_status(d, "s1.0")[0] == "stale"
    # re-adopting REFUSES and leaves the record stale
    with pytest.raises(SystemExit):
        stage_adopt_baselines(d, allowlist=allow)
    assert _adoption_status(d, "s1.0")[0] == "stale"
    # nor does deleting the stale sidecar reopen the route: the live pair no
    # longer matches the audited migration record
    os.remove(_adoption_path(d, "s1.0"))
    assert _adoption_status(d, "s1.0")[0] == "absent"
    with pytest.raises(SystemExit):
        stage_adopt_baselines(d, allowlist=allow)
    assert _adoption_status(d, "s1.0")[0] == "absent"
    # and a tag with no audited entry at all is refused rather than adopted
    with pytest.raises(SystemExit):
        stage_adopt_baselines(d, allowlist={})
    assert _adoption_status(d, "s1.0")[0] == "absent"


def test_adoption_record_is_ruled_closed_over_its_fields(tmp_path,
                                                         monkeypatch):
    # the same fail-open shape as the manifest, one level down: a record
    # with its dns entry removed and its identity rebuilt over the reduced
    # body was self-consistent, and the digest loop then iterated nothing
    from UQ.reproduce_sbli_apriori import (_adoption_status, _fields_path,
                                           _adoption_path, _write_adoption)
    root = _synthetic_dns_root(str(tmp_path / "root"))
    monkeypatch.setenv("QBTM_DNS_DATA", root)
    dm.forget(tuple(dm.DATASETS))
    d = str(tmp_path / "results")
    os.makedirs(d)
    cfp.savez_atomic(_fields_path(d, "s1.0"),
                     cfp.attach({"primitive": np.zeros((3, 6))},
                                sbli_ident("sbli_fields", "s1.0")))
    _write_adoption(d, "s1.0")
    assert _adoption_status(d, "s1.0")[0] == "current"
    path = _adoption_path(d, "s1.0")
    good = json.load(open(path))

    def _reforge(body):
        cfp.json_atomic(path, cfp.attach_json(
            body, sbli_ident("baseline-dns", "s1.0", adopt=body)))

    # (a) the dns entry removed and the identity reforged over what is left
    body = {k: good[k] for k in ("tag", "campaign", "fields")}
    _reforge(body)
    status, why = _adoption_status(d, "s1.0")
    assert status == "stale" and "dns" in why
    # (b) an empty dns map, self-consistent
    body = {k: good[k] for k in ("tag", "campaign", "dns", "fields")}
    body["dns"] = {}
    _reforge(body)
    assert _adoption_status(d, "s1.0")[0] == "stale"
    # (c) a null digest (adopted while the campaign was absent)
    body["dns"] = {good["campaign"]: None}
    _reforge(body)
    status, why = _adoption_status(d, "s1.0")
    assert status == "stale" and "no" in why
    # (d) a digest recorded under the WRONG campaign
    body["dns"] = {"gv_channel": dm.dataset_digest("gv_channel")}
    _reforge(body)
    assert _adoption_status(d, "s1.0")[0] == "stale"
    # (e) the record renamed to another baseline
    body = {k: good[k] for k in ("tag", "campaign", "dns", "fields")}
    body["tag"] = "s1.9"
    _reforge(body)
    assert _adoption_status(d, "s1.0")[0] == "stale"


def test_run_token_is_bound_to_the_verified_manifest(tmp_path):
    # the worker-side cheap gate: a parent that fresh-verified writes the
    # token, workers re-check it instead of re-hashing every dataset, and it
    # stops authorizing the moment the manifest it names moves
    root = _synthetic_dns_root(str(tmp_path / "root"))
    mpath = str(tmp_path / "dns_manifest.json")
    tpath = str(tmp_path / "token.json")
    dm.write_manifest(mpath, root=root)
    assert dm.verify_manifest(mpath, root=root)[0]
    dm.write_run_token(tpath, mpath)
    ok, why = dm.verify_run_token(tpath, mpath)
    assert ok, why
    # a re-manifested (even identical) record is a different file: the token
    # names the exact manifest its parent verified
    with open(_dataset_file(root, "interaction_heated"), "wb") as f:
        f.write(b"corrected campaign")
    dm.write_manifest(mpath, root=root)
    ok, why = dm.verify_run_token(tpath, mpath)
    assert not ok and "manifest changed" in why
    # a token whose digests are edited and identity reforged still refuses:
    # it must agree with the manifest record digest for digest
    dm.write_run_token(tpath, mpath)
    tok = json.load(open(tpath))
    body = {k: tok[k] for k in dm.TOKEN_FIELDS}
    body["digests"] = dict(body["digests"])
    body["digests"]["interaction_heated"] = "0000000000000000"
    cfp.json_atomic(tpath, cfp.attach_json(body, dm.token_ident(body)))
    ok, why = dm.verify_run_token(tpath, mpath)
    assert not ok and "digests" in why
    # and a narrowed token cannot claim coverage it does not have
    dm.write_run_token(tpath, mpath)
    tok = json.load(open(tpath))
    body = {k: tok[k] for k in dm.TOKEN_FIELDS}
    body["digests"] = {"interaction_heated":
                       body["digests"]["interaction_heated"]}
    cfp.json_atomic(tpath, cfp.attach_json(body, dm.token_ident(body)))
    assert not dm.verify_run_token(tpath, mpath)[0]
    # a missing token never passes
    os.remove(tpath)
    ok, why = dm.verify_run_token(tpath, mpath)
    assert not ok and "absent" in why


def test_faradiab_targets_bind_the_attached_training_pool(tmp_path,
                                                          monkeypatch):
    # the exploratory far-transfer targets TRAIN on the attached channel
    # matrix, a direct DNS consumer with no extraction intermediary
    d = str(tmp_path / "results")
    os.makedirs(d)
    for tag, content in (("dataA", b"gv-v1"), ("dataB", b"gv-v2")):
        r = _synthetic_dns_root(str(tmp_path / tag))
        with open(_dataset_file(r, "gv_channel"), "wb") as f:
            f.write(content)
    for tag in ("adiabatic", "s1.0"):
        cfp.savez_atomic(os.path.join(d, f"fields_{tag}.npz"),
                         {"primitive": np.zeros((3, 6))})

    def _fp(fold):
        return cfp.fingerprint(apo._targets_config(
            d, fold, "flow", 0, 3, 2, "objective-basis"))

    monkeypatch.setenv("QBTM_DNS_DATA", str(tmp_path / "dataA"))
    far_a, fold_a = _fp("faradiab"), _fp("s1.0")
    monkeypatch.setenv("QBTM_DNS_DATA", str(tmp_path / "dataB"))
    far_b, fold_b = _fp("faradiab"), _fp("s1.0")
    assert far_a != far_b      # replacing the training pool invalidates them
    assert fold_a == fold_b    # and only the leg that actually reads it


def test_atomic_writers_use_per_pid_temps(tmp_path):
    d = str(tmp_path)
    p = os.path.join(d, "x.npz")
    cfp.savez_atomic(p, {"a": np.arange(3)})
    q = os.path.join(d, "y.json")
    cfp.json_atomic(q, {"a": 1})
    # no shared-temp residue survives, and the temp naming is per-writer
    assert os.listdir(d) == sorted(["x.npz", "y.json"]) or \
        sorted(os.listdir(d)) == ["x.npz", "y.json"]
    assert str(os.getpid()) not in "".join(os.listdir(d))


def _concurrent_writer(npz_path, json_path, tag, n, barrier):
    """One writer process of the atomic-write race: a payload identifying
    this writer, written repeatedly to the SAME paths as the other."""
    arr = np.full(8192, float(tag))
    barrier.wait()
    for _ in range(n):
        cfp.savez_atomic(npz_path, {"a": arr})
        cfp.json_atomic(json_path, {"tag": tag, "pad": [tag] * 4096})


def test_atomic_writers_survive_concurrent_processes(tmp_path):
    # the sequential check above would also pass the shared-temp writer it
    # replaced; the race is what the per-writer temp name exists for, so two
    # barrier-synchronized processes publish to one path and the survivor
    # must be ONE writer's complete payload (a shared temp let one writer
    # truncate or publish another's half-written bytes)
    ctx = multiprocessing.get_context("fork")   # children only write arrays
    d = str(tmp_path)
    p, q = os.path.join(d, "x.npz"), os.path.join(d, "y.json")
    barrier = ctx.Barrier(2)
    procs = [ctx.Process(target=_concurrent_writer,
                         args=(p, q, tag, 40, barrier)) for tag in (1, 2)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join(120)
    assert [pr.exitcode for pr in procs] == [0, 0]
    a = np.load(p)["a"]
    assert a.shape == (8192,) and len(set(a.tolist())) == 1
    assert a[0] in (1.0, 2.0)
    rec = json.load(open(q))
    assert rec["pad"] == [rec["tag"]] * 4096
    # and no temp residue survives either writer
    assert sorted(os.listdir(d)) == ["x.npz", "y.json"]
