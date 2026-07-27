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
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "python"))

from UQ import cache_fingerprint as cfp
from UQ.datasets.sbli_apriori import (extraction_ident,
                                      extraction_fields_path,
                                      conformal_case_split)
from UQ.datasets.heatflux_apriori import mach_family
from UQ.datasets.gv_channel import GV_CASES
from UQ import reproduce_sbli_aposteriori as apo


def _write_npz(path, cfg, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfp.savez_atomic(path, cfp.attach(arrays, cfg))


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
    from UQ.datasets import dns_manifest as dm
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
    # manifest roundtrip and change detection
    mpath = str(tmp_path / "dns_manifest.json")
    dm.write_manifest(mpath, names=("interaction_adiabatic",), root=r1)
    ok, _ = dm.verify_manifest(mpath, root=r1)
    assert ok
    ok, why = dm.verify_manifest(mpath, root=r2)
    assert not ok and "interaction_adiabatic" in why


def test_extraction_identity_binds_dns_digest(tmp_path, monkeypatch):
    from UQ.datasets import dns_manifest as dm
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


def test_strict_numbers_validation(tmp_path):
    from UQ.reproduce_sbli_apriori import (numbers_ident,
                                           validate_apriori_numbers,
                                           _gate_ident)
    from UQ.datasets.sbli_apriori import sbli_ident
    d = str(tmp_path)
    # a current gate adjudication with identity lineage
    cfp.savez_atomic(os.path.join(d, "fields_s1.0.npz"),
                     {"primitive": np.zeros((2, 6))})
    gpath = os.path.join(d, "gate_b_s1.0.json")
    cfp.json_atomic(gpath, cfp.attach_json({"pass": True},
                                           _gate_ident(d, "s1.0")))
    lin = {"gate_b_s1.0.json": cfp.file_sha(gpath)}
    cfp.json_atomic(os.path.join(d, "gates_adjudication.json"),
                    cfp.attach_json({"gate_a_pass": True},
                                    sbli_ident("gates-adjudication",
                                               "all-cases",
                                               adjud={"lineage": lin})))
    full = {"gates": {}, "complete": True, "missing": [],
            "loso": {k: {} for k in ("dq_y", "dq_joint", "db",
                                     "dq_y_history")},
            "insample": {k: {} for k in ("dq_y", "dq_joint", "db")},
            "far": {k: {} for k in ("transfer", "control", "conformal")}}
    # importing the solver binding registers the binding provenance
    # process-wide; clear it to exercise the no-provenance refusal, then
    # restore the real value at the end
    saved_binding = cfp._BINDING_SHA
    cfp.set_binding_provenance(None)
    cfp.json_atomic(os.path.join(d, "apriori_numbers.json"),
                    cfp.attach_json(full, numbers_ident(d, (0,), 2, False,
                                                        {})))
    # non-strict passes on lineage alone; strict needs binding provenance
    ok, _ = validate_apriori_numbers(d)
    assert ok
    ok, why = validate_apriori_numbers(d, strict=True, expected_seeds=(0,),
                                       expected_epochs=2)
    assert not ok and "binding" in why
    cfp.set_binding_provenance("abc123")
    cfp.json_atomic(os.path.join(d, "apriori_numbers.json"),
                    cfp.attach_json(full, numbers_ident(d, (0,), 2, False,
                                                        {})))
    ok, why = validate_apriori_numbers(d, strict=True, expected_seeds=(0,),
                                       expected_epochs=2)
    assert ok, why
    # wrong protocol settings refuse
    ok, why = validate_apriori_numbers(d, strict=True,
                                       expected_seeds=(0, 1, 2),
                                       expected_epochs=2)
    assert not ok and "seeds" in why
    # an incomplete result refuses under strict
    partial = dict(full)
    partial["complete"] = False
    partial["missing"] = ["far"]
    cfp.json_atomic(os.path.join(d, "apriori_numbers.json"),
                    cfp.attach_json(partial, numbers_ident(d, (0,), 2,
                                                           False, {})))
    ok, why = validate_apriori_numbers(d, strict=True, expected_seeds=(0,),
                                       expected_epochs=2)
    assert not ok and "incomplete" in why
    cfp.set_binding_provenance(saved_binding)


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
