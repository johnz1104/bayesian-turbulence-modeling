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
