"""Cache identity: fingerprints reuse-if-identical and refuse-if-changed.

Pins the audit fix: ensemble caches carry the scientifically relevant
configuration inside the file, loaders classify match / mismatch / legacy,
and a full run can no longer silently reuse a quick-mode cache that shares
its filename. (The archived scaling-study harness keeps its own documented
blind-resume contract; the production calibration caches are governed by
this module.)
"""
import numpy as np

from UQ import cache_fingerprint as cfp


def _cfg(n_ensemble=48):
    return {"kind": "channel_ensemble", "case": 5200, "n_ensemble": n_ensemble,
            "seed": 0, "cfg": {"nx": 40, "ny": 56, "conv_tol": 1.0e-3}}


def test_fingerprint_stable_and_numpy_normalised():
    a = cfp.fingerprint(_cfg())
    b = cfp.fingerprint({"kind": "channel_ensemble", "case": np.int64(5200),
                         "n_ensemble": np.int32(48), "seed": 0,
                         "cfg": {"nx": 40, "ny": 56,
                                 "conv_tol": np.float64(1.0e-3)}})
    assert a == b, "numpy scalar types must not change the fingerprint"
    assert a != cfp.fingerprint(_cfg(n_ensemble=12))


def test_roundtrip_match_through_npz(tmp_path):
    path = tmp_path / "ensemble.npz"
    arrays = {"X": np.arange(6.0).reshape(3, 2), "loglik": np.zeros(3)}
    np.savez(path, **cfp.attach(arrays, _cfg()))
    d = dict(np.load(path))
    status, reason = cfp.check(d, _cfg())
    assert status == "match", reason
    assert np.array_equal(d["X"], arrays["X"])


def test_quick_cache_is_refused_by_full_config(tmp_path):
    path = tmp_path / "ensemble.npz"
    np.savez(path, **cfp.attach({"X": np.zeros((12, 2))}, _cfg(n_ensemble=12)))
    d = dict(np.load(path))
    status, reason = cfp.check(d, _cfg(n_ensemble=48))
    assert status == "mismatch"
    assert "n_ensemble" in reason and "12" in reason and "48" in reason


def test_pre_fingerprint_cache_classifies_legacy(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, X=np.zeros((4, 2)), loglik=np.zeros(4))
    d = dict(np.load(path))
    status, _ = cfp.check(d, _cfg())
    assert status == "legacy"


def test_legacy_reuse_is_env_gated(monkeypatch):
    # default: refuse; explicit opt-in: allow
    monkeypatch.delenv("QBTM_ALLOW_LEGACY_CACHE", raising=False)
    assert not cfp.legacy_reuse_allowed()
    monkeypatch.setenv("QBTM_ALLOW_LEGACY_CACHE", "1")
    assert cfp.legacy_reuse_allowed()
    monkeypatch.setenv("QBTM_ALLOW_LEGACY_CACHE", "0")
    assert not cfp.legacy_reuse_allowed()


def test_attach_records_code_revision():
    stamped = cfp.attach({"x": np.arange(3)}, {"kind": "t"})
    assert cfp.CODE_REV_KEY in stamped
    rev = str(np.asarray(stamped[cfp.CODE_REV_KEY])[()])
    assert rev  # short hash in a repo, "unknown" outside one
    # a code-revision drift on a config match stays a match (provenance only)
    stamped[cfp.CODE_REV_KEY] = np.array("0000000")
    status, reason = cfp.check(stamped, {"kind": "t"})
    assert status == "match"
    assert "0000000" in reason
