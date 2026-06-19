"""Tests for forward_model_interface.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from forward_model_interface import (
    EvaluationResult,
    ForwardModelBase,
    InternalIncompressibleForwardModel,
    InternalCompressibleForwardModel,
    PrecomputedEnsembleForwardModel,
    ExternalSolverForwardModel,
    SurrogateForwardModel,
    save_ensemble_npz,
    load_forward_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_xy(n=30, d=2, n_obs=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.1, 0.5, size=(n, d))
    Y = np.stack([
        0.003 + 0.001 * X[:, 0],
        0.004 * X[:, 1],
        50000 + 1000 * X[:, 0],
        0.8 + 0.1 * X[:, 1],
    ], axis=1)[:, :n_obs]
    return X, Y


class _MockCppResult:
    """Mimics rans_sst_py ForwardModel evaluate() result."""
    def __init__(self, preds, status="converged"):
        self.predictions = preds
        self.status = status


class _MockCppForwardModel:
    def __init__(self, n_obs=4, fail=False):
        self._n_obs = n_obs
        self._fail  = fail

    def evaluate(self, theta):
        if self._fail:
            return _MockCppResult([], status="diverged")
        preds = [float(t) * 0.01 for t in theta[:self._n_obs]]
        return _MockCppResult(preds)


class _MockParamSet:
    def active_names(self):
        return ["a1", "betaStar"]


class _MockSurrogate:
    """Minimal stand-in for MultiOutputSurrogate."""
    def __init__(self, n_obs=4, d=2):
        self.trained = True
        self.n_outputs = n_obs
        self.gps = [type("GP", (), {"X": np.zeros((10, d))})()]

    def predict(self, theta):
        mu  = np.array([0.003, 0.003, 50000.0, 0.8])[:self.n_outputs]
        var = np.zeros(self.n_outputs)
        return mu, var


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------

class TestEvaluationResult:
    def test_truthy_when_converged_with_preds(self):
        r = EvaluationResult(predictions=[0.003, 0.004], converged=True)
        assert bool(r)

    def test_falsy_when_empty_preds(self):
        r = EvaluationResult(predictions=[], converged=True)
        assert not bool(r)

    def test_falsy_when_not_converged(self):
        r = EvaluationResult(predictions=[0.003], converged=False)
        assert not bool(r)

    def test_default_status(self):
        r = EvaluationResult([1.0, 2.0])
        assert r.status == "ok"
        assert r.converged is True

    def test_metadata_default_empty(self):
        r = EvaluationResult([1.0])
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# ForwardModelBase (via concrete subclass check)
# ---------------------------------------------------------------------------

class TestForwardModelBase:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ForwardModelBase()

    def test_precomputed_ensemble_returns_none_by_default(self):
        fm = InternalIncompressibleForwardModel(
            _MockCppForwardModel(), _MockParamSet()
        )
        assert fm.precomputed_ensemble() is None


# ---------------------------------------------------------------------------
# InternalIncompressibleForwardModel
# ---------------------------------------------------------------------------

class TestInternalIncompressibleForwardModel:
    def test_evaluate_returns_evaluation_result(self):
        fm = InternalIncompressibleForwardModel(
            _MockCppForwardModel(n_obs=2), _MockParamSet()
        )
        r = fm.evaluate([0.31, 0.09])
        assert isinstance(r, EvaluationResult)
        assert len(r.predictions) == 2
        assert r.converged

    def test_evaluate_failed_cpp(self):
        fm = InternalIncompressibleForwardModel(
            _MockCppForwardModel(fail=True), _MockParamSet()
        )
        r = fm.evaluate([0.31, 0.09])
        assert not r.converged
        assert r.predictions == []

    def test_parameter_names_from_param_set(self):
        fm = InternalIncompressibleForwardModel(
            _MockCppForwardModel(), _MockParamSet()
        )
        assert fm.parameter_names() == ["a1", "betaStar"]

    def test_status_propagated(self):
        fm = InternalIncompressibleForwardModel(
            _MockCppForwardModel(), _MockParamSet()
        )
        r = fm.evaluate([0.31, 0.09])
        assert r.status == "converged"


# ---------------------------------------------------------------------------
# InternalCompressibleForwardModel
# ---------------------------------------------------------------------------

class TestInternalCompressibleForwardModel:
    def test_evaluate_basic(self):
        fm = InternalCompressibleForwardModel(
            _MockCppForwardModel(n_obs=3), _MockParamSet()
        )
        r = fm.evaluate([0.31, 0.09])
        assert isinstance(r, EvaluationResult)
        assert len(r.predictions) == 2  # mock returns min(len(theta), n_obs)

    def test_parameter_names(self):
        fm = InternalCompressibleForwardModel(
            _MockCppForwardModel(), _MockParamSet()
        )
        assert "a1" in fm.parameter_names()


# ---------------------------------------------------------------------------
# PrecomputedEnsembleForwardModel
# ---------------------------------------------------------------------------

class TestPrecomputedEnsembleForwardModel:
    @pytest.fixture
    def fm(self):
        X, Y = _make_xy(n=40, d=2, n_obs=4, seed=1)
        return PrecomputedEnsembleForwardModel(X, Y, param_names=["a1", "betaStar"])

    def test_evaluate_returns_result(self, fm):
        r = fm.evaluate([0.31, 0.09])
        assert isinstance(r, EvaluationResult)
        assert r.converged
        assert len(r.predictions) == 4

    def test_evaluate_nearest_neighbor(self, fm):
        X, Y = _make_xy(n=40, d=2, n_obs=4, seed=1)
        theta_exact = X[5].tolist()
        r = fm.evaluate(theta_exact)
        assert np.allclose(r.predictions, Y[5].tolist())

    def test_precomputed_ensemble_returns_arrays(self, fm):
        result = fm.precomputed_ensemble()
        assert result is not None
        X_out, Y_out = result
        assert X_out.shape[1] == 2
        assert Y_out.shape[1] == 4
        assert len(X_out) == 40

    def test_parameter_names(self, fm):
        assert fm.parameter_names() == ["a1", "betaStar"]

    def test_n_samples_and_n_outputs(self, fm):
        assert fm.n_samples == 40
        assert fm.n_outputs == 4

    def test_status_is_precomputed(self, fm):
        r = fm.evaluate([0.3, 0.09])
        assert r.status == "precomputed"

    def test_default_param_names(self):
        X, Y = _make_xy(n=10, d=3, n_obs=2)
        fm = PrecomputedEnsembleForwardModel(X, Y)
        names = fm.parameter_names()
        assert names == ["theta_0", "theta_1", "theta_2"]

    def test_from_npz_round_trip(self, tmp_path):
        X, Y = _make_xy(n=20, d=2, n_obs=3, seed=7)
        path = tmp_path / "ens.npz"
        np.savez(str(path), X=X, Y=Y,
                 param_names=np.array(["a1", "betaStar"]))
        fm = PrecomputedEnsembleForwardModel.from_npz(path)
        assert fm.n_samples == 20
        assert fm.n_outputs == 3
        assert fm.parameter_names() == ["a1", "betaStar"]

    def test_from_dict(self):
        X, Y = _make_xy(n=10, d=2, n_obs=2)
        d = {"X": X.tolist(), "Y": Y.tolist(), "param_names": ["p0", "p1"]}
        fm = PrecomputedEnsembleForwardModel.from_dict(d)
        r = fm.evaluate([0.3, 0.1])
        assert len(r.predictions) == 2

    def test_nan_rows_are_nearest_but_still_returned(self):
        """NaN rows in Y are stored as-is; calling code should filter them."""
        X = np.array([[0.1, 0.1], [0.3, 0.09], [0.5, 0.12]])
        Y = np.array([[0.003, 0.004], [np.nan, 0.005], [0.002, 0.006]])
        fm = PrecomputedEnsembleForwardModel(X, Y, param_names=["a1", "bs"])
        r = fm.evaluate([0.3, 0.09])
        # Nearest neighbor is row 1 (NaN); result is returned and caller filters
        assert len(r.predictions) == 2

    def test_precomputed_returns_copy(self):
        """precomputed_ensemble() must return copies, not views."""
        X, Y = _make_xy(n=15, d=2, n_obs=3)
        fm = PrecomputedEnsembleForwardModel(X, Y)
        X_out, Y_out = fm.precomputed_ensemble()
        X_out[0, 0] = 9999.0
        X_out2, _ = fm.precomputed_ensemble()
        assert X_out2[0, 0] != pytest.approx(9999.0)

    def test_normalisation_handles_zero_std_column(self):
        """A constant column (std=0) should not cause division errors."""
        X = np.ones((10, 2)) * 0.31
        X[:, 1] = np.linspace(0.05, 0.15, 10)
        Y = np.random.default_rng(0).uniform(0, 0.01, (10, 3))
        fm = PrecomputedEnsembleForwardModel(X, Y)
        r = fm.evaluate([0.31, 0.09])
        assert r.converged

    def test_shape_1d_X_raises(self):
        """1-D X (no column dimension) should raise on construction."""
        X = np.array([0.31, 0.09])   # shape (2,) not (2, d)
        Y = np.array([[0.003, 0.004], [0.002, 0.005]])
        with pytest.raises((ValueError, IndexError, AttributeError)):
            PrecomputedEnsembleForwardModel(X, Y)


# ---------------------------------------------------------------------------
# save_ensemble_npz + load_forward_model
# ---------------------------------------------------------------------------

class TestSaveEnsembleNPZ:
    def test_creates_file(self, tmp_path):
        class _SimpleFM(ForwardModelBase):
            def evaluate(self, theta):
                return EvaluationResult([float(theta[0]) * 2], converged=True)
            def parameter_names(self):
                return ["x"]

        fm = _SimpleFM()
        path = save_ensemble_npz(
            fm, bounds=([0.0], [1.0]), n_samples=10,
            path=tmp_path / "e.npz", rng_seed=0, verbose=False
        )
        assert path.exists()
        data = np.load(str(path), allow_pickle=True)
        assert "X" in data and "Y" in data
        assert len(data["X"]) == 10

    def test_load_forward_model_from_npz(self, tmp_path):
        X, Y = _make_xy(n=15, d=2)
        path = tmp_path / "e.npz"
        np.savez(str(path), X=X, Y=Y)
        fm = load_forward_model(path)
        assert isinstance(fm, PrecomputedEnsembleForwardModel)
        r = fm.evaluate([0.3, 0.1])
        assert len(r.predictions) == Y.shape[1]

    def test_load_forward_model_from_json_config(self, tmp_path):
        X, Y = _make_xy(n=12, d=2)
        np.savez(str(tmp_path / "ens.npz"), X=X, Y=Y)
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({
            "type": "precomputed_npz",
            "path": str(tmp_path / "ens.npz"),
        }))
        fm = load_forward_model(cfg_path)
        assert isinstance(fm, PrecomputedEnsembleForwardModel)


# ---------------------------------------------------------------------------
# ExternalSolverForwardModel
# ---------------------------------------------------------------------------

class TestExternalSolverForwardModel:
    def test_queued_pending_when_no_result(self, tmp_path):
        fm = ExternalSolverForwardModel(
            queue_dir=tmp_path / "q",
            results_dir=tmp_path / "r",
            param_names=["a1", "betaStar"],
            poll_timeout_s=0.0,
        )
        r = fm.evaluate([0.31, 0.09])
        assert not r.converged
        assert r.status == "queued_pending"
        assert "job_id" in r.metadata

    def test_queue_file_written(self, tmp_path):
        q_dir = tmp_path / "q"
        fm = ExternalSolverForwardModel(
            queue_dir=q_dir,
            results_dir=tmp_path / "r",
            param_names=["a1", "betaStar"],
            poll_timeout_s=0.0,
        )
        fm.evaluate([0.31, 0.09])
        queue_files = list(q_dir.glob("*.json"))
        assert len(queue_files) == 1
        payload = json.loads(queue_files[0].read_text())
        assert "theta" in payload
        assert payload["theta"] == pytest.approx([0.31, 0.09])

    def test_result_delivered_synchronously(self, tmp_path):
        q_dir = tmp_path / "q"
        r_dir = tmp_path / "r"
        fm = ExternalSolverForwardModel(
            queue_dir=q_dir,
            results_dir=r_dir,
            param_names=["a1", "betaStar"],
            poll_timeout_s=0.0,
        )
        # Pre-write result before evaluate so poll finds it immediately
        job_id = "test-job-123"
        (r_dir / f"{job_id}.json").write_text(json.dumps({
            "job_id": job_id,
            "predictions": [0.003, 0.004],
            "converged": True,
            "status": "external",
        }))
        # The ExternalSolverForwardModel generates its own UUID, so we
        # test deliver_result helper instead
        import uuid
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("uuid.uuid4", lambda: job_id)
            r = fm.evaluate([0.31, 0.09])
        assert r.converged
        assert r.predictions == pytest.approx([0.003, 0.004])

    def test_parameter_names(self, tmp_path):
        fm = ExternalSolverForwardModel(
            tmp_path / "q", tmp_path / "r",
            param_names=["p0", "p1", "p2"],
        )
        assert fm.parameter_names() == ["p0", "p1", "p2"]

    def test_deliver_result_helper(self, tmp_path):
        q_dir = tmp_path / "q"
        r_dir = tmp_path / "r"
        fm = ExternalSolverForwardModel(
            queue_dir=q_dir, results_dir=r_dir,
            param_names=["a1"], poll_timeout_s=0.0,
        )
        fm.deliver_result("abc123", [0.005, 0.006], converged=True)
        result_file = r_dir / "abc123.json"
        assert result_file.exists()
        data = json.loads(result_file.read_text())
        assert data["predictions"] == pytest.approx([0.005, 0.006])


# ---------------------------------------------------------------------------
# SurrogateForwardModel
# ---------------------------------------------------------------------------

class TestSurrogateForwardModel:
    def test_evaluate_returns_mu(self):
        surr = _MockSurrogate(n_obs=4, d=2)
        fm = SurrogateForwardModel(surr, param_names=["a1", "betaStar"])
        r = fm.evaluate([0.31, 0.09])
        assert isinstance(r, EvaluationResult)
        assert r.converged
        assert len(r.predictions) == 4
        assert r.status == "surrogate"

    def test_raises_on_untrained(self):
        surr = _MockSurrogate()
        surr.trained = False
        with pytest.raises(ValueError, match="trained"):
            SurrogateForwardModel(surr)

    def test_parameter_names(self):
        surr = _MockSurrogate()
        fm = SurrogateForwardModel(surr, param_names=["x", "y"])
        assert fm.parameter_names() == ["x", "y"]

    def test_precomputed_ensemble_returns_none(self):
        surr = _MockSurrogate()
        fm = SurrogateForwardModel(surr)
        assert fm.precomputed_ensemble() is None


# ---------------------------------------------------------------------------
# BayesianInferenceKOH integration with PrecomputedEnsembleForwardModel
# ---------------------------------------------------------------------------

class TestBayesianInferenceKOHIntegration:
    """
    Verify that BayesianInferenceKOH.run_ensemble() uses precomputed data
    when the forward model provides it, without any KOH-side knowledge of type.
    """

    def test_precomputed_short_circuit(self):
        from bayesian_inference import BayesianInferenceKOH, Prior, KOHLikelihood

        X, Y = _make_xy(n=25, d=2, n_obs=3, seed=42)
        fm = PrecomputedEnsembleForwardModel(X, Y,
                                             param_names=["a1", "betaStar"])

        prior = Prior(
            means=[0.31, 0.09],
            stds=[0.05, 0.015],
            lower=[0.20, 0.05],
            upper=[0.50, 0.15],
        )
        koh = KOHLikelihood(
            obs_locations=np.linspace(0, 1, 3),
            obs_values=Y[0],
            obs_sigmas=np.full(3, 0.0002),
            mode="diagonal",
        )

        param_set_dict = {
            "defaults": [0.31, 0.09],
            "lower": [0.20, 0.05],
            "upper": [0.50, 0.15],
            "names": ["a1", "betaStar"],
        }

        koh_bi = BayesianInferenceKOH(
            forward_model=fm,
            param_set=param_set_dict,
            koh_likelihood=koh,
            theta_prior=prior,
        )
        # run_ensemble() should use precomputed data, not run LHC
        eX, eY = koh_bi.run_ensemble(n_samples=999, verbose=False)

        assert eX.shape == X.shape
        assert eY.shape == Y.shape
        assert np.allclose(eX, X)

    def test_internal_style_fm_uses_lhc(self):
        """When precomputed_ensemble returns None, LHC is used normally."""
        from bayesian_inference import BayesianInferenceKOH, Prior, KOHLikelihood

        class _InlineFM(ForwardModelBase):
            def evaluate(self, theta):
                return EvaluationResult(
                    [float(theta[0]) * 0.01, float(theta[1]) * 0.01,
                     float(theta[0]) + float(theta[1])],
                    converged=True
                )
            def parameter_names(self):
                return ["a1", "betaStar"]

        fm = _InlineFM()
        prior = Prior(
            means=[0.31, 0.09],
            stds=[0.05, 0.015],
            lower=[0.20, 0.05],
            upper=[0.50, 0.15],
        )
        obs_vals = np.array([0.003, 0.001, 0.4])
        koh = KOHLikelihood(
            obs_locations=np.array([0.0, 0.5, 1.0]),
            obs_values=obs_vals,
            obs_sigmas=np.full(3, 0.0002),
            mode="diagonal",
        )
        param_set_dict = {
            "defaults": [0.31, 0.09],
            "lower": [0.20, 0.05],
            "upper": [0.50, 0.15],
            "names": ["a1", "betaStar"],
        }

        koh_bi = BayesianInferenceKOH(
            forward_model=fm,
            param_set=param_set_dict,
            koh_likelihood=koh,
            theta_prior=prior,
        )
        eX, eY = koh_bi.run_ensemble(n_samples=15, verbose=False)
        assert eX.shape[1] == 2
        assert eY.shape[1] == 3
        assert len(eX) == 15
