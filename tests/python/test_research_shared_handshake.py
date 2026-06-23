"""Validate the FROZEN research.shared interfaces (foundation freeze, core-v1.0).

Two jobs:
  1. Exercise the pure interfaces (inference handshake, constraints, metrics) on
     their own.
  2. Prove the inference handshake's shape against the known-working SST closure:
     its parameter spec, prior (with the inert fluctuation-dissipation hook and the
     realizability + stability bounds), and likelihood hook theta -> predicted
     statistics, driving the real C++ ForwardModel.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.shared.inference import (
    ClosurePrior, EvaluationStatus, Likelihood, ParameterSpec, Prediction,
)
from research.shared.constraints import (
    BarycentricRealizability, IntegrityBasis,
)
from research.shared.metrics import (
    GaussianNLL, NormalizedRMSE, ood_gap,
)


# ---------------------------------------------------------------------------
# Inference handshake: pure types
# ---------------------------------------------------------------------------

def test_parameter_spec_bounds_and_membership():
    spec = ParameterSpec(["a1", "betaStar"], [0.31, 0.09], [0.2, 0.07], [0.4, 0.11])
    assert spec.n == 2
    assert spec.in_bounds(spec.defaults)
    assert not spec.in_bounds([0.5, 0.09])          # a1 above upper
    assert np.allclose(spec.clip([1.0, 0.0]), [0.4, 0.07])


def test_closure_prior_truncates_and_fd_hook_optional():
    spec = ParameterSpec(["a1", "betaStar"], [0.31, 0.09], [0.2, 0.07], [0.4, 0.11])
    prior = ClosurePrior(spec)                       # memoryless: no FD coupling
    assert prior.fd_coupling is None
    assert prior.log_prior(spec.defaults) == pytest.approx(0.0)   # at the mean
    assert prior.log_prior([0.5, 0.09]) == -np.inf               # outside the box
    samples = prior.sample(16, rng=np.random.default_rng(0))
    assert samples.shape == (16, 2)
    assert np.all(samples >= spec.lower) and np.all(samples <= spec.upper)


def test_evaluation_status_coercion():
    assert EvaluationStatus.coerce("Converged") is EvaluationStatus.Converged
    assert EvaluationStatus.coerce("precomputed") is EvaluationStatus.Converged
    assert EvaluationStatus.coerce("Unconverged") is EvaluationStatus.Unconverged
    assert EvaluationStatus.coerce("nonsense") is EvaluationStatus.Unknown
    assert EvaluationStatus.Converged.is_usable()
    assert not EvaluationStatus.Unconverged.is_usable()


def test_prediction_converged_semantics():
    ok = Prediction([1.0, 2.0], status="Converged")
    assert ok.converged
    bad = Prediction([1.0, np.nan], status="Converged")
    assert not bad.converged                         # non-finite is not usable
    unconv = Prediction([1.0, 2.0], status="Unconverged")
    assert not unconv.converged


# ---------------------------------------------------------------------------
# Constraints: TWO separate entry points
# ---------------------------------------------------------------------------

def _sym_traceless_from_eigs(eigs, seed=0):
    """A symmetric traceless 3x3 with the given eigenvalues in a rotated frame."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]                           # make it a proper rotation
    return Q @ np.diag(eigs) @ Q.T


def test_realizability_projection_repairs_and_is_idempotent():
    proj = BarycentricRealizability()
    # eigenvalues (1, 0, -1): l3 = -1 < -1/3 so C3c = 3*l3 + 1 = -2 < 0 -> unrealizable
    b_bad = _sym_traceless_from_eigs([1.0, 0.0, -1.0])
    assert not proj.is_realizable(b_bad)
    b_fixed = proj.project(b_bad)
    assert proj.is_realizable(b_fixed)
    # symmetric, traceless preserved
    assert np.allclose(b_fixed, b_fixed.T, atol=1e-10)
    assert abs(np.trace(b_fixed)) < 1e-10
    # idempotent on an already-realizable tensor
    b_ok = _sym_traceless_from_eigs([0.2, 0.0, -0.2])
    assert proj.is_realizable(b_ok)
    assert np.allclose(proj.project(b_ok), b_ok, atol=1e-8)


def test_galilean_basis_is_rotation_invariant():
    basis = IntegrityBasis()
    G = np.array([[0.0, 2.0, 0.0],
                  [0.5, 0.0, 1.0],
                  [0.0, 0.3, 0.0]])
    S, W = basis.strain_rotation(G)
    assert np.allclose(S, S.T) and np.allclose(W, -W.T)   # S symmetric, W skew
    assert np.allclose(S + W, G)
    inv = basis.invariants(S, W)
    # integrity-basis invariants are unchanged under a frame rotation Q
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    inv_rot = basis.invariants(Q @ S @ Q.T, Q @ W @ Q.T)
    assert np.allclose(inv, inv_rot, atol=1e-10)
    assert len(basis.tensors(S, W)) == 3


def test_basis_invariance_does_not_imply_realizability():
    # The CLAUDE.md point: the invariant basis delivers invariance, NOT
    # realizability; the two constraints are separate. A basis-built anisotropy
    # b = alpha * T1 (T1 = S) can be unrealizable and needs the projection.
    basis = IntegrityBasis()
    proj = BarycentricRealizability()
    S, W = basis.strain_rotation(np.diag([1.0, -0.5, -0.5]))   # pure strain
    b = 3.0 * basis.tensors(S, W)[0]                            # large -> leaves triangle
    assert not proj.is_realizable(b)
    assert proj.is_realizable(proj.project(b))


# ---------------------------------------------------------------------------
# Metrics: deterministic error + probabilistic UQ score + OOD gap
# ---------------------------------------------------------------------------

def test_metrics_behaviour():
    obs = np.array([1.0, 2.0, 3.0])
    nrmse = NormalizedRMSE()
    assert nrmse(obs, obs) == pytest.approx(0.0)
    assert nrmse(obs + 0.1, obs) > 0.0
    nll = GaussianNLL()
    # a calibrated, accurate forecast scores better (lower) than an overconfident one
    good = nll(obs, np.full(3, 0.2), obs)
    overconfident = nll(obs + 0.5, np.full(3, 0.01), obs)
    assert good < overconfident
    # OOD gap: worse out-of-distribution score is positive for lower-is-better
    assert ood_gap(0.1, 0.3) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# SST handshake against the real, known-working forward model
# ---------------------------------------------------------------------------

def test_sst_closure_parameter_spec_and_prior(rs):
    from research.shared.closures.sst import SSTClosure
    sst = SSTClosure()                               # default a1_betaStar channel
    assert sst.name == "sst_menter1994"
    assert (sst.has_memory, sst.is_stochastic, sst.is_nonlocal) == (False, False, False)
    spec = sst.parameter_spec()
    assert spec.names == ["a1", "betaStar"]
    assert np.allclose(spec.defaults, [0.31, 0.09])  # Menter 1994
    assert np.all(spec.lower <= spec.defaults) and np.all(spec.defaults <= spec.upper)
    prior = sst.prior()
    assert isinstance(prior, ClosurePrior)
    assert prior.fd_coupling is None                 # memoryless baseline
    assert np.isfinite(prior.log_prior(spec.defaults))


def test_sst_likelihood_hook_predicts_statistics(rs):
    from research.shared.closures.sst import SSTClosure
    # Small mesh keeps the real solver eval fast; Menter defaults is the robust point.
    sst = SSTClosure.channel(nx=16, ny=12)
    spec = sst.parameter_spec()
    lik = sst.likelihood()
    assert isinstance(lik, Likelihood)
    pred = lik.predict(spec.defaults)
    # the handshake carries predicted statistics + a real status classification
    assert pred.statistics.shape == (5,)             # Cf at 5 stations
    assert np.all(np.isfinite(pred.statistics))
    assert isinstance(pred.status, EvaluationStatus)
    # the scalar likelihood (the C++ penalized log-likelihood) is finite at defaults
    assert np.isfinite(lik.log_likelihood(spec.defaults))


def test_sst_handshake_is_forward_model_agnostic(rs):
    # The same Likelihood interface drives a precomputed (no-solver) forward model,
    # whose "precomputed" status coerces to Converged.
    from forward_model_interface import PrecomputedEnsembleForwardModel
    from research.shared.closures.sst import _ForwardModelLikelihood
    spec = ParameterSpec(["a1", "betaStar"], [0.31, 0.09], [0.2, 0.07], [0.4, 0.11])
    X = np.array([[0.31, 0.09], [0.35, 0.10]])
    Y = np.array([[0.0024, 0.0022], [0.0026, 0.0023]])
    fm = PrecomputedEnsembleForwardModel(X, Y, param_names=spec.names)
    lik = _ForwardModelLikelihood(fm, spec)
    pred = lik.predict([0.31, 0.09])
    assert pred.status is EvaluationStatus.Converged
    assert pred.converged
    assert np.allclose(pred.statistics, [0.0024, 0.0022])
