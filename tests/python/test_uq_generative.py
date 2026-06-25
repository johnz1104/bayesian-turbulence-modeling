"""Synthetic-data tests for the generative, realizability-constrained model-form
model. Skipped when PyTorch is unavailable so the rest of the suite is unaffected.

Two properties are pinned:
  1. the conditional normalizing flow recovers a planted non-Gaussian,
     heteroscedastic, bimodal conditional p(y | x); and
  2. every generated discrepancy sample, after the barycentric projection, is
     strictly realizable (and the projection is doing real work: a fraction of
     the raw samples are not realizable).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from UQ import synthetic as syn
from UQ import realizability as rz
from UQ.generative import GenerativeDiscrepancyModel


def test_flow_recovers_bimodal_heteroscedastic_conditional():
    x, y = syn.conditional_discrepancy_dataset(n=3000, seed=2)
    model = GenerativeDiscrepancyModel(n_features=1, n_targets=2,
                                       n_layers=6, hidden=48, seed=0)
    model.fit(x, y, epochs=180, lr=2e-3, batch=512)

    # x = 0 (modes closer) and x = 1.5 (modes well separated) both bimodal
    for xv in (0.0, 1.5):
        gen = model.sample(np.full((3000, 1), xv), n_per=1)[:, 0, :]
        true = syn.true_conditional_samples(xv, 3000)
        mid = np.sin(1.5 * xv)
        # both modes populated -> the flow is not collapsing to one Gaussian
        up = np.mean(gen[:, 0] > mid + 0.5)
        lo = np.mean(gen[:, 0] < mid - 0.5)
        assert up > 0.25 and lo > 0.25, f"bimodality not recovered at x={xv}"
        # first two moments of the recovered conditional match the truth
        assert abs(gen[:, 0].mean() - true[:, 0].mean()) < 0.35
        assert abs(gen[:, 0].std() - true[:, 0].std()) / true[:, 0].std() < 0.30

    # heteroscedasticity: the conditional spread changes with x as planted
    s0 = model.sample(np.zeros((2000, 1)), 1)[:, 0, 0].std()
    s15 = model.sample(np.full((2000, 1), 1.5), 1)[:, 0, 0].std()
    assert s15 > s0                     # separation grows with |x|


def _anisotropy_component_dataset(n, seed):
    """Synthetic (features -> 5 anisotropy components) data whose raw draws can
    leave the realizable set, so the projection has something to do."""
    rng = np.random.default_rng(seed)
    feat = rng.uniform(-1, 1, size=(n, 2))
    # components scaled large enough that base + discrepancy is often unrealizable
    comp = np.stack([
        0.45 * np.tanh(2 * feat[:, 0]) + 0.1 * rng.normal(size=n),
        -0.30 * feat[:, 1] + 0.1 * rng.normal(size=n),
        0.35 * feat[:, 0] * feat[:, 1] + 0.08 * rng.normal(size=n),
        0.05 * rng.normal(size=n),
        0.05 * rng.normal(size=n),
    ], axis=-1)
    return feat, comp


def test_generated_samples_are_realizable_after_projection():
    feat, comp = _anisotropy_component_dataset(1500, seed=5)
    model = GenerativeDiscrepancyModel(n_features=2, n_targets=5,
                                       n_layers=6, hidden=48, seed=1)
    model.fit(feat, comp, epochs=120, lr=2e-3, batch=512)

    # a base (Boussinesq-like) anisotropy to add the discrepancy onto
    rng = np.random.default_rng(9)
    base = np.zeros((len(feat), 3, 3))
    bd = 0.2 * rng.normal(size=(len(feat), 2))
    base[:, 0, 0] = bd[:, 0]; base[:, 1, 1] = bd[:, 1]
    base[:, 2, 2] = -(bd[:, 0] + bd[:, 1])

    # raw (unprojected) samples: some must be NOT realizable, else the
    # projection is untested. Build the raw anisotropy and check.
    raw_comp = model.sample(feat, n_per=2)
    raw_b = model.components_to_anisotropy(raw_comp) + base[:, None, :, :]
    R_raw = 2.0 * (raw_b + np.eye(3) / 3.0)            # k = 1
    frac_unrealizable = 1.0 - np.mean(rz.is_realizable(R_raw))
    assert frac_unrealizable > 0.02, "construct a case where projection matters"

    # projected samples: every one must be realizable
    bp = model.sample_realizable_anisotropy(feat, base_anisotropy=base, n_per=2)
    R_proj = 2.0 * (bp + np.eye(3) / 3.0)
    assert np.all(rz.is_realizable(R_proj, tol=1e-6)), "projection failed to realize"
