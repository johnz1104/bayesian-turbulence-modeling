"""Gaussian model-form baseline: heteroscedastic Gaussian conditional.

Skipped when PyTorch is unavailable. Three properties are pinned:
  1. the model recovers a planted heteroscedastic GAUSSIAN conditional
     (its own model class, so recovery must be accurate);
  2. on the planted BIMODAL conditional it fills in the probability trough
     between the modes (the distribution-family limitation the generative
     flow is compared against; the flow test shows the flow does not);
  3. realizable-anisotropy sampling mirrors the generative API: every
     projected sample is realizable and the projection does real work.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from UQ import synthetic as syn
from UQ import realizability as rz
from UQ.gaussian_modelform import GaussianDiscrepancyModel


def test_recovers_planted_heteroscedastic_gaussian():
    rng = np.random.default_rng(4)
    x = rng.uniform(-2.0, 2.0, size=(4000, 1))
    mean = np.sin(1.5 * x[:, 0])
    std = 0.10 + 0.20 * np.abs(x[:, 0])
    y = np.stack([mean + std * rng.normal(size=x.shape[0]),
                  0.5 * x[:, 0] + 0.15 * rng.normal(size=x.shape[0])], axis=-1)

    model = GaussianDiscrepancyModel(n_features=1, n_targets=2, hidden=48, seed=0)
    model.fit(x, y, epochs=200, lr=2e-3, batch=512)

    for xv in (0.0, 1.5):
        s = model.sample(np.full((4000, 1), xv), n_per=1)[:, 0, :]
        assert abs(s[:, 0].mean() - np.sin(1.5 * xv)) < 0.08
        planted = 0.10 + 0.20 * abs(xv)
        assert abs(s[:, 0].std() - planted) / planted < 0.30


def test_gaussian_fills_bimodal_trough():
    # the property under test in the study: a Gaussian conditional CANNOT
    # represent the bimodal discrepancy law and concentrates mass exactly in
    # the trough between the modes (the flow keeps the trough nearly empty,
    # see test_uq_generative)
    x, y = syn.conditional_discrepancy_dataset(n=3000, seed=2)
    model = GaussianDiscrepancyModel(n_features=1, n_targets=2, hidden=48, seed=0)
    model.fit(x, y, epochs=200, lr=2e-3, batch=512)

    xv = 1.5
    gen = model.sample(np.full((3000, 1), xv), n_per=1)[:, 0, :]
    true = syn.true_conditional_samples(xv, 3000)
    mid = np.sin(1.5 * xv)
    frac_gauss = np.mean(np.abs(gen[:, 0] - mid) < 0.5)
    frac_true = np.mean(np.abs(true[:, 0] - mid) < 0.5)
    assert frac_true < 0.02                 # the true conditional has a trough
    assert frac_gauss > 0.15                # the Gaussian fills it in


def test_realizable_sampling_mirrors_generative_api():
    rng = np.random.default_rng(9)
    feat = rng.uniform(-1, 1, size=(600, 2))
    # wide planted components so raw draws leave the realizable set
    comp = 0.45 * rng.normal(size=(600, 5)) + 0.25 * feat[:, :1]
    model = GaussianDiscrepancyModel(n_features=2, n_targets=5, hidden=32, seed=1)
    model.fit(feat, comp, epochs=120, lr=2e-3, batch=256)

    raw = model.components_to_anisotropy(model.sample(feat, n_per=4))
    R_raw = 2.0 * (raw + np.eye(3) / 3.0)
    raw_ok = np.mean(rz.is_realizable(R_raw, tol=1e-8))

    proj = model.sample_realizable_anisotropy(feat, n_per=4)
    R_proj = 2.0 * (proj + np.eye(3) / 3.0)
    assert np.all(rz.is_realizable(R_proj, tol=1e-6))
    assert raw_ok < 0.999          # the projection is doing real work


def test_sampling_is_seed_reproducible():
    # torch's stream is global (as for the generative flow), so reproducibility
    # is at the script level: identical seeds and call order give identical
    # fits, and re-seeding before sampling gives identical draws
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, size=(500, 1))
    y = np.stack([x[:, 0], -x[:, 0]], axis=-1) + 0.1 * rng.normal(size=(500, 2))
    a = GaussianDiscrepancyModel(1, 2, hidden=16, seed=7).fit(x, y, epochs=40)
    b = GaussianDiscrepancyModel(1, 2, hidden=16, seed=7).fit(x, y, epochs=40)
    torch.manual_seed(0)
    sa = a.sample(x[:50], n_per=2)
    torch.manual_seed(0)
    sb = b.sample(x[:50], n_per=2)
    assert np.allclose(sa, sb)
