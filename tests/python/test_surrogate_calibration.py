"""
Surrogate-trust fix: the noise floor prevents the degenerate noise→0 GP fit that
made the surrogate posterior overconfident/biased.  Locked here so it can't regress.
"""

from __future__ import annotations

import numpy as np

from bayesian_inference import GPSurrogate, MultiOutputSurrogate


def test_noise_floor_prevents_variance_collapse():
    # Noise-free smooth target -> the unconstrained GP drives noise ~0 (interpolates,
    # overconfident); the floored GP keeps a calibrated predictive variance.
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (40, 2))
    y = np.sin(2 * X[:, 0]) + 0.3 * X[:, 1] ** 2          # deterministic (no noise)

    gp0 = GPSurrogate(); gp0.train(X, y, optimize_restarts=4, noise_floor=None)
    gpf = GPSurrogate(); gpf.train(X, y, optimize_restarts=4, noise_floor=1e-2)

    # floored GP's likelihood variance respects the floor; unconstrained collapses lower
    v0 = float(np.asarray(gp0.gp.likelihood.variance).ravel()[0])
    vf = float(np.asarray(gpf.gp.likelihood.variance).ravel()[0])
    assert vf >= 1e-2 - 1e-9
    assert v0 < vf                                        # unconstrained is smaller

    # away from training points the floored GP is less overconfident (larger sigma)
    Xt = rng.uniform(-0.9, 0.9, (50, 2))
    _, var0 = gp0.predict_batch(Xt)
    _, varf = gpf.predict_batch(Xt)
    assert np.mean(varf) > np.mean(var0)


def test_noise_floor_multioutput():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (40, 2))
    Y = np.column_stack([X[:, 0] ** 2, np.sin(X[:, 1])])
    mo = MultiOutputSurrogate(); mo.train(X, Y, optimize_restarts=3, noise_floor=1e-2)
    for gp in mo.gps:
        assert float(np.asarray(gp.likelihood.variance).ravel()[0]) >= 1e-2 - 1e-9
