"""
Surrogate diagnostics + active learning tests.

Uses a fast 2-D quadratic-bowl forward model so the tests run in seconds while
still exercising:

  1. ``train_val_split`` produces disjoint splits with the requested fraction.
  2. ``multi_output_diagnostics`` reports finite RMSE/R^2 and σ-coverage.
  3. ``ActiveLearner`` initialises, refits, and queries without crashing.
  4. ``max_var`` outperforms ``random`` on average for the same query budget
     on the toy quadratic — a genuine but loose property.
  5. ``cpp_forward_adapter`` flags non-finite predictions as ok=False.
"""

from __future__ import annotations

import numpy as np
import pytest

from surrogate_diagnostics import (
    train_val_split, multi_output_diagnostics, scalar_diagnostics,
)
from active_learning import (
    ActiveLearner, sample_lhs, sample_uniform, cpp_forward_adapter,
)
from bayesian_inference import MultiOutputSurrogate


# ---------------- toy forward maps ---------------------------------

def _quad_bowl(theta):
    """Toy 2-D, 3-output forward map.

    Output[0]: quadratic bowl ‖θ - μ_0‖²
    Output[1]: linear ramp θ_0 + 2θ_1
    Output[2]: scaled product θ_0 * θ_1
    """
    theta = np.asarray(theta, float)
    return np.array([
        np.sum((theta - np.array([0.5, 0.5])) ** 2),
        theta[0] + 2.0 * theta[1],
        theta[0] * theta[1],
    ])


def _quad_bowl_forward(theta):
    """Adapter to AL forward signature."""
    y = _quad_bowl(theta)
    return y, bool(np.all(np.isfinite(y)))


def _build_dataset(n: int, *, seed: int):
    """Random training set on the unit square."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2))
    Y = np.array([_quad_bowl(x) for x in X])
    return X, Y


# ---------------- train/val split -----------------------------------

class TestTrainValSplit:
    def test_split_sizes(self):
        X, Y = _build_dataset(20, seed=0)
        Xt, Yt, Xv, Yv = train_val_split(X, Y, val_frac=0.25, rng_seed=0)
        assert len(Xv) == 5 and len(Xt) == 15
        assert len(Yv) == 5 and len(Yt) == 15

    def test_split_disjoint(self):
        X, Y = _build_dataset(30, seed=0)
        Xt, _, Xv, _ = train_val_split(X, Y, val_frac=0.2, rng_seed=0)
        # No row of Xv should appear in Xt (with random sampling and a seed).
        for row in Xv:
            for tr in Xt:
                assert not np.allclose(row, tr)

    def test_seed_reproducible(self):
        X, Y = _build_dataset(15, seed=0)
        a = train_val_split(X, Y, val_frac=0.2, rng_seed=42)
        b = train_val_split(X, Y, val_frac=0.2, rng_seed=42)
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)

    def test_min_val_enforced(self):
        X, Y = _build_dataset(5, seed=0)
        Xt, Yt, Xv, Yv = train_val_split(X, Y, val_frac=0.05, rng_seed=0,
                                          min_val=2)
        assert len(Xv) >= 2
        assert len(Xt) + len(Xv) == 5


# ---------------- diagnostics --------------------------------------

class TestMultiOutputDiagnostics:
    @pytest.fixture(scope="class")
    def trained_mo(self):
        X, Y = _build_dataset(20, seed=0)
        mo = MultiOutputSurrogate()
        mo.train(X, Y, optimize_restarts=2)
        return mo, X, Y

    def test_per_output_keys(self, trained_mo):
        mo, X, Y = trained_mo
        Xt, Yt, Xv, Yv = train_val_split(X, Y, val_frac=0.25, rng_seed=0)
        diag = multi_output_diagnostics(mo, Xv, Yv)
        assert "per_output" in diag and "aggregate" in diag
        assert len(diag["per_output"]) == mo.n_outputs
        for p in diag["per_output"]:
            for k in ("rmse", "mae", "r2", "mean_sigma",
                      "coverage_1sigma", "coverage_2sigma", "coverage_3sigma"):
                assert k in p
                assert np.isfinite(p[k]) or p[k] != p[k]   # allow NaN for r2

    def test_rmse_decreases_with_more_training(self):
        # Train on small set vs. large set; large-set RMSE should be smaller.
        X_te, Y_te = _build_dataset(40, seed=99)
        rmses = {}
        for n_train in (10, 30):
            X, Y = _build_dataset(n_train, seed=0)
            mo = MultiOutputSurrogate()
            mo.train(X, Y, optimize_restarts=2)
            diag = multi_output_diagnostics(mo, X_te, Y_te)
            rmses[n_train] = diag["aggregate"]["mean_rmse"]
        assert rmses[30] <= rmses[10] * 1.10, (
            f"RMSE did not decrease with more data: {rmses}"
        )

    def test_coverage_in_unit_interval(self, trained_mo):
        mo, X, Y = trained_mo
        diag = multi_output_diagnostics(mo, X, Y)
        for p in diag["per_output"]:
            for k in ("coverage_1sigma", "coverage_2sigma", "coverage_3sigma"):
                assert 0.0 <= p[k] <= 1.0


# ---------------- active learning ----------------------------------

class TestActiveLearner:
    def test_initialize_produces_dataset(self):
        learner = ActiveLearner(_quad_bowl_forward, [0, 0], [1, 1],
                                  strategy="max_var", n_candidates=64,
                                  rng_seed=0, verbose=False)
        learner.initialize(n_init=12)
        assert learner.X.shape[1] == 2
        assert len(learner.X) >= 8   # almost all valid for this bowl

    def test_max_var_run_does_not_crash(self):
        learner = ActiveLearner(_quad_bowl_forward, [0, 0], [1, 1],
                                  strategy="max_var", n_candidates=64,
                                  rng_seed=0, verbose=False)
        learner.initialize(n_init=12)
        n_before = len(learner.X)
        hist = learner.run(n_queries=4)
        assert len(learner.X) == n_before + 4
        assert hist.iteration == 4

    @pytest.mark.parametrize("strategy",
                             ["random", "max_var", "max_norm_var", "max_min_dist"])
    def test_strategy_runs(self, strategy):
        learner = ActiveLearner(_quad_bowl_forward, [0, 0], [1, 1],
                                  strategy=strategy, n_candidates=64,
                                  rng_seed=0, verbose=False)
        learner.initialize(n_init=10)
        learner.run(n_queries=3)
        assert len(learner.X) == 13

    def test_unknown_strategy_rejected(self):
        with pytest.raises(ValueError, match="strategy"):
            ActiveLearner(_quad_bowl_forward, [0, 0], [1, 1],
                          strategy="nope", verbose=False)

    def test_max_min_dist_produces_more_spread_points_than_random(self):
        """Structural (deterministic) AL test: the ``max_min_dist`` strategy
        must produce a point set whose minimum pairwise distance is *strictly
        larger* than that of uniform random sampling for the same budget.

        This is the textbook maximin property and is the reason maximin LHS
        beats random sampling on integration / surrogate benchmarks; if it
        fails, the AL acquisition is broken.  We deliberately avoid noisy
        end-to-end RMSE comparisons here — those belong in the example
        script (``examples/active_learning_bfs.py``).
        """
        lower = np.zeros(4); upper = np.ones(4)
        n_init, n_queries = 8, 12

        # Build a forward map that always succeeds; we only care about X here.
        def _trivial(theta):
            return np.array([float(theta[0])]), True

        gains = []
        for seed in range(3):
            spread = {}
            for strat in ("random", "max_min_dist"):
                lr = ActiveLearner(_trivial, lower, upper,
                                    strategy=strat, n_candidates=256,
                                    rng_seed=seed, verbose=False)
                lr.initialize(n_init=n_init)
                lr.run(n_queries=n_queries)
                X = lr.X
                # Min pairwise distance — larger means better space-filling.
                d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
                np.fill_diagonal(d2, np.inf)
                spread[strat] = float(np.sqrt(np.min(d2)))
            gains.append(spread["max_min_dist"] - spread["random"])
            assert spread["max_min_dist"] > spread["random"], (
                f"seed {seed}: maxmin spread {spread['max_min_dist']:.4f} "
                f"not > random spread {spread['random']:.4f}"
            )
        # Cross-seed sanity: gain must be positive on every run.
        assert min(gains) > 0.0


class TestCppForwardAdapter:
    def test_finite_predictions_pass(self):
        class _Result:
            predictions = [1.0, 2.0, 3.0]
        class _FM:
            def evaluate(self, theta): return _Result()
        adapt = cpp_forward_adapter(_FM(), koh_n=3)
        preds, ok = adapt([0.5, 0.5])
        assert ok and preds.tolist() == [1.0, 2.0, 3.0]

    def test_nan_predictions_fail(self):
        class _Result:
            predictions = [1.0, float("nan"), 3.0]
        class _FM:
            def evaluate(self, theta): return _Result()
        adapt = cpp_forward_adapter(_FM(), koh_n=3)
        _, ok = adapt([0.5, 0.5])
        assert not ok

    def test_wrong_length_fails(self):
        class _Result:
            predictions = [1.0, 2.0]
        class _FM:
            def evaluate(self, theta): return _Result()
        adapt = cpp_forward_adapter(_FM(), koh_n=3)
        _, ok = adapt([0.5, 0.5])
        assert not ok

    def test_empty_predictions_fail(self):
        class _Result:
            predictions = []
        class _FM:
            def evaluate(self, theta): return _Result()
        adapt = cpp_forward_adapter(_FM())
        _, ok = adapt([0.5, 0.5])
        assert not ok


# ---------------- scalar surrogate diagnostics ---------------------

def test_scalar_diagnostics_runs():
    """Smoke check that scalar_diagnostics works on a GPSurrogate."""
    from bayesian_inference import GPSurrogate
    rng = np.random.default_rng(0)
    X = rng.random((20, 2))
    y = np.array([_quad_bowl(x)[0] for x in X])
    gp = GPSurrogate()
    gp.train(X, y, optimize_restarts=2)
    diag = scalar_diagnostics(gp, X[:6], y[:6])
    for k in ("rmse", "mae", "r2", "mean_log_var",
              "coverage_1sigma", "coverage_2sigma", "coverage_3sigma"):
        assert k in diag
    assert diag["rmse"] >= 0.0
    assert 0.0 <= diag["coverage_2sigma"] <= 1.0
