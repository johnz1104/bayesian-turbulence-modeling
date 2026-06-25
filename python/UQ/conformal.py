"""Conformal prediction: distribution-free predictive intervals with
finite-sample marginal coverage.

Implements split (inductive) conformal, cross-conformal, and conformalized
quantile regression (CQR). The coverage guarantee requires only exchangeability
of the calibration data, so the honest treatment of a distribution shift is to
report the coverage gap rather than hide it; ``conformal_coverage_gap`` measures
exactly that. The routines are model-agnostic: they consume point or quantile
predictions, so any regressor (or none, in the synthetic tests) can drive them.

References: Vovk, Gammerman and Shafer (2005); Lei et al. (2018); Romano,
Patterson and Candes (2019) for CQR; Angelopoulos and Bates (2023).
"""
import numpy as np


def conformal_quantile(scores, alpha):
    """Finite-sample-adjusted (1 - alpha) quantile of nonconformity scores.

    Uses the ceil((n+1)(1-alpha))/n level that yields the marginal coverage
    guarantee P(y in C) >= 1 - alpha. If the level exceeds 1 (tiny calibration
    set) the interval is unbounded (returns +inf), the honest finite-sample
    behaviour.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return np.inf
    return float(np.quantile(scores, level, method="higher"))


def split_conformal_intervals(pred_cal, y_cal, pred_test, alpha=0.1):
    """Split-conformal intervals around a point predictor.

    Nonconformity score = absolute residual on the calibration set. Returns
    (lower, upper) for the test predictions with marginal coverage >= 1 - alpha.
    """
    scores = np.abs(np.asarray(y_cal) - np.asarray(pred_cal))
    q = conformal_quantile(scores, alpha)
    pred_test = np.asarray(pred_test, dtype=float)
    return pred_test - q, pred_test + q


def cqr_intervals(qlo_cal, qhi_cal, y_cal, qlo_test, qhi_test, alpha=0.1):
    """Conformalized quantile regression (Romano et al. 2019).

    Given lower/upper conditional-quantile predictions on a calibration and a
    test set, conformalise them so the marginal coverage guarantee holds while
    keeping the adaptivity (varying width) of the quantile model.
    """
    y_cal = np.asarray(y_cal, dtype=float)
    E = np.maximum(np.asarray(qlo_cal) - y_cal, y_cal - np.asarray(qhi_cal))
    q = conformal_quantile(E, alpha)
    return np.asarray(qlo_test) - q, np.asarray(qhi_test) + q


def cross_conformal_intervals(predict_fn, X, y, X_test, alpha=0.1, k=5, seed=0):
    """K-fold cross-conformal intervals around a point predictor.

    predict_fn(X_train, y_train, X_eval) returns predictions for X_eval after
    fitting on (X_train, y_train). Cross-conformal pools the out-of-fold
    nonconformity scores, using the data more efficiently than a single split.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)

    scores = []
    test_preds = []
    for f in folds:
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        pred_oof = predict_fn(X[mask], y[mask], X[f])
        scores.append(np.abs(y[f] - pred_oof))
        test_preds.append(predict_fn(X[mask], y[mask], X_test))
    scores = np.concatenate(scores)
    q = conformal_quantile(scores, alpha)
    pred_test = np.mean(test_preds, axis=0)
    return pred_test - q, pred_test + q


def conformal_coverage_gap(pred_cal, y_cal, pred_test, y_test, alpha=0.1):
    """Empirical test coverage of a split-conformal interval and its gap to
    the nominal 1 - alpha. A positive gap (under-coverage) on shifted data is a
    finding to report, not to hide.
    """
    lo, hi = split_conformal_intervals(pred_cal, y_cal, pred_test, alpha)
    cov = float(np.mean((y_test >= lo) & (y_test <= hi)))
    return cov, (1.0 - alpha) - cov
