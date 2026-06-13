"""
Gaussian process surrogates for the forward model.

GPSurrogate        — scalar surrogate of the log-likelihood surface.
MultiOutputSurrogate — independent GP per observable (θ → η), for KOH.

Extracted from bayesian_inference.py (2026-06-12 modularization).
"""

import time

import numpy as np
import GPy


# Gaussian Process Surrogate

class GPSurrogate:
    """
    Gaussian process surrogate for the forward model log-likelihood surface.

    Trained on (θ, loglik) pairs from the CFD ensemble.  Once trained,
    provides cheap (μ, σ²) predictions at any θ, enabling MCMC with
    ~10^4 posterior evaluations at negligible cost.

    Uses GPy with ARD (automatic relevance determination) RBF kernel
    to automatically identify which SST parameters dominate the
    log-likelihood variation.
    """

    def __init__(self):
        self.gp = None
        self.X_train = None
        self.y_train = None
        self.trained = False
        self._train_time = 0.0

    def train(self, X, y, optimize_restarts=3, noise_floor=None):
        """
        Train GP on ensemble data

        Parameters
        X: ndarray, shape (n_train, ndim)
            Parameter vectors from ensemble
        y: ndarray, shape (n_train,)
            Log-likelihood values from forward model
        optimize_restarts : int
            Number of random restarts for kernel hyperparameter optimization
            More restarts reduce risk of local optima in marginal likelihood
        noise_floor : float | None
            Lower bound on the Gaussian-likelihood variance (in normalized-target
            units).  CFD log-likelihoods are deterministic, so the optimizer drives the
            noise to ~0 and the GP *interpolates* — giving a wildly overconfident
            posterior (the V.1 surrogate breakdown).  A small floor (e.g. 1e-3) keeps the
            predictive variance calibrated and the mean lightly regularized.  Default
            ``None`` preserves the legacy (interpolating) behaviour.
        """
        self.X_train = X.copy()

        # Standardize y to zero mean / unit std so kernel hyperparameters
        # are on a sensible scale regardless of absolute log-likelihood values.
        self._y_mean = float(np.mean(y))
        self._y_std  = float(np.std(y)) if np.std(y) > 1e-10 else 1.0
        y_norm = (y - self._y_mean) / self._y_std

        self.y_train = y_norm.reshape(-1, 1)

        ndim = X.shape[1]
        kernel = GPy.kern.RBF(ndim, ARD=True)

        t0 = time.time()
        self.gp = GPy.models.GPRegression(X, self.y_train, kernel)
        if noise_floor is not None:
            # bound the noise variance below so it cannot collapse to interpolation
            self.gp.likelihood.variance.constrain_bounded(
                float(noise_floor), 1e6, warning=False)
        self.gp.optimize_restarts(
            num_restarts=optimize_restarts,
            messages=False, verbose=False
        )
        self._train_time = time.time() - t0
        self.trained = True

    def predict(self, theta):
        """
        Predict (mean, variance) at a single parameter vector
        Returns
        mu: float
            Predicted log-likelihood (in original scale)
        var: float
            Predictive variance (in original scale)
        """
        assert self.trained, "Surrogate not trained"
        X = np.asarray(theta).reshape(1, -1)
        mu_n, var_n = self.gp.predict(X)
        mu  = float(mu_n[0, 0])  * self._y_std + self._y_mean
        var = float(var_n[0, 0]) * self._y_std ** 2
        return mu, var

    def predict_batch(self, Theta):
        """Predict (means, variances) for a batch of parameter vectors (original scale)"""
        assert self.trained, "Surrogate not trained"
        Theta = np.atleast_2d(Theta)
        mu_n, var_n = self.gp.predict(Theta)
        return mu_n.ravel() * self._y_std + self._y_mean, var_n.ravel() * self._y_std ** 2

    def log_likelihood(self, theta):
        """Surrogate log-likelihood (mean prediction, ignoring variance)"""
        mu, _ = self.predict(theta)
        return mu

    def gradient(self, theta):
        """
        PHASE 2 — analytic gradient ∂μ/∂θ of the surrogate mean (original scale).

        The bulk of all sampling runs on this surrogate, so its derivative is the
        highest-leverage gradient in the program and costs nothing extra.  GPy's
        ``predictive_gradients`` gives ∂μ_norm/∂θ in *normalized-target* units (the
        inputs θ are passed to GPy unscaled); we rescale by the stored target std to
        return the gradient in original log-likelihood units.

        Verified against finite differences (rel. err < 1e-6) in the Phase-2 gate.

        Returns
        -------
        grad : ndarray, shape (ndim,)
            ∂(mean log-lik)/∂θ at ``theta``.
        """
        assert self.trained, "Surrogate not trained"
        X = np.asarray(theta, dtype=float).reshape(1, -1)
        dmu_n, _ = self.gp.predictive_gradients(X)   # (1, ndim, 1)
        return np.asarray(dmu_n[0, :, 0], dtype=float) * self._y_std

    def mean_and_gradient(self, theta):
        """Convenience: (mean log-lik, ∂mean/∂θ) in original scale."""
        return self.log_likelihood(theta), self.gradient(theta)

    def rmse(self, X_test, y_test):
        """Root mean squared error on holdout set"""
        mu, _ = self.predict_batch(X_test)
        return float(np.sqrt(np.mean((mu - y_test) ** 2)))

    def lengthscales(self):
        """
        ARD lengthscales from the RBF kernel.

        Short lengthscale -> log-likelihood varies rapidly with that parameter(high sensitivity)
        Long lengthscale -> parameter has little effect
        Useful for identifying which SST coefficients actually matter
        """
        if not self.trained:
            return None
        return self.gp.kern.lengthscale.values.copy()


# ── Kennedy-O'Hagan Model-Form Uncertainty ──────────────────────────────────

class MultiOutputSurrogate:
    """
    Independent Gaussian process per observable output.

    Maps θ → η vector (raw CFD predictions at observation locations), training
    one GP per output with ARD RBF kernel.  Normalises each output channel
    independently so poorly-conditioned kernel hyperparameters are avoided.

    Use in place of GPSurrogate when the KOH likelihood needs the prediction
    vector η rather than the scalar log-likelihood.
    """

    def __init__(self):
        self.gps = []
        self.n_outputs = 0
        self._y_means = None
        self._y_stds  = None
        self.trained = False
        self._train_time = 0.0

    def train(self, X, Y, optimize_restarts=3, noise_floor=None):
        """
        X: (n, d_theta)
        Y: (n, n_obs)  — raw CFD predictions, one column per observable
        noise_floor : float | None
            Per-output lower bound on the GP noise variance (normalized units) — the
            V.1 surrogate-trust fix (prevents the interpolating/overconfident η fit).
        """
        n, d = X.shape
        self.n_outputs = Y.shape[1]
        self._y_means = np.mean(Y, axis=0)
        self._y_stds  = np.std(Y, axis=0)
        self._y_stds[self._y_stds < 1e-10] = 1.0

        Y_norm = (Y - self._y_means) / self._y_stds

        import warnings
        t0 = time.time()
        self.gps = []
        for k in range(self.n_outputs):
            kernel = GPy.kern.RBF(d, ARD=True)
            gp = GPy.models.GPRegression(X, Y_norm[:, k:k+1], kernel)
            if noise_floor is not None:
                gp.likelihood.variance.constrain_bounded(
                    float(noise_floor), 1e6, warning=False)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                gp.optimize_restarts(num_restarts=optimize_restarts,
                                     messages=False, verbose=False)
            self.gps.append(gp)

        self._train_time = time.time() - t0
        self.trained = True

    def predict(self, theta):
        """Predict η at a single θ. Returns (mu, var) each shape (n_obs,)."""
        assert self.trained
        X = np.asarray(theta).reshape(1, -1)
        mu  = np.zeros(self.n_outputs)
        var = np.zeros(self.n_outputs)
        for k, gp in enumerate(self.gps):
            mu_k, var_k = gp.predict(X)
            mu[k]  = float(mu_k[0, 0])  * self._y_stds[k] + self._y_means[k]
            var[k] = float(var_k[0, 0]) * self._y_stds[k] ** 2
        return mu, var

    def predict_batch(self, Theta):
        """Returns (means, vars) each shape (n, n_obs)."""
        assert self.trained
        Theta = np.atleast_2d(Theta)
        n = len(Theta)
        means = np.zeros((n, self.n_outputs))
        vars_ = np.zeros((n, self.n_outputs))
        for k, gp in enumerate(self.gps):
            mu_k, var_k = gp.predict(Theta)
            means[:, k] = mu_k.ravel() * self._y_stds[k] + self._y_means[k]
            vars_[:, k] = var_k.ravel() * self._y_stds[k] ** 2
        return means, vars_

    def rmse(self, X_test, Y_test):
        """Per-output RMSE (original scale) on a holdout set."""
        means, _ = self.predict_batch(X_test)
        return np.sqrt(np.mean((means - Y_test) ** 2, axis=0))
