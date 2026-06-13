"""
Kennedy–O'Hagan model-form uncertainty likelihood.

Extracted from bayesian_inference.py (2026-06-12 modularization).
"""

import numpy as np


KOH_MODES = ("diagonal", "physical_gp")


class KOHLikelihood:
    """
    Kennedy-O'Hagan (2001) model-form uncertainty likelihood.

    Assumes:
        y_obs(x) = η(θ, x) + δ(x) + ε
        δ ~ GP(0, σ_δ² · k(x, x'; l_δ))   model discrepancy
        ε ~ N(0, σ_ε²)                      measurement noise

    Marginalising δ gives a Gaussian with covariance
        C = σ_δ² · K_δ + diag(σ_ε²)

    and log-likelihood
        log p(y | η, σ_δ, l_δ) = -½ [r^T C⁻¹ r + log|C| + n log 2π]

    where r = y_obs − η.

    Modes
    -----
    The structure of K_δ is controlled by ``mode``:

      - ``"diagonal"`` (PHASE 3 baseline)
            K_δ = I_n.  Each observation gets an independent discrepancy of
            common amplitude σ_δ; lengthscale has no effect and is fixed.
            Inference space is therefore just θ + log σ_δ.
      - ``"physical_gp"`` (PHASE 3 default)
            K_δ(x, x') = exp(-½ |x-x'|² / l_δ²) on the supplied physical
            locations.  This is the most-correct long-term form; inference
            space is θ + log σ_δ + log l_δ.

    Use ``"diagonal"`` as the simple baseline and ``"physical_gp"`` once you
    have spatially-aware observation locations.

    Parameters
    ----------
    obs_locations : array-like, shape (n,) or (n, d)
        Spatial coordinates of observations (e.g. x/h values for BFS).
    obs_values : array-like, shape (n,)
        Experimental or synthetic observations.
    obs_sigmas : array-like, shape (n,)
        Per-observation measurement-noise standard deviations.
    mode : {"diagonal", "physical_gp"}, default "physical_gp"
        Discrepancy kernel structure.
    """

    def __init__(self, obs_locations, obs_values, obs_sigmas,
                 mode: str = "physical_gp"):
        locs = np.asarray(obs_locations, dtype=float)
        if locs.ndim == 1:
            locs = locs.reshape(-1, 1)
        self.x          = locs                          # (n, d)
        self.y          = np.asarray(obs_values, float) # (n,)
        self.sigma_eps  = np.asarray(obs_sigmas, float) # (n,)
        self.n          = len(self.y)

        if self.x.shape[0] != self.n or self.sigma_eps.shape[0] != self.n:
            raise ValueError(
                f"KOHLikelihood: obs_locations ({self.x.shape[0]}), "
                f"obs_values ({self.n}), and obs_sigmas "
                f"({self.sigma_eps.shape[0]}) must have the same length"
            )
        if not np.all(np.isfinite(self.y)):
            raise ValueError("KOHLikelihood: obs_values must be finite")
        if not np.all(np.isfinite(self.sigma_eps)) or np.any(self.sigma_eps <= 0):
            raise ValueError(
                "KOHLikelihood: obs_sigmas must be finite and strictly positive"
            )
        if mode not in KOH_MODES:
            raise ValueError(
                f"KOHLikelihood mode {mode!r} not in {KOH_MODES}"
            )
        self.mode = mode

        self._Sigma_eps = np.diag(self.sigma_eps ** 2)
        self._I = np.eye(self.n)

    def _kernel(self, l_delta):
        """Squared-exponential kernel matrix K(x, x'; l_δ).

        For ``diagonal`` mode the matrix is identity regardless of l_delta;
        for ``physical_gp`` mode it uses the supplied physical locations.
        """
        if self.mode == "diagonal":
            return self._I
        l = np.atleast_1d(np.asarray(l_delta, float))
        diff = (self.x[:, None, :] - self.x[None, :, :]) / l  # (n, n, d)
        return np.exp(-0.5 * np.sum(diff ** 2, axis=-1))

    @property
    def n_extra_params(self) -> int:
        """Number of extra hyperparameters this mode adds to MCMC.

        ``diagonal``    -> 1 (just log σ_δ)
        ``physical_gp`` -> 2 (log σ_δ and log l_δ)
        """
        return 1 if self.mode == "diagonal" else 2

    def __call__(self, eta, log_sigma_delta, log_l_delta):
        """
        Compute KOH log-likelihood.

        Returns -np.inf for non-finite eta/hyperparameters or length mismatch,
        so MCMC proposals can rely on -inf to reject the sample without a
        hard exception.

        Parameters
        ----------
        eta : array (n_obs,)
            Surrogate or forward-model predictions at observation locations.
        log_sigma_delta : float
            Log of discrepancy amplitude σ_δ.
        log_l_delta : float or array
            Log of discrepancy lengthscale(s) l_δ.
        """
        eta = np.asarray(eta, float)
        if eta.shape != self.y.shape or not np.all(np.isfinite(eta)):
            return -np.inf

        lsd = float(log_sigma_delta)
        lld = np.asarray(log_l_delta, float)
        if not (np.isfinite(lsd) and np.all(np.isfinite(lld))):
            return -np.inf

        sigma_delta = np.exp(lsd)
        l_delta     = np.exp(lld)
        if not np.all(l_delta > 0) or not np.isfinite(sigma_delta):
            return -np.inf

        K = self._kernel(l_delta)
        C = sigma_delta ** 2 * K + self._Sigma_eps
        C += 1e-10 * np.eye(self.n)   # jitter

        r = self.y - eta

        try:
            L     = np.linalg.cholesky(C)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, r))
            ld    = 2.0 * np.sum(np.log(np.diag(L)))
            ll    = -0.5 * (r @ alpha + ld + self.n * np.log(2.0 * np.pi))
            return float(ll) if np.isfinite(ll) else -np.inf
        except np.linalg.LinAlgError:
            return -np.inf

    def gradient(self, eta, log_sigma_delta, log_l_delta):
        """
        PHASE 5 — analytic gradient of the KOH log-likelihood (it is differentiable).

        Returns ``(dL_deta, dL_dlog_sigma_delta, dL_dlog_l_delta)`` where

            dL/dη         = C⁻¹ r = α            (chain with ∂η/∂θ for dL/dθ = Jᵀα)
            dL/dlog σ_δ   = ½[ αᵀ(∂C/∂logσ)α − tr(C⁻¹ ∂C/∂logσ) ],  ∂C/∂logσ = 2σ_δ²K
            dL/dlog l_δ   = ½[ αᵀ(∂C/∂logl)α − tr(C⁻¹ ∂C/∂logl) ],  ∂K/∂logl = K·(D²/l²)

        (the standard GP marginal-likelihood hyperparameter gradient).  In ``diagonal``
        mode l_δ has no effect so dL/dlog l_δ = 0.  Non-finite inputs return zeros so a
        NUTS step rejects cleanly.  Verified vs. finite differences (rel err < 1e-4).
        """
        eta = np.asarray(eta, float)
        if eta.shape != self.y.shape or not np.all(np.isfinite(eta)):
            return np.zeros(self.n), 0.0, 0.0
        lsd = float(log_sigma_delta)
        lld = np.asarray(log_l_delta, float)
        if not (np.isfinite(lsd) and np.all(np.isfinite(lld))):
            return np.zeros(self.n), 0.0, 0.0

        sigma_delta = np.exp(lsd)
        l_delta = np.exp(lld)
        K = self._kernel(l_delta)
        C = sigma_delta ** 2 * K + self._Sigma_eps + 1e-10 * self._I
        r = self.y - eta
        try:
            Cinv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            return np.zeros(self.n), 0.0, 0.0
        alpha = Cinv @ r

        # log σ_δ  (∂C/∂logσ = 2 σ_δ² K)
        dC_ds = 2.0 * sigma_delta ** 2 * K
        g_s = 0.5 * (alpha @ dC_ds @ alpha - np.sum(Cinv * dC_ds))

        # log l_δ  (physical_gp only)
        if self.mode == "physical_gp":
            l = float(np.atleast_1d(l_delta)[0])
            D2 = np.sum((self.x[:, None, :] - self.x[None, :, :]) ** 2, axis=-1)
            dC_dl = sigma_delta ** 2 * (K * (D2 / l ** 2))
            g_l = 0.5 * (alpha @ dC_dl @ alpha - np.sum(Cinv * dC_dl))
        else:
            g_l = 0.0

        return alpha, float(g_s), float(g_l)

    def discrepancy_mean(self, eta, log_sigma_delta, log_l_delta):
        """
        PHASE 6 / V.5 — posterior-mean model-form discrepancy at the observation
        locations (the *irreducible* residual after the best parameters):

            δ̂(x_obs) = σ_δ² K_δ(l_δ) C⁻¹ (y − η),   C = σ_δ² K_δ + diag(σ_ε²).

        Averaged over the KOH posterior this localizes *where along the wall* the
        evidence-preferred model is most wrong.  Returns zeros on non-finite inputs.
        """
        eta = np.asarray(eta, float)
        if eta.shape != self.y.shape or not np.all(np.isfinite(eta)):
            return np.zeros(self.n)
        lsd = float(log_sigma_delta)
        lld = np.asarray(log_l_delta, float)
        if not (np.isfinite(lsd) and np.all(np.isfinite(lld))):
            return np.zeros(self.n)
        sigma_delta = np.exp(lsd)
        K = self._kernel(np.exp(lld))
        C = sigma_delta ** 2 * K + self._Sigma_eps + 1e-10 * self._I
        r = self.y - eta
        try:
            alpha = np.linalg.solve(C, r)
        except np.linalg.LinAlgError:
            return np.zeros(self.n)
        return sigma_delta ** 2 * (K @ alpha)
