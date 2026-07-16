"""Compressible-channel SST and turbulent-Prandtl calibration with calibrated
predictive UQ (the pre-registered study harness).

Mirrors the incompressible calibration pattern (ensemble, surrogates, power
posterior, conformal) with the 1-D compressible SST solve as the forward map,
per the confirmed scope decisions and the committed pre-registration:

- Inference parameters: theta = (a1, betaStar, Pr_t). The SST pair carries
  the incompressible studies' Menter-centred truncated-normal prior (15
  percent relative standard deviation, support mean +/- 3 sigma inside the
  physical box); Pr_t carries the pre-registered EXACTLY uniform prior on
  [0.5, 1.5] (CompressiblePrior overrides the Gaussian term for it).
- Likelihood observables (never the thermal targets): the mean velocity
  profile U+ and the Favre temperature profile T/T_w at log-spaced stations,
  plus the skin friction cf. Observation sigma is the MODELED relative value
  (0.5 percent primary, 1 percent sensitivity), per case floored at the
  loader-measured physics-anchor rms, never called a DNS uncertainty.
- HELD OUT (the pre-registered coverage targets, never in the likelihood):
  the wall-heat-flux parameter B_q and the wall-normal turbulent heat-flux
  profile q_hat at its own stations. The forward map predicts them alongside
  the likelihood observables, so the posterior predictive covers them
  without them ever informing the posterior.
- Corrections composed exactly as the incompressible phases: the power
  (Gibbs) posterior at a learning rate moment-matched on a calibration split
  only, with predictive observation variance inflated by 1/eta; split
  conformal from calibration-station residual quantiles. Nothing is tuned
  toward any evaluated coverage.

The forward map is fast (a fraction of a second per solve), but the ensemble
plus GP-surrogate route is kept so the machinery, diagnostics and failure
handling stay identical to the incompressible studies; surrogate quality is
reported, not assumed.
"""
import numpy as np

from priors import Prior
from design import latin_hypercube
from gp_surrogate import GPSurrogate, MultiOutputSurrogate
from parallel_mcmc import run_emcee

from .. import generalized_bayes as gb
from .. import conformal as cf
from .. import evaluation as ev
from . import observation_sigma
from .compressible_baseline import CompressibleChannelSST

# the Menter-centred SST prior of the incompressible studies (a1, betaStar),
# extended by the pre-registered uniform Pr_t box
SST_PRIOR = {
    "means": (0.31, 0.09),
    "rel_std": 0.15,
    "bounds": ((0.1, 0.8), (0.01, 0.2)),   # physical box (InferenceParameters)
}
PRT_BOX = (0.5, 1.5)


class CompressiblePrior(Prior):
    """(a1, betaStar) truncated normal; Pr_t exactly uniform on its box.

    The base Prior's Gaussian log-density applies to the SST pair only; the
    Pr_t dimension contributes zero inside the box (the pre-registered
    uniform), and sampling draws it uniformly.
    """

    @staticmethod
    def build(k_sigma=3.0):
        means = np.array(SST_PRIOR["means"] + (0.5 * sum(PRT_BOX),))
        stds = np.array([SST_PRIOR["rel_std"] * m
                         for m in SST_PRIOR["means"]] + [1.0])
        lo = np.array([max(m - k_sigma * s, b[0]) for m, s, b in
                       zip(SST_PRIOR["means"], stds, SST_PRIOR["bounds"])]
                      + [PRT_BOX[0]])
        hi = np.array([min(m + k_sigma * s, b[1]) for m, s, b in
                       zip(SST_PRIOR["means"], stds, SST_PRIOR["bounds"])]
                      + [PRT_BOX[1]])
        return CompressiblePrior(means, stds, lo, hi)

    def log_prior(self, theta):
        theta = np.asarray(theta, dtype=float)
        if np.any(theta < self.lower) or np.any(theta > self.upper) \
                or np.any(~np.isfinite(theta)):
            return -np.inf
        z = (theta[:2] - self.means[:2]) / self.stds[:2]
        return -0.5 * float(np.sum(z * z))

    def sample(self, n=1):
        out = np.empty((n, self.ndim))
        out[:, :2] = Prior(self.means[:2], self.stds[:2], self.lower[:2],
                           self.upper[:2]).sample(n)
        out[:, 2] = np.random.uniform(PRT_BOX[0], PRT_BOX[1], size=n)
        return out


class CompressibleCalibration:
    """Calibrate (a1, betaStar, Pr_t) on one compressible channel case and
    quantify held-out thermal coverage.

    Typical use mirrors the incompressible harness:
        c = CompressibleCalibration(dns)
        c.run_ensemble(n=64, seed=0)
        c.fit_surrogates()
        post = c.sample_posterior(eta=1.0)
        c.coverage_vs_truth(c.posterior_predictive(post,
                            qoi_index=c.heldout_index), c.heldout_index)
    """

    def __init__(self, dns, rel_sigma=0.005, n_stations=16, n_q_stations=10,
                 model_kwargs=None):
        self.dns = dns
        self.rel_sigma = float(rel_sigma)
        self.prior = CompressiblePrior.build()
        self.ndim = self.prior.ndim
        self.model_kwargs = dict(model_kwargs or {})

        # per-case effective relative level: max(level, loader anchor rms),
        # the pre-registered floor rule
        anchor = observation_sigma.physics_anchor(dns)
        self.rel_eff = max(self.rel_sigma, anchor["rms"])
        self.anchor = anchor

        self._select_stations(n_stations, n_q_stations)
        self.X = self.loglik = self.preds = None
        self.gp = None
        self.mo = None

    # ---- observation construction -------------------------------------------

    def _select_stations(self, n_stations, n_q_stations):
        d = self.dns
        targets = np.geomspace(3.0, 0.95 * d.re_tau, n_stations)
        idx = sorted(set(int(np.argmin(np.abs(d.yplus - t)))
                         for t in targets))
        self.idx = np.array(idx)
        q_targets = np.geomspace(10.0, 0.8 * d.re_tau, n_q_stations)
        qi = sorted(set(int(np.argmin(np.abs(d.yplus - t)))
                        for t in q_targets))
        self.q_idx = np.array(qi)

        u_true = d.U[self.idx]
        t_true = d.T[self.idx]
        q_true = d.q_hat[self.q_idx, 1]
        # skin friction on the centreline dynamic head. The GV cases state it
        # in their global table; the CKM record carries none, so it is derived
        # from the record's own centreline state, cf = 2/(rho_CL+ U_CL+^2),
        # the same definition the GV table uses (identical for GV to its
        # stated value, asserted by the harness tests)
        cf_true = d.wall.get("cf")
        if cf_true is None:
            cf_true = 2.0 / (float(d.rho[-1]) * float(d.U[-1]) ** 2)
        bq_true = d.wall["b_q"]

        # sigma: the modeled relative level on the local scale; the heat-flux
        # profile takes its scale from the profile peak (the local value
        # crosses zero), everything else from the local value itself
        u_sig = self.rel_eff * np.abs(u_true)
        t_sig = self.rel_eff * np.abs(t_true)
        q_scale = np.max(np.abs(d.q_hat[:, 1]))
        q_sig = np.full(q_true.size, self.rel_eff * q_scale)
        self.qoi_truth = np.concatenate(
            [[cf_true], u_true, t_true, [bq_true], q_true])
        self.qoi_sigma = np.concatenate(
            [[self.rel_eff * abs(cf_true)], u_sig, t_sig,
             [max(self.rel_eff * abs(bq_true), self.rel_eff * 1e-3)], q_sig])
        n_lik = 1 + 2 * self.idx.size
        self.lik_index = np.arange(n_lik)
        self.heldout_index = np.arange(n_lik, self.qoi_truth.size)
        self.qoi_names = (
            ["cf"] + [f"U+@y+={d.yplus[i]:.0f}" for i in self.idx]
            + [f"T@y+={d.yplus[i]:.0f}" for i in self.idx]
            + ["B_q"] + [f"q@y+={d.yplus[i]:.0f}" for i in self.q_idx])
        self.n_qoi = self.qoi_truth.size

    # ---- forward map ----------------------------------------------------------

    def _model(self, theta):
        d = self.dns
        pr = d.pr_molecular if d.pr_molecular is not None \
            else np.full(d.n, d.wall["pr_w"])
        coeffs = {"a1": float(theta[0]), "betaStar": float(theta[1]),
                  "Pr_t": float(theta[2])}
        return CompressibleChannelSST(
            re_tau_w=d.re_tau, m_tau=d.wall["m_tau"], gamma=d.wall["gamma_w"],
            T_mu_samples=(d.T, d.mu), T_pr_samples=(d.T, pr),
            coeffs=coeffs, **self.model_kwargs)

    def evaluate(self, theta):
        """theta -> (status, prediction vector over ALL QoIs)."""
        out = self._model(theta).solve()
        if out["status"] != "converged":
            return out["status"], None
        d = self.dns
        u_pred = np.interp(d.yplus[self.idx], out["y_plus"], out["U_plus"])
        t_pred = np.interp(d.yplus[self.idx], out["y_plus"], out["T_hat"])
        q_pred = np.interp(d.yplus[self.q_idx], out["y_plus"],
                           out["q_gdh_hat"][:, 1])
        pred = np.concatenate([[out["cf"]], u_pred, t_pred,
                               [out["b_q"]], q_pred])
        return "converged", pred

    def log_likelihood_direct(self, pred):
        """Gaussian log-likelihood over the LIKELIHOOD block only."""
        r = (pred[self.lik_index] - self.qoi_truth[self.lik_index]) \
            / self.qoi_sigma[self.lik_index]
        return -0.5 * float(np.sum(r * r))

    # ---- ensemble and surrogates ----------------------------------------------

    def run_ensemble(self, n=64, seed=0, verbose=False):
        np.random.seed(seed)
        X = latin_hypercube(n, self.ndim, self.prior.lower, self.prior.upper)
        loglik = np.full(n, -np.inf)
        preds = np.full((n, self.n_qoi), np.nan)
        for i in range(n):
            status, p = self.evaluate(X[i])
            if p is not None:
                preds[i] = p
                loglik[i] = self.log_likelihood_direct(p)
            if verbose and (i + 1) % 16 == 0:
                print(f"  ensemble {i + 1}/{n}", flush=True)
        valid = np.isfinite(loglik) & np.all(np.isfinite(preds), axis=1)
        self.X, self.loglik, self.preds = X[valid], loglik[valid], preds[valid]
        self.n_valid = int(valid.sum())
        return self.n_valid

    def fit_surrogates(self, noise_floor=1e-3, seed=0):
        """Fit both surrogates with the fit RNG pinned and the ARD
        lengthscales bounded.

        The seed makes the GPy restart draws reproducible (they otherwise
        consume the ambient global RNG, so a refit from cache is
        run-to-run nondeterministic). The lengthscale bounds span raw theta
        units (the box is order 0.1 to 1), making the zero-lengthscale
        overflow corner infeasible; on healthy fits the optimum is interior
        and the bound never binds.
        """
        np.random.seed(seed)
        bounds = (1e-3, 1e2)
        self.gp = GPSurrogate()
        self.gp.train(self.X, self.loglik, noise_floor=noise_floor,
                      lengthscale_bounds=bounds)
        self.mo = MultiOutputSurrogate()
        self.mo.train(self.X, self.preds, noise_floor=noise_floor,
                      lengthscale_bounds=bounds)

    def to_cache(self):
        return {"X": self.X, "loglik": self.loglik, "preds": self.preds,
                "qoi_truth": self.qoi_truth, "qoi_sigma": self.qoi_sigma,
                "rel_eff": self.rel_eff}

    def load_cache(self, d):
        """Rebuild surrogates from a cache dict; returns True on success.

        Refuses (returns False, loads nothing) when the cached observation
        identity (truth vector, sigma vector, relative sigma level) disagrees
        with this calibration's: such a cache describes a different
        likelihood. File-level configuration identity (ensemble size, seed) is
        the caller's cache_fingerprint check.
        """
        checks = (("qoi_truth", np.asarray(self.qoi_truth, dtype=float)),
                  ("qoi_sigma", np.asarray(self.qoi_sigma, dtype=float)),
                  ("rel_eff", np.asarray(self.rel_eff, dtype=float)))
        for key, mine in checks:
            if key in d:
                theirs = np.asarray(d[key], dtype=float)
                if theirs.shape != mine.shape or not np.allclose(theirs, mine,
                                                                 rtol=1e-10):
                    print(f"  cache REFUSED: stored {key} differs from the "
                          f"current calibration (different observation identity)")
                    return False
        self.X, self.loglik, self.preds = d["X"], d["loglik"], d["preds"]
        self.n_valid = len(self.X)
        self.fit_surrogates()
        return True

    # ---- posterior, predictive, coverage --------------------------------------

    def sample_posterior(self, eta=1.0, n_walkers=24, n_steps=2000,
                         burn_in=500, seed=0):
        log_post = gb.tempered_log_posterior(
            log_likelihood=self.gp.log_likelihood,
            log_prior=self.prior.log_prior, eta=eta)
        samples, _ = run_emcee(log_posterior=log_post, prior=self.prior,
                               n_walkers=n_walkers, n_steps=n_steps,
                               burn_in=burn_in, thin=1, progress=False,
                               rng_seed=seed)
        return samples

    def posterior_predictive(self, theta_samples, eta=1.0, qoi_index=None,
                             seed=0):
        rng = np.random.default_rng(seed)
        mu, _ = self.mo.predict_batch(theta_samples)
        sigma = self.qoi_sigma
        if qoi_index is not None:
            mu = mu[:, qoi_index]
            sigma = sigma[qoi_index]
        noise = rng.normal(size=mu.shape) * (sigma / np.sqrt(eta))[None, :]
        return mu + noise

    def coverage_vs_truth(self, predictive, qoi_index=None, level=0.9):
        truth = self.qoi_truth if qoi_index is None \
            else self.qoi_truth[qoi_index]
        return ev.coverage_from_samples(truth, predictive.T, level=level)

    def point_prediction(self, theta_samples, qoi_index=None):
        mu, _ = self.mo.predict_batch(theta_samples)
        mu = mu.mean(axis=0)
        return mu if qoi_index is None else mu[qoi_index]

    def calibrate_eta(self, theta_samples, cal_index):
        """Moment-matched learning rate from calibration-split residuals only."""
        pred = self.point_prediction(theta_samples, cal_index)
        resid = self.qoi_truth[cal_index] - pred
        sigma2 = np.mean(self.qoi_sigma[cal_index] ** 2)
        eta = sigma2 / max(np.mean(resid ** 2), 1e-300)
        return float(np.clip(eta, 1e-6, 1.0))

    def marginal_prt(self, theta_samples):
        """The Pr_t marginal posterior summary (the measured-profile check)."""
        prt = np.asarray(theta_samples)[:, 2]
        return {"mean": float(prt.mean()), "sd": float(prt.std()),
                "q05": float(np.quantile(prt, 0.05)),
                "q95": float(np.quantile(prt, 0.95)),
                "prior_sd": float((PRT_BOX[1] - PRT_BOX[0]) / np.sqrt(12.0))}
