"""
Bayesian calibration with Kennedy–O'Hagan model-form discrepancy.

Extracted from bayesian_inference.py (2026-06-12 modularization).
"""

import time

import numpy as np

from priors import Prior, make_prior_from_param_set
from design import latin_hypercube
from gp_surrogate import MultiOutputSurrogate
from param_utils import _get_param_names


class BayesianInferenceKOH:
    """
    Bayesian calibration with Kennedy-O'Hagan model-form discrepancy.

    Extended parameter space: [θ₁, …, θₙ, log σ_δ, log l_δ]

    The ensemble evaluates the C++ forward model to collect (θ, η) pairs.
    A MultiOutputSurrogate approximates θ → η cheaply.  MCMC then samples
    the full joint posterior over physical parameters and discrepancy
    hyperparameters simultaneously.

    Parameters
    ----------
    forward_model : ForwardModel
        C++ forward model; must expose evaluate() returning result.predictions.
    param_set : InferenceParameterSet
        Active SST parameters.
    koh_likelihood : KOHLikelihood
        Configured with the observation locations, values, and noise.
    theta_prior : Prior, optional
        Prior over θ (default: truncated-normal from param_set defaults).
    log_sigma_delta_prior : (mean, std)
        Normal prior on log σ_δ.  Default centres σ_δ ≈ 0.14.
    log_l_delta_prior : (mean, std)
        Normal prior on log l_δ (in observation-location units).
    """

    def __init__(self, forward_model, param_set, koh_likelihood,
                 theta_prior=None,
                 log_sigma_delta_prior=(-2.0, 2.0),
                 log_l_delta_prior=(0.0, 2.0)):
        self.forward_model = forward_model
        self.param_set     = param_set
        self.koh           = koh_likelihood

        if theta_prior is None:
            theta_prior = make_prior_from_param_set(param_set)
        self.theta_prior = theta_prior
        self.n_theta     = theta_prior.ndim

        lsd_m, lsd_s = log_sigma_delta_prior
        lld_m, lld_s = log_l_delta_prior

        # Extra hyperparameter dimension depends on KOH mode:
        #   diagonal    -> [log σ_δ]                    (1 extra dim)
        #   physical_gp -> [log σ_δ, log l_δ]           (2 extra dims)
        self.n_extra = self.koh.n_extra_params
        if self.n_extra == 1:
            extra_means = [lsd_m]
            extra_stds  = [lsd_s]
            extra_lo    = [lsd_m - 5*lsd_s]
            extra_hi    = [lsd_m + 5*lsd_s]
        else:
            extra_means = [lsd_m, lld_m]
            extra_stds  = [lsd_s, lld_s]
            extra_lo    = [lsd_m - 5*lsd_s, lld_m - 5*lld_s]
            extra_hi    = [lsd_m + 5*lsd_s, lld_m + 5*lld_s]

        ext_means = np.append(theta_prior.means, extra_means)
        ext_stds  = np.append(theta_prior.stds,  extra_stds)
        ext_lower = np.append(theta_prior.lower, extra_lo)
        ext_upper = np.append(theta_prior.upper, extra_hi)

        self.prior          = Prior(ext_means, ext_stds, ext_lower, ext_upper)
        self.multi_surrogate = MultiOutputSurrogate()

        self.ensemble_X = None   # (n_valid, n_theta)
        self.ensemble_Y = None   # (n_valid, n_obs) — raw predictions
        self.samples    = None   # (n_samples, n_theta + n_extra)
        self.sampler    = None

    # Ensemble

    def run_ensemble(self, n_samples=200, verbose=True):
        """
        Latin-hypercube sample over θ, evaluate forward model, collect η vectors.

        If the forward model exposes a precomputed_ensemble() method that returns
        (X, Y) arrays, those are used directly and LHC sampling is skipped.
        """
        pre = getattr(self.forward_model, "precomputed_ensemble", None)
        if pre is not None:
            xy = pre()
            if xy is not None:
                self.ensemble_X, self.ensemble_Y = xy[0], xy[1]
                if verbose:
                    print(f"  Using precomputed ensemble: "
                          f"{len(self.ensemble_X)} samples")
                return self.ensemble_X, self.ensemble_Y

        X      = latin_hypercube(n_samples, self.n_theta,
                                  self.theta_prior.lower, self.theta_prior.upper)
        valid_X, valid_Y = [], []
        t0     = time.time()

        for i in range(n_samples):
            theta_list = X[i].tolist()
            result     = self.forward_model.evaluate(theta_list)
            preds      = np.array(result.predictions) if result.predictions else None
            ok         = (preds is not None
                          and len(preds) == self.koh.n
                          and np.all(np.isfinite(preds)))
            if ok:
                valid_X.append(X[i])
                valid_Y.append(preds)

            if verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  Ensemble {i+1}/{n_samples}  "
                      f"valid={len(valid_X)}  [{(i+1)/elapsed:.1f} eval/s]")

        elapsed    = time.time() - t0
        n_valid    = len(valid_X)
        n_diverged = n_samples - n_valid

        self.ensemble_X = np.array(valid_X)
        self.ensemble_Y = np.array(valid_Y)

        if verbose:
            print(f"\n  Ensemble complete: {n_valid}/{n_samples} valid "
                  f"({n_diverged} diverged)  [{elapsed:.1f}s total]")

        return self.ensemble_X, self.ensemble_Y

    # Surrogate

    def train_surrogate(self, holdout_frac=0.1, verbose=True, optimize_restarts=3,
                        noise_floor=None):
        """Train multi-output surrogate θ → η (noise_floor = V.1 surrogate-trust fix)."""
        assert self.ensemble_X is not None and len(self.ensemble_X) >= 10

        n      = len(self.ensemble_X)
        n_test = max(1, int(n * holdout_frac))
        idx    = np.random.permutation(n)

        X_tr = self.ensemble_X[idx[n_test:]]
        Y_tr = self.ensemble_Y[idx[n_test:]]
        X_te = self.ensemble_X[idx[:n_test]]
        Y_te = self.ensemble_Y[idx[:n_test]]

        self.multi_surrogate.train(X_tr, Y_tr, optimize_restarts=optimize_restarts,
                                   noise_floor=noise_floor)
        rmse = self.multi_surrogate.rmse(X_te, Y_te)

        if verbose:
            names = _get_param_names(self.param_set)
            print(f"  KOH surrogate: {len(X_tr)} train, {n_test} holdout  "
                  f"[{self.multi_surrogate._train_time:.2f}s]")
            print(f"  Per-output RMSE: {' '.join(f'{r:.4f}' for r in rmse)}")

        return rmse

    # Log-posterior

    def log_posterior(self, extended_theta):
        """log p([θ, KOH hyperparams] | y) ∝ log prior + KOH log-lik.

        The KOH hyperparam vector is mode-dependent:
          - diagonal    -> [log σ_δ]                  (l_δ ignored)
          - physical_gp -> [log σ_δ, log l_δ]
        """
        lp = self.prior.log_prior(extended_theta)
        if not np.isfinite(lp):
            return -np.inf

        theta          = extended_theta[:self.n_theta]
        log_sigma_delta = extended_theta[self.n_theta]
        if self.n_extra == 2:
            log_l_delta = extended_theta[self.n_theta + 1]
        else:
            # diagonal mode: lengthscale not in the inference space.
            log_l_delta = 0.0

        eta, _ = self.multi_surrogate.predict(theta)
        ll      = self.koh(eta, log_sigma_delta, log_l_delta)

        return lp + ll if np.isfinite(ll) else -np.inf

    # MCMC

    def run_mcmc(self, n_walkers=None, n_steps=500, burn_in=100, thin=1,
                 verbose=True, *,
                 parallel=False, pool=None, n_processes=None, rng_seed=None):
        """Sample [θ, KOH hyperparams] with affine-invariant ensemble MCMC.

        Parallelism (PHASE 5) is opt-in: ``parallel=True`` evaluates walker
        log-posteriors with a ``multiprocess.Pool``.  See
        ``parallel_mcmc.run_emcee`` for trade-offs.
        """
        from parallel_mcmc import run_emcee

        ndim = self.n_theta + self.n_extra
        if n_walkers is None:
            n_walkers = max(16, 2 * ndim)

        if verbose:
            mode = "parallel" if (parallel or pool is not None) else "serial"
            print(f"  KOH MCMC ({mode}): {n_walkers} walkers × {n_steps} steps "
                  f"(burn-in={burn_in}, ndim={ndim})")

        self.samples, info = run_emcee(
            log_posterior=self.log_posterior,
            prior=self.prior,
            n_walkers=n_walkers, n_steps=n_steps,
            burn_in=burn_in, thin=thin,
            parallel=parallel, pool=pool, n_processes=n_processes,
            progress=verbose, rng_seed=rng_seed,
        )
        self.sampler = info["sampler"]

        if verbose:
            print(f"\n  MCMC complete: {len(self.samples)} samples "
                  f"[{info['elapsed_s']:.1f}s]")
            print(f"  Acceptance fraction: {info['acceptance_mean']:.3f} "
                  f"(range [{info['acceptance_min']:.3f}, "
                  f"{info['acceptance_max']:.3f}])")

        return self.samples

    # Summaries

    def _extra_param_names(self):
        return (['log_sigma_delta'] if self.n_extra == 1
                else ['log_sigma_delta', 'log_l_delta'])

    def posterior_summary(self):
        """Posterior statistics for all extended parameters."""
        names = _get_param_names(self.param_set) + self._extra_param_names()
        summary = {}
        for i, name in enumerate(names):
            s = self.samples[:, i]
            shift = ((np.mean(s) - self.prior.means[i])
                     / max(self.prior.stds[i], 1e-10))
            summary[name] = {
                'mean':       float(np.mean(s)),
                'std':        float(np.std(s)),
                'ci_2.5':     float(np.percentile(s, 2.5)),
                'ci_97.5':    float(np.percentile(s, 97.5)),
                'prior_mean': float(self.prior.means[i]),
                'shift':      float(shift),
            }
        return summary

    def print_summary(self):
        """Formatted posterior table including discrepancy hyperparameters."""
        summary = self.posterior_summary()
        names   = _get_param_names(self.param_set) + self._extra_param_names()

        print(f"\n  {'Parameter':>16s}  {'Prior μ':>8s}  {'Posterior μ':>11s}  "
              f"{'±σ':>7s}  {'95% CI':>18s}  {'Shift':>6s}")
        print("  " + "-" * 80)

        for name in names:
            s = summary[name]
            print(f"  {name:>16s}  {s['prior_mean']:8.4f}  {s['mean']:11.4f}  "
                  f"{s['std']:7.4f}  [{s['ci_2.5']:.4f}, {s['ci_97.5']:.4f}]  "
                  f"{s['shift']:+6.2f}σ")

        # Natural-scale discrepancy summary
        print(f"\n  KOH mode: {self.koh.mode}")
        sigma_d = np.exp(self.samples[:, self.n_theta])
        print(f"  σ_δ (natural scale):  {np.mean(sigma_d):.4f} ± {np.std(sigma_d):.4f}")
        if self.n_extra == 2:
            l_d = np.exp(self.samples[:, self.n_theta + 1])
            print(f"  l_δ (natural scale):  {np.mean(l_d):.4f} ± {np.std(l_d):.4f}")
        else:
            print("  l_δ:  (not used in 'diagonal' mode)")
