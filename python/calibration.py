"""
Bayesian calibration pipeline (ensemble → GP surrogate → MCMC).

Extracted from bayesian_inference.py (2026-06-12 modularization).
"""

import time

import numpy as np

from priors import make_prior_from_param_set
from design import latin_hypercube
from gp_surrogate import GPSurrogate
from param_utils import _get_param_names


class BayesianInference:
    """
    Bayesian calibration pipeline for SST turbulence coefficients.

    Orchestration:
    - Prior specification
    - Ensemble generation
    - GP surrogate training
    - MCMC sampling
    - Posterior diagnostics

    Parameters:
    - forward_model: object
        C++ ForwardModel (via pybind11) or any callable with penalized_log_likelihood(theta) -> float.
    param_set: object
        InferenceParameterSet - defines which SST coefficients are active
    prior: Prior, optional
        If None, constructs truncated normal prior from param_set defaults
    """

    def __init__(self, forward_model, param_set, prior = None):
        self.forward_model = forward_model
        self.param_set = param_set
        self.prior = prior if prior is not None else make_prior_from_param_set(param_set)
        self.surrogate = GPSurrogate()

        # ensemble storage
        self.ensemble_X = None      # shape (n_valid, ndim)
        self.ensemble_y = None      # shape (n_valid,)
        self.ensemble_status = None # list of EvaluationStatus per run

        #MCMC storage
        self.samples = None         # shape (n_samples, ndim)
        self.sampler = None         # emcee sampler object (for diagnostics)


    def run_ensemble(self, n_samples = 200, verbose = True):
        """
        Builds training dataset for Gaussian Process surrogate
        Generate Latin hypercube design and evaluate forward model.

        Each evaluation runs the ful C++ pipeline:
        * given theta -> unpack SST coefficients -> SIMPLE solve (AMG pressure, BiCGSTAB momentum)
        -> observation operator -> Gaussian log-likelihood

        Failed evaluations (diverged, unconverged, invalid parameters) are filtered before surrogatr training, but count is reported

        Parameters:
        - n_samples: int
            Number of ensemble members, with 200 as default
                - number of ensemble members should be proportional to number of active parameters
        - verbose: bool
            Prints progress every 20 evalutations

        Returns:
        - X: ndarray, shape (n_valid, ndim)
        - Y: ndarray, shape (n_valid)
        """
        ndim = self.prior.ndim
        X = latin_hypercube(n_samples, ndim, self.prior.lower, self.prior.upper)
        y = np.full(n_samples, -np.inf)
        statuses = []

        t0 = time.time()
        for i in range(n_samples):
            theta_list = X[i].tolist()

            # use full evaluate() if available to get status info
            if hasattr(self.forward_model, 'evaluate'):
                result = self.forward_model.evaluate(theta_list)
                y[i] = result.log_lik
                statuses.append(str(result.status))
            else:
                y[i] = self.forward_model.penalized_log_likelihood(theta_list)
                statuses.append("Unknown")

            if verbose and (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  Ensemble {i+1}/{n_samples}  "
                      f"loglik={y[i]:.4f}  "
                      f"[{rate:.1f} eval/s]")

        # filter out failed evaluations
        valid = y>-1e5
        self.ensemble_X = X[valid]
        self.ensemble_y = y[valid]
        self.ensemble_status = statuses

        n_valid = int(np.sum(valid))
        n_diverged = n_samples - n_valid
        elapsed = time.time() - t0

        if verbose:
            print(f"\n  Ensemble complete: {n_valid}/{n_samples} valid "
                  f"({n_diverged} diverged/invalid)  "
                  f"[{elapsed:.1f}s total]")

        return self.ensemble_X, self.ensemble_y

    # Surrogate training

    def train_surrogate(self, holdout_frac = 0.1, verbose = True):
        """
        Train GP surrogate on ensemble data without holdout validation.

        Parameters:
        - holdout_frac: float
            Fraction of ensemble reserved for validation (default 10%)
        - verbose: bool
            Print training summary

        Returns:
        rmse: float
            Holdout RMSE (units of log-likelihood)
        """

        assert self.ensemble_X is not None, \
            "No ensemble data — call run_ensemble() first"

        n = len(self.ensemble_X)
        assert n >= 10, \
            f"Only {n} valid ensemble points — need at least 10"

        n_test = max(1, int(n * holdout_frac))
        idx = np.random.permutation(n)

        X_train = self.ensemble_X[idx[n_test:]]
        y_train = self.ensemble_y[idx[n_test:]]
        X_test  = self.ensemble_X[idx[:n_test]]
        y_test  = self.ensemble_y[idx[:n_test]]

        self.surrogate.train(X_train, y_train)
        rmse = self.surrogate.rmse(X_test, y_test)

        if verbose:
            print(f"  Surrogate trained: {len(X_train)} train, "
                  f"{n_test} holdout")
            print(f"  Holdout RMSE:      {rmse:.4f}")
            print(f"  Training time:     {self.surrogate._train_time:.2f}s")

            ls = self.surrogate.lengthscales()
            if ls is not None:
                names = _get_param_names(self.param_set)
                print(f"  ARD lengthscales:")
                for name, l in zip(names, ls):
                    print(f"    {name:12s}  {l:.4f}")

        return rmse


    # Log-Posterior MCMC

    def log_posterior(self, theta):
        """
        log p(θ|y) ∝ log p(θ) + log p(y|θ)

        Prior: truncated normal on SST coefficients.
        Likelihood: GP surrogate prediction.
        """
        lp = self.prior.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.surrogate.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll


    # MCMC Sampling

    def run_mcmc(self, n_walkers=32, n_steps=5000, burn_in=1000, thin=1,
                 verbose=True, *,
                 parallel=False, pool=None, n_processes=None, rng_seed=None):
        """
        Run emcee affine-invariant ensemble sampler.

        Walkers are initialised near the prior mean with small random
        perturbations.  The sampler uses the GP surrogate for likelihood
        evaluations (~μs per call instead of ~minutes for full CFD).

        Parameters
        ----------
        n_walkers : int
            Number of walkers (must be >= 2 * ndim).
        n_steps : int
            Total MCMC steps per walker.
        burn_in : int
            Steps to discard as burn-in.
        thin : int
            Thinning factor for final samples.
        verbose : bool
            Print progress and diagnostics.
        parallel : bool, default False  (PHASE 5)
            If True, evaluate walker log-posteriors via a multiprocess.Pool.
            Pickling overhead is non-trivial; only beneficial when
            ``log_posterior`` is expensive.  See ``parallel_mcmc.py``.
        pool : optional
            Externally-managed pool that overrides ``parallel``.
        n_processes : int, optional
            Worker count when ``parallel=True``.  Defaults to ``cpu_count - 1``.
        rng_seed : int, optional
            Seed for walker initialisation.

        Returns
        -------
        samples : ndarray, shape (n_effective, ndim)
            Posterior samples after burn-in and thinning.
        """
        from parallel_mcmc import run_emcee

        assert self.surrogate.trained, \
            "Surrogate not trained — call train_surrogate() first"

        ndim = self.prior.ndim
        if n_walkers < 2 * ndim:
            n_walkers = 2 * ndim
            if verbose:
                print(f"  Increased n_walkers to {n_walkers} (need >= 2*ndim)")

        if verbose:
            mode = "parallel" if (parallel or pool is not None) else "serial"
            print(f"  MCMC ({mode}): {n_walkers} walkers x {n_steps} steps "
                  f"(burn-in={burn_in}, thin={thin})")

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
            print(f"\n  MCMC complete: {len(self.samples)} posterior samples "
                  f"[{info['elapsed_s']:.1f}s]")
            self._print_diagnostics()

        return self.samples

    def _print_diagnostics(self):
        """Print MCMC convergence diagnostics."""
        chain = self.sampler.get_chain()
        n_steps = chain.shape[0]
        ndim = chain.shape[2]

        # heuristic: need at least 50 steps per dimension for a meaningful autocorrelation estimate
        if n_steps >= 50 * ndim:
            tau = self.sampler.get_autocorr_time(quiet=True)
            if np.all(np.isfinite(tau)):
                names = _get_param_names(self.param_set)
                print(f"  Autocorrelation times:")
                for name, t in zip(names, tau):
                    print(f"    {name:12s}  τ = {t:.1f}")
                n_eff = len(self.samples) / np.max(tau)
                print(f"  Effective samples:  ~{int(n_eff)}")
            else:
                print("  (Autocorrelation times contain non-finite values — "
                      "chain may need more steps)")
        else:
            print(f"  (Chain too short for autocorrelation estimate: "
                  f"{n_steps} steps, need ~{50 * ndim}+)")

        # acceptance fraction
        af = self.sampler.acceptance_fraction
        print(f"  Acceptance fraction: {np.mean(af):.3f} "
              f"(range [{np.min(af):.3f}, {np.max(af):.3f}])")


    #  Posterior analysis

    def posterior_summary(self):
        """
        Compute posterior statistics: mean, std, 95% credible interval.

        Returns
        summary : dict
            Keyed by parameter name.
            Each entry contains: mean, std, ci_2.5, ci_97.5, prior_mean, shift
            shift = (posterior_mean - prior_mean) / prior_std
                - quantifies how much the data moved the parameter.
        """

        names = _get_param_names(self.param_set)
        summary = {}

        for i, name in enumerate(names):
            s = self.samples[:, i]
            shift = (np.mean(s) - self.prior.means[i]) / self.prior.stds[i]
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
        """Print formatted posterior summary table."""
        summary = self.posterior_summary()
        names = _get_param_names(self.param_set)

        print(f"\n  {'Parameter':>12s}  {'Prior':>8s}  {'Posterior':>10s}  "
              f"{'±σ':>7s}  {'95% CI':>18s}  {'Shift':>6s}")
        print("  " + "-" * 72)

        for name in names:
            s = summary[name]
            print(f"  {name:>12s}  {s['prior_mean']:8.4f}  {s['mean']:10.4f}  "
                  f"{s['std']:7.4f}  "
                  f"[{s['ci_2.5']:.4f}, {s['ci_97.5']:.4f}]  "
                  f"{s['shift']:+6.2f}σ")

    def plot_posterior(self, save_path=None):
        """
        Corner plot of posterior with prior means marked.
        """

        import corner
        names = _get_param_names(self.param_set)
        fig = corner.corner(
            self.samples, labels=names,
            truths=self.prior.means.tolist(),
            show_titles=True, title_fmt='.4f',
            quantiles=[0.16, 0.5, 0.84]
        )
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Corner plot saved to {save_path}")
        return fig


    #  Posterior predictive check

    def posterior_predictive(self, n_samples=50, verbose=True):
        """
        Run forward model at posterior samples to get predictive distribution.

        This is expensive (n_samples full CFD solves) but gives the
        actual posterior predictive uncertainty without surrogate error.
        Use for final validation, not routine analysis.

        Parameters
        n_samples : int
            Number of posterior samples to evaluate (subset of full chain).
        Returns
        predictions : list of ndarray
            Each entry is the H(fields) vector from one posterior sample.
        """

        # subsample from posterior
        idx = np.random.choice(len(self.samples), size=n_samples, replace=False)
        predictions = []
        statuses = []

        for i, si in enumerate(idx):
            theta = self.samples[si].tolist()
            result = self.forward_model.evaluate(theta)
            predictions.append(np.array(result.predictions))
            statuses.append(str(result.status))

            if verbose and (i + 1) % 10 == 0:
                print(f"  Predictive check {i+1}/{n_samples}: "
                      f"{result.status}")

        n_converged = sum(1 for s in statuses
                          if 'Converged' in s or 'Unconverged' in s)
        if verbose:
            print(f"  Predictive check complete: "
                  f"{n_converged}/{n_samples} converged")

        return predictions

    # Automated report

    def report(self, save_dir, mesh=None, forward_model=None, fmt='png'):
        """
        Write a complete inference report to save_dir.

        Generates all available inference plots via InferenceVisualizer, then
        (if mesh and forward_model are supplied) renders flow field screenshots
        at the posterior mean using FlowVisualizer.  Skips stages that have not
        yet been run.

        Parameters
        save_dir : str or Path
            Directory to write figures and summary.  Created if absent.
        mesh : rans_sst_py.Mesh, optional
            Mesh object used for flow field screenshots.
        forward_model : rans_sst_py.ForwardModel, optional
            Forward model used for the posterior-mean CFD solve.
        fmt : str
            Image format ('png', 'pdf', 'svg').
        """
        import os
        from pathlib import Path

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Writing inference report to {save_dir}/")

        # 1. Inference diagnostics via InferenceVisualizer
        if self.ensemble_X is not None:
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from inference_visualizer import InferenceVisualizer
                vis = InferenceVisualizer(self)
                burn_in = None
                if self.sampler is not None:
                    burn_in = max(0, self.sampler.get_chain().shape[0] // 5)
                vis.plot_full_report(
                    save_dir=str(save_dir),
                    prefix='inference',
                    burn_in=burn_in,
                    fmt=fmt
                )
            except Exception as e:
                print(f"  [report] InferenceVisualizer skipped: {e}")

        # 2. Posterior summary text
        if self.samples is not None:
            summary_path = save_dir / 'summary.txt'
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.print_summary()
            summary_path.write_text(buf.getvalue())
            print(f"  Posterior summary -> {summary_path}")

        # 3. Flow field screenshots at posterior mean
        if mesh is not None and forward_model is not None and self.samples is not None:
            try:
                from visualization import FlowVisualizer, flow_data_from_solver
                theta_mean = np.mean(self.samples, axis=0).tolist()
                print(f"  Solving at posterior mean theta...")
                result = forward_model.evaluate(theta_mean)
                print(f"  Status: {result.status}")
                data = flow_data_from_solver(mesh, forward_model)

                vis_flow = FlowVisualizer(data, theme='document')
                vis_flow.screenshot(
                    field='U_mag',
                    path=str(save_dir / f'velocity.{fmt}'),
                    camera_position='xy'
                )
                vis_flow.screenshot_turbulence(
                    path=str(save_dir / f'turbulence.{fmt}'),
                    camera_position='xy'
                )
            except Exception as e:
                print(f"  [report] Flow field screenshots skipped: {e}")

        print(f"  Report complete. Files in {save_dir}/\n")
