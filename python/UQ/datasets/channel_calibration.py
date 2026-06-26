"""Real-DNS SST calibration on the plane channel, with calibrated predictive UQ.

Replaces the Dean-correlation likelihood of the existing pipeline with the real
Lee-Moser observables (the mean-velocity profile in outer units and the wall
skin friction, with per-point uncertainty from the DNS _stdev), and wires the
calibrated-UQ techniques (generalized Bayes and conformal) onto the posterior so
the pipeline emits calibrated predictive intervals on the channel QoIs.

Reuse, not reimplementation. The expensive forward map and the inference are the
existing layers: latin_hypercube (design), the C++ ForwardModel (SIMPLE SST
solve + Gaussian likelihood), GPSurrogate (theta -> log-likelihood),
MultiOutputSurrogate (theta -> QoI predictions, for the posterior predictive),
run_emcee (sampler). The UQ corrections are the existing UQ modules
(generalized_bayes, conformal, evaluation). The new code here is only the
channel-specific glue: building the DNS observation operator, collecting the
prediction vector during the ensemble, and the posterior-predictive coverage.

Units. The solver runs at the DNS bulk Reynolds number (half-height convention,
Re_b = U_b^+ Re_tau, with U_b = h = 1) and the velocity profile is compared in
outer units U/U_b versus y/delta, which is friction-velocity free. The wall skin
friction Cf = 2/(U_b^+)^2 is then a predicted QoI, so its misfit (the developing
channel sits a little below the fully-developed friction) is part of the
misspecification the UQ must cover, not a matched input.

Overconfidence mechanism (H1, research.md / compressible_research.md). The DNS
mean velocity is known to about 0.1 percent, while the Boussinesq SST profile is
off by a few percent, so the standard (eta = 1) posterior contracts on the
Kullback-Leibler-closest coefficients and its predictive band (coefficient spread
plus the tiny DNS noise) is far too narrow to contain the DNS. The generalized-
Bayes power posterior inflates the effective observation variance by 1/eta, and
conformal sets the interval half-width from the calibration-residual quantile;
both restore coverage. The learning rate and the conformal score are calibrated on
a calibration split only, never on the evaluated number.
"""
import numpy as np

from solver_bindings import _rs
from priors import make_prior_from_param_set, make_sampling_prior
from design import latin_hypercube
from gp_surrogate import GPSurrogate, MultiOutputSurrogate
from parallel_mcmc import run_emcee

from .. import generalized_bayes as gb
from .. import conformal as cf
from .. import evaluation as ev


# one channel configuration, shared with the baseline generator for consistency
DEFAULT_CFG = {"nx": 48, "ny": 64, "Lx": 20.0,
               "max_iter": 6000, "conv_tol": 1.0e-4, "yplus_target": 0.5}

# parameter sets the existing pipeline exposes (DNS_plan.md Step 1: re-run both)
PARAM_SETS = {
    "a1_betaStar": lambda rs: rs.InferenceParameterSet.a1_betaStar(),
    "near_wall4":  lambda rs: rs.InferenceParameterSet.near_wall4(),
}


class ChannelCalibration:
    """Calibrate SST coefficients on one channel DNS case and quantify coverage.

    Typical use:
        c = ChannelCalibration(dns)
        c.run_ensemble(n=48, seed=0)      # forward solves (the expensive step)
        c.fit_surrogates()
        post = c.sample_posterior(eta=1.0) # standard Bayes
        c.coverage_report(...)             # vs the DNS QoIs
    """

    def __init__(self, dns, param_set="a1_betaStar", n_stations=20,
                 yplus_lo=1.0, cfg=None, sigma_floor=0.005, seed=0):
        self.dns = dns
        self.param_set_name = param_set
        self.cfg = dict(DEFAULT_CFG if cfg is None else cfg)
        self.rs = _rs()
        self.param_set = PARAM_SETS[param_set](self.rs)
        self.prior = make_sampling_prior(self.param_set)
        self.ndim = self.prior.ndim

        # bulk velocity (wall units) and the matched half-height bulk Reynolds number
        self.ub_plus = dns.bulk_velocity()
        self.re_b = self.ub_plus * dns.re_tau
        self.nu = 1.0 / self.re_b                      # U_b = h = 1

        # select observation stations, log-spaced in y^+ from yplus_lo to the
        # centerline; map to outer units (U/U_b at y/delta) and DNS _stdev sigma
        self._select_stations(n_stations, yplus_lo, sigma_floor)

        # ensemble / surrogate / posterior storage
        self.X = self.loglik = self.preds = None
        self.gp = None
        self.mo = None

    # ---- observation construction -----------------------------------------

    def _select_stations(self, n_stations, yplus_lo, sigma_floor):
        d = self.dns
        targets = np.geomspace(yplus_lo, 0.99 * d.re_tau, n_stations)
        idx = sorted(set(int(np.argmin(np.abs(d.yplus - t))) for t in targets))
        self.idx = np.array(idx)
        self.y_delta = d.y_delta[self.idx]
        # QoI truth: Cf first, then U/U_b at each station (matching prediction order)
        u_out = d.U[self.idx] / self.ub_plus
        # observation sigma is the DNS statistical uncertainty (outer units),
        # floored at sigma_floor (a fraction of U_b) so a few near-wall stations
        # with a vanishing _stdev do not dominate and destabilise the likelihood
        u_sig = np.maximum(d.U_std[self.idx] / self.ub_plus, sigma_floor)
        self.cf_obs = d.skin_friction()
        self.cf_sig = 0.05 * self.cf_obs               # 5% on the friction QoI
        self.qoi_truth = np.concatenate([[self.cf_obs], u_out])
        self.qoi_sigma = np.concatenate([[self.cf_sig], u_sig])
        self.qoi_names = ["Cf"] + [f"U/Ub@y+={d.yplus[i]:.0f}" for i in self.idx]
        self.n_qoi = self.qoi_truth.size

    def _observation_operator(self):
        rs = self.rs
        Lx = self.cfg["Lx"]
        obs = rs.ObservationOperator()
        # Cf at the developed (near-outlet) wall, then the velocity profile
        obs.add_skin_friction("bottom", rs.Vec3(0.9 * Lx, 0.0, 0.0),
                              self.cf_obs, self.cf_sig, 1.0)
        for yd, u, s in zip(self.y_delta, self.qoi_truth[1:], self.qoi_sigma[1:]):
            obs.add_velocity_profile(rs.Vec3(0.9 * Lx, float(yd), 0.0), 0,
                                     float(u), float(s))
        return obs

    def _forward_model(self):
        rs = self.rs
        Ub = 1.0
        mesh = rs.Mesh.make_channel_2d(nx=self.cfg["nx"], ny=self.cfg["ny"],
                                       Lx=self.cfg["Lx"], Ly=2.0,
                                       Re=self.re_b, yPlusTarget=self.cfg["yplus_target"])
        mesh.compute_wall_distance()
        kIn = 1.5 * (Ub * 0.05) ** 2
        omIn = kIn / (self.nu * 100.0)
        bcs = rs.FlowBoundaryConditions.channel_defaults(mesh, Ub, kIn, omIn)
        obs = self._observation_operator()
        settings = rs.SolverSettings()
        settings.max_iterations = self.cfg["max_iter"]
        settings.convergence_tol = self.cfg["conv_tol"]
        settings.verbose = False
        settings.alpha_u = 0.7
        settings.alpha_p = 0.5
        fm = rs.ForwardModel(mesh, self.param_set, obs, bcs, self.nu, settings,
                             rs.Vec3(Ub, 0.0, 0.0), 0.0, kIn, omIn)
        # keep references alive (pybind11 stores refs, not copies; see auto-memory)
        self._alive = (mesh, obs, bcs, settings)
        return fm

    # ---- ensemble and surrogates (the expensive step) ---------------------

    def run_ensemble(self, n=48, seed=0, verbose=False):
        """Latin-hypercube ensemble of forward solves; collect loglik + predictions."""
        fm = self._forward_model()
        np.random.seed(seed)   # latin_hypercube draws from the global RNG
        X = latin_hypercube(n, self.ndim, self.prior.lower, self.prior.upper)
        loglik = np.full(n, -np.inf)
        preds = np.full((n, self.n_qoi), np.nan)
        for i in range(n):
            res = fm.evaluate(X[i].tolist())
            loglik[i] = res.log_lik
            p = np.array(res.predictions)
            if p.size == self.n_qoi:
                preds[i] = p
            if verbose and (i + 1) % 10 == 0:
                print(f"  ensemble {i+1}/{n}  loglik={loglik[i]:.1f}", flush=True)
        valid = (loglik > -1e5) & np.all(np.isfinite(preds), axis=1)
        self.X = X[valid]
        self.loglik = loglik[valid]
        self.preds = preds[valid]
        self.n_valid = int(valid.sum())
        return self.n_valid

    def fit_surrogates(self, noise_floor=1e-3):
        """GP surrogate of the log-likelihood and a multi-output QoI surrogate."""
        self.gp = GPSurrogate()
        self.gp.train(self.X, self.loglik, noise_floor=noise_floor)
        self.mo = MultiOutputSurrogate()
        self.mo.train(self.X, self.preds, noise_floor=noise_floor)

    def to_cache(self):
        """Arrays sufficient to rebuild the surrogates without re-solving."""
        return {"X": self.X, "loglik": self.loglik, "preds": self.preds,
                "qoi_truth": self.qoi_truth, "qoi_sigma": self.qoi_sigma,
                "y_delta": self.y_delta, "re_tau": self.dns.re_tau}

    def load_cache(self, d):
        self.X, self.loglik, self.preds = d["X"], d["loglik"], d["preds"]
        self.n_valid = len(self.X)
        self.fit_surrogates()

    # ---- posterior and posterior predictive -------------------------------

    def sample_posterior(self, eta=1.0, n_walkers=24, n_steps=3000, burn_in=600,
                         seed=0):
        """Power (Gibbs) posterior at learning rate eta via the existing sampler.

        eta = 1 is standard Bayes; eta < 1 tempers the likelihood to match the
        degree of misspecification (generalized_bayes.tempered_log_posterior).
        """
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
        """Posterior-predictive QoI samples = surrogate mean + inflated noise.

        The generalized-Bayes predictive inflates the observation variance by
        1/eta (the power posterior scales the data precision by eta), so the
        predictive draws use sigma/sqrt(eta). qoi_index selects a QoI subset
        (e.g. held-out stations); default all QoIs.
        """
        rng = np.random.default_rng(seed)
        mu, _ = self.mo.predict_batch(theta_samples)        # (n_theta, n_qoi)
        if qoi_index is not None:
            mu = mu[:, qoi_index]
            sigma = self.qoi_sigma[qoi_index]
        else:
            sigma = self.qoi_sigma
        noise = rng.normal(size=mu.shape) * (sigma / np.sqrt(eta))[None, :]
        return mu + noise                                   # (n_theta, n_qoi)

    # ---- coverage on the DNS QoIs -----------------------------------------

    def coverage_vs_truth(self, predictive, qoi_index=None, level=0.9):
        """Coverage and sharpness of the central predictive intervals vs DNS QoIs.

        predictive: (n_theta, n_qoi_subset). Returns (coverage, sharpness). The
        QoI ensemble is transposed so each station is one observation with its
        own predictive ensemble (evaluation.coverage_from_samples).
        """
        truth = self.qoi_truth if qoi_index is None else self.qoi_truth[qoi_index]
        return ev.coverage_from_samples(truth, predictive.T, level=level)

    def point_prediction(self, theta_samples, qoi_index=None):
        """Posterior-mean QoI prediction (the point predictor conformal wraps)."""
        mu, _ = self.mo.predict_batch(theta_samples)
        mu = mu.mean(axis=0)
        return mu if qoi_index is None else mu[qoi_index]

    def calibrate_eta(self, theta_samples, cal_index):
        """Moment-matched learning rate from the calibration-station residuals.

        eta = mean(sigma^2) / mean(residual^2): the factor that inflates the
        effective observation variance to the dispersion the data actually show
        (generalized_bayes.calibrate_eta_moment, aggregated over the calibration
        stations). Calibrated on the calibration split only, never on the
        evaluated coverage.
        """
        pred = self.point_prediction(theta_samples, cal_index)
        resid = self.qoi_truth[cal_index] - pred
        sigma2 = np.mean(self.qoi_sigma[cal_index] ** 2)
        eta = sigma2 / max(np.mean(resid ** 2), 1e-300)
        return float(np.clip(eta, 1e-6, 1.0))

    def conformal_coverage(self, point_pred_cal, point_pred_test,
                           cal_index, test_index, alpha=0.1):
        """Split-conformal interval from calibration residuals; coverage on test.

        Stations are the exchangeable units: the (1 - alpha) quantile of the
        calibration absolute residuals sets a constant half-width, applied to the
        test point predictions (conformal.split_conformal_intervals). Returns
        (coverage, half_width, coverage_gap). Under a cross-Re shift the gap is
        the honest exchangeability-violation measure.
        """
        y_cal = self.qoi_truth[cal_index]
        y_test = self.qoi_truth[test_index]
        lo, hi = cf.split_conformal_intervals(point_pred_cal, y_cal,
                                              point_pred_test, alpha=alpha)
        cov = ev.empirical_coverage(y_test, lo, hi)
        half = 0.5 * float(np.mean(hi - lo))
        return cov, half, (1.0 - alpha) - cov


class CrossReStudy:
    """Cross-Reynolds-number generalization on the calibrated channel UQ.

    Holds one fitted ChannelCalibration per Reynolds number and forms a pooled
    (tempered) posterior over the training Reynolds numbers, then predicts a
    held-out Reynolds number and scores coverage. The pre-registered splits
    (DNS_plan.md Step 1) are leave-one-Reynolds-out and a held-out-high-Re split;
    the conformal coverage gap under the shift is reported, not hidden.

    All cases share one parameter set and prior, so their log-likelihood
    surrogates combine additively: log p(theta | train) = log prior +
    eta * sum_{r in train} loglik_r(theta).
    """

    def __init__(self, calibrations):
        self.cals = calibrations                 # dict {re_nominal: ChannelCalibration}
        any_cal = next(iter(calibrations.values()))
        self.prior = any_cal.prior

    def pooled_log_posterior(self, train, eta):
        gps = [self.cals[r].gp for r in train]

        def log_post(theta):
            lp = self.prior.log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            ll = sum(gp.log_likelihood(theta) for gp in gps)
            if not np.isfinite(ll):
                return -np.inf
            return lp + eta * ll
        return log_post

    def pooled_posterior_samples(self, train, eta, n_walkers=24, n_steps=3000,
                                 burn_in=600, seed=0):
        samples, _ = run_emcee(
            log_posterior=self.pooled_log_posterior(train, eta), prior=self.prior,
            n_walkers=n_walkers, n_steps=n_steps, burn_in=burn_in, thin=1,
            progress=False, rng_seed=seed)
        return samples

    def predict_heldout(self, train, test, eta, level=0.9, seed=0):
        """Calibrate on train Reynolds numbers, predict the held-out one.

        Returns a dict with the predictive coverage and sharpness at the held-out
        Re for standard Bayes (eta = 1) and the tempered posterior, plus the
        conformal coverage and gap (calibration set = a training Re's stations).
        """
        test_cal = self.cals[test]
        out = {"test": test, "train": list(train)}

        # standard Bayes (eta = 1) and tempered posterior predictive at held-out Re
        for tag, e in (("standard", 1.0), ("tempered", eta)):
            post = self.pooled_posterior_samples(train, e, seed=seed)
            pp = test_cal.posterior_predictive(post, eta=e, seed=seed + 1)
            cov, sharp = test_cal.coverage_vs_truth(pp, level=level)
            out[f"{tag}_coverage"] = cov
            out[f"{tag}_sharpness"] = sharp
            out[f"{tag}_eta"] = e

        # conformal: calibration residuals from a training Re, applied at held-out Re
        cal_re = train[0]
        cal = self.cals[cal_re]
        post = self.pooled_posterior_samples(train, eta, seed=seed)
        idx_all = np.arange(test_cal.n_qoi)
        pred_cal = cal.point_prediction(post, idx_all)
        pred_test = test_cal.point_prediction(post, idx_all)
        y_cal, y_test = cal.qoi_truth, test_cal.qoi_truth
        lo, hi = cf.split_conformal_intervals(pred_cal, y_cal, pred_test,
                                              alpha=1.0 - level)
        out["conformal_coverage"] = ev.empirical_coverage(y_test, lo, hi)
        out["conformal_gap"] = level - out["conformal_coverage"]
        return out
