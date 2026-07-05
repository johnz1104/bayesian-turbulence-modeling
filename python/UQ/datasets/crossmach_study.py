"""Cross-Mach generalization study on the calibrated compressible channel UQ.

Mirrors the cross-Reynolds study of the incompressible phase: one fitted
CompressibleCalibration per channel case, a pooled (tempered) posterior over
the training cases, prediction of held-out cases, and coverage scored
SEPARATELY on the likelihood block (U, T, cf) and on the pre-registered
HELD-OUT thermal block (B_q and the heat-flux profile), which never entered
any likelihood. The pre-registered splits: calibrate on the M_CLx < 1.0 cases
and predict the M_CLx >= 1.47 cases; leave-one-Mach-family-out within the
matrix; the never-calibrated CKM cases as the external cross-check. The
learning rate is moment-matched on the training cases' likelihood stations
only; the conformal calibration residuals come from training-case stations;
nothing is tuned toward any evaluated coverage.
"""
import numpy as np

from parallel_mcmc import run_emcee

from .. import conformal as cf
from .. import evaluation as ev


class CrossMachStudy:
    """Pooled calibration across channel cases; held-out-case prediction.

    calibrations: dict {case_tag: CompressibleCalibration}, every case with
    its ensemble and surrogates fitted (training cases inform the posterior;
    held-out cases only supply their forward map and truth).
    """

    def __init__(self, calibrations):
        self.cals = calibrations
        any_cal = next(iter(calibrations.values()))
        self.prior = any_cal.prior

    def pooled_log_posterior(self, train, eta):
        gps = [self.cals[t].gp for t in train]

        def log_post(theta):
            lp = self.prior.log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            ll = sum(gp.log_likelihood(theta) for gp in gps)
            if not np.isfinite(ll):
                return -np.inf
            return lp + eta * ll
        return log_post

    def pooled_posterior_samples(self, train, eta, n_walkers=24, n_steps=2000,
                                 burn_in=500, seed=0):
        samples, _ = run_emcee(
            log_posterior=self.pooled_log_posterior(train, eta),
            prior=self.prior, n_walkers=n_walkers, n_steps=n_steps,
            burn_in=burn_in, thin=1, progress=False, rng_seed=seed)
        return samples

    def predict_heldout(self, train, test, eta=None, level=0.9, seed=0):
        """Calibrate on the training cases, predict one held-out case.

        Coverage and sharpness are reported per block: 'lik' (the U, T, cf
        observables) and 'thermal' (the held-out B_q and heat-flux profile,
        the pre-registered coverage targets). The conformal interval is
        calibrated per block on a training case's residuals and its
        cross-Mach gap reported. If eta is None it is moment-matched on the
        training cases' likelihood stations from the eta = 1 posterior.
        """
        test_cal = self.cals[test]
        out = {"test": test, "train": list(train)}

        post1 = self.pooled_posterior_samples(train, 1.0, seed=seed)
        if eta is None:
            eta = float(np.mean([
                self.cals[t].calibrate_eta(post1, self.cals[t].lik_index)
                for t in train]))
        post_t = self.pooled_posterior_samples(train, eta, seed=seed)
        out["eta"] = eta
        out["prt_posterior"] = test_cal.marginal_prt(post_t)

        blocks = {"lik": test_cal.lik_index, "thermal": test_cal.heldout_index}
        for tag, e, post in (("standard", 1.0, post1),
                             ("tempered", eta, post_t)):
            for block, idx in blocks.items():
                pp = test_cal.posterior_predictive(post, eta=e, qoi_index=idx,
                                                   seed=seed + 1)
                cov, sharp = test_cal.coverage_vs_truth(pp, qoi_index=idx,
                                                        level=level)
                out[f"{tag}_{block}_coverage"] = cov
                out[f"{tag}_{block}_sharpness"] = sharp

        # conformal per block: calibration residuals from the first training
        # case's stations, applied to the held-out case's point predictions
        cal = self.cals[train[0]]
        for block in blocks:
            idx_cal = cal.lik_index if block == "lik" else cal.heldout_index
            idx_test = blocks[block]
            pred_cal = cal.point_prediction(post_t, idx_cal)
            pred_test = test_cal.point_prediction(post_t, idx_test)
            lo, hi = cf.split_conformal_intervals(
                pred_cal, cal.qoi_truth[idx_cal], pred_test,
                alpha=1.0 - level)
            cov = ev.empirical_coverage(test_cal.qoi_truth[idx_test], lo, hi)
            out[f"conformal_{block}_coverage"] = cov
            out[f"conformal_{block}_gap"] = level - cov
        return out
