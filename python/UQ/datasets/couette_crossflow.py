"""Cross-flow generalization: channel calibration -> Couette prediction (Step 2).

The pre-registered Step-2 question (DNS_plan.md): does the Step-1 channel
calibration plus its calibrated UQ transfer to the held-out flow type, and does the
predictive interval cover the Couette truth at the strict 0.5 percent observation
band. The protocol mirrors the Step-1 cross-Reynolds study exactly, with flow type
as the held-out axis instead of Reynolds number:

  1. Pool the channel calibrations into a posterior over the SST coefficients (the
     Step-1 channel posterior), standard (eta = 1) and tempered (the channel-
     calibrated generalized-Bayes learning rate).
  2. Push that channel posterior THROUGH the a-posteriori moving-wall Couette
     forward model (CouetteCalibration, the same SST closure) to predict the
     Couette QoIs.
  3. Score coverage of the Couette DNS QoIs for standard Bayes, generalized Bayes,
     and conformal, at the 0.5 percent and 1 percent modeled observation bands, and
     report the conformal cross-flow coverage gap (the honest OOD-shift measure).

The correction is calibrated ON THE CHANNEL (the in-distribution data) and applied
to the held-out Couette, so the gap measures how the channel-calibrated UQ degrades
under the cross-flow shift, never tuned on the Couette coverage. The within-Couette
in-distribution and cross-Re coverage (the capability check and the bonus cross-Re
axis) reuse the Step-1 CrossReStudy directly on the Couette calibrations.
"""
import numpy as np

from .. import conformal as cf
from .. import evaluation as ev
from .channel_calibration import CrossReStudy


class CrossFlowStudy:
    """Channel-calibrated UQ predicting the held-out Couette flow type.

    Holds the fitted channel calibrations (one per channel Re_tau) and the fitted
    Couette calibrations (one per Couette Re_tau). All share the one parameter set
    and prior, so the channel log-likelihood surrogates pool additively into the
    channel posterior, which the Couette forward model then propagates.
    """

    def __init__(self, channel_cals, couette_cals):
        self.channel = CrossReStudy(channel_cals)        # channel posterior machinery
        self.channel_cals = channel_cals
        self.couette_cals = couette_cals
        self.prior = self.channel.prior

    def channel_posteriors(self, seed=0):
        """Pooled channel posterior over the SST coefficients: standard and tempered.

        Returns (post1, post_t, eta). eta is the generalized-Bayes learning rate
        moment-matched on the channel QoIs (the in-distribution calibration data),
        the same channel-calibrated correction Step 1 used, applied unchanged to the
        cross-flow prediction.
        """
        train = tuple(self.channel_cals)
        post1 = self.channel.pooled_posterior_samples(train, 1.0, seed=seed)
        eta = float(np.mean([self.channel_cals[r].calibrate_eta(
            post1, np.arange(self.channel_cals[r].n_qoi)) for r in train]))
        post_t = self.channel.pooled_posterior_samples(train, eta, seed=seed)
        return post1, post_t, eta

    def predict_couette(self, re, rel, post1, post_t, eta, level=0.9, seed=0):
        """Coverage of the Couette DNS QoIs from the channel posterior, one band.

        Standard Bayes (channel posterior, eta = 1), generalized Bayes (channel
        posterior tempered by the channel eta, predictive noise inflated by 1/eta),
        and conformal (channel calibration residuals set the interval half-width,
        applied to the Couette point predictions). Returns the coverages, the
        sharpnesses, and the conformal cross-flow gap at the given observation band.
        """
        cou = self.couette_cals[re]
        cou.set_observation_band(rel)
        out = {"re": re, "rel": rel, "eta": eta, "level": level}

        pp1 = cou.posterior_predictive(post1, eta=1.0, seed=seed + 1)
        out["standard_coverage"], out["standard_sharpness"] = \
            cou.coverage_vs_truth(pp1, level=level)

        pp_t = cou.posterior_predictive(post_t, eta=eta, seed=seed + 2)
        out["tempered_coverage"], out["tempered_sharpness"] = \
            cou.coverage_vs_truth(pp_t, level=level)

        # conformal: a channel case supplies the exchangeable calibration residuals,
        # whose (1 - level) quantile is the interval half-width applied to the Couette
        # point predictions; the gap to nominal is the cross-flow exchangeability cost
        cal = self.channel_cals[min(self.channel_cals)]
        pred_cal = cal.point_prediction(post_t, np.arange(cal.n_qoi))
        pred_test = cou.point_prediction(post_t, np.arange(cou.n_qoi))
        lo, hi = cf.split_conformal_intervals(pred_cal, cal.qoi_truth, pred_test,
                                              alpha=1.0 - level)
        out["conformal_coverage"] = ev.empirical_coverage(cou.qoi_truth, lo, hi)
        out["conformal_gap"] = level - out["conformal_coverage"]
        return out

    def run(self, rels=(0.005, 0.010), level=0.9, seed=0):
        """Cross-flow coverage for every Couette Re at every observation band.

        Returns a dict keyed by band (rel) of per-Re coverage rows, plus the channel
        learning rate. One channel posterior is drawn and reused across all Couette
        cases and bands (the observation band rescales the predictive noise only).
        """
        post1, post_t, eta = self.channel_posteriors(seed=seed)
        result = {"eta": eta, "bands": {}}
        for rel in rels:
            rows = []
            for re in self.couette_cals:
                rows.append(self.predict_couette(re, rel, post1, post_t, eta,
                                                 level=level, seed=seed))
            result["bands"][rel] = rows
        return result
