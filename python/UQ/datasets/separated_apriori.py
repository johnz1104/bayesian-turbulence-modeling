"""A-priori model-form checkpoint on the real separated discrepancy.

The precursor checkpoint of the separated-flow model-form study: before any
a-posteriori claim, the conditional models must reproduce their
synthetic-validation behaviour on the real (feature, db) pairs of the
backward-facing step, sampling realizable predictive anisotropies whose
coverage, CRPS and energy score against the held-out DNS anisotropy can be
scored by the standard harness. Both the generative flow and the Gaussian
model-form baseline run through the identical path (fit / sample / project /
score), so the a-priori comparison isolates the distribution family.

Protocol (pinned in UQ-RANS_research/separated_modelform/
METHODS_OPERATIONALIZATION.md before any result was computed): the held-out
unit is a profile STATION (wall-normal profiles are internally correlated, so
point-level splits would leak), evaluated leave-one-station-out over the six
stations plus the train-on-all in-distribution machinery check, at the nominal
90 percent level with fixed seeds.
"""
import numpy as np

from ..generative import GenerativeDiscrepancyModel
from ..gaussian_modelform import GaussianDiscrepancyModel
from .. import realizability as rz
from .. import evaluation as ev
from .separated_discrepancy import BFSDiscrepancy

COMPONENT_NAMES = ("xx", "yy", "xy", "xz", "yz")


def anisotropy_to_components(b):
    """(N, 3, 3) symmetric traceless to the five independent components.

    Layout [b_xx, b_yy, b_xy, b_xz, b_yz], the exact inverse of
    GenerativeDiscrepancyModel.components_to_anisotropy (b_zz = -(b_xx+b_yy)).
    """
    b = np.asarray(b, float)
    return np.stack([b[..., 0, 0], b[..., 1, 1], b[..., 0, 1],
                     b[..., 0, 2], b[..., 1, 2]], axis=-1)


class SeparatedAPriori:
    """The (feature, db) training set and the a-priori scoring loop.

    Arrays are restricted to the valid points of the discrepancy record:
      features (N, 5), db_comp (N, 5), b_base and b_dns (N, 3, 3),
      station (N,) integer station index.
    """

    def __init__(self, disc):
        m = disc.mask
        self.features = disc.features[m]
        self.db_comp = anisotropy_to_components(disc.db[m])
        self.b_base = disc.b_baseline[m]
        # b_DNS = b_baseline + db by construction of the discrepancy
        self.b_dns = disc.b_baseline[m] + disc.db[m]
        # the held-out unit: profile stations on the backward-facing step,
        # streamwise bands on the dense periodic-hills field (the pinned
        # spatially-correlated grouping of each geometry's protocol)
        if hasattr(disc, "station"):
            self.station = disc.station[m]
            self.station_xh = disc.dns.station_xh
        else:
            self.station = disc.band[m]
            self.station_xh = np.arange(int(self.station.max()) + 1,
                                        dtype=float)
        self.disc = disc

    @staticmethod
    def build(cfg=None, dns=None, baseline=None):
        return SeparatedAPriori(BFSDiscrepancy.build(dns=dns, baseline=baseline,
                                                     cfg=cfg))

    @property
    def n(self):
        return self.features.shape[0]

    def station_split(self, held_out):
        """(train_mask, test_mask) with one station index held out."""
        te = self.station == int(held_out)
        return ~te, te

    def fit(self, kind, train_mask, seed=0, epochs=400, **kw):
        """Fit one conditional model ("flow" or "gauss") on the training rows."""
        n_feat = self.features.shape[1]
        if kind == "flow":
            model = GenerativeDiscrepancyModel(n_feat, 5, n_layers=8, hidden=64,
                                               seed=seed)
        elif kind == "gauss":
            model = GaussianDiscrepancyModel(n_feat, 5, hidden=64, seed=seed)
        else:
            raise ValueError(f"unknown model kind '{kind}'")
        model.fit(self.features[train_mask], self.db_comp[train_mask],
                  epochs=epochs, **kw)
        return model

    @staticmethod
    def _score(model, feats, b_base, b_true, n_samples, level):
        """Score a fitted model against DNS anisotropy at arbitrary points."""
        b_samp = model.sample_realizable_anisotropy(feats, base_anisotropy=b_base,
                                                    n_per=n_samples)
        R = 2.0 * (b_samp + np.eye(3) / 3.0)
        realizable_fraction = float(np.mean(rz.is_realizable(R, tol=1e-6)))

        y = anisotropy_to_components(b_true)            # (N, 5)
        s = anisotropy_to_components(b_samp)            # (N, M, 5)

        coverage, sharpness, crps = {}, {}, {}
        for j, name in enumerate(COMPONENT_NAMES):
            cov, shp = ev.coverage_from_samples(y[:, j], s[:, :, j], level=level)
            coverage[name] = cov
            sharpness[name] = shp
            crps[name] = ev.crps_ensemble(y[:, j], s[:, :, j])

        return {
            "level": level,
            "n_test": int(feats.shape[0]),
            "n_samples": int(n_samples),
            "coverage": coverage,
            "sharpness": sharpness,
            "crps": crps,
            "crps_mean": float(np.mean(list(crps.values()))),
            "energy_score": ev.energy_score(y, s),
            "realizable_fraction": realizable_fraction,
        }

    def evaluate(self, model, test_mask, n_samples=128, level=0.9):
        """Score realizable predictive anisotropies against the DNS anisotropy.

        Samples db from the model at the test features, adds the baseline
        anisotropy, projects every draw into the realizable set (the same
        projection every method's injection targets pass through), and scores
        component-wise coverage/sharpness/CRPS plus the multivariate energy
        score and the realizable fraction (1.0 by construction).
        """
        return self._score(model, self.features[test_mask],
                           self.b_base[test_mask], self.b_dns[test_mask],
                           n_samples, level)

    def evaluate_external(self, model, other, n_samples=128, level=0.9):
        """Score a model fitted on THIS geometry against ANOTHER geometry.

        The cross-geometry transfer of the pinned protocol (methods note
        section 8): the model conditions only on the five invariant features
        (no geometry label), and its realizable predictive anisotropy is scored
        on every valid point of the other geometry's record.
        """
        return self._score(model, other.features, other.b_base, other.b_dns,
                           n_samples, level)

    def leave_one_station_out(self, kinds=("flow", "gauss"), seed=0,
                              epochs=400, n_samples=128, level=0.9):
        """The pinned cross-station protocol: per held-out station, per model."""
        out = {}
        for si, xh in enumerate(self.station_xh):
            tr, te = self.station_split(si)
            per = {}
            for kind in kinds:
                model = self.fit(kind, tr, seed=seed, epochs=epochs)
                per[kind] = self.evaluate(model, te, n_samples=n_samples,
                                          level=level)
            out[float(xh)] = per
        return out
