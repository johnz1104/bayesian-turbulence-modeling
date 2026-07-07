"""A-priori conditional model-form study on the compressible heat-flux
discrepancy.

The study module of the committed pre-registration
(UQ-RANS_research/heatflux_modelform/PRE_REGISTRATION.md). Two legs share the
infrastructure here:

- The conditional-model leg assembles the (features, dq) pairs of the
  compressible attached matrix through the committed extraction
  (UQ.datasets.compressible_discrepancy: Favre moments in wall units, the
  gradient-diffusion baseline at the fixed Pr_t = 0.9, the M_t-extended
  invariant features), pools them over the pinned case splits, and fits the
  conditional normalizing flow and the Gaussian conditional baseline on
  identical inputs, scoring per held-out CASE through UQ.evaluation. The
  held-out unit is the case (profiles are internally correlated; point-level
  splits would leak). Targets: the wall-normal component dq_y as primary
  (defined for every dataset) and the joint (dq_x, dq_y) on the channel
  matrix as the family leg; the spanwise component is a symmetry null and is
  never modeled.

- The secondary axis re-scores the committed global-correction residuals on
  the held-out thermal block with the pinned physically normalized conformal
  scores (residual over the case's baseline-predicted wall flux as primary;
  the semi-local sqrt(rho+) rescaling as sensitivity). It reuses the cached
  calibration ensembles and the committed pooled-posterior protocol; it does
  NOT recalibrate.

Splits, hyperparameters, seeds, masks and score definitions are the
pre-registered ones; nothing here is tuned toward any evaluated number.
"""
import numpy as np

from .. import conformal as cf
from .. import evaluation as ev
from ..gaussian_modelform import GaussianDiscrepancyModel
from ..generative import GenerativeDiscrepancyModel
from .compressible_baseline import CompressibleChannelSST
from .compressible_discrepancy import channel_study, flatplate_study
from .ckm_channel import CKMChannelDNS, CKM_CASES
from .crossmach_study import CrossMachStudy
from .gv_channel import GVChannelDNS, GV_CASES
from .zdc_flatplate import FlatPlateDNS, ZDC_CASES

# the pre-registered cross-Mach split (identical to the committed calibration
# study): train on the M_CLx < 1.0 channel cases, hold out M_CLx >= 1.47
LOW_MACH = tuple(c for c in GV_CASES if GVChannelDNS.parse_tag(c)[1] < 1.0)
HIGH_MACH = tuple(c for c in GV_CASES if GVChannelDNS.parse_tag(c)[1] >= 1.47)

# leave-one-Mach-family-out: families pinned by nominal M_CLx level
MACH_FAMILY_EDGES = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.7), (1.7, 2.15),
                     (2.15, np.inf))

# pre-registered target definitions: dq column indices per leg
TARGETS = {"wall_normal": (1,), "joint_xy": (0, 1)}

# pinned protocol constants (PRE_REGISTRATION.md)
MODEL_SEEDS = (0, 1, 2)
N_SAMPLES = 128
EPOCHS = 400
LEVELS = (0.9, 0.5)

# the nominal bulk Mach of the independent-code cross-check cases (stated in
# the dataset provenance; the record itself carries no directory-name tag)
CKM_BULK_MACH = {"M1p5": 1.5, "M3p0": 3.0}


def mach_family(case):
    """Family index of a channel-matrix case under the pinned Mach edges."""
    m = GVChannelDNS.parse_tag(case)[1]
    for i, (lo, hi) in enumerate(MACH_FAMILY_EDGES):
        if lo <= m < hi:
            return i
    raise ValueError(case)


class PooledGaussianDiagnostic:
    """Unconditional Gaussian on the training dq: constant mean and variance,
    no features. The labeled what-conditioning-buys diagnostic of the
    pre-registration, never a criterion baseline. API-compatible with the
    conditional models (fit / sample) so it runs through the same loop.
    """

    def __init__(self, seed=0):
        self.seed = int(seed)
        self.mu = None
        self.sd = None

    def fit(self, features, targets, **_):
        Y = np.asarray(targets, dtype=float).reshape(len(targets), -1)
        self.mu = Y.mean(axis=0)
        self.sd = Y.std(axis=0) + 1e-12
        return self

    def sample(self, features, n_per=1, shared_latent=False, seed=None):
        # one pooled N(mu, sd^2) per component, independent of the features
        rng = np.random.default_rng(self.seed if seed is None else seed)
        n = len(np.asarray(features))
        eps = rng.standard_normal((n, n_per, self.mu.size))
        return self.mu[None, None, :] + self.sd[None, None, :] * eps


class HeatFluxAPriori:
    """The (features, dq) study set over the compressible attached matrix.

    cases: dict tag -> record with the masked per-case arrays
      features (n, 6)   five Pope invariants + M_t (the committed feature set)
      dq       (n, 3)   q_hat - q_GDH in (u_tau, T_w) units, per component
    plus metadata (nominal condition, M_t span, the baseline wall flux).
    gv / ckm / plates list the converged tags per dataset; skipped records
    the excluded cases with their status classification.
    """

    def __init__(self, cases, gv, ckm, plates, skipped):
        self.cases = cases
        self.gv = tuple(gv)
        self.ckm = tuple(ckm)
        self.plates = tuple(plates)
        self.skipped = dict(skipped)

    # ---- assembly ------------------------------------------------------------

    @staticmethod
    def interior_mask(dns):
        """The committed extraction convention: outside the buffer layer
        (y+ > 30, where the baseline interpolation and the anchor are
        trusted) and inside the outer 90 percent (y/delta < 0.9)."""
        return (dns.yplus > 30.0) & (dns.y_outer < 0.9)

    @staticmethod
    def _finish_record(dns, study, kind, extra_valid=None):
        mask = HeatFluxAPriori.interior_mask(dns)
        if extra_valid is not None:
            mask = mask & extra_valid
        feats = study["features"]
        dq = study["dq"]
        # guard: only rows where every feature and target is finite enter
        finite = np.all(np.isfinite(feats), axis=1) \
            & np.all(np.isfinite(dq), axis=1)
        n_dropped = int(np.sum(mask & ~finite))
        mask = mask & finite
        m_t = feats[mask, 5]
        rec = {
            "status": "converged",
            "kind": kind,
            "features": feats[mask],
            "dq": dq[mask],
            "n": int(mask.sum()),
            "n_dropped": n_dropped,
            "m_t_min": float(m_t.min()) if mask.any() else np.nan,
            "m_t_max": float(m_t.max()) if mask.any() else np.nan,
            "realizable_fraction": study["summary"]["realizable_fraction"],
        }
        return rec

    @staticmethod
    def _channel_record(dns, kind, m_nom, re_nom):
        study = channel_study(dns)
        if study["status"] != "converged":
            return {"status": study["status"], "kind": kind}
        rec = HeatFluxAPriori._finish_record(dns, study, kind)
        rec["m_nom"] = float(m_nom)
        rec["re_nom"] = float(re_nom)
        rec["b_q_base"] = float(study["meta"]["baseline_qois"]["b_q"])
        return rec

    @staticmethod
    def _plate_record(dns):
        study = flatplate_study(dns)
        if study["status"] != "converged":
            return {"status": study["status"], "kind": "plate"}
        rec = HeatFluxAPriori._finish_record(dns, study, "plate",
                                             extra_valid=dns.q_hat_valid)
        rec["m_nom"] = float(dns.wall["m_inf"])
        rec["tw_tr"] = float(dns.wall["tw_tr"])
        # the frozen-mean plate baseline predicts no wall flux (stated scope)
        rec["b_q_base"] = None
        return rec

    @staticmethod
    def build(root=None, gv_cases=GV_CASES, ckm_cases=CKM_CASES,
              zdc_cases=ZDC_CASES):
        cases, gv, ckm, plates, skipped = {}, [], [], [], {}
        for tag in gv_cases:
            if not GVChannelDNS.is_available(tag, root):
                skipped[tag] = "data_missing"
                continue
            re_nom, m_nom = GVChannelDNS.parse_tag(tag)
            rec = HeatFluxAPriori._channel_record(
                GVChannelDNS.load(tag, root), "gv", m_nom, re_nom)
            if rec["status"] != "converged":
                skipped[tag] = rec["status"]
                continue
            cases[tag] = rec
            gv.append(tag)
        for tag in ckm_cases:
            if not CKMChannelDNS.is_available(tag, root):
                skipped[tag] = "data_missing"
                continue
            dns = CKMChannelDNS.load(tag, root)
            rec = HeatFluxAPriori._channel_record(
                dns, "ckm", CKM_BULK_MACH[tag], dns.re_tau)
            if rec["status"] != "converged":
                skipped[tag] = rec["status"]
                continue
            cases[tag] = rec
            ckm.append(tag)
        for tag in zdc_cases:
            if not FlatPlateDNS.is_available(tag, root):
                skipped[tag] = "data_missing"
                continue
            rec = HeatFluxAPriori._plate_record(FlatPlateDNS.load(tag, root))
            if rec["status"] != "converged":
                skipped[tag] = rec["status"]
                continue
            cases[tag] = rec
            plates.append(tag)
        return HeatFluxAPriori(cases, gv, ckm, plates, skipped)

    # ---- splits ---------------------------------------------------------------

    def low_mach(self):
        return tuple(t for t in self.gv if t in LOW_MACH)

    def high_mach(self):
        return tuple(t for t in self.gv if t in HIGH_MACH)

    def loco_splits(self):
        """Leave-one-case-out within the low-Mach training cases (the
        pre-registered in-distribution held-out control)."""
        low = self.low_mach()
        return [(t, tuple(u for u in low if u != t)) for t in low]

    def lomo_splits(self):
        """Leave-one-Mach-family-out over the channel matrix: (family index,
        held-out tags, training tags) per pinned family."""
        fams = sorted(set(mach_family(t) for t in self.gv))
        return [(f,
                 tuple(t for t in self.gv if mach_family(t) == f),
                 tuple(t for t in self.gv if mach_family(t) != f))
                for f in fams]

    # ---- fit and score ---------------------------------------------------------

    @staticmethod
    def target_columns(components):
        if isinstance(components, str):
            return TARGETS[components]
        return tuple(int(c) for c in components)

    def rows(self, tags, components):
        """Pool the (features, dq[:, components]) rows of the given cases."""
        comps = list(self.target_columns(components))
        X = np.concatenate([self.cases[t]["features"] for t in tags], axis=0)
        Y = np.concatenate([self.cases[t]["dq"][:, comps] for t in tags],
                           axis=0)
        return X, Y

    def fit(self, kind, tags, components, seed=0, epochs=EPOCHS, lr=1e-3,
            batch=256):
        """Fit one model kind on the pooled training rows (pinned settings)."""
        X, Y = self.rows(tags, components)
        if kind == "flow":
            model = GenerativeDiscrepancyModel(X.shape[1], Y.shape[1],
                                               n_layers=8, hidden=64,
                                               seed=seed)
        elif kind == "gauss":
            model = GaussianDiscrepancyModel(X.shape[1], Y.shape[1],
                                             hidden=64, seed=seed)
        elif kind == "pooled":
            model = PooledGaussianDiagnostic(seed=seed)
        else:
            raise ValueError(f"unknown model kind '{kind}'")
        model.fit(X, Y, epochs=epochs, lr=lr, batch=batch)
        return model

    def evaluate(self, model, tag, components, n_samples=N_SAMPLES,
                 levels=LEVELS, sample_seed=None):
        """Score one case: per-component coverage and sharpness at the pinned
        nominal levels, CRPS, reliability error, the wall-normal PIT, and the
        energy score on multivariate targets. Returns a JSON-ready dict; the
        PIT array rides under 'arrays' for the figure cache. sample_seed pins
        the predictive draws per evaluation, so a score depends only on the
        (model, case, seed) triple and never on the call order."""
        comps = list(self.target_columns(components))
        rec = self.cases[tag]
        X = rec["features"]
        y = rec["dq"][:, comps]
        if isinstance(model, PooledGaussianDiagnostic):
            s = model.sample(X, n_per=n_samples, seed=sample_seed)
        else:
            if sample_seed is not None:
                import torch  # the conditional models already require torch
                torch.manual_seed(sample_seed)
            s = model.sample(X, n_per=n_samples)
        out = {"n_test": rec["n"], "m_t_max": rec["m_t_max"]}
        for j, c in enumerate(comps):
            name = "xyz"[c]
            for lv in levels:
                cov, shp = ev.coverage_from_samples(y[:, j], s[:, :, j],
                                                    level=lv)
                out[f"coverage_{name}_{lv:g}"] = cov
                out[f"sharpness_{name}_{lv:g}"] = shp
            out[f"crps_{name}"] = ev.crps_ensemble(y[:, j], s[:, :, j])
            out[f"reliability_error_{name}"] = ev.reliability_error(
                y[:, j], s[:, :, j])
        if len(comps) > 1:
            out["energy_score"] = ev.energy_score(y, s)
        if 1 in comps:
            jy = comps.index(1)
            pit = ev.pit_values(y[:, jy], s[:, :, jy])
            out["pit_ks_pvalue"] = ev.pit_uniformity_pvalue(pit)
            out["arrays"] = {"pit_y": pit}
        return out


# ---- the secondary axis: normalized conformal re-scoring ---------------------


def baseline_wall_flux(dns):
    """The case's own deterministic 1-D SST wall-flux prediction at the fixed
    nominal coefficients (the pinned primary conformal scale; a model
    quantity, so it uses no held-out truth). None when the solve does not
    converge (classified, not raised)."""
    pr = dns.pr_molecular if dns.pr_molecular is not None \
        else np.full(dns.n, dns.wall["pr_w"])
    out = CompressibleChannelSST(
        re_tau_w=dns.re_tau, m_tau=dns.wall["m_tau"],
        gamma=dns.wall["gamma_w"], T_mu_samples=(dns.T, dns.mu),
        T_pr_samples=(dns.T, pr)).solve()
    if out["status"] != "converged":
        return None
    return float(out["b_q"])


def thermal_score_weights(cal, score, b_q_base=None):
    """Per-QoI weights of the pinned conformal scores on the thermal block.

    The weighted nonconformity score is s_i = w_i |y_i - m_i|; the conformal
    interval at a test point is m_i +/- q / w_i, so coverage is
    mean(w |resid| <= q). The thermal block is [B_q, q profile stations].

    - absolute: w = 1 (the committed comparison line).
    - bq_base: w = 1 / max(|B_q_base|, 1e-4), the residual in units of the
      case's baseline-predicted wall flux; the floor guards the thermally
      quiescent limit and never binds on this matrix.
    - semilocal: w = sqrt(rho+) per profile station, the (u_tau, T_w) to
      semi-local flux-unit conversion q* = q_hat sqrt(rho+); the wall-flux
      entry keeps rho+ = 1 at the wall.
    """
    n_thermal = cal.heldout_index.size
    if score == "absolute":
        return np.ones(n_thermal)
    if score == "bq_base":
        return np.full(n_thermal, 1.0 / max(abs(b_q_base), 1e-4))
    if score == "semilocal":
        return np.concatenate([[1.0], np.sqrt(cal.dns.rho[cal.q_idx])])
    raise ValueError(f"unknown score '{score}'")


class ThermalConformalRescore:
    """Normalized split-conformal re-scoring of the committed
    global-correction residuals on the held-out thermal block.

    Reuses the committed cross-Mach protocol exactly: the pooled posterior
    over the training cases with the learning rate moment-matched on their
    likelihood stations, and the surrogate point predictions from the cached
    ensembles. Only the nonconformity score changes (the pinned weights
    above); nothing is recalibrated. The absolute score runs through the
    identical pooled-calibration path as the control, so the normalized
    against absolute comparison isolates the normalization itself.
    """

    SCORES = ("absolute", "bq_base", "semilocal")

    def __init__(self, cals, train, test, seed=0):
        self.cals = cals
        self.train = [t for t in train if t in cals]
        self.test = [t for t in test if t in cals]
        study = CrossMachStudy(cals)
        post1 = study.pooled_posterior_samples(self.train, 1.0, seed=seed)
        # eta moment-matched on the TRAINING cases' likelihood stations only,
        # exactly the committed protocol (never on any evaluated coverage)
        self.eta = float(np.mean([
            cals[t].calibrate_eta(post1, cals[t].lik_index)
            for t in self.train]))
        self.post_t = study.pooled_posterior_samples(self.train, self.eta,
                                                     seed=seed)
        self._resid = {}
        self._bq = {}

    def residuals(self, tag):
        """|truth - pooled-posterior point prediction| on the thermal block."""
        if tag not in self._resid:
            cal = self.cals[tag]
            pred = cal.point_prediction(self.post_t, cal.heldout_index)
            self._resid[tag] = np.abs(
                cal.qoi_truth[cal.heldout_index] - pred)
        return self._resid[tag]

    def weights(self, tag, score):
        if score == "bq_base" and tag not in self._bq:
            self._bq[tag] = baseline_wall_flux(self.cals[tag].dns)
        return thermal_score_weights(self.cals[tag], score,
                                     b_q_base=self._bq.get(tag))

    def run(self, score, level=0.9):
        """Calibrate the weighted score on the training cases, evaluate the
        held-out cases; per-case coverage, mean width in raw units, and the
        conformal gap (the honest exchangeability-violation measure)."""
        cal_scores = np.concatenate([
            self.residuals(t) * self.weights(t, score) for t in self.train])
        q = cf.conformal_quantile(cal_scores, alpha=1.0 - level)
        out = {"score": score, "level": level, "eta": self.eta,
               "quantile": float(q), "n_cal": int(cal_scores.size),
               "per_case": {}}
        covs = []
        for t in self.test:
            w = self.weights(t, score)
            cov = float(np.mean(self.residuals(t) * w <= q))
            out["per_case"][t] = {
                "coverage": cov,
                # interval width back in raw thermal units: 2 q / w per QoI
                "sharpness": float(np.mean(2.0 * q / w)),
            }
            covs.append(cov)
        out["mean_coverage"] = float(np.mean(covs))
        out["gap"] = float(level - out["mean_coverage"])
        return out
