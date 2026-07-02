"""A-posteriori model-form ensembles on the separated geometries.

The coupled leg of the separated-flow model-form study: sampled realizable
closures from the generative flow and the Gaussian baseline, and the corner
family of the eigenspace-perturbation method, are all propagated through the
SAME Reynolds-stress injection to predicted quantities of interest, and scored
against the DNS truths with the standard harness. The backward-facing step
scores reattachment and the measured station Cf; the periodic hills score the
wall-shear bubble geometry and the pinned mean-velocity probes (the dataset
carries no measured wall Cf/Cp).

The ensemble construction follows METHODS_OPERATIONALIZATION.md sections 6 and
9, pinned before any ensemble ran: each probabilistic member is a COHERENT
field realization (one shared latent pushed through the conditional model at
every cell), members are projected into the realizable set before injection
and re-checked in the running solve, non-converged members are reported by
count, and the eigenspace family is scored both as its own envelope and under
the charitable uniform-ensemble reading. Cross-geometry propagation reuses
these drivers unchanged: a model trained on one geometry's (feature, db) pairs
is handed to the other geometry's run_ensemble, since the conditioning is the
five invariant features with no geometry label.
"""
import numpy as np

from .bfs_injection import BFSInjection
from .hills_injection import HillsInjection
from .hills_discrepancy import HillsDiscrepancy
from .separated_apriori import SeparatedAPriori
from ..eigenspace import EigenspacePerturbation
from .. import evaluation as ev


class BFSAPosteriori:
    """Ensemble driver over one shared injection wrapper and training set.

    The conditioning features and base anisotropy at the SOLVER CELLS come
    from the injection wrapper; the training pairs at the DNS profile points
    come from the a-priori set. Models are trained on all stations (the
    a-posteriori target is the predicted quantity of interest; cross-geometry
    generalization is a separate leg of the study).
    """

    def __init__(self, apriori, injection):
        self.apriori = apriori
        self.inj = injection

    @staticmethod
    def build(cfg=None, dns=None, baseline=None):
        ap = SeparatedAPriori.build(cfg=cfg, dns=dns, baseline=baseline)
        inj = BFSInjection(cfg=cfg, dns=dns, baseline=ap.disc.baseline)
        return BFSAPosteriori(ap, inj)

    def train(self, kind, seed=0, epochs=400):
        """Fit one conditional model on all stations' (feature, db) pairs."""
        all_mask = np.ones(self.apriori.n, dtype=bool)
        return self.apriori.fit(kind, all_mask, seed=seed, epochs=epochs)

    def run_ensemble(self, model, n_members=24, shared_latent=True):
        """Propagate n_members sampled closures through the coupled solve.

        Each member: one realizable target field b_baseline + db_m at the
        solver cells (coherent shared-latent draw per the methods note),
        injected and solved. Returns the member records (reattachment, Cf
        along the downstream wall, status, iterations, diagnostics).
        """
        b_targets = model.sample_realizable_anisotropy(
            self.inj.features, base_anisotropy=self.inj.b_baseline,
            n_per=n_members, shared_latent=shared_latent)   # (N, M, 3, 3)
        members = []
        for m in range(n_members):
            r = self.inj.run(b_targets[:, m])
            xw, cf = self.inj.wall_cf(r["fields"])
            members.append({
                "reattachment": r["reattachment"],
                "status": r["status"],
                "iterations": r["iterations"],
                "all_realizable": bool(r["diagnostics"]["all_realizable"]),
                "cf_x": xw,
                "cf": cf,
            })
        return members

    def run_eigenspace(self, delta_b=1.0):
        """The corner family through the same injection (the envelope)."""
        family = EigenspacePerturbation.corner_set(self.inj.b_baseline,
                                                   delta_b=delta_b)
        members = {}
        for name, b_pert in family.items():
            bt, _ = self.inj.target_from_db(b_pert - self.inj.b_baseline)
            r = self.inj.run(bt)
            xw, cf = self.inj.wall_cf(r["fields"])
            members[name] = {
                "reattachment": r["reattachment"],
                "status": r["status"],
                "iterations": r["iterations"],
                "all_realizable": bool(r["diagnostics"]["all_realizable"]),
                "cf_x": xw,
                "cf": cf,
            }
        return members

    # ---- scoring -----------------------------------------------------------

    @staticmethod
    def score_reattachment(members, truth, level=0.9):
        """Coverage / sharpness / CRPS of the reattachment ensemble vs truth.

        Non-converged members are excluded from the scores and reported by
        count (the exclusion is part of the record, per the methods note).
        """
        ok = [m for m in members if m["status"] == "Converged"]
        xr = np.array([m["reattachment"] for m in ok])
        n_bad = len(members) - len(ok)
        if xr.size == 0:
            return {"n_members": len(members), "n_nonconverged": n_bad}
        cov, shp = ev.coverage_from_samples(np.array([truth]), xr[None, :],
                                            level=level)
        return {
            "n_members": len(members),
            "n_nonconverged": n_bad,
            "all_realizable": bool(all(m["all_realizable"] for m in ok)),
            "mean": float(xr.mean()),
            "std": float(xr.std()),
            "band": [float(np.quantile(xr, (1 - level) / 2)),
                     float(np.quantile(xr, 1 - (1 - level) / 2))],
            "contains_truth": bool(cov == 1.0),
            "sharpness": shp,
            "crps": ev.crps_ensemble(np.array([truth]), xr[None, :]),
            "abs_error_of_mean": float(abs(xr.mean() - truth)),
        }

    @staticmethod
    def score_envelope(members, truth):
        """The eigenspace method's own claim: does [min, max] contain the truth."""
        ok = {k: m for k, m in members.items() if m["status"] == "Converged"}
        xr = np.array([m["reattachment"] for m in ok.values()])
        out = {
            "corners": {k: float(m["reattachment"]) for k, m in ok.items()},
            "n_nonconverged": len(members) - len(ok),
        }
        if xr.size:
            out["envelope"] = [float(xr.min()), float(xr.max())]
            out["contains_truth"] = bool(xr.min() <= truth <= xr.max())
            # charitable uniform-ensemble reading for score comparability
            out["crps_uniform_reading"] = ev.crps_ensemble(np.array([truth]),
                                                           xr[None, :])
        return out

    def score_cf(self, members, level=0.9):
        """Coverage of the measured downstream-station Cf by the Cf ensemble.

        The DNS supplies Cf at the five downstream stations (x/h = 4, 6, 10,
        15, 19); each member's wall-Cf curve is interpolated to those stations
        and the per-station ensemble is scored against the measured value.
        """
        dns = self.apriori.disc.dns
        xs, cf_dns = dns.cf_stations()
        keep = xs > 0.0                      # downstream stations only
        xs, cf_dns = xs[keep], cf_dns[keep]
        ok = [m for m in members if m["status"] == "Converged"]
        if not ok:
            return {"n_used": 0}
        samples = np.stack([np.interp(xs, m["cf_x"], m["cf"]) for m in ok],
                           axis=1)           # (n_stations, n_members)
        cov, shp = ev.coverage_from_samples(cf_dns, samples, level=level)
        return {
            "n_used": len(ok),
            "stations_xh": xs.tolist(),
            "coverage": cov,
            "sharpness": shp,
            "crps": ev.crps_ensemble(cf_dns, samples),
            "energy_score": ev.energy_score(cf_dns[None, :],
                                            samples.T[None, :, :]),
        }


class HillsAPosteriori:
    """Ensemble driver on the periodic hills (methods note section 9).

    Same construction as the backward-facing-step driver over the hills
    injection wrapper: coherent shared-latent members, projected targets,
    per-iteration realizability, exclusion by count. Scored quantities are the
    wall-shear bubble geometry (separation, reattachment, bubble length) and
    the mean streamwise velocity at the pinned probes.
    """

    SCALAR_KEYS = ("separation", "reattachment", "bubble_length")

    def __init__(self, apriori, injection):
        self.apriori = apriori
        self.inj = injection

    @staticmethod
    def build(cfg=None, dns=None, baseline=None, case="1p0"):
        disc = HillsDiscrepancy.build(dns=dns, baseline=baseline, cfg=cfg,
                                      case=case)
        ap = SeparatedAPriori(disc)
        inj = HillsInjection(cfg=cfg, dns=dns, baseline=disc.baseline,
                             case=case)
        return HillsAPosteriori(ap, inj)

    def train(self, kind, seed=0, epochs=400):
        """Fit one conditional model on all bands' (feature, db) pairs."""
        all_mask = np.ones(self.apriori.n, dtype=bool)
        return self.apriori.fit(kind, all_mask, seed=seed, epochs=epochs)

    def _member_record(self, r):
        rec = {
            "separation": r["separation"],
            "reattachment": r["reattachment"],
            "bubble_length": r["bubble_length"],
            "status": r["status"],
            "iterations": r["iterations"],
            "all_realizable": bool(r["diagnostics"]["all_realizable"]),
        }
        if r["fields"] is not None:
            rec["u_probe"] = self.inj.probe_velocity(r["fields"])
        return rec

    def run_ensemble(self, model, n_members=24, shared_latent=True):
        """Propagate n_members sampled closures through the coupled solve.

        The model may be trained on EITHER geometry (cross-geometry
        propagation passes the other geometry's fit); sampling happens at
        this geometry's solver cells through the invariant features.
        """
        b_targets = model.sample_realizable_anisotropy(
            self.inj.features, base_anisotropy=self.inj.b_baseline,
            n_per=n_members, shared_latent=shared_latent)   # (N, M, 3, 3)
        members = []
        for m in range(n_members):
            r = self.inj.run(b_targets[:, m])
            members.append(self._member_record(r))
        return members

    def run_eigenspace(self, delta_b=1.0):
        """The corner family through the same injection (the envelope)."""
        family = EigenspacePerturbation.corner_set(self.inj.b_baseline,
                                                   delta_b=delta_b)
        members = {}
        for name, b_pert in family.items():
            bt, _ = self.inj.target_from_db(b_pert - self.inj.b_baseline)
            r = self.inj.run(bt)
            members[name] = self._member_record(r)
        return members

    # ---- scoring -----------------------------------------------------------

    @staticmethod
    def score_scalar(members, key, truth, level=0.9):
        """Coverage / sharpness / CRPS of one scalar-QoI ensemble vs truth.

        Converged members with a defined crossing (finite value) enter the
        scores; members without one (bubble absent) are counted separately,
        alongside the non-converged count, per the pinned exclusion handling.
        """
        ok = [m for m in members if m["status"] == "Converged"]
        vals = np.array([m[key] for m in ok], float)
        finite = np.isfinite(vals)
        xr = vals[finite]
        out = {
            "n_members": len(members),
            "n_nonconverged": len(members) - len(ok),
            "n_no_crossing": int(np.sum(~finite)),
        }
        if xr.size == 0:
            return out
        cov, shp = ev.coverage_from_samples(np.array([truth]), xr[None, :],
                                            level=level)
        out.update({
            "all_realizable": bool(all(m["all_realizable"] for m in ok)),
            "mean": float(xr.mean()),
            "std": float(xr.std()),
            "band": [float(np.quantile(xr, (1 - level) / 2)),
                     float(np.quantile(xr, 1 - (1 - level) / 2))],
            "contains_truth": bool(cov == 1.0),
            "sharpness": shp,
            "crps": ev.crps_ensemble(np.array([truth]), xr[None, :]),
            "abs_error_of_mean": float(abs(xr.mean() - truth)),
        })
        return out

    @staticmethod
    def score_envelope(members, key, truth):
        """The eigenspace method's own claim on one scalar quantity."""
        ok = {k: m for k, m in members.items()
              if m["status"] == "Converged" and np.isfinite(m[key])}
        xr = np.array([m[key] for m in ok.values()], float)
        out = {
            "corners": {k: float(m[key]) for k, m in ok.items()},
            "n_excluded": len(members) - len(ok),
        }
        if xr.size:
            out["envelope"] = [float(xr.min()), float(xr.max())]
            out["contains_truth"] = bool(xr.min() <= truth <= xr.max())
            # charitable uniform-ensemble reading for score comparability
            out["crps_uniform_reading"] = ev.crps_ensemble(np.array([truth]),
                                                           xr[None, :])
        return out

    def score_probes(self, members, level=0.9):
        """Coverage of the DNS mean velocity at the pinned probes."""
        truth = self.inj.probe_truth()
        ok = [m for m in members if m["status"] == "Converged"
              and "u_probe" in m]
        if not ok:
            return {"n_used": 0}
        samples = np.stack([np.asarray(m["u_probe"], float) for m in ok],
                           axis=1)            # (n_probes, n_members)
        keep = np.all(np.isfinite(samples), axis=1) & np.isfinite(truth)
        truth, samples = truth[keep], samples[keep]
        cov, shp = ev.coverage_from_samples(truth, samples, level=level)
        return {
            "n_used": len(ok),
            "n_probes": int(truth.size),
            "probe_x": self.inj.probe_x[keep].tolist(),
            "probe_y": self.inj.probe_y[keep].tolist(),
            "coverage": cov,
            "sharpness": shp,
            "crps": ev.crps_ensemble(truth, samples),
            "energy_score": ev.energy_score(truth[None, :],
                                            samples.T[None, :, :]),
        }
