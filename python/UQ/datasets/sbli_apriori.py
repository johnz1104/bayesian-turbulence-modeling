"""A-priori model-form study over the impinging-shock interaction.

The pre-registered splits and scoring for the discrepancy-distribution leg:

- within-interaction: leave-one-wall-thermal-out over the five heated-set
  fields (the primary conditional-transfer axis; the held-out unit is the s
  condition), plus the train-on-all in-sample machinery check;
- far transfer: train on the attached compressible pools (the channel matrix
  through the committed extraction for dq; the channel matrix plus the twelve
  attached boundary layers for db), score the interaction records, gated by
  the attached leave-one-Mach-family-out control;
- targets: dq_y (scalar primary), the joint (dq_x, dq_y) leg, and the
  db-vector leg (b11, b22, b12; the family surfaces);
- the scoped-in strain-history ablation: the same splits with the seventh
  conditioning feature;
- region-graded aggregation (pre_switch, upstream, interaction, relaxation
  labels from the records' own landmarks) alongside the case means;
- models and protocol exactly as committed: the conditional flow and the
  Gaussian conditional on identical inputs, the pooled unconditional
  diagnostic, seeds {0, 1, 2}, 128 samples per point, criterion on seed
  means.

Test rows use the pinned per-grid-family strides; training rows use twice the
stride per direction. Extraction caches (per case, per stride, per history
flag) live under the gitignored results tree so the expensive baseline solves
and tracings run once.
"""
import json
import os

import numpy as np

from .. import evaluation
from ..generative import GenerativeDiscrepancyModel
from ..gaussian_modelform import GaussianDiscrepancyModel
from .heatflux_apriori import HeatFluxAPriori, PooledGaussianDiagnostic, \
    mach_family
from .sbli_discrepancy import interaction_study

EPOCHS = 400
SAMPLES_PER_POINT = 128
SEEDS = (0, 1, 2)

# pre-registered strides per grid family (test; train is twice each direction)
TEST_STRIDE = {"s0.5": (8, 4), "s0.75": (8, 4),
               "s1.0": (4, 4), "s1.4": (4, 4), "s1.9": (4, 4),
               "adiabatic": (8, 4)}


def _train_stride(case):
    sx, sy = TEST_STRIDE[case]
    return (2 * sx, 2 * sy)


class SBLIAPriori:
    """Assembled study set: interaction extractions at both strides plus the
    attached far-transfer pools; the split and scoring loops."""

    def __init__(self, test_sets, train_sets, attached, results_dir):
        self.test_sets = test_sets      # case -> extraction dict (test stride)
        self.train_sets = train_sets    # case -> extraction dict (train stride)
        self.attached = attached        # HeatFluxAPriori (the committed pools)
        self.results_dir = results_dir

    # ---- assembly with caching ------------------------------------------------

    @staticmethod
    def _cache_path(results_dir, case, stride, history):
        tag = f"extract_{case}_s{stride[0]}x{stride[1]}" \
              f"{'_hist' if history else ''}.npz"
        return os.path.join(results_dir, tag)

    @staticmethod
    def _extract_cached(record, baseline, stride, history, results_dir):
        path = SBLIAPriori._cache_path(results_dir, record.case, stride,
                                       history)
        if os.path.isfile(path):
            z = np.load(path, allow_pickle=True)
            if "g_basis" in z.files:
                out = {k: z[k] for k in z.files if k != "meta"}
                out["meta"] = json.loads(str(z["meta"]))
                out["dq"] = None if out["dq"].size == 0 else out["dq"]
                out["region"] = out["region"].astype(object)
                return out
            # pre-basis cache (no objective-representation keys): stale under
            # the amendment; regenerate from the cached converged baseline
            print(f"  [extract] {os.path.basename(path)} predates the "
                  f"objective-basis representation; regenerating")
        study = interaction_study(record, baseline, stride=stride,
                                  history=history)
        os.makedirs(results_dir, exist_ok=True)
        np.savez_compressed(
            path,
            features=study["features"], db=study["db"],
            db_free=study["db_free"],
            g_basis=study["g_basis"],
            basis_M=study["basis_M"],
            dq=study["dq"] if study["dq"] is not None else np.zeros(0),
            x=study["x"], y=study["y"],
            region=np.asarray(study["region"], dtype=str),
            realizable_fraction=study["realizable_fraction"],
            meta=json.dumps(study["meta"]))
        return study

    @staticmethod
    def build(records, baselines, results_dir, history=True, root=None):
        """records/baselines: dicts case -> loader record / SOLVED baseline.
        The attached pools come from the committed heat-flux assembly."""
        test_sets, train_sets = {}, {}
        for case, record in records.items():
            base = baselines[case]
            test_sets[case] = SBLIAPriori._extract_cached(
                record, base, TEST_STRIDE[case], history, results_dir)
            train_sets[case] = SBLIAPriori._extract_cached(
                record, base, _train_stride(case), history, results_dir)
        attached = HeatFluxAPriori.build(root=root)
        return SBLIAPriori(test_sets, train_sets, attached, results_dir)

    # ---- target and feature views ----------------------------------------------

    @staticmethod
    def basis_feasibility(extraction, gate_median=0.20):
        """The pre-registered objective-basis feasibility gate, evaluated on
        the DEPLOYED representation (the cached normalized three-tensor maps):
        the relative residual of reconstructing db_free from the stored
        coefficients through the stored per-sample map, over the
        interaction-region samples. Returns the record the drivers persist;
        gate_pass False triggers the registered raw-component reversion
        (flow and Gaussian identically), stated wherever scores are reported.
        """
        keep = extraction["region"] == "interaction"
        db_free = np.asarray(extraction["db_free"], float)[keep]
        g = np.asarray(extraction["g_basis"], float)[keep]
        M = np.asarray(extraction["basis_M"], float)[keep]
        recon = np.einsum("nij,nj->ni", M, g)
        num = np.linalg.norm(recon - db_free, axis=1)
        den = np.maximum(np.linalg.norm(db_free, axis=1), 1e-300)
        rel = num / den
        med = float(np.median(rel))
        return {"n_interaction_samples": int(keep.sum()),
                "rel_residual_median": med,
                "rel_residual_p90": float(np.percentile(rel, 90)),
                "gate_median_max": float(gate_median),
                "gate_pass": bool(med <= gate_median)}

    @staticmethod
    def _target(extraction, leg, db_raw=False):
        if leg == "dq_y":
            return extraction["dq"][:, 1:2]
        if leg == "dq_joint":
            return extraction["dq"][:, 0:2]
        if leg == "db":
            # the amendment's objective representation: models train on the
            # integrity-basis coefficients; scoring reconstructs components.
            # db_raw True is the registered feasibility-gate reversion (raw
            # free components, flow and Gaussian identically).
            return extraction["db_free"] if db_raw else extraction["g_basis"]
        raise ValueError(f"unknown leg '{leg}'")

    @staticmethod
    def _features(extraction, history):
        f = extraction["features"]
        return f if history else f[:, :6]

    # ---- model zoo --------------------------------------------------------------

    @staticmethod
    def _make(kind, n_features, n_targets, seed):
        if kind == "flow":
            return GenerativeDiscrepancyModel(
                n_features=n_features, n_targets=n_targets,
                n_layers=8, hidden=64, seed=seed)
        if kind == "gauss":
            return GaussianDiscrepancyModel(
                n_features=n_features, n_targets=n_targets,
                hidden=64, seed=seed)
        if kind == "pooled":
            return PooledGaussianDiagnostic(seed=seed)
        raise ValueError(kind)

    def _fit_and_score(self, kind, X_tr, Y_tr, X_te, Y_te, seed,
                       epochs=EPOCHS, component_map=None):
        model = self._make(kind, X_tr.shape[1], Y_tr.shape[1], seed)
        model.fit(X_tr, Y_tr, epochs=epochs, lr=1e-3, batch=256)
        # the committed draw-seeding pattern: the pooled diagnostic takes a
        # seed, the torch models are seeded through the global generator
        if kind == "pooled":
            S = np.asarray(model.sample(X_te, n_per=SAMPLES_PER_POINT,
                                        seed=seed))
        else:
            import torch
            torch.manual_seed(seed)
            S = np.asarray(model.sample(X_te, n_per=SAMPLES_PER_POINT))
        if component_map is not None:
            # draws live in basis-coefficient space; the pre-registered
            # clauses are evaluated on the reconstructed tensor components,
            # so the per-sample exact linear map is applied BEFORE scoring
            # and Y_te is already the component-space truth
            S = np.einsum("nij,nmj->nmi", component_map, S)
        out = {}
        for level in (0.9, 0.5):
            covs, shps = [], []
            for d in range(Y_te.shape[1]):
                cov, shp = evaluation.coverage_from_samples(
                    Y_te[:, d], S[:, :, d], level=level)
                covs.append(float(cov))
                shps.append(float(shp))
            out[f"coverage_{level:g}"] = covs
            out[f"sharpness_{level:g}"] = shps
        out["crps"] = [float(evaluation.crps_ensemble(Y_te[:, d], S[:, :, d]))
                       for d in range(Y_te.shape[1])]
        if Y_te.shape[1] > 1:
            out["energy_score"] = float(evaluation.energy_score(Y_te, S))
        return out, S

    @staticmethod
    def _region_coverage(Y, S, regions, level=0.9):
        out = {}
        for name in ("pre_switch", "upstream", "interaction", "relaxation"):
            m = regions == name
            if m.sum() < 5:
                continue
            out[name] = [float(evaluation.coverage_from_samples(
                Y[m][:, d], S[m][:, :, d], level=level)[0])
                for d in range(Y.shape[1])]
        return out

    # ---- pre-registered splits ---------------------------------------------------

    def db_gate(self):
        """Evaluate the deployed-basis feasibility gate on every heated test
        extraction; the db leg reverts to raw components unless EVERY fold
        passes (one convention for the whole leg, as registered)."""
        gates = {c: self.basis_feasibility(self.test_sets[c])
                 for c in self.test_sets
                 if self.test_sets[c]["dq"] is not None}
        all_pass = all(g["gate_pass"] for g in gates.values())
        return {"per_fold": gates, "all_pass": bool(all_pass),
                "representation": ("objective-basis" if all_pass
                                   else "raw-components (registered "
                                        "feasibility reversion)")}

    def loso(self, leg, history=False, seeds=SEEDS, epochs=EPOCHS):
        """Leave-one-wall-thermal-out over the FIVE heated-set cases for
        every leg (the pre-registered split: train on four s conditions,
        score the held-out fifth). The adiabatic campaign never trains; for
        the db leg it is scored as the independent-campaign second truth
        surface on the s = 1.0 fold, reported separately."""
        cases = [c for c in self.test_sets
                 if self.test_sets[c]["dq"] is not None]
        db_raw = leg == "db" and not self.db_gate()["all_pass"]
        results = {}
        for held in cases:
            train_cases = [c for c in cases if c != held]
            X_tr = np.concatenate([
                self._features(self.train_sets[c], history)
                for c in train_cases])
            Y_tr = np.concatenate([
                self._target(self.train_sets[c], leg, db_raw=db_raw)
                for c in train_cases])
            X_te = self._features(self.test_sets[held], history)
            cmap = (self.test_sets[held]["basis_M"]
                    if leg == "db" and not db_raw else None)
            Y_te = (self.test_sets[held]["db_free"] if leg == "db"
                    else self._target(self.test_sets[held], leg))
            regions = self.test_sets[held]["region"]
            per_model = {}
            for kind in ("flow", "gauss", "pooled"):
                per_seed = []
                for seed in seeds:
                    scores, S = self._fit_and_score(
                        kind, X_tr, Y_tr, X_te, Y_te, seed, epochs,
                        component_map=cmap)
                    scores["region_coverage_0.9"] = self._region_coverage(
                        Y_te, S, regions)
                    per_seed.append(scores)
                per_model[kind] = per_seed
            results[held] = {"n_train": int(len(X_tr)),
                             "n_test": int(len(X_te)),
                             "models": per_model}
            if leg == "db" and held == "s1.0" and "adiabatic" in self.test_sets:
                # the independent-campaign second truth surface: the same
                # s1.0-fold-trained models scored on the adiabatic extraction
                adia = self.test_sets["adiabatic"]
                X_ad = self._features(adia, history)
                Y_ad = adia["db_free"]
                cmap_ad = adia["basis_M"]
                surface = {}
                for kind in ("flow", "gauss", "pooled"):
                    per_seed = []
                    for seed in seeds:
                        scores, _ = self._fit_and_score(
                            kind, X_tr, Y_tr, X_ad, Y_ad, seed, epochs,
                            component_map=cmap_ad)
                        per_seed.append(scores)
                    surface[kind] = per_seed
                results[held]["independent_campaign_surface"] = {
                    "n_test": int(len(X_ad)), "models": surface}
        return results

    def insample(self, leg, history=False, seeds=SEEDS, epochs=EPOCHS):
        """Train on every record at the train stride, score every record at
        the test stride (the machinery check)."""
        cases = [c for c in self.test_sets
                 if self.test_sets[c]["dq"] is not None]
        db_raw = leg == "db" and not self.db_gate()["all_pass"]
        X_tr = np.concatenate([
            self._features(self.train_sets[c], history) for c in cases])
        Y_tr = np.concatenate([
            self._target(self.train_sets[c], leg, db_raw=db_raw)
            for c in cases])
        X_te = np.concatenate([
            self._features(self.test_sets[c], history) for c in cases])
        if leg == "db":
            Y_te = np.concatenate([self.test_sets[c]["db_free"]
                                   for c in cases])
            cmap = (None if db_raw else
                    np.concatenate([self.test_sets[c]["basis_M"]
                                    for c in cases]))
        else:
            Y_te = np.concatenate([
                self._target(self.test_sets[c], leg) for c in cases])
            cmap = None
        out = {}
        for kind in ("flow", "gauss", "pooled"):
            out[kind] = [self._fit_and_score(kind, X_tr, Y_tr, X_te, Y_te,
                                             seed, epochs,
                                             component_map=cmap)[0]
                         for seed in seeds]
        return {"n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                "models": out}

    # ---- attached far-transfer pools ----------------------------------------------

    def _attached_dq_pool(self):
        """The committed channel-matrix rows (features, dq): the dq
        far-transfer training pool (the only attached source with the full
        flux vector)."""
        X, Y = [], []
        for tag in self.attached.gv:
            rec = self.attached.cases[tag]
            X.append(rec["features"])
            Y.append(rec["dq"])
        return np.concatenate(X), np.concatenate(Y)

    def far_transfer(self, leg, history=False, seeds=SEEDS, epochs=EPOCHS):
        """Train on the attached pool, score every interaction record.

        The attached rows carry no history feature, so this axis always runs
        on the six local features (stated in the numbers JSON); dq legs use
        the channel matrix, the db leg is out of scope for this loop until a
        matched attached-db assembly exists and raises if requested."""
        if leg not in ("dq_y", "dq_joint"):
            raise ValueError("far transfer is pinned to the dq legs; the "
                             "attached db pool rides the stress sensitivity")
        X_tr, dq_tr = self._attached_dq_pool()
        cols = [1] if leg == "dq_y" else [0, 1]
        Y_tr = dq_tr[:, cols]
        results = {}
        for case, ext in self.test_sets.items():
            if ext["dq"] is None:
                continue
            X_te = self._features(ext, history=False)
            Y_te = self._target(ext, leg)
            regions = ext["region"]
            per_model = {}
            for kind in ("flow", "gauss", "pooled"):
                per_seed = []
                for seed in seeds:
                    scores, S = self._fit_and_score(
                        kind, X_tr, Y_tr, X_te, Y_te, seed, epochs)
                    scores["region_coverage_0.9"] = self._region_coverage(
                        Y_te, S, regions)
                    per_seed.append(scores)
                per_model[kind] = per_seed
            results[case] = {"n_train": int(len(X_tr)),
                             "n_test": int(len(X_te)),
                             "models": per_model}
        return results

    def attached_control(self, leg="dq_y", seeds=SEEDS, epochs=EPOCHS):
        """Leave-one-Mach-family-out over the channel matrix (the committed
        families): the health gate of the far-transfer clause."""
        tags = list(self.attached.gv)
        fams = sorted({mach_family(t) for t in tags})
        cols = [1] if leg == "dq_y" else [0, 1]
        results = {}
        for fam in fams:
            te = [t for t in tags if mach_family(t) == fam]
            tr = [t for t in tags if t not in te]
            X_tr = np.concatenate([self.attached.cases[t]["features"]
                                   for t in tr])
            Y_tr = np.concatenate([self.attached.cases[t]["dq"][:, cols]
                                   for t in tr])
            per_case = {}
            for t in te:
                X_te = self.attached.cases[t]["features"]
                Y_te = self.attached.cases[t]["dq"][:, cols]
                per_model = {}
                for kind in ("flow", "gauss", "pooled"):
                    per_model[kind] = [self._fit_and_score(
                        kind, X_tr, Y_tr, X_te, Y_te, seed, epochs)[0]
                        for seed in seeds]
                per_case[t] = per_model
            results[f"family_{fam}"] = per_case
        return results


def seed_mean(per_seed, key, component=None):
    """Mean over the per-seed score dicts of one metric (a list per
    component); the criterion is always read on this mean."""
    vals = []
    for s in per_seed:
        v = s[key]
        vals.append(v if np.isscalar(v) else
                    (v[component] if component is not None else v))
    return float(np.mean(vals)) if np.isscalar(vals[0]) \
        else [float(np.mean([v[i] for v in vals]))
              for i in range(len(vals[0]))]
