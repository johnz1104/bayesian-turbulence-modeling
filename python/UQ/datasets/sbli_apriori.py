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
            out = {k: z[k] for k in z.files if k != "meta"}
            out["meta"] = json.loads(str(z["meta"]))
            out["dq"] = None if out["dq"].size == 0 else out["dq"]
            out["region"] = out["region"].astype(object)
            return out
        study = interaction_study(record, baseline, stride=stride,
                                  history=history)
        os.makedirs(results_dir, exist_ok=True)
        np.savez_compressed(
            path,
            features=study["features"], db=study["db"],
            db_free=study["db_free"],
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
    def _target(extraction, leg):
        if leg == "dq_y":
            return extraction["dq"][:, 1:2]
        if leg == "dq_joint":
            return extraction["dq"][:, 0:2]
        if leg == "db":
            return extraction["db_free"]
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
                       epochs=EPOCHS):
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

    def loso(self, leg, history=False, seeds=SEEDS, epochs=EPOCHS,
             progress=None):
        """Leave-one-wall-thermal-out over the heated-set cases (dq legs) or
        all six records (db leg). progress, when given, is called with
        (leg, held, fold_result) after each fold, so a long production run
        checkpoints and reports incrementally."""
        cases = [c for c in self.test_sets
                 if (leg == "db") or self.test_sets[c]["dq"] is not None]
        results = {}
        for held in cases:
            train_cases = [c for c in cases if c != held]
            X_tr = np.concatenate([
                self._features(self.train_sets[c], history)
                for c in train_cases])
            Y_tr = np.concatenate([
                self._target(self.train_sets[c], leg) for c in train_cases])
            X_te = self._features(self.test_sets[held], history)
            Y_te = self._target(self.test_sets[held], leg)
            regions = self.test_sets[held]["region"]
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
            results[held] = {"n_train": int(len(X_tr)),
                             "n_test": int(len(X_te)),
                             "models": per_model}
            if progress is not None:
                progress(leg, held, results[held])
        return results

    def insample(self, leg, history=False, seeds=SEEDS, epochs=EPOCHS):
        """Train on every record at the train stride, score every record at
        the test stride (the machinery check)."""
        cases = [c for c in self.test_sets
                 if (leg == "db") or self.test_sets[c]["dq"] is not None]
        X_tr = np.concatenate([
            self._features(self.train_sets[c], history) for c in cases])
        Y_tr = np.concatenate([
            self._target(self.train_sets[c], leg) for c in cases])
        X_te = np.concatenate([
            self._features(self.test_sets[c], history) for c in cases])
        Y_te = np.concatenate([
            self._target(self.test_sets[c], leg) for c in cases])
        out = {}
        for kind in ("flow", "gauss", "pooled"):
            out[kind] = [self._fit_and_score(kind, X_tr, Y_tr, X_te, Y_te,
                                             seed, epochs)[0]
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
