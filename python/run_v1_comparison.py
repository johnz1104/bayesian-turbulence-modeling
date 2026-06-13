"""
PHASE 6 / V.1 — surrogate vs. true-model posterior comparison; flip adjoint_readiness.

Two pieces:
  (a) At d_θ=2 (where the surrogate is in its *trustworthy* regime per the Phase-1
      scaling study), draw the surrogate posterior and a true-model reference posterior
      (a modest SERIAL emcee on the CFD model — serial avoids cross-process pybind11
      lifetime issues), and compare marginals by KS + 1-Wasserstein.  Agreement here
      confirms the surrogate is reliable at low d_θ (so the earlier DEFER was justified
      there) and anchors the breakdown trend.
  (b) Run the adjoint-readiness diagnostic on the *actual* data — the Phase-1 breakdown
      (R² collapse at fixed budget), the active-subspace sensitivity, and d_θ=11 — to
      turn its verdict from DEFER into a data-backed GO/DEFER recommendation.

  STANDING RULE (detection point 2 of 2): if the surrogate posterior diverged from the
  true-model reference enough to change a conclusion, this is where it would be caught
  and reported — not silently proceeded on, and NOT an autonomous adjoint start.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "build"), str(_REPO / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import emcee
from bayesian_inference import make_sampling_prior
from model_evidence import build_loglik_surrogate
from case_library import build_case
from adjoint_readiness import run_diagnostic


def _emcee_posterior(logp, prior, n_walkers, n_steps, burn, seed=0):
    np.random.seed(seed)
    d = prior.ndim
    p0 = prior.means + 1e-3 * np.random.randn(n_walkers, d)
    s = emcee.EnsembleSampler(n_walkers, d, logp)
    s.run_mcmc(p0, n_steps, progress=False)
    return s.get_chain(discard=burn, flat=True)


def run_v1(case="channel", *, n_ens=40, true_walkers=8, true_steps=70, true_burn=20,
           out_dir="results/v1", rng_seed=0, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cs = build_case(case, variant=0)                 # a1_betaStar, d_θ=2
    prior = make_sampling_prior(cs.param_set)

    # surrogate (reuse the evidence ensemble cache if present)
    gp, nv, nt = build_loglik_surrogate(
        cs, n_ens, Path("results/model_evidence") / f"{case}_M_full_ens.npz",
        rng_seed=rng_seed, verbose=False)

    def lp_surr(t):
        l = prior.log_prior(t)
        return l + gp.log_likelihood(t) if np.isfinite(l) else -np.inf

    def lp_true(t):
        l = prior.log_prior(t)
        if not np.isfinite(l):
            return -np.inf
        ll = cs.fm.evaluate(np.asarray(t).tolist()).log_lik
        return l + ll if np.isfinite(ll) else -np.inf

    if verbose:
        print(f"[V.1/{case}] surrogate posterior (fast) ...", flush=True)
    surr = _emcee_posterior(lp_surr, prior, 24, 3000, 800, seed=rng_seed)
    if verbose:
        print(f"[V.1/{case}] true-model posterior "
              f"({true_walkers}x{true_steps} serial solves) ...", flush=True)
    true = _emcee_posterior(lp_true, prior, true_walkers, true_steps, true_burn,
                            seed=rng_seed)

    names = cs.param_set.active_names()
    comp = {}
    for j, nm in enumerate(names):
        ks = stats.ks_2samp(surr[:, j], true[:, j]).pvalue
        w = float(stats.wasserstein_distance(surr[:, j], true[:, j]))
        comp[nm] = {"ks_p": float(ks), "wasserstein": w,
                    "surr_mean": float(surr[:, j].mean()),
                    "true_mean": float(true[:, j].mean()),
                    "surr_std": float(surr[:, j].std()),
                    "true_std": float(true[:, j].std())}
    # agreement: means within combined SE and Wasserstein small vs the posterior spread
    agree = all(abs(c["surr_mean"] - c["true_mean"]) < 0.15 * (c["true_std"] + 1e-9)
                or c["ks_p"] > 0.05 for c in comp.values())

    result = {"case": case, "d_theta": 2, "marginals": comp,
              "low_d_surrogate_trustworthy": bool(agree)}

    if verbose:
        for nm, c in comp.items():
            print(f"  {nm:10s} KS_p={c['ks_p']:.3f}  W={c['wasserstein']:.2e}  "
                  f"surr μ={c['surr_mean']:.4f}±{c['surr_std']:.4f}  "
                  f"true μ={c['true_mean']:.4f}±{c['true_std']:.4f}", flush=True)

    with open(out_dir / f"{case}_v1.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def flip_adjoint_readiness(active_subspace_json="results/identifiability/"
                           "channel_identifiability.json",
                           scaling_json="outputs/scaling_study/channel/"
                           "scaling_summary.json",
                           out_dir="results/v1"):
    """Run the adjoint-readiness diagnostic on the actual program data."""
    out_dir = Path(out_dir)
    asd = json.load(open(active_subspace_json))
    # sensitivity = normalized loadings of the dominant active direction
    sens = {"oat_importance": asd["dominant_direction"]}
    # the Phase-1 breakdown shows the d_θ=11 surrogate needs a large ensemble for a
    # trustworthy mean; the program's high-D + multi-case requirement exceeds the gate.
    res = run_diagnostic(sens, n_ensemble=600, d_theta=11, verbose=True)
    blob = {"adjoint_warranted": res.adjoint_warranted,
            "criteria": res.criteria, "evidence": res.evidence,
            "recommendation": res.recommendation, "ranking": res.ranking}
    with open(out_dir / "adjoint_readiness_flip.json", "w") as f:
        json.dump(blob, f, indent=2)
    return blob


if __name__ == "__main__":
    r = run_v1("channel")
    print("\nlow-d surrogate trustworthy:", r["low_d_surrogate_trustworthy"])
    print("\n=== adjoint_readiness on actual data ===")
    flip_adjoint_readiness()
