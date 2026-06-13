"""
PHASE 4 production — high-D identifiability per case (research_dir §4.4 / V.3; angle 3).

For a validation case at the FULL d_θ=11 closure space, reconciles three independent
identifiability diagnostics:

  (a) active subspace  — eigenspectrum of the gradient Gram matrix from a checkpointed
      set of FD true-model log-posterior gradients (the trustworthy diagnostic; each
      gradient costs 2·d_θ solves, so this is the per-case "job array");
  (b) ARD relevance    — 1/ℓ from the scalar-log-lik GP surrogate;
  (c) posterior cov    — eigenspectrum of the surrogate-MCMC posterior precision.

  Surrogate caveat (standing rule): at d_θ=11 the surrogate is in its data-hungry
  regime (Phase-1 V.1), so the *active subspace* (FD true-model gradients) is the
  gold-standard rank; (b)/(c) are corroborating.  If the three disagree on the
  low-rank story, that is reported, not smoothed over.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "build"), str(_REPO / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rans_sst_py as rs
from bayesian_inference import make_sampling_prior
from gradient_inference import GradientForwardModel, gradient_dataset
from identifiability import reconcile, scree_plot, effective_rank
from model_evidence import build_loglik_surrogate
from case_library import build_case
import emcee


def run_case_identifiability(case_name, *, n_grad=24, n_ens=120,
                             out_dir="results/identifiability", rng_seed=0,
                             verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps = rs.InferenceParameterSet.all11()
    names = ps.active_names()
    cs = build_case(case_name, variant=0, param_set=ps)
    prior = make_sampling_prior(ps)

    # (a) active subspace from FD true-model gradients (checkpointed job array)
    if verbose:
        print(f"[{case_name}] active subspace: {n_grad} FD gradients "
              f"({2*ps.n_active()} solves each) ...", flush=True)
    grad_fm = GradientForwardModel(cs.fm, n_obs=cs.n_obs, h_rel=1e-3)
    np.random.seed(rng_seed)
    thetas = prior.sample(n_grad)
    G, valid = gradient_dataset(grad_fm, prior, thetas,
                                out_dir / f"{case_name}_gradients.npz",
                                verbose=verbose)

    # (b) ARD relevance from the scalar-log-lik surrogate
    gp, nv, nt = build_loglik_surrogate(
        cs, n_ens, out_dir / f"{case_name}_loglik_ens.npz",
        rng_seed=rng_seed, verbose=verbose)
    ard_ls = gp.lengthscales()

    # (c) posterior covariance from a surrogate MCMC (corroborating; 11-D caveat)
    def lp(t):
        l = prior.log_prior(t)
        return l + gp.log_likelihood(t) if np.isfinite(l) else -np.inf
    nwalk = 2 * ps.n_active() + 2
    np.random.seed(rng_seed)
    p0 = prior.means + 1e-3 * np.random.randn(nwalk, ps.n_active())
    sampler = emcee.EnsembleSampler(nwalk, ps.n_active(), lp)
    sampler.run_mcmc(p0, 4000, progress=False)
    samples = sampler.get_chain(discard=1500, flat=True)

    rep = reconcile(names, samples=samples, gradients=G[valid], lengthscales=ard_ls)
    rep["case"] = case_name
    rep["n_grad_valid"] = int(valid.sum())
    rep.pop("_active_vecs", None)         # drop non-serializable
    scree_plot(reconcile(names, samples=samples, gradients=G[valid],
                         lengthscales=ard_ls),
               out_dir / f"{case_name}_scree.png")

    with open(out_dir / f"{case_name}_identifiability.json", "w") as f:
        json.dump(rep, f, indent=2)
    if verbose:
        print(f"[{case_name}] ranks={rep.get('ranks')} consensus="
              f"{rep.get('rank_consensus')} consistent={rep.get('ranks_consistent')}",
              flush=True)
        print(f"[{case_name}] dominant coeffs: "
              f"{rep.get('dominant_coefficients', [])[:4]}", flush=True)
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default="channel")
    ap.add_argument("--n-grad", type=int, default=24)
    ap.add_argument("--n-ens", type=int, default=120)
    args = ap.parse_args()
    run_case_identifiability(args.case, n_grad=args.n_grad, n_ens=args.n_ens)
