"""
PHASE 3 production — model evidence + Bayes factors for term-toggled closures
(research_dir.md §4.3 / V.4; angle 7).

For each validation case and each closure variant (M_full / M_nolim / M_kw) it builds a
checkpointed CFD ensemble over the narrowed prior, trains a scalar-log-likelihood GP
surrogate (the evidence integrand is then microsecond-cheap), and estimates log Z by
BOTH thermodynamic integration and stepping-stone (the mandated dual cross-check).
Bayes factors vs. M_full with a Jeffreys/Kass-Raftery label follow, plus the mandatory
prior sweep (Bayes factors are prior-sensitive; Lindley-Bartlett).

The active subset is held to a1_betaStar (d_θ=2): the question is *structural* (which
closure), and at d_θ=2 the surrogate is in its trustworthy regime (Phase-1 V.1), so the
evidence rests on a reliable integrand.

  Surrogate-trust note: the scalar-loglik GP is used for its MEAN (evidence integrand);
  its mean is accurate at d_θ=2 (V.1 R²≳0.95 for n≳40).  If a Bayes-factor *direction*
  flips under the prior sweep or between TI and SS beyond their SE, that is reported
  (per the standing rule) rather than silently trusted.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "build"), str(_REPO / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bayesian_inference import GPSurrogate, latin_hypercube, make_sampling_prior
from evidence import log_evidence, bayes_factor
from case_library import build_case, VARIANTS


def build_loglik_surrogate(case_spec, n_ens, cache_path, *, rng_seed=0, verbose=True,
                           noise_floor=None):
    """Checkpointed LHS ensemble -> scalar-log-likelihood GPSurrogate over the prior.

    ``noise_floor`` (passed to ``GPSurrogate.train``) bounds the GP noise away from the
    interpolating/overconfident regime — the V.1 surrogate-trust fix.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prior = case_spec.prior
    if cache_path.exists():
        d = np.load(cache_path)
        X, y = d["X"], d["y"]
    else:
        np.random.seed(rng_seed)
        X = latin_hypercube(n_ens, prior.ndim, prior.lower, prior.upper)
        y = np.full(n_ens, -np.inf)
        t0 = time.time()
        for i in range(n_ens):
            y[i] = case_spec.fm.evaluate(X[i].tolist()).log_lik
            if verbose and (i + 1) % 20 == 0:
                print(f"    [{case_spec.name}/{case_spec.variant_name}] "
                      f"{i+1}/{n_ens}  [{(i+1)/(time.time()-t0):.2f} solve/s]",
                      flush=True)
        np.savez(cache_path, X=X, y=y)
    valid = np.isfinite(y) & (y > -1e5)
    gp = GPSurrogate()
    gp.train(X[valid], y[valid], optimize_restarts=6, noise_floor=noise_floor)
    return gp, int(valid.sum()), int(len(y))


def run_case_evidence(case_name, *, n_ens=80, out_dir="results/model_evidence",
                      prior_rel_stds=(0.10, 0.15, 0.20), rng_seed=0, verbose=True,
                      noise_floor=None):
    """
    Estimate log Z (TI + SS) for all three closure variants on one case, with a prior
    sweep, and compute Bayes factors vs. M_full.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"case": case_name, "variants": {}, "bayes_factors": {},
              "prior_sweep": {}}

    # --- baseline prior (rel_std=0.15) logZ for each variant ---
    base_logZ = {}
    for v, vname in VARIANTS.items():
        cs = build_case(case_name, variant=v)
        cache = out_dir / f"{case_name}_{vname}_ens.npz"
        gp, nv, nt = build_loglik_surrogate(cs, n_ens, cache, rng_seed=rng_seed,
                                            verbose=verbose, noise_floor=noise_floor)
        loglik = gp.log_likelihood
        logprior = cs.prior.log_prior
        res = log_evidence(loglik, logprior, cs.prior.means, rng_seed=rng_seed)
        base_logZ[vname] = res["ti_logZ"]
        result["variants"][vname] = {
            "ti_logZ": res["ti_logZ"], "ti_se": res["ti_se"],
            "ss_logZ": res["ss_logZ"], "ss_se": res["ss_se"],
            "ti_ss_agree": res["agree"], "ti_ss_absdiff": res["abs_diff"],
            "ensemble_valid": nv, "ensemble_total": nt,
        }
        if verbose:
            print(f"  [{case_name}/{vname}] TI logZ={res['ti_logZ']:.3f}±{res['ti_se']:.3f}"
                  f"  SS logZ={res['ss_logZ']:.3f}±{res['ss_se']:.3f}  "
                  f"agree={res['agree']}", flush=True)

    # --- Bayes factors vs M_full ---
    for vname in ("M_nolim", "M_kw"):
        log10B, label = bayes_factor(base_logZ["M_full"], base_logZ[vname])
        result["bayes_factors"][f"M_full_vs_{vname}"] = {
            "log10_B": log10B, "interpretation": label}

    # --- prior sweep: BF(M_full vs M_kw) stability across prior widths ---
    sweep = {}
    for rel in prior_rel_stds:
        lz = {}
        for v, vname in VARIANTS.items():
            cs = build_case(case_name, variant=v)
            # re-narrow the prior to this rel_std (reuse cached ensemble; only the
            # prior support/density changes — the ensemble box used rel_std=0.15).
            cs.prior = make_sampling_prior(cs.param_set, relative_std=rel)
            cache = out_dir / f"{case_name}_{vname}_ens.npz"
            gp, _, _ = build_loglik_surrogate(cs, n_ens, cache, rng_seed=rng_seed,
                                              verbose=False, noise_floor=noise_floor)
            res = log_evidence(gp.log_likelihood, cs.prior.log_prior,
                               cs.prior.means, rng_seed=rng_seed)
            lz[vname] = res["ti_logZ"]
        log10B, label = bayes_factor(lz["M_full"], lz["M_kw"])
        sweep[f"rel_std={rel}"] = {"log10_B_full_vs_kw": log10B, "label": label}
    result["prior_sweep"] = sweep
    # stable if the BF sign (preferred model) is unchanged across the sweep
    signs = [np.sign(s["log10_B_full_vs_kw"]) for s in sweep.values()]
    result["prior_sweep_stable"] = bool(len(set(signs)) == 1)

    with open(out_dir / f"{case_name}_evidence.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def summarize(results):
    """Print a compact logZ + Bayes-factor table across cases."""
    print("\n" + "=" * 74)
    print("MODEL EVIDENCE — log Z (TI) and Bayes factors (Jeffreys)")
    print("=" * 74)
    for r in results:
        print(f"\n{r['case']}:")
        for vname, d in r["variants"].items():
            print(f"  {vname:8s} logZ = {d['ti_logZ']:8.3f} ± {d['ti_se']:.3f}"
                  f"   (TI/SS agree: {d['ti_ss_agree']})")
        for k, bf in r["bayes_factors"].items():
            print(f"  BF {k}: log10 B = {bf['log10_B']:+.2f}  [{bf['interpretation']}]")
        print(f"  prior-sweep stable: {r['prior_sweep_stable']}")


if __name__ == "__main__":
    res = [run_case_evidence("channel", n_ens=40, verbose=True)]
    summarize(res)
