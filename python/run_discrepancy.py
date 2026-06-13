"""
PHASE 6 / V.5 production — irreducible model-form discrepancy δ(x) along the wall
(research_dir §4.4/§4.6; angle 1).

For a case with spatial wall observations (channel / flat-plate Cf profiles), runs the
``physical_gp`` KOH augmented posterior over (θ, log σ_δ, log l_δ), then reconstructs the
posterior-mean discrepancy δ(x) at the observation locations — localizing *where* the
evidence-preferred closure is most wrong, the residual that remains after the best
parameters (the thesis's irreducible limit).
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

from bayesian_inference import (BayesianInferenceKOH, KOHLikelihood,
                                make_sampling_prior)
from case_library import build_case, VARIANTS


def run_case_discrepancy(case_name, *, variant=0, n_ens=60, n_steps=2000,
                         out_dir="results/discrepancy", rng_seed=0, verbose=True,
                         noise_floor=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cs = build_case(case_name, variant=variant)
    if cs.n_obs < 3:
        raise ValueError(f"{case_name}: δ(x) needs spatial wall obs (got {cs.n_obs})")

    koh = KOHLikelihood(cs.obs_locations, cs.obs_values, cs.obs_sigmas,
                        mode="physical_gp")
    prior = make_sampling_prior(cs.param_set)
    bik = BayesianInferenceKOH(cs.fm, cs.param_set, koh, theta_prior=prior)
    bik.run_ensemble(n_samples=n_ens, verbose=verbose)
    bik.train_surrogate(verbose=verbose, noise_floor=noise_floor)
    bik.run_mcmc(n_steps=n_steps, burn_in=n_steps // 4, verbose=verbose)

    n_theta = bik.n_theta
    n_extra = koh.n_extra_params
    samples = bik.samples                       # (N, n_theta + n_extra)

    # reconstruct δ(x_obs) for each posterior draw
    deltas = []
    for s in samples:
        theta = s[:n_theta]
        lsd = s[n_theta]
        lld = s[n_theta + 1] if n_extra == 2 else 0.0
        eta, _ = bik.multi_surrogate.predict(theta)
        deltas.append(koh.discrepancy_mean(eta, lsd, lld))
    deltas = np.asarray(deltas)
    delta_mean = deltas.mean(0)
    delta_lo, delta_hi = np.percentile(deltas, [2.5, 97.5], axis=0)

    # discrepancy hyperposterior
    log_sd = samples[:, n_theta]
    sigma_delta = np.exp(log_sd)
    rel_to_data = float(np.median(sigma_delta) / np.mean(np.abs(cs.obs_values)))

    result = {
        "case": case_name, "variant": VARIANTS[variant],
        "obs_locations": cs.obs_locations.tolist(),
        "obs_values": cs.obs_values.tolist(),
        "delta_mean": delta_mean.tolist(),
        "delta_ci95": [delta_lo.tolist(), delta_hi.tolist()],
        "sigma_delta_median": float(np.median(sigma_delta)),
        "sigma_delta_rel_to_data": rel_to_data,
        "peak_discrepancy_location": float(
            cs.obs_locations[int(np.argmax(np.abs(delta_mean)))]),
    }
    with open(out_dir / f"{case_name}_{VARIANTS[variant]}_discrepancy.json", "w") as f:
        json.dump(result, f, indent=2)
    _plot(cs, delta_mean, delta_lo, delta_hi,
          out_dir / f"{case_name}_{VARIANTS[variant]}_delta.png")
    if verbose:
        print(f"[{case_name}/{VARIANTS[variant]}] median σ_δ="
              f"{result['sigma_delta_median']:.4g} "
              f"({100*rel_to_data:.1f}% of data); peak |δ| at x="
              f"{result['peak_discrepancy_location']:.3g}", flush=True)
    return result


def _plot(cs, dm, lo, hi, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x = cs.obs_locations
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="k", lw=0.8)
    ax.fill_between(x, lo, hi, alpha=0.3, label="95% CI")
    ax.plot(x, dm, "o-", label="posterior-mean δ(x)")
    ax.set_xlabel("wall location x")
    ax.set_ylabel(f"discrepancy δ ({cs.obs_kind})")
    ax.set_title(f"Irreducible model-form discrepancy — {cs.name}/{cs.variant_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default="flat_plate")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--n-ens", type=int, default=60)
    args = ap.parse_args()
    run_case_discrepancy(args.case, variant=args.variant, n_ens=args.n_ens)
