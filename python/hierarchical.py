"""
PHASE 5 — hierarchical / transfer utilities + KOH integration (research_dir §4.5/4.6;
angles 5, 1).

Provides:
  * ``fit_transfer_prior`` — fit a tractable (diagonal-Gaussian) density to a completed
    posterior p(θ|y_low) and return it as an informative ``Prior`` for a higher-regime
    inference (posterior-as-prior transfer; the antidote to a cold start).
  * ``koh_evidence_loglik`` — wrap a per-case ``KOHLikelihood`` + an η-surrogate into a
    ``loglik_fn`` over the augmented [θ…, log σ_δ, (log l_δ)] space, so the KOH marginal
    likelihood composes as the per-β integrand inside ``evidence.py`` (KOH ↔ evidence).
  * ``koh_logpost_grad`` — augmented gradient log-posterior (analytic KOH gradient ×
    surrogate/FD Jacobian for θ, analytic for the discrepancy hyperparameters) so NUTS
    can sample the KOH-augmented posterior (KOH ↔ gradient path).
  * ``compare_discrepancies`` — compare per-case δ_c hyperposteriors to test whether the
    model-form discrepancy is *shared* (a systematic SST deficiency) or *case-specific*.
"""

from __future__ import annotations

import numpy as np

from bayesian_inference import Prior


# --------------------------------------------------------------------------- #
# Posterior-as-prior transfer
# --------------------------------------------------------------------------- #
def fit_transfer_prior(samples, *, param_set=None, lower=None, upper=None,
                       inflate=1.0, min_std=1e-4, k_sigma=4.0):
    """
    Fit a diagonal-Gaussian density to posterior samples and return it as an
    *informative, concentrated* Prior for transfer to a higher regime.

    ``inflate`` widens the fitted std (a mild hedge against over-confidence in the new
    regime).  Crucially, the Prior's support (the box ``run_ensemble`` samples) is
    narrowed to ``mean ± k_sigma·σ`` intersected with the physical bounds — so the
    transferred high-fidelity ensemble concentrates where the low-regime posterior lived
    instead of being spread across the full physical box (that concentration is the
    eval-reduction benefit).
    """
    samples = np.atleast_2d(np.asarray(samples, float))
    means = samples.mean(0)
    stds = np.maximum(samples.std(0) * inflate, min_std)
    if param_set is not None:
        plo = np.asarray(param_set.lower_bounds(), float)
        phi = np.asarray(param_set.upper_bounds(), float)
    else:
        plo = np.asarray(lower if lower is not None else means - 1e3 * stds, float)
        phi = np.asarray(upper if upper is not None else means + 1e3 * stds, float)
    box_lo = np.maximum(means - k_sigma * stds, plo)
    box_hi = np.minimum(means + k_sigma * stds, phi)
    return Prior(means, stds, box_lo, box_hi)


# --------------------------------------------------------------------------- #
# KOH ↔ evidence  and  KOH ↔ gradient
# --------------------------------------------------------------------------- #
def koh_evidence_loglik(koh, eta_fn, n_theta):
    """
    Build ``loglik_fn(extended_theta)`` = KOH(η(θ), log σ_δ, [log l_δ]) so the KOH
    marginal likelihood is the per-β integrand for ``evidence.thermodynamic_integration``
    / ``stepping_stone``.  ``extended_theta = [θ…, log σ_δ, (log l_δ)]``.
    """
    n_extra = koh.n_extra_params

    def loglik(extended_theta):
        theta = np.asarray(extended_theta[:n_theta], float)
        lsd = float(extended_theta[n_theta])
        lld = float(extended_theta[n_theta + 1]) if n_extra == 2 else 0.0
        eta = np.asarray(eta_fn(theta), float)
        return koh(eta, lsd, lld)

    return loglik


def koh_logpost_grad(koh, eta_fn, eta_jac_fn, prior, n_theta):
    """
    Augmented gradient log-posterior for NUTS over [θ…, log σ_δ, (log l_δ)].

    Uses the analytic KOH gradient: dL/dθ = Jᵀ α (J = ∂η/∂θ from the surrogate or FD
    oracle), and the analytic discrepancy-hyperparameter gradients.  ``prior`` is the
    augmented Prior (θ-prior ⊕ hyperparameter priors).
    """
    n_extra = koh.n_extra_params

    def logp_and_grad(ext):
        ext = np.asarray(ext, float)
        lp = prior.log_prior(ext)
        if not np.isfinite(lp):
            return -np.inf, np.zeros(len(ext))
        theta = ext[:n_theta]
        lsd = ext[n_theta]
        lld = ext[n_theta + 1] if n_extra == 2 else 0.0
        eta = np.asarray(eta_fn(theta), float)
        ll = koh(eta, lsd, lld)
        if not np.isfinite(ll):
            return -np.inf, np.zeros(len(ext))
        deta, g_s, g_l = koh.gradient(eta, lsd, lld)
        J = np.atleast_2d(eta_jac_fn(theta))             # (n_obs, n_theta)
        g_theta = J.T @ deta
        # prior gradient (truncated normal)
        z = (ext - prior.means) / prior.stds
        g_prior = -z / prior.stds
        g_ll = np.concatenate([g_theta, [g_s] + ([g_l] if n_extra == 2 else [])])
        return lp + ll, g_prior + g_ll

    return logp_and_grad


# --------------------------------------------------------------------------- #
# Shared vs. case-specific discrepancy
# --------------------------------------------------------------------------- #
def compare_discrepancies(sigma_delta_samples_by_case, *, overlap_k=2.0):
    """
    Test whether the discrepancy amplitude σ_δ is *shared* across cases or
    *case-specific*, from per-case hyperposterior samples of σ_δ (linear scale).

    Returns per-case mean/std and a verdict: "shared" if every pair of case marginals
    overlaps within ``overlap_k`` combined std (a systematic, learnable SST deficiency),
    else "case-specific" (geometry-dependent).
    """
    cases = list(sigma_delta_samples_by_case)
    stats = {}
    for c, s in sigma_delta_samples_by_case.items():
        s = np.asarray(s, float)
        stats[c] = {"mean": float(np.mean(s)), "std": float(np.std(s))}
    shared = True
    pairs = {}
    for i in range(len(cases)):
        for j in range(i + 1, len(cases)):
            a, b = cases[i], cases[j]
            dm = abs(stats[a]["mean"] - stats[b]["mean"])
            comb = overlap_k * np.hypot(stats[a]["std"], stats[b]["std"])
            ov = bool(dm <= comb)
            pairs[f"{a}|{b}"] = {"abs_mean_diff": float(dm),
                                 "overlap_band": float(comb), "overlap": ov}
            shared = shared and ov
    return {"per_case": stats, "pairs": pairs,
            "verdict": "shared" if shared else "case-specific"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # transfer prior from synthetic low-regime posterior
    samp = rng.normal([0.31, 0.09], [0.02, 0.005], size=(2000, 2))

    class _PS:
        def lower_bounds(self): return [0.1, 0.01]
        def upper_bounds(self): return [0.8, 0.2]
    pr = fit_transfer_prior(samp, param_set=_PS())
    print("transfer prior means:", pr.means.round(4), "stds:", pr.stds.round(4))

    # shared vs case-specific
    shared = {"channel": rng.normal(0.05, 0.01, 500),
              "plate": rng.normal(0.052, 0.01, 500),
              "bfs": rng.normal(0.051, 0.01, 500)}
    print("shared case:", compare_discrepancies(shared)["verdict"])
    specific = {"channel": rng.normal(0.02, 0.005, 500),
                "bfs": rng.normal(0.15, 0.01, 500)}
    print("specific case:", compare_discrepancies(specific)["verdict"])
