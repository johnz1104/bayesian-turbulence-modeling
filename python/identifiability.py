"""
PHASE 4 — identifiability suite (research_dir.md §4.4; angle 3).

Quantifies how many directions of the closure-coefficient space the wall data actually
constrain, three independent ways, and reconciles them into one report:

  1. **Posterior-covariance eigenspectrum** — eigendecomposition of Cov(θ|y) from a
     completed posterior.  Few large eigenvalues ⇒ few well-determined directions.
     (Reported on the *precision* Σ⁻¹ so "large eigenvalue = well-constrained",
     matching the active-subspace convention.)
  2. **Active subspace** — eigendecomposition of the gradient Gram matrix
     C = (1/N) Σ_i g_i g_iᵀ with g_i = ∇_θ log p(θ|y) (the Phase-2 FD gradients).
     Its dominant eigenvectors span the directions the data constrain; the eigenvalue
     spectrum quantifies how many (Constantine 2015).
  3. **ARD-lengthscale relevance** — 1/ℓ_j from the surrogate's ARD-RBF kernel, an
     axis-aligned (per-coefficient) relevance ranking.

The three are reconciled into a single low-rank story (a shared effective rank and a
consistent set of dominant coefficients), with a scree plot per case.  On a planted
synthetic case with known inactive directions the active-subspace rank must match the
design and the planted inactive directions must be recovered (V.3 acceptance).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Individual diagnostics
# --------------------------------------------------------------------------- #
def posterior_covariance_eigenspectrum(samples, *, use_precision=True):
    """
    Eigenspectrum of the posterior (co)variance.

    With ``use_precision`` the spectrum is of the precision Σ⁻¹ so that *large*
    eigenvalues mark *well-constrained* directions (consistent with the active
    subspace).  Returns (eigvals_desc, eigvecs, normalized_eigvals).
    """
    samples = np.atleast_2d(np.asarray(samples, float))
    cov = np.cov(samples.T)
    cov = np.atleast_2d(cov)
    cov += 1e-12 * np.eye(cov.shape[0])
    M = np.linalg.inv(cov) if use_precision else cov
    M = 0.5 * (M + M.T)                          # symmetrize
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return vals, vecs, vals / np.sum(np.abs(vals))


def active_subspace(gradients):
    """
    Active subspace from the gradient Gram matrix C = (1/N) Σ g_i g_iᵀ.

    Parameters
    ----------
    gradients : (N, d) array of log-posterior (or log-lik) gradients.

    Returns (eigvals_desc, eigvecs, normalized_eigvals).  NaN rows are dropped.
    """
    G = np.atleast_2d(np.asarray(gradients, float))
    G = G[np.all(np.isfinite(G), axis=1)]
    if len(G) == 0:
        raise ValueError("active_subspace: no finite gradient rows")
    C = (G.T @ G) / len(G)
    C = 0.5 * (C + C.T)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.maximum(vals, 0.0)
    s = np.sum(vals)
    return vals, vecs, vals / s if s > 0 else vals


def ard_relevance(lengthscales):
    """Per-coefficient relevance r_j = 1/ℓ_j (normalized).  Short ℓ ⇒ influential."""
    ls = np.asarray(lengthscales, float)
    r = 1.0 / np.maximum(ls, 1e-12)
    return r / np.sum(r)


# --------------------------------------------------------------------------- #
# Rank estimation
# --------------------------------------------------------------------------- #
def effective_rank(eigvals, *, energy=0.95, gap_factor=10.0):
    """
    Estimate the number of constrained directions from a (descending) spectrum.

    Reports both the energy-based rank (smallest k with cumulative ≥ ``energy``) and
    the spectral-gap rank (largest gap λ_k/λ_{k+1} ≥ ``gap_factor``); the headline
    rank is their min (the more conservative low-rank claim).
    """
    v = np.maximum(np.asarray(eigvals, float), 0.0)
    total = np.sum(v)
    if total <= 0:
        return {"energy_rank": 0, "gap_rank": 0, "rank": 0, "cum_energy": []}
    cum = np.cumsum(v) / total
    energy_rank = int(np.searchsorted(cum, energy) + 1)
    # spectral gap
    gap_rank = len(v)
    for k in range(len(v) - 1):
        if v[k + 1] <= 0 or (v[k] / max(v[k + 1], 1e-30)) >= gap_factor:
            gap_rank = k + 1
            break
    return {
        "energy_rank": energy_rank,
        "gap_rank": int(gap_rank),
        "rank": int(min(energy_rank, gap_rank)),
        "cum_energy": cum.tolist(),
    }


def principal_angles(A, B):
    """Principal angles (radians) between the column spaces of A and B."""
    Qa, _ = np.linalg.qr(np.atleast_2d(A))
    Qb, _ = np.linalg.qr(np.atleast_2d(B))
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #
def reconcile(names, *, samples=None, gradients=None, lengthscales=None,
              energy=0.95):
    """
    Reconcile the available diagnostics into one identifiability report.

    Any subset of (samples, gradients, lengthscales) may be supplied.  Returns a dict
    with each spectrum, each effective-rank estimate, the dominant active direction
    (coefficient loadings), and a consistency summary (do the ranks agree?).
    """
    names = list(names)
    d = len(names)
    report = {"names": names, "methods": []}

    ranks = {}
    if samples is not None:
        pv, pvec, pn = posterior_covariance_eigenspectrum(samples)
        report["posterior_precision_eigvals"] = pv.tolist()
        report["posterior_rank"] = effective_rank(pv, energy=energy)
        ranks["posterior"] = report["posterior_rank"]["rank"]
        report["methods"].append("posterior")

    if gradients is not None:
        av, avec, an = active_subspace(gradients)
        report["active_subspace_eigvals"] = av.tolist()
        report["active_subspace_norm"] = an.tolist()
        report["active_rank"] = effective_rank(av, energy=energy)
        ranks["active_subspace"] = report["active_rank"]["rank"]
        # dominant active direction loadings (|component| per coefficient)
        dom = np.abs(avec[:, 0])
        report["dominant_direction"] = {n: float(c) for n, c in zip(names, dom)}
        report["dominant_coefficients"] = [names[i] for i in np.argsort(dom)[::-1]]
        report["_active_vecs"] = avec
        report["methods"].append("active_subspace")

    if lengthscales is not None:
        r = ard_relevance(lengthscales)
        report["ard_relevance"] = {n: float(c) for n, c in zip(names, r)}
        report["ard_ranking"] = [names[i] for i in np.argsort(r)[::-1]]
        report["methods"].append("ard")

    # consistency: do the spectral ranks agree?
    if ranks:
        rvals = list(ranks.values())
        report["ranks"] = ranks
        report["rank_consensus"] = int(round(np.median(rvals)))
        report["ranks_consistent"] = bool(max(rvals) - min(rvals) <= 1)
    report["dimension"] = d
    return report


def scree_plot(report, save_path):
    """Save a scree plot of the available eigenspectra (normalized)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if "active_subspace_eigvals" in report:
        v = np.array(report["active_subspace_eigvals"])
        ax.semilogy(range(1, len(v) + 1), v / v[0], "o-", label="active subspace")
    if "posterior_precision_eigvals" in report:
        v = np.array(report["posterior_precision_eigvals"])
        v = np.maximum(v, 1e-30)
        ax.semilogy(range(1, len(v) + 1), v / v[0], "s--",
                    label="posterior precision")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("normalized eigenvalue (log)")
    rc = report.get("rank_consensus")
    if rc:
        ax.axvline(rc + 0.5, color="r", ls=":", alpha=0.7,
                   label=f"consensus rank ≈ {rc}")
    ax.set_title("Identifiability scree — constrained directions")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


if __name__ == "__main__":
    # Planted synthetic: gradients of a function active in only 2 directions.
    rng = np.random.default_rng(0)
    d = 6
    # two planted active directions
    a1 = np.zeros(d); a1[0] = 1.0
    a2 = np.zeros(d); a2[1] = 1.0; a2[2] = 1.0; a2 /= np.linalg.norm(a2)
    thetas = rng.uniform(-1, 1, (300, d))
    # ∇ of f = sin(a1·θ) + 0.5(a2·θ)^2  -> lies in span{a1, a2}
    G = (np.cos(thetas @ a1)[:, None] * a1[None, :]
         + (thetas @ a2)[:, None] * a2[None, :])
    names = [f"c{i}" for i in range(d)]
    rep = reconcile(names, gradients=G)
    print("active eigvals:", np.round(rep["active_subspace_eigvals"], 4))
    print("active rank:", rep["active_rank"]["rank"], "(expect 2)")
    ang = principal_angles(rep["_active_vecs"][:, :2], np.column_stack([a1, a2]))
    print("principal angles to planted subspace (deg):",
          np.round(np.degrees(ang), 3))
