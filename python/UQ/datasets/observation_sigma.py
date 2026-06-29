"""Modeled per-point observation uncertainty for the Step-2 cross-flow datasets.

Step 1 took its observation uncertainty from the Lee-Moser DNS _stdev, but it was
floored at 0.5 percent of U_b, so the floor (not the raw _stdev) was operative at
essentially every station. The Step-2 files (Couette, pipe, rotating channel)
carry NO per-point statistical _stdev, and the true sampling error is
unrecoverable from a flat profile file (it needs the averaging time / integral
scale). So observation uncertainty here is a MODELED relative value, a small
fraction of the local velocity / friction scale, explicitly labeled modeled and
never presented as the DNS statistical _stdev.

The magnitude is ANCHORED and sanity-checked per case by a data-only physics
residual, an exact identity the converged DNS must satisfy:

  - plane Couette: constant total stress, dU^+/dy^+ - <u'v'>^+ = 1.
  - channel / pipe: linear total stress, dU^+/dy^+ - <u'v'>^+ = 1 - y^+/Re_tau.
  - rotating channel: the file's own RSTE budget residual columns res_*^+.

The rms of that residual is the DNS's own convergence level; the modeled sigma is
set at or above it (and at the 0.5 percent level Step 1 effectively used), so the
cross-flow coverage comparison stays apples-to-apples with Step 1. Setting the
modeled sigma BELOW the physics-residual level would claim the data is known more
precisely than its own identities close, so the anchor is the floor the level
check enforces.
"""
import numpy as np

# the relative level Step 1 effectively ran on (0.5 percent of the local scale);
# the single place the cross-flow modeled-sigma convention is documented
DEFAULT_REL = 0.005


def relative(values, rel=DEFAULT_REL, floor=0.0):
    """Modeled observation sigma = rel * |values|, optionally floored.

    A MODELED relative uncertainty (a fraction of the local scale), not the DNS
    statistical _stdev (which these files do not carry). `values` is the local
    scale the QoI is measured against (e.g. the mean velocity at a station).
    """
    s = rel * np.abs(np.asarray(values, dtype=float))
    return np.maximum(s, floor) if floor else s


def _interior_mask(dns):
    """Stations where the finite-differenced stress identity is well-resolved.

    Excludes the first few wall stations (the gradient is steep) and the near-edge
    stations (the gradient is small and the FD is least accurate), so the residual
    rms reflects the DNS convergence rather than the differencing stencil.
    """
    return (dns.yplus > 5.0) & (np.abs(dns.y_outer) < 0.9)


def physics_anchor(dns):
    """Data-only physics-residual anchor for the modeled observation sigma.

    Returns a dict {kind, rms, description}. The residual is an exact identity the
    converged DNS satisfies; its rms over the interior is the data's own
    convergence level, the floor below which a modeled sigma would understate the
    data noise. Dispatches on the loader's meta['case'] tag.
    """
    case = dns.meta.get("case", "")
    if case == "plane_couette":
        residual = dns.total_stress_plus() - 1.0
        rms = _interior_rms(dns, residual)
        return {"kind": "constant_total_stress", "rms": rms,
                "description": "rms| dU+/dy+ - <u'v'>+ - 1 | (Couette)"}
    if case in ("pipe_flow", "plane_channel_profile"):
        target = 1.0 - dns.yplus / dns.re_tau
        residual = dns.total_stress_plus() - target
        rms = _interior_rms(dns, residual)
        return {"kind": "linear_total_stress", "rms": rms,
                "description": "rms| dU+/dy+ - <u'v'>+ - (1 - y+/Re_tau) | (pipe)"}
    if case == "streamwise_rotating_channel":
        rms = dns.budget_residual_level()
        return {"kind": "rste_budget_residual", "rms": rms,
                "description": "rms of the file's res_*+ budget-closure columns"}
    raise ValueError(f"no physics anchor defined for case '{case}'")


def _interior_rms(dns, residual):
    mask = _interior_mask(dns)
    return float(np.sqrt(np.mean(np.asarray(residual)[mask] ** 2)))


def report(dns, rel=DEFAULT_REL):
    """Per-case modeled-sigma report: the chosen relative level, the physics-
    residual anchor, and whether the level sits at or above the anchor.

    `anchored_ok` is True when rel >= anchor rms, i.e. the modeled relative sigma
    is no smaller than the DNS's own convergence level (so the observation noise
    is not claimed tighter than the data's identities close). This is a sanity
    check reported alongside the number, not a value tuned toward any result.
    """
    anchor = physics_anchor(dns)
    return {
        "case": dns.meta.get("case", ""),
        "label": _case_label(dns),
        "rel": float(rel),
        "anchor_kind": anchor["kind"],
        "anchor_rms": anchor["rms"],
        "anchor_description": anchor["description"],
        "anchored_ok": bool(rel >= anchor["rms"]),
        "note": "modeled observation uncertainty, not the DNS statistical _stdev",
    }


def _case_label(dns):
    """A short per-case label for reports (Re_tau, or Ro_tau for the rotating case)."""
    if dns.meta.get("case") == "streamwise_rotating_channel":
        return f"Ro_tau={getattr(dns, 'ro_tau', float('nan')):g}"
    return f"Re_tau={dns.re_tau:.0f}"
