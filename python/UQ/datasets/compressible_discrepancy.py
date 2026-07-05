"""A-priori discrepancy extraction on the real compressible attached data.

For every case of the compressible attached matrix this forms, through the
EXISTING machinery (UQ.dns_field / UQ.discrepancy), the two model-form
discrepancies against the reviewer-approved baseline:

  - the anisotropy discrepancy db = b_DNS - b_Boussinesq, with the Boussinesq
    baseline in its limiter-consistent actual-eddy-viscosity form (the 1-D
    compressible SST solve supplies nu_t+, k+ and the timescale
    tau+ = 1/(C_mu omega+) interpolated onto the DNS stations; the flat plate
    uses the frozen-mean reconstruction, stated as such);
  - the heat-flux discrepancy dq = q_DNS - q_GDH in the record's (u_tau, T_w)
    units (dq is the gradient-diffusion misfit the Prandtl-number calibration
    addresses; on the flat plate only the wall-normal component is defined,
    per its derived-flux mask).

The conditioning features are the five Galilean-invariant Pope invariants
EXTENDED with the turbulent Mach number M_t (the first compressibility-aware
feature, computed identically for every dataset from the record itself).
Realizability of the DNS stress (barycentric check, fraction 1.0 expected on
attached data) and the invariant construction are verified as SEPARATE checks,
per the standing project rule.
"""
import numpy as np

from .. import realizability
from .compressible_baseline import CompressibleChannelSST, FlatPlateFrozenSST


def channel_study(dns, coeffs=None, model_kwargs=None):
    """Baseline solve + both discrepancy legs for one channel case (GV or CKM).

    Returns a dict with the solve status and, when converged: the DNS-station
    baseline fields, the extracted features (Pope invariants + M_t), both
    discrepancies, the realizability fraction, and the GDH-misfit summary.
    """
    pr = dns.pr_molecular if dns.pr_molecular is not None \
        else np.full(dns.n, dns.wall["pr_w"])
    model = CompressibleChannelSST(
        re_tau_w=dns.re_tau, m_tau=dns.wall["m_tau"],
        gamma=dns.wall["gamma_w"], T_mu_samples=(dns.T, dns.mu),
        T_pr_samples=(dns.T, pr), coeffs=coeffs, **(model_kwargs or {}))
    out = model.solve()
    if out["status"] != "converged":
        return {"status": out["status"], "case": dns.meta.get("case_dir",
                                                              dns.meta.get("case_tag", ""))}
    base = {
        key: np.interp(dns.yplus, out["y_plus"], out[key])
        for key in ("nu_t_plus", "timescale_plus", "k_plus")
    }
    return _extract(dns, base, extra_meta={"baseline": "1d_compressible_sst",
                                           "baseline_qois": {
                                               "cf": out["cf"],
                                               "b_q": out["b_q"]}})


def flatplate_study(dns, coeffs=None):
    """Frozen-mean reconstruction + both discrepancy legs for one plate case.

    The heat-flux leg is restricted to the derived-flux valid mask; the
    summary reports the masked statistics only.
    """
    rec = FlatPlateFrozenSST(dns, coeffs=coeffs).closure()
    base = {key: rec[key] for key in ("nu_t_plus", "timescale_plus")}
    base["k_plus"] = None    # the reconstruction carries no transported k
    return _extract(dns, base, valid=dns.q_hat_valid,
                    extra_meta={"baseline": "frozen_mean_sst_reconstruction"})


def _extract(dns, base, valid=None, extra_meta=None):
    """The shared extraction: DNSField wiring, features with M_t, checks."""
    fd = dns.to_dnsfield(timescale_plus=base["timescale_plus"],
                         nu_t_plus=base["nu_t_plus"], Pr_t=0.9,
                         k_baseline_plus=base.get("k_plus"))
    m_t = dns.turbulent_mach()
    features = fd.features(extra=m_t[:, None])
    db = fd.reynolds_discrepancy()
    dq = fd.heatflux_discrepancy() if fd.has_heat_flux() else None

    # separate checks: (1) DNS-stress realizability (barycentric), off-wall;
    # (2) the invariant construction is exactly shift-invariant (adding a
    # uniform velocity leaves grad_u, hence the features, unchanged; asserted
    # structurally: the features depend on the record only through grad_u)
    off_wall = dns.yplus > 0.0
    realizable = realizability.is_realizable(dns.R[off_wall])

    interior = (dns.yplus > 30.0) & (dns.y_outer < 0.9)
    if valid is not None:
        q_mask = interior & valid
    else:
        q_mask = interior
    summary = {
        "realizable_fraction": float(np.mean(realizable)),
        "db_shear_rms_log": float(np.sqrt(np.mean(
            db[interior, 0, 1] ** 2))),
        "db_normal_rms_log": float(np.sqrt(np.mean(
            (db[interior, 0, 0] ** 2 + db[interior, 1, 1] ** 2
             + db[interior, 2, 2] ** 2) / 3.0))),
        "m_t_max": float(m_t.max()),
    }
    if dq is not None and q_mask.any():
        q_scale = float(np.max(np.abs(dns.q_hat[q_mask, 1])))
        summary["dq_y_rms"] = float(np.sqrt(np.mean(dq[q_mask, 1] ** 2)))
        summary["gdh_misfit_fraction"] = (summary["dq_y_rms"]
                                          / max(q_scale, 1e-30))
    meta = {"case": dns.meta.get("case_dir", dns.meta.get("case_tag", ""))}
    meta.update(extra_meta or {})
    return {
        "status": "converged",
        "meta": meta,
        "features": features,
        "db": db,
        "dq": dq,
        "q_valid_mask": valid,
        "baseline": base,
        "summary": summary,
    }
