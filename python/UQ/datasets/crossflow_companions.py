"""A-priori Step-2 companions: pipe (cross-geometry) and rotating channel.

These reuse the existing discrepancy machinery (UQ.datasets.channel_discrepancy is
generic over the canonical record) on the two companion flows. They are a-priori
diagnostics, not a-posteriori coverage tests:

  - Pipe: the cross-geometry Boussinesq discrepancy, the same shear-dominated,
    normal-anisotropy-dominated, near-wall-growing structure the channel and
    Couette show, validating the machinery transfers across geometry.
  - Rotating channel: the model-stress-test. Streamwise rotation drives a mean
    spanwise velocity and the out-of-plane stresses <uw>, <vw>. The <uw> component
    is a structural model-form failure a linear eddy-viscosity cannot represent at
    all: in this mean flow S_13 = 0 (no dU/dz or dW/dx), so the Boussinesq
    anisotropy b_13 = -C_mu S_13 is exactly zero, yet the DNS <uw> is large. The
    diagnostic then shows that a GLOBAL correction (the Step-1/2 generalized-Bayes
    scalar tempering and pooled conformal) can only widen the predictive band
    uniformly, so it cannot localise to the <uw>-failing region: covering the <uw>
    failure forces gross over-coverage of the components the closure already gets
    right. That diffuse inflation is exactly what the Step-3 feature-conditioned
    generative model-form is built to fix.
"""
import numpy as np

from .. import discrepancy as dq
from .. import realizability as rz
from .channel_discrepancy import channel_discrepancy, diagnostics


def _mixing_length_timescale(dns, wall_distance_plus=None):
    """Wall-damped mixing-length eddy viscosity -> turbulence timescale (wall units).

    nu_t^+ = kappa d^+ (1 - exp(-d^+/26))^2, tau^+ = nu_t^+/(C_mu k^+), with d^+ the
    distance from the nearest wall (defaults to y^+ for a half-profile). A
    self-contained analytic baseline so the a-priori discrepancy needs no solver,
    the same stand-in the Step-1 channel discrepancy test uses; for the full-channel
    rotating case the caller passes the nearest-wall distance so the mixing length
    vanishes at both walls rather than growing toward them.
    """
    dplus = np.abs(dns.yplus) if wall_distance_plus is None else wall_distance_plus
    nu_t = 0.41 * dplus * (1.0 - np.exp(-dplus / 26.0)) ** 2
    tau = nu_t / (0.09 * np.maximum(dns.k, 1e-6))
    return tau, nu_t


def pipe_discrepancy(pipe_dns):
    """A-priori Boussinesq discrepancy diagnostics for one pipe case (cross-geometry)."""
    tau, nu_t = _mixing_length_timescale(pipe_dns)
    res = channel_discrepancy(pipe_dns, tau, nu_t_plus=nu_t)
    d = diagnostics(res)
    d["re_tau"] = pipe_dns.re_tau
    return d


def rotating_diagnostic(rot_dns, level=0.9):
    """A-priori rotation diagnostic: the <uw> structural failure and the diffuse
    inflation of a global correction.

    Returns a dict with the out-of-plane structural-error measures and the per-
    component coverage of a single GLOBAL predictive band (the (1-alpha) quantile of
    the pooled discrepancy magnitude). A global band that covers the rotation-driven
    out-of-plane failure necessarily over-covers the in-plane components: that
    coverage asymmetry is the diffuse-inflation signature.
    """
    # nearest-wall distance for the full channel (y/h in [-1, 1], walls at +/-1)
    dplus = (1.0 - np.abs(rot_dns.y_outer)) * rot_dns.re_tau
    tau, nu_t = _mixing_length_timescale(rot_dns, wall_distance_plus=dplus)
    res = channel_discrepancy(rot_dns, tau, nu_t_plus=nu_t)
    b_dns = res["b_dns"]
    b_bouss = res["b_bouss"]
    db = res["db"]
    valid = res["valid"]

    # the <uw> anisotropy: a linear eddy-viscosity gives b_13 = -C_mu S_13 = 0 here
    # (S_13 = 0), so the whole DNS <uw> anisotropy is structural model-form error
    b13_bouss_max = float(np.max(np.abs(b_bouss[valid, 0, 2])))    # ~0 by construction
    b13_dns = b_dns[:, 0, 2]
    b23_dns = b_dns[:, 1, 2]
    in_plane = np.sqrt(db[:, 0, 0] ** 2 + db[:, 1, 1] ** 2 + db[:, 2, 2] ** 2
                       + db[:, 0, 1] ** 2)
    out_of_plane = np.sqrt(db[:, 0, 2] ** 2 + db[:, 1, 2] ** 2)

    # the discrepancy is unevenly distributed across the six stress components, so a
    # feature-conditioned correction would assign each its own band; a GLOBAL
    # correction (the Step-1/2 scalar tempering / pooled conformal) has one width.
    names = ["uu", "vv", "ww", "uv", "uw", "vw"]
    comp = np.stack([db[:, 0, 0], db[:, 1, 1], db[:, 2, 2],
                     db[:, 0, 1], db[:, 0, 2], db[:, 1, 2]], axis=1)[valid]
    # the per-component band each would need for `level` coverage of its own |db|
    band_per_component = {nm: float(np.quantile(np.abs(comp[:, j]), level))
                          for j, nm in enumerate(names)}
    bands = np.array(list(band_per_component.values()))
    # a global band must equal the worst component's need to cover it; that single
    # width then over-inflates the median component by this localisation cost
    global_band = float(np.max(bands))
    localisation_cost = float(global_band / np.median(bands))
    # marginal-`level` global band (pooled quantile) -> unequal per-component coverage
    pooled_band = float(np.quantile(np.abs(comp), level))
    per_component_coverage = {nm: float(np.mean(np.abs(comp[:, j]) <= pooled_band))
                              for j, nm in enumerate(names)}
    cov_values = np.array(list(per_component_coverage.values()))

    return {
        "ro_tau": rot_dns.ro_tau,
        "b13_boussinesq_max": b13_bouss_max,            # ~0: structurally unrepresentable
        "b13_dns_rms": float(np.sqrt(np.mean(b13_dns[valid] ** 2))),
        "b23_dns_rms": float(np.sqrt(np.mean(b23_dns[valid] ** 2))),
        "out_of_plane_over_in_plane": float(np.mean(out_of_plane[valid])
                                            / (np.mean(in_plane[valid]) + 1e-30)),
        "band_per_component": band_per_component,
        # the diffuse-inflation cost: one global band over-inflates the median
        # component by this factor to cover the worst (a feature-conditioned model
        # avoids it), and the marginal-coverage global band cannot equalise the
        # per-component coverage (it spreads across components)
        "localisation_cost": localisation_cost,
        "per_component_coverage_pooled_band": per_component_coverage,
        "coverage_spread_pooled_band": float(cov_values.max() - cov_values.min()),
        "dns_realizable": bool(np.all(rz.is_realizable(rot_dns.R[valid]))),
        "n_valid": int(np.sum(valid)),
    }
