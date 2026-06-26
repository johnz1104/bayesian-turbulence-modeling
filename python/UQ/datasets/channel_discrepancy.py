"""Boussinesq anisotropy discrepancy on real channel profiles (machinery validation).

Combines a ChannelDNS case with a baseline turbulence timescale and runs the
existing UQ dns_field / discrepancy machinery (reuse, not reimplement) to produce
the anisotropy discrepancy db = b_DNS - b_Boussinesq and the Galilean-invariant
features at every DNS station, plus the diagnostics the channel must show.

Channel flow is attached and the Boussinesq discrepancy is modest, so this step
validates the extraction machinery on real data; the generative model-form is not
trained here (it is first exercised on the separated case, DNS_plan.md Step 3).

Convention. The DNS supplies the mean velocity gradient directly (exact, no
numerical differentiation), so the strain uses the DNS dU/dy^+; the turbulence
timescale tau^+ = nu_t^+/(C_mu k^+) comes from the baseline RANS at the matched
Re_tau. With that timescale the linear-eddy-viscosity anisotropy b = -C_mu dev(S)
reproduces the actual baseline Boussinesq anisotropy -nu_t/k dev(strain), so the
discrepancy is exactly what the baseline closure leaves unmodelled. Everything is
in wall units; the anisotropy is a pure ratio, so the units carry through.

What the channel must show (the discrepancy is a real-data sanity check):
  - the shear-stress component u'v' dominates the off-diagonal discrepancy (the
    spanwise off-diagonals u'w', v'w' vanish in a parallel shear flow);
  - the discrepancy is largest near the wall and decays toward the centerline,
    because the Reynolds-stress anisotropy is strongest in the buffer layer;
  - the normal-stress anisotropy (b_11, b_22, b_33) is the dominant structural
    error: the Boussinesq baseline predicts zero normal anisotropy in a parallel
    shear flow, so the entire DNS normal anisotropy appears in the discrepancy.
"""
import numpy as np

from .. import discrepancy as dq
from .. import realizability as rz


def channel_discrepancy(dns, timescale_plus, nu_t_plus=None, k_floor=1e-6):
    """Extract the anisotropy discrepancy and features at the DNS stations.

    dns:            a ChannelDNS case (wall units)
    timescale_plus: baseline turbulence timescale tau^+ per DNS station (array)
    k_floor:        stations with k^+ below this are flagged invalid; at the wall
                    k -> 0 so the anisotropy b = R/(2k) - I/3 is undefined there.
    Returns a dict of (N, ...) arrays in wall units plus a boolean `valid` mask.
    """
    field = dns.to_dnsfield(timescale_plus=timescale_plus, nu_t_plus=nu_t_plus)
    S, W = field.strain_rotation()
    b_dns = dq.reynolds_anisotropy(dns.R)            # b = R/(2k) - I/3
    b_bouss = dq.boussinesq_anisotropy(S)            # -C_mu dev(S)
    db = b_dns - b_bouss                             # the learnable discrepancy
    valid = dns.k > k_floor                          # exclude the degenerate wall row
    return {
        "yplus": dns.yplus.copy(),
        "re_tau": dns.re_tau,
        "valid": valid,
        "features": field.features(),                # Pope invariants
        "b_dns": b_dns,
        "b_bouss": b_bouss,
        "db": db,
        "db_norm": np.linalg.norm(db, axis=(1, 2)),  # Frobenius per station
    }


def from_baseline(dns, baseline):
    """Convenience: discrepancy from a ChannelBaselineRANS at the DNS stations."""
    tau = baseline.timescale_plus_at(dns.yplus)
    nu_t = baseline.profiles_at(dns.yplus)["nu_t"]
    return channel_discrepancy(dns, tau, nu_t_plus=nu_t)


def diagnostics(result):
    """Quantitative sanity diagnostics for a channel discrepancy result.

    Returns a dict of scalars and the booleans the channel is expected to honour,
    so the reproduce script and the test can both report and assert them.
    """
    valid = result["valid"]
    yp = result["yplus"]
    db = result["db"]
    b_dns = result["b_dns"]
    dbn = result["db_norm"]

    # off-diagonal discrepancy magnitudes (mean over the valid profile)
    uv = np.abs(db[:, 0, 1])      # streamwise-wall-normal shear (dominant)
    uw = np.abs(db[:, 0, 2])      # spanwise off-diagonals (vanish in channel)
    vw = np.abs(db[:, 1, 2])
    shear_dom = float(np.mean(uv[valid])) > 5.0 * float(
        np.mean(uw[valid]) + np.mean(vw[valid]) + 1e-30)

    # normal-stress anisotropy: the structural error the baseline cannot represent
    normal_norm = np.sqrt(db[:, 0, 0] ** 2 + db[:, 1, 1] ** 2 + db[:, 2, 2] ** 2)
    normal_dominates = float(np.mean(normal_norm[valid])) > float(np.mean(uv[valid]))

    # near-wall vs outer discrepancy magnitude (grows toward the wall)
    near = valid & (yp > 1.0) & (yp < 30.0)
    outer = valid & (yp > 100.0)
    near_mag = float(np.mean(dbn[near])) if near.any() else float("nan")
    outer_mag = float(np.mean(dbn[outer])) if outer.any() else float("nan")
    grows_to_wall = (np.isnan(outer_mag) or near_mag > outer_mag)

    # the DNS Reynolds stress is realizable at every valid station (real turbulence)
    realizable = bool(np.all(rz.is_realizable(_stress_from_aniso(b_dns[valid]))))

    imax = int(np.argmax(np.where(valid, dbn, -np.inf)))

    return {
        "shear_uv_mean": float(np.mean(uv)),
        "shear_uw_mean": float(np.mean(uw)),
        "shear_vw_mean": float(np.mean(vw)),
        "uv_dominates_offdiagonal": shear_dom,
        "normal_aniso_mean": float(np.mean(normal_norm)),
        "normal_dominates_discrepancy": normal_dominates,
        "db_norm_near_wall": near_mag,
        "db_norm_outer": outer_mag,
        "discrepancy_grows_toward_wall": bool(grows_to_wall),
        "dns_stress_realizable": realizable,
        "db_norm_max": float(dbn[imax]),
        "yplus_at_db_norm_max": float(yp[imax]),
        "n_valid": int(np.sum(valid)),
    }


def _stress_from_aniso(b):
    """Reconstruct a Reynolds-stress batch (2k=1) from an anisotropy batch.

    Realizability is a property of the anisotropy alone, so any positive 2k works;
    we use 2k = 1 to feed the (R-based) realizability check.
    """
    return b + np.eye(3) / 3.0
