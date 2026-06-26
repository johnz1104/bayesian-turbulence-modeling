"""Real-data discrepancy extraction on the channel profiles (machinery validation).

Drives the existing UQ dns_field / discrepancy machinery with the real Lee-Moser
Reynolds-stress profiles and an analytic mixing-length turbulence timescale (so
the test needs no solver run), and asserts the qualitative structure the channel
must show. The reproduce script substitutes the real baseline RANS timescale; the
qualitative checks here are baseline-insensitive because the dominant discrepancy
is the normal-stress anisotropy that the Boussinesq baseline cannot represent at
all.
"""
import numpy as np
import pytest

from UQ.datasets import ChannelDNS, CHANNEL_CASES
from UQ.datasets.channel_discrepancy import channel_discrepancy, diagnostics

pytestmark = pytest.mark.skipif(
    not ChannelDNS.is_available(180),
    reason="channel DNS_data not present (bulk data is local/gitignored)",
)


def _stand_in_timescale(dns):
    """A bounded, wall-damped eddy viscosity -> turbulence timescale (wall units).

    nu_t^+ = kappa y^+ (1 - exp(-y^+/26))^2, tau^+ = nu_t^+/(C_mu k^+). A
    self-contained analytic stand-in for the baseline RANS timescale so the
    machinery test needs no solver run; the reproduce script uses the real SST
    baseline. The form is wall-damped and grows mildly outward (it does not blow
    up near the centerline where k is small), so the discrepancy structure it
    produces matches the real baseline qualitatively.
    """
    yp = dns.yplus
    nu_t = 0.41 * yp * (1.0 - np.exp(-yp / 26.0)) ** 2
    tau = nu_t / (0.09 * np.maximum(dns.k, 1e-6))
    return tau, nu_t


@pytest.mark.parametrize("n", CHANNEL_CASES)
def test_discrepancy_qualitative_structure(n):
    """Across all five Re_tau: uv dominates, normal anisotropy dominates, grows to wall."""
    if not ChannelDNS.is_available(n):
        pytest.skip(f"Re_tau={n} not present")
    dns = ChannelDNS.load(n)
    tau, nu_t = _stand_in_timescale(dns)
    res = channel_discrepancy(dns, tau, nu_t_plus=nu_t)
    d = diagnostics(res)

    # u'v' is the dominant off-diagonal; spanwise off-diagonals vanish in channel
    assert d["uv_dominates_offdiagonal"]
    assert d["shear_uw_mean"] < 1e-2 * d["shear_uv_mean"]
    assert d["shear_vw_mean"] < 1e-2 * d["shear_uv_mean"]
    # normal-stress anisotropy is the dominant structural error (Boussinesq blind)
    assert d["normal_dominates_discrepancy"]
    # discrepancy grows toward the wall and peaks in the buffer layer (y+ ~ 5-20)
    assert d["discrepancy_grows_toward_wall"]
    assert 3.0 < d["yplus_at_db_norm_max"] < 30.0
    # the DNS Reynolds stress is realizable at every valid station
    assert d["dns_stress_realizable"]
    # the degenerate near-wall stations (k -> 0) are excluded, almost all retained
    assert dns.n - 3 <= d["n_valid"] <= dns.n - 1


def test_boussinesq_predicts_zero_normal_anisotropy():
    """The baseline anisotropy is pure shear, so its normal components are zero.

    This is the structural statement behind the discrepancy: in a parallel shear
    flow the linear eddy-viscosity model puts the entire normal-stress anisotropy
    into the discrepancy.
    """
    dns = ChannelDNS.load(180)
    tau, nu_t = _stand_in_timescale(dns)
    res = channel_discrepancy(dns, tau, nu_t_plus=nu_t)
    b_bouss = res["b_bouss"]
    # only the 0-1 shear entry is nonzero in the Boussinesq anisotropy
    assert np.allclose(np.diagonal(b_bouss, axis1=1, axis2=2), 0.0, atol=1e-12)
    assert np.allclose(b_bouss[:, 0, 2], 0.0) and np.allclose(b_bouss[:, 1, 2], 0.0)
    assert np.max(np.abs(b_bouss[:, 0, 1])) > 0.0


def test_discrepancy_equals_dns_minus_baseline():
    """db = b_DNS - b_Boussinesq exactly (the extraction is a pure difference)."""
    from UQ import discrepancy as dq
    dns = ChannelDNS.load(550)
    tau, nu_t = _stand_in_timescale(dns)
    res = channel_discrepancy(dns, tau, nu_t_plus=nu_t)
    assert np.allclose(res["db"], res["b_dns"] - res["b_bouss"])
    # b_DNS is the schema anisotropy of the DNS Reynolds-stress tensor
    assert np.allclose(res["b_dns"], dq.reynolds_anisotropy(dns.R))


def test_features_are_finite_and_invariant_count():
    """The Galilean-invariant feature set is the five Pope invariants, all finite."""
    dns = ChannelDNS.load(1000)
    tau, nu_t = _stand_in_timescale(dns)
    res = channel_discrepancy(dns, tau, nu_t_plus=nu_t)
    feats = res["features"]
    assert feats.shape == (dns.n, 5)
    assert np.all(np.isfinite(feats))
