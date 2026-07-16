"""Baseline RANS field generator (matched-Re_tau SST channel) structural tests.

These run a tiny, fast SST channel solve (not a production-resolution one) to
exercise the Re_tau matcher and the wall-unit profile extraction. Accuracy
(Re_tau match to a few percent, Cf versus Dean) is validated in the reproduce
script at production resolution, not here. Skips without the built extension or
the DNS data.
"""
import numpy as np
import pytest

from UQ.datasets import ChannelDNS

# tiny, fast config: a few hundred cells and a short march, enough to produce a
# sensible developing-channel field in a couple of seconds
# max_iter sized for GENUINE convergence on this mesh (measured 5634
# iterations, ~9 s): baselines now refuse unconverged profiles, so the old
# 1500 budget, which never truly converged and was silently accepted before
# the audit, would fail every solve here
FAST_CFG = {"nx": 24, "ny": 32, "Lx": 10.0,
            "max_iter": 12000, "conv_tol": 1.0e-3, "yplus_target": 0.5}


@pytest.fixture(scope="module")
def baseline(rs):
    if not ChannelDNS.is_available(180):
        pytest.skip("channel DNS_data not present")
    from UQ.datasets.channel_baseline import ChannelBaselineRANS
    dns = ChannelDNS.load(180)
    # loose tol / few solves so the tiny-mesh test stays fast
    return ChannelBaselineRANS.match(dns.re_tau, tol=0.20, max_solves=3, cfg=FAST_CFG)


def test_profiles_are_physical(baseline):
    """Wall-unit profiles are positive, finite, and on a monotone y^+ grid."""
    b = baseline
    assert b.u_tau > 0.0 and b.nu > 0.0 and b.re_tau > 0.0
    assert np.all(np.diff(b.yplus) > 0)              # monotone wall-normal grid
    assert b.yplus[0] >= 0.0
    assert np.all(b.k >= 0.0)                         # tke non-negative
    assert np.all(b.nu_t >= 0.0)                      # eddy viscosity non-negative
    assert np.all(b.omega > 0.0)                      # specific dissipation positive
    assert np.all(np.isfinite(b.U))
    # mean velocity increases from the wall outward
    assert b.U[-1] > b.U[0]


def test_developing_channel_cf_below_dean(baseline):
    """The developing channel sits below Dean's fully-developed Cf (documented gap)."""
    b = baseline
    # the inlet/outlet channel under-predicts the fully-developed friction; assert
    # it is in the right ballpark and on the low side, never above by a wide margin
    assert 0.5 * b.cf_dean < b.cf < 1.1 * b.cf_dean


def test_interpolation_to_dns_stations(baseline):
    """profiles_at / timescale_plus_at return finite, positive fields at DNS y^+."""
    b = baseline
    dns = ChannelDNS.load(180)
    p = b.profiles_at(dns.yplus)
    for key in ("U", "k", "nu_t", "omega"):
        assert p[key].shape == (dns.n,)
        assert np.all(np.isfinite(p[key]))
    tau = b.timescale_plus_at(dns.yplus)
    assert tau.shape == (dns.n,)
    assert np.all(tau > 0.0)                          # tau = nu_t/(C_mu k) > 0


def test_match_moves_toward_target(baseline):
    """The matcher lands near the requested Re_tau even on the tiny mesh.

    The tolerance is loose because the test mesh is deliberately coarse; the
    production reproduce script matches to a few percent.
    """
    dns = ChannelDNS.load(180)
    assert abs(baseline.re_tau - dns.re_tau) / dns.re_tau < 0.35
