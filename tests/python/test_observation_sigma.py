"""Modeled observation-sigma helper for the Step-2 cross-flow datasets.

Checks that the modeled relative sigma scales as advertised and that the data-only
physics-residual anchor is small (a fraction of a percent) for every case, so the
default 0.5 percent modeled level sits at or above the DNS's own convergence level
(apples-to-apples with the Step-1 floor). Skips where the gitignored data is
absent.
"""
import numpy as np
import pytest

from UQ.datasets import (CouetteDNS, PipeDNS, RotatingChannelDNS,
                         COUETTE_CASES, PIPE_CASES, ROTATING_CASES)
from UQ.datasets import observation_sigma as obs

pytestmark = pytest.mark.skipif(
    not CouetteDNS.is_available(986),
    reason="Step-2 DNS_data not present (bulk data is local/gitignored)",
)


def test_relative_sigma_scales_and_floors():
    """sigma = rel * |values|, floored; it is modeled, not a DNS _stdev."""
    values = np.array([2.0, -4.0, 0.0, 10.0])
    s = obs.relative(values, rel=0.01)
    assert np.allclose(s, [0.02, 0.04, 0.0, 0.10])
    s_floored = obs.relative(values, rel=0.01, floor=0.03)
    assert np.allclose(s_floored, [0.03, 0.04, 0.03, 0.10])


@pytest.mark.parametrize("n", COUETTE_CASES)
def test_couette_anchor_constant_total_stress(n):
    """Couette anchor is the constant-total-stress residual, small for all Re."""
    if not CouetteDNS.is_available(n):
        pytest.skip(f"Re_tau={n} not present")
    a = obs.physics_anchor(CouetteDNS.load(n))
    assert a["kind"] == "constant_total_stress"
    assert a["rms"] < 3.0e-3


@pytest.mark.parametrize("n", PIPE_CASES)
def test_pipe_anchor_linear_total_stress(n):
    """Pipe anchor is the linear-total-stress residual, small for all Re."""
    if not PipeDNS.is_available(n):
        pytest.skip(f"Re_tau={n} not present")
    a = obs.physics_anchor(PipeDNS.load(n))
    assert a["kind"] == "linear_total_stress"
    assert a["rms"] < 5.0e-3


@pytest.mark.parametrize("ro", ROTATING_CASES)
def test_rotating_anchor_budget_residual(ro):
    """Rotating-channel anchor is the file's own RSTE budget residual, small."""
    if not RotatingChannelDNS.is_available(ro):
        pytest.skip(f"Ro_tau={ro} not present")
    a = obs.physics_anchor(RotatingChannelDNS.load(ro))
    assert a["kind"] == "rste_budget_residual"
    assert a["rms"] < 1.0e-2


def test_default_level_sits_at_or_above_every_anchor():
    """The default 0.5 percent modeled level is at least the data convergence
    level for every compiled case (so the modeled sigma never understates the
    data noise). This is a reported sanity check, not a tuned value."""
    reports = []
    for n in COUETTE_CASES:
        if CouetteDNS.is_available(n):
            reports.append(obs.report(CouetteDNS.load(n)))
    for n in PIPE_CASES:
        if PipeDNS.is_available(n):
            reports.append(obs.report(PipeDNS.load(n)))
    for ro in ROTATING_CASES:
        if RotatingChannelDNS.is_available(ro):
            reports.append(obs.report(RotatingChannelDNS.load(ro)))
    assert reports                                    # at least one case present
    for r in reports:
        assert r["anchored_ok"], f"{r['label']}: anchor {r['anchor_rms']} > rel {r['rel']}"
        assert "modeled" in r["note"]
