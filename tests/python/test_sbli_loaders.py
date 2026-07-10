"""Tests for the impinging-shock interaction loaders (UQ.datasets.sbli_interaction).

Data-gated: without the bulk data under DNS_data these skip. The numeric
bounds are parse guards set from values measured at loader bring-up (recorded
in the pre-registration addendum), placed with margin so they catch
column-swap and orientation bugs, not statistical drift.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "python"))

from UQ.datasets.sbli_interaction import (SBLIInteractionDNS, SBLI_S_CASES,
                                          cross_campaign_wall_residual)

pytestmark = pytest.mark.skipif(
    not (SBLIInteractionDNS.is_available("adiabatic")
         and SBLIInteractionDNS.is_available("1.0")),
    reason="SBLI interaction data not present")

_RECOVERY = 1.9318  # measured recovery wall temperature, T_w/T_inf


@pytest.fixture(scope="module")
def adiabatic():
    return SBLIInteractionDNS.adiabatic()


@pytest.fixture(scope="module")
def s10():
    return SBLIInteractionDNS.wall_thermal("1.0")


@pytest.fixture(scope="module")
def s05():
    if not SBLIInteractionDNS.is_available("0.5"):
        pytest.skip("s = 0.5 field not present")
    return SBLIInteractionDNS.wall_thermal("0.5")


# ---- adiabatic dataset -------------------------------------------------------

def test_adiabatic_dimensions_and_tiling(adiabatic):
    # 32 blocks of 61 columns sharing edges: 32*61 - 31 unique columns
    assert adiabatic.U.shape == (1921, 344)
    assert adiabatic.R.shape == (1921, 344, 3, 3)
    assert adiabatic.q is None
    # strictly increasing x after dedup, wall row at y = 0
    assert np.all(np.diff(adiabatic.x) > 0)
    assert adiabatic.y[0] == 0.0
    # re-origined interaction coordinate covers the downstream half
    assert -12.0 < adiabatic.x[0] < -10.0
    assert 28.0 < adiabatic.x[-1] < 31.0


def test_adiabatic_wall_series_and_landmarks(adiabatic):
    s = adiabatic.series
    # deduped full-domain series in the re-origined coordinate
    assert s.x.size == 3841
    assert s.pw_valid and s.cp is not None
    # upstream Cp sits at zero by construction of the plateau rule
    assert abs(float(np.median(s.cp[s.x < -20]))) < 0.02
    # the re-origin puts the series half-rise at zero; the field row agrees
    assert abs(s.shock_position()) < 0.1
    assert abs(adiabatic.shock_position()) < 0.5
    onset = adiabatic.onset()
    assert -4.0 < onset < -1.0
    x_s, x_r = s.separation_reattachment()
    assert x_s is not None and x_s < x_r
    assert -4.0 < x_s < 0.0 < x_r < 3.0
    # no thermal switch on an adiabatic wall
    assert adiabatic.thermal_switch_position() is None


def test_adiabatic_reference_state(adiabatic):
    ref = adiabatic.reference
    assert ref["x_star"] == -7.0
    # recovery wall temperature from the field's own wall row
    assert abs(ref["T_w"] - _RECOVERY) < 0.02
    # the cf-and-wall-density friction velocity agrees with the series' own
    # tabulated u_tau column (measured 0.02 percent; guard at 1 percent)
    assert abs(ref["u_tau_over_uinf"] / ref["u_tau_series"] - 1.0) < 0.01


def test_adiabatic_stress_record_and_realizability(adiabatic):
    m = adiabatic.interior_mask()
    assert m.sum() > 100000
    # measured exactly 1.0; realizability of the DNS record is a parse guard
    assert adiabatic.realizable_fraction() > 0.999
    # favre-versus-reynolds convention gap, measured 1.5 percent outer layer
    assert adiabatic.favre_vs_reynolds_gap() < 0.05


def test_adiabatic_physics_anchors(adiabatic):
    # momentum-integral residual, measured 1.0 percent (guard 5)
    assert adiabatic.series.momentum_integral_residual() < 0.05
    # budget closure per station, measured 0.5 to 4.3 percent of peak
    # production (guard 10)
    closure = adiabatic.budget_closure_residual()
    assert set(closure) == {-1.93, -0.05, 2.10}
    for value in closure.values():
        assert value < 0.10
    # incoming-profile van Driest residual, measured 0.4 percent (guard 2)
    blinc = adiabatic.blinc
    resid = blinc.van_driest_residual(adiabatic.reference["u_tau_over_uinf"])
    assert float(np.median(np.abs(resid[blinc.yplus > 10]))) < 0.02


# ---- wall-thermal sweep ------------------------------------------------------

def test_s10_dimensions_and_quirks(s10):
    assert s10.U.shape == (1610, 230)
    assert s10.q is not None and s10.q.shape == (1610, 230, 3)
    # the documented s = 1.0 wall-series quirks
    assert s10.series.St is None
    assert not s10.series.pw_valid
    assert s10.series.cp is None
    with pytest.raises(ValueError):
        s10.series.shock_position()
    # the record's field-row landmarks serve instead
    assert -4.5 < s10.onset() < -2.0
    assert -3.0 < s10.shock_position() < 0.0
    x_s, x_r = s10.series.separation_reattachment()
    assert x_s is not None and x_s < x_r
    # adiabatic member of the sweep: recovery wall, no thermal switch
    assert abs(s10.reference["T_w"] - _RECOVERY) < 0.02
    assert s10.thermal_switch_position() is None


def test_s05_thermal_state_and_heat_flux(s05):
    assert s05.U.shape == (3001, 284)
    # the imposed wall-to-recovery ratio from the field's own wall row at the
    # reference station (measured 0.50002)
    assert abs(s05.reference["T_w"] / _RECOVERY - 0.5) < 0.01
    # the thermal switch sits inside the window (measured half-switch -8.96)
    switch = s05.thermal_switch_position()
    assert switch is not None and -10.5 < switch < -8.0
    # the reference station is post-switch and pre-onset
    assert switch < -7.0 < s05.onset()
    # both live heat-flux components carried and finite on the interior
    q_hat = s05.q_hat_wall_units()
    m = s05.interior_mask()
    assert np.all(np.isfinite(q_hat[m]))
    assert np.abs(q_hat[m][:, 0]).max() > 0.0
    assert np.abs(q_hat[m][:, 1]).max() > 0.0
    # spanwise component identically zero (span homogeneity)
    assert np.all(q_hat[:, :, 2] == 0.0)
    assert s05.realizable_fraction() > 0.999


def test_wall_series_st_where_present(s05):
    # Stanton carried for the heated and cooled members
    assert s05.series.St is not None
    st_down = s05.series.interpolate("St", 5.0)
    assert 1e-4 < abs(st_down) < 1e-2


def test_cross_campaign_anchor(adiabatic, s10):
    # between-campaigns spread, measured cf 9.3 percent and Cp 0.013
    # (guards 20 percent and 0.05): the mixed-campaign sigma floor
    resid = cross_campaign_wall_residual(adiabatic, s10)
    assert resid["cf_median_rel"] < 0.20
    assert resid["cp_median_abs"] < 0.05


def test_available_cases_enumerated():
    assert SBLI_S_CASES == ("0.5", "0.75", "1.0", "1.4", "1.9")
    for s in SBLI_S_CASES:
        assert SBLIInteractionDNS.is_available(s)
