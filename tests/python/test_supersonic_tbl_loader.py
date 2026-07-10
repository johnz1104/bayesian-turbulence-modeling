"""Tests for the attached supersonic boundary-layer loader (dnsm2 family).

Data-gated: without the bulk data under DNS_data these skip. Bounds are parse
guards set from values measured at loader bring-up: the van-Driest residual is
at machine-trapezoid level on these files, the implied wall temperature sits
within one percent of the turbulent recovery estimate, and every stress record
is realizable.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "python"))

from UQ import realizability
from UQ.datasets.supersonic_tbl import SupersonicTBLDNS, TBL_CASES

pytestmark = pytest.mark.skipif(
    not SupersonicTBLDNS.is_available("M2_Retau450"),
    reason="dnsm2 data not present")


@pytest.fixture(scope="module")
def cases():
    loaded = {name: SupersonicTBLDNS.load(name) for name in TBL_CASES
              if SupersonicTBLDNS.is_available(name)}
    if not loaded:
        pytest.skip("no dnsm2 cases present")
    return loaded


def test_all_cases_load(cases):
    assert len(cases) == len(TBL_CASES)
    for name, d in cases.items():
        assert d.n > 150
        assert d.yplus[0] == 0.0
        assert np.all(d.k >= 0.0)
        assert d.q_hat is None            # no heat flux in this dataset
        assert d.mu is None               # no viscosity profile given


def test_header_consistency(cases):
    for name, d in cases.items():
        nominal = float(name.split("Retau")[1])
        assert abs(d.re_tau / nominal - 1.0) < 0.08
        m_nominal = float(name.split("_")[0][1:])
        assert d.m_inf == m_nominal
        assert 1e-3 < d.wall["cf"] < 4e-3
        assert 0.05 < d.wall["m_tau"] < 0.12


def test_van_driest_anchor(cases):
    for name, d in cases.items():
        resid = d.van_driest_residual()
        interior = d.yplus > 10
        assert float(np.median(np.abs(resid[interior]))) < 0.005, name


def test_temperature_reconstruction(cases):
    # implied wall temperature against the turbulent recovery estimate
    # (r = 0.89); measured within one percent, guarded at three
    for name, d in cases.items():
        recovery = 1.0 + 0.89 * 0.2 * d.m_inf ** 2
        assert abs(d.implied_wall_temperature_ratio() / recovery - 1.0) \
            < 0.03, name
        # temperature profile decreases from the (hot) wall to the free stream
        assert d.T[0] == 1.0
        assert d.T[-1] < 1.0


def test_stress_record_realizable(cases):
    for name, d in cases.items():
        ok = realizability.is_realizable(d.R[1:])
        assert float(np.mean(ok)) > 0.999, name
        # shear stress negative through the log layer
        log_layer = (d.yplus > 30) & (d.y_outer < 0.5)
        assert np.all(d.uv[log_layer] < 0.0), name


def test_dnsfield_adapter_smoke(cases):
    d = cases["M2_Retau450"]
    tau = np.full(d.n, 10.0)
    field = d.to_dnsfield(timescale_plus=tau)
    assert field.R.shape == (d.n, 3, 3)
    m_t = d.turbulent_mach()
    assert np.all(np.isfinite(m_t)) and m_t.max() < 0.6
