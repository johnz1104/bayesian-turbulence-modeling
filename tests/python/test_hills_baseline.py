"""Periodic-hills SST baseline on the real dataset geometry.

Runs a real (coarse, fast) coupled solve on the streamwise-periodic
curved-bottom mesh, so it skips where the binding or the gitignored DNS_data is
absent. Verifies the geometry is extracted from the data itself (the blanking
mask), the body-force-driven solve converges to a separated field with the
matched bulk-flow convention, and the field record honours the same sampling
contract as the backward-facing-step baseline.
"""
import numpy as np
import pytest

pytest.importorskip("rans_sst_py", reason="C++ binding not built")
from UQ.datasets.periodic_hills import PeriodicHillsDNS
from UQ.datasets.hills_baseline import HillsBaselineRANS

pytestmark = pytest.mark.skipif(
    not PeriodicHillsDNS.is_available("1p0"),
    reason="periodic-hills DNS_data not present (bulk data is local/gitignored)",
)

# coarse, fast solve config for the test (the science runs a finer grid);
# body_force is the coarse-grid matched value (crest bulk 0.995)
FAST_CFG = {"nx": 48, "ny": 32, "max_iter": 15000, "conv_tol": 1.0e-4,
            "body_force": 0.0095}


@pytest.fixture(scope="module")
def dns():
    return PeriodicHillsDNS.load("1p0")


@pytest.fixture(scope="module")
def baseline(dns):
    return HillsBaselineRANS.solve(FAST_CFG, dns=dns)


def test_hill_curve_from_data(dns):
    Lx, yTop = HillsBaselineRANS.domain_extent(dns)
    assert 8.5 < Lx < 9.5                    # the alpha = 1 period is ~9 h
    assert abs(yTop - 3.036) < 0.01
    xn = np.linspace(0.0, Lx, 49)
    yb = HillsBaselineRANS.hill_curve(dns, xn)
    assert yb[0] == yb[-1]                   # periodic closure
    assert 0.9 < yb[0] < 1.1                 # crest height ~ h at the ends
    mid = yb[(xn > 3.0) & (xn < 6.0)]
    assert np.all(mid < 0.05)                # flat valley floor between hills
    assert np.all(yb >= 0.0) and np.all(yb < yTop)


def test_baseline_converges_separated_with_matched_bulk(baseline):
    assert baseline.status == "Converged"
    # the bulk-flow convention: crest-column bulk velocity is U_b = 1
    assert abs(baseline.bulk_crest - 1.0) < 0.05
    # the solve is genuinely separated on the lee side and the reattachment is
    # physical; the (large) gap to the benchmark 4.7 is the model-form signal,
    # reported by the study, not asserted here
    assert np.isfinite(baseline.reattachment)
    assert 2.0 < baseline.reattachment < 12.0


def test_field_record_contract(baseline):
    for f in (baseline.U, baseline.V, baseline.k, baseline.omega, baseline.nu_t):
        assert np.all(np.isfinite(f))
    assert np.all(baseline.k >= 0.0)
    # interior sampling, gradients and timescale (the discrepancy consumes these)
    xq = np.array([2.0, 4.5])
    yq = np.array([0.5, 1.0])
    s = baseline.sample_at(xq, yq)
    assert np.all(np.isfinite(s["U"])) and np.all(np.isfinite(s["nu_t"]))
    g = baseline.velocity_gradient_at(xq, yq, dx=0.05,
                                      dy=np.array([0.02, 0.05]))
    assert g.shape == (2, 3, 3) and np.all(np.isfinite(g))
    assert np.any(np.abs(g[:, 0, 1]) > 1e-6)
    tau = baseline.timescale_at(xq, yq)
    assert np.all(tau > 0.0)
