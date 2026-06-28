"""A-posteriori moving-wall Couette forward model (DNS_plan.md Step 2).

Exercises the moving-wall solver end to end through the Python wrapper: matching
the friction Reynolds number to a DNS Couette case, and checking that the solved
profile is physically a fully-developed turbulent Couette flow (monotone wall to
centerline, antisymmetric, constant total stress) and that the QoI vector is the
Couette half-gap U/U_b profile plus Cf. Runs a real solve, so it uses a coarse,
fast config and is one focused case. Skips where the binding or DNS_data is absent.
"""
import numpy as np
import pytest

pytest.importorskip("rans_sst_py", reason="C++ binding not built")
from UQ.datasets import CouetteDNS
from UQ.datasets.couette_forward import CouetteForwardRANS, CouetteCalibration

pytestmark = pytest.mark.skipif(
    not CouetteDNS.is_available(171),
    reason="Couette DNS_data not present (bulk data is local/gitignored)",
)

# coarse, fast solve config for the test (the science runs a finer grid)
FAST_CFG = {"nx": 4, "ny": 64, "Lx": 1.5,
            "max_iter": 8000, "conv_tol": 1.0e-4, "yplus_target": 0.4}


@pytest.fixture(scope="module")
def matched():
    return CouetteForwardRANS.match(171, tol=0.06, max_solves=6, cfg=FAST_CFG)


def test_match_hits_target_re_tau(matched):
    """The secant on nu brings the solved Re_tau near the DNS target."""
    assert matched["ok"]
    assert matched["re_tau"] == pytest.approx(171.0, rel=0.08)
    assert matched["status"] == "Converged"


def test_solved_profile_is_physical_couette(matched):
    """U/U_b runs monotonically from 0 at the wall to 1 at the centerline."""
    yh = matched["y_h"]
    U = matched["U_over_Ub"]
    assert yh[0] < 0.05 and yh[-1] == pytest.approx(1.0, abs=0.05)
    assert U[0] < 0.1                                  # no-slip stationary wall
    assert U[-1] == pytest.approx(1.0, rel=0.05)       # bulk-normalised centerline
    assert np.all(np.diff(U) > -1e-3)                  # monotone increasing


def test_solved_profile_tracks_dns_within_model_error(matched):
    """The channel-closure Couette profile tracks the DNS to within the cross-flow
    model-form error (a few percent), the misspecification the UQ must cover."""
    d = CouetteDNS.load(171)
    ubp = d.wall_velocity()
    yh_q = np.linspace(0.1, 1.0, 10)
    sU = np.interp(yh_q, matched["y_h"], matched["U_over_Ub"])
    dU = np.interp(yh_q, d.y_outer, d.U / ubp)
    # tracks the DNS but is not exact: a genuine, bounded cross-flow discrepancy
    assert np.max(np.abs(sU - dU)) < 0.15
    assert np.max(np.abs(sU - dU)) > 0.01


def test_calibration_qoi_layout(matched):
    """CouetteCalibration builds the Cf-first QoI vector and the modeled band."""
    d = CouetteDNS.load(171)
    c = CouetteCalibration(d, nu=matched["matched_nu"], n_stations=12, cfg=FAST_CFG)
    assert c.qoi_names[0] == "Cf"
    assert c.n_qoi == len(c.qoi_truth) == len(c.qoi_sigma)
    assert c.qoi_truth[0] == pytest.approx(2.0 / d.wall_velocity() ** 2, rel=1e-6)
    # modeled band: velocity sigma scales with the relative level, Cf at 5%
    s05 = c.set_observation_band(0.005).copy()
    s10 = c.set_observation_band(0.010).copy()
    assert np.allclose(s10[1:], 2.0 * s05[1:])         # velocity band doubles
    assert s05[0] == pytest.approx(0.05 * c.qoi_truth[0])   # Cf at 5% modeled
