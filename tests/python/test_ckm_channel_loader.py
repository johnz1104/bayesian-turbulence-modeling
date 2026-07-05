"""Coleman-Kim-Moser supersonic channel loader verified on the real raw files.

These tests run only where the gitignored DNS_data is present (locally); on CI
without the bulk data they skip. They check the group-table parsing against
the header (u_tau, Re_tau) and the filename bulk Mach (cross-checked against
the wall pressure, a data-only identity of the file's units), the canonical
Favre wall-unit record, realizability, the natively-carried temperature-
velocity correlation, the total-stress anchor with the forcing form pinned
from the data, and the independent-code cross-check against the
Gerolymos-Vallet matrix at the overlapping condition.
"""
import numpy as np
import pytest

from UQ.datasets import CKMChannelDNS, CKM_CASES, GVChannelDNS
from UQ.datasets import observation_sigma

# per-case header values: u_tau (bulk units), Re_tau, bulk Mach
KNOWN = {
    "M1p5": (0.0545, 221.6, 1.5),
    "M3p0": (0.0387, 451.2, 3.0),
}

pytestmark = pytest.mark.skipif(
    not CKMChannelDNS.is_available("M1p5"),
    reason="CKM channel DNS_data not present (bulk is local/gitignored)",
)


@pytest.fixture(scope="module")
def cases():
    return {name: CKMChannelDNS.load(name) for name in CKM_CASES
            if CKMChannelDNS.is_available(name)}


@pytest.mark.parametrize("name", CKM_CASES)
def test_header_and_mach_cross_check(cases, name):
    """u_tau and Re_tau match the header line; the filename bulk Mach is
    reproduced by the wall state (M_B = sqrt(rho_w/(gamma p_w)) in the file's
    bulk units, a data-only identity of the TSF9/JFM95 normalization)."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    u_tau, re_tau, m_bulk = KNOWN[name]
    assert c.u_tau_bulk == pytest.approx(u_tau, abs=1e-6)
    assert c.re_tau == pytest.approx(re_tau, abs=0.05)
    assert c.m_bulk == m_bulk
    assert c.bulk_mach_from_wall_pressure() == pytest.approx(m_bulk, rel=2e-3)


@pytest.mark.parametrize("name", CKM_CASES)
def test_wall_and_centerline_rows(cases, name):
    """No-slip isothermal wall row; the lower-half record spans wall to
    centreline; the core runs hotter than the wall; heat flows into the wall
    (derived B_q < 0) at the level of the matched Gerolymos-Vallet family."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    assert c.yplus[0] == 0.0 and c.U[0] == 0.0
    assert c.T[0] == pytest.approx(1.0, abs=1e-6)
    assert c.rho[0] == pytest.approx(1.0, abs=1e-12)
    assert c.mu[0] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(c.yplus) > 0)
    assert c.y_outer[-1] == pytest.approx(1.0, abs=2e-2)
    assert c.T[-1] > 1.3
    assert c.wall["b_q"] < 0.0
    # the GV matrix at comparable conditions has B_qw of -0.049 (M 1.5) and
    # about -0.2 at M 2.5; the CKM derived values must sit in that family
    assert -0.35 < c.wall["b_q"] < -0.02


@pytest.mark.parametrize("name", CKM_CASES)
def test_favre_record_and_realizability(cases, name):
    """R is the file's native Favre tensor in wall units, symmetric,
    realizable off the wall, with negative interior shear."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    from UQ import realizability
    c = cases[name]
    assert c.R.shape == (c.n, 3, 3)
    assert np.allclose(c.R, np.swapaxes(c.R, 1, 2))
    assert np.allclose(0.5 * np.trace(c.R, axis1=1, axis2=2), c.k)
    assert np.all(c.uv[1:-1] < 0.0)
    assert np.all(realizability.is_realizable(c.R[1:]))


@pytest.mark.parametrize("name", CKM_CASES)
def test_native_temperature_velocity_correlation(cases, name):
    """The files carry the temperature-velocity covariance natively (Favre
    and Reynolds forms): the record's q_hat is populated from <rho.v"T">/<rho>
    and carries heat toward the cooled wall, and the Reynolds <v'T'> view
    agrees with the Favre form at the density-correlation level."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    assert c.q_hat.shape == (c.n, 3)
    assert c.q_hat[0, 1] == pytest.approx(0.0, abs=1e-10)
    peak = np.abs(c.q_hat[:, 1]).max()
    assert peak > 0.01
    assert np.min(c.q_hat[:, 1]) == pytest.approx(-peak)
    # Reynolds <v'T'> (bulk units) against the Favre column, interior only
    vT_reynolds = c.reynolds["<v'T'>"] / c.u_tau_bulk
    favre = c.q_hat[:, 1]
    interior = (c.yplus > 30) & (c.y_outer < 0.9)
    ratio = vT_reynolds[interior] / favre[interior]
    assert np.median(np.abs(ratio - 1.0)) < 0.15


@pytest.mark.parametrize("name", CKM_CASES)
def test_total_stress_anchor(cases, name):
    """The variable-density total-stress balance anchors the modeled sigma.
    The forcing form is pinned from the data: at bulk M 3.0 the per-unit-mass
    target closes five times better than the uniform-force target (0.5 versus
    2.8 percent); at bulk M 1.5 the closure is form-insensitive at the case's
    own ~1 percent convergence level (its three-digit header u_tau is a
    stated contributor), which is that case's honest anchor."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    rep = observation_sigma.report(cases[name])
    assert rep["anchor_kind"] == "variable_density_total_stress"
    assert rep["anchor_rms"] < 2.0e-2


def test_half_to_half_asymmetry_is_small(cases):
    """The full-channel file is statistically symmetric; the half-to-half
    asymmetry of the Favre mean velocity is a convergence diagnostic."""
    for c in cases.values():
        assert c.asymmetry < 0.02


def test_cross_check_against_gv_matrix(cases):
    """Independent-code consistency: CKM bulk-M 1.5 (Re_tau 221.6) against
    the Gerolymos-Vallet case at M_CLx 1.503, Re_tau_w 227.6. Different
    codes, gas models and averaging windows; the transformed mean profile
    and the Favre shear peak must agree at the few-percent level."""
    gv_case = "Retaus_0151_MCLx_1p50_isoTw_0298_MB_AIR0"
    if "M1p5" not in cases or not GVChannelDNS.is_available(gv_case):
        pytest.skip("overlapping cases not present")
    ckm = cases["M1p5"]
    gv = GVChannelDNS.load(gv_case)
    # centreline temperature ratio: the same wall-cooled physics
    assert ckm.T[-1] == pytest.approx(gv.T[-1], rel=0.05)
    # mean velocity in wall units on the overlapping log region
    grid = np.linspace(30.0, 150.0, 25)
    u_ckm = np.interp(grid, ckm.yplus, ckm.U)
    u_gv = np.interp(grid, gv.yplus, gv.U)
    rel = np.abs(u_ckm - u_gv) / u_gv
    assert np.median(rel) < 0.04
    assert rel.max() < 0.08
    # Favre shear-stress peak (momentum-flux form)
    peak_ckm = np.abs(ckm.rho * ckm.uv).max()
    peak_gv = np.abs(gv.rho * gv.uv).max()
    assert peak_ckm == pytest.approx(peak_gv, rel=0.10)


def test_builds_dnsfield_with_both_discrepancy_legs(cases):
    """The record drives DNSField.extract end to end (native heat flux)."""
    name = "M3p0"
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    fd = c.to_dnsfield(timescale_plus=np.ones(c.n), nu_t_plus=np.ones(c.n))
    out = fd.extract()
    assert out["features"].shape == (c.n, 5)
    assert out["reynolds_discrepancy"].shape == (c.n, 3, 3)
    assert out["heatflux_discrepancy"].shape == (c.n, 3)
    assert np.all(np.isfinite(out["heatflux_discrepancy"]))
