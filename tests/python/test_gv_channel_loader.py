"""Gerolymos-Vallet compressible channel loader verified on the real raw files.

These tests run only where the gitignored DNS_data is present (locally); on CI
without the bulk data they skip. They check the header-driven parsing against
the files' own redundant statements (the simulation-data block, the GD inline
values, the tabulated GD row, the directory-name tags), the unit conversions
against the files' own conversion columns, the internal consistency of the
canonical Favre wall-unit record, realizability of the DNS stress, and the
data-only physics identity that anchors the modeled observation sigma (the
per-unit-mass-forced variable-density total-stress balance).
"""
import numpy as np
import pytest

from UQ.datasets import GVChannelDNS, GV_CASES
from UQ.datasets import observation_sigma

pytestmark = pytest.mark.skipif(
    not GVChannelDNS.is_available(GV_CASES[0]),
    reason="GV compressible channel DNS_data not present (bulk is local/gitignored)",
)

# three representative cases for the heavier checks: the lowest-Mach anchor,
# a mid-matrix supersonic case, and the strongest-cooling highest-Mach case
_REPRESENTATIVE = (
    "Retaus_0105_MCLx_0p32_isoTw_0298_MB_AIR0",
    "Retaus_0342_MCLx_1p51_isoTw_0298_MB_AIR0",
    "Retaus_0113_MCLx_2p49_isoTw_0298_MB_AIR0",
)


@pytest.fixture(scope="module")
def cases():
    return {name: GVChannelDNS.load(name) for name in GV_CASES
            if GVChannelDNS.is_available(name)}


@pytest.mark.parametrize("name", GV_CASES)
def test_headers_tags_and_globals_agree(cases, name):
    """The directory tag, the simulation-data block, and the GD table state the
    same Re_tau*, M_CLx and Re_tau_w (three independent statements per case)."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    re_nom, m_nom = GVChannelDNS.parse_tag(name)
    assert c.wall["re_tau_star"] == pytest.approx(re_nom, abs=1.0)
    assert c.wall["m_clx"] == pytest.approx(m_nom, abs=0.005)
    assert c.sim["re_tau_star"] == pytest.approx(c.wall["re_tau_star"], rel=1e-12)
    assert c.sim["m_clx"] == pytest.approx(c.wall["m_clx"], rel=1e-12)
    assert c.sim["re_tau_w"] == pytest.approx(c.re_tau, rel=1e-12)


@pytest.mark.parametrize("name", GV_CASES)
def test_gd_inline_matches_tabulated_row(cases, name):
    """The GD header restates every global inline; both parses must agree."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    gd_path = GVChannelDNS._paths(name)[0]
    gd = GVChannelDNS._globals(gd_path)
    tabulated = list(gd["all"].values())
    assert len(gd["inline"]) == len(tabulated)
    for col, inline_value in gd["inline"].items():
        assert inline_value == pytest.approx(tabulated[col], rel=1e-12)


@pytest.mark.parametrize("name", GV_CASES)
def test_wall_and_centerline_rows(cases, name):
    """Wall row is the exact no-slip isothermal state; the profile spans the
    half channel with y+ reaching Re_tau_w and the Mach reaching M_CLx."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    assert c.yplus[0] == 0.0 and c.U[0] == 0.0
    assert c.rho[0] == pytest.approx(1.0, abs=1e-12)
    assert c.T[0] == pytest.approx(1.0, abs=1e-12)
    assert c.mu[0] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(c.yplus) > 0)
    assert c.y_outer[-1] == pytest.approx(1.0, abs=1e-10)
    assert c.yplus[-1] == pytest.approx(c.re_tau, rel=1e-10)
    assert c.mach[-1] == pytest.approx(c.wall["m_clx"], rel=1e-10)
    # isothermal cooled walls: heat flows into the wall (B_q < 0) and the
    # core runs hotter than the wall
    assert c.wall["b_q"] < 0.0
    assert c.T[-1] > 1.0


@pytest.mark.parametrize("name", GV_CASES)
def test_favre_stress_record(cases, name):
    """R is the symmetric density-scaled Favre tensor, k its half trace, the
    shear negative off the wall, and every off-wall station realizable."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    from UQ import realizability
    c = cases[name]
    assert c.R.shape == (c.n, 3, 3)
    assert np.allclose(c.R, np.swapaxes(c.R, 1, 2))
    assert np.allclose(0.5 * np.trace(c.R, axis1=1, axis2=2), c.k)
    # the shear vanishes exactly at the wall and the centreline (symmetry)
    assert np.all(c.uv[1:-1] < 0.0)
    assert c.uv[-1] == 0.0
    assert np.all(realizability.is_realizable(c.R[1:]))


@pytest.mark.parametrize("name", GV_CASES)
def test_total_stress_anchor(cases, name):
    """The per-unit-mass-forced total-stress balance closes at the data's own
    convergence level outside the buffer layer (the modeled-sigma anchor)."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    rep = observation_sigma.report(cases[name])
    assert rep["anchor_kind"] == "variable_density_total_stress"
    # measured across the 24 cases: 0.0002 to 0.0056 (worst: the largest box,
    # Re_tau* 985); the bound is a data-quality guard, not a tuned number
    assert rep["anchor_rms"] < 7.0e-3


@pytest.mark.parametrize("name", _REPRESENTATIVE)
def test_enthalpy_flux_structure(cases, name):
    """The Favre temperature flux vanishes at the wall, its spanwise component
    is statistically zero, and its wall-normal component carries heat toward
    the cooled wall (negative) where it peaks."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    assert c.q_hat.shape == (c.n, 3)
    assert np.allclose(c.q_hat[0], 0.0)
    peak = np.max(np.abs(c.q_hat[:, 1]))
    assert peak > 0.0
    assert np.min(c.q_hat[:, 1]) == pytest.approx(-peak)
    assert np.max(np.abs(c.q_hat[:, 2])) < 0.05 * peak


@pytest.mark.parametrize("name", _REPRESENTATIVE)
def test_conversion_columns_are_consistent(cases, name):
    """The file's own *-to-outer conversion column reproduces the meanflow
    centreline velocity: <u_CL>/V_unit*(y) / sqrt(rho+) = u_CL+ at every y."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    u_cl_plus = c.ucl_over_vunit[1:] / np.sqrt(c.rho[1:])
    assert np.allclose(u_cl_plus, c.U_reynolds[-1], rtol=1e-6)


@pytest.mark.parametrize("name", _REPRESENTATIVE)
def test_favre_and_reynolds_means_are_close(cases, name):
    """The Favre and Reynolds means differ by the density-velocity correlation,
    which grows with Mach squared: negligible at M 0.32 and a few percent in
    the buffer layer at M 2.49 (measured 3.6 percent in U, 1.7 in T). The
    record stays internally consistent if both views sit within that envelope."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    du = np.abs(c.U[1:] - c.U_reynolds[1:]) / np.abs(c.U_reynolds[1:])
    dT = np.abs(c.T - c.T_reynolds) / c.T_reynolds
    assert du.max() < 0.05
    assert dT.max() < 0.025
    if c.wall["m_clx"] < 0.5:
        assert du.max() < 1e-3 and dT.max() < 1e-3


def test_turbulent_mach_tracks_the_matrix(cases):
    """M_t grows with the centreline Mach number across the matrix and stays
    in the physically expected attached-channel range."""
    if len(cases) < 2:
        pytest.skip("need at least two cases")
    low = cases.get(_REPRESENTATIVE[0])
    high = cases.get(_REPRESENTATIVE[2])
    if low is None or high is None:
        pytest.skip("representative cases not present")
    assert high.turbulent_mach().max() > 2.0 * low.turbulent_mach().max()
    assert high.turbulent_mach().max() < 0.6


def test_builds_dnsfield_with_both_discrepancy_legs(cases):
    """The record drives DNSField.extract end to end: invariant features, the
    anisotropy discrepancy, and (with a baseline nu_t supplied) the heat-flux
    discrepancy."""
    name = _REPRESENTATIVE[1]
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    fd = c.to_dnsfield(timescale_plus=np.ones(c.n), nu_t_plus=np.ones(c.n))
    out = fd.extract()
    assert out["features"].shape == (c.n, 5)
    assert out["reynolds_discrepancy"].shape == (c.n, 3, 3)
    assert out["heatflux_discrepancy"].shape == (c.n, 3)
    assert np.all(np.isfinite(out["features"]))
    # without a baseline eddy viscosity only the anisotropy leg is active
    fd_no_nut = c.to_dnsfield(timescale_plus=np.ones(c.n))
    assert not fd_no_nut.has_heat_flux()
