"""Zhang-Duan-Choudhari flat-plate loader verified on the real raw files.

These tests run only where the gitignored DNS_data is present (locally); on CI
without the bulk data they skip. They check the alias-line-driven parsing
against the header condition tables, the coordinate mapping (source z
wall-normal to the pipeline's y), the Favre-anisotropy record construction,
the DERIVED wall-normal heat flux (its mask, sign structure and magnitude),
the measured turbulent-Prandtl reference profiles, and the data-only
identities: the exact-by-construction anisotropy trace (parse guard) and the
van-Driest reconstruction residual (the modeled-sigma anchor).
"""
import numpy as np
import pytest

from UQ.datasets import FlatPlateDNS, ZDC_CASES
from UQ.datasets import observation_sigma

# per-case header values (DNS_data/README.md): M_inf, Tw/Tr, Re_tau, -B_q,
# M_tau, stations
KNOWN = {
    "M2p5": (2.5, 1.00, 510, 0.00, 0.08, 260),
    "M6Tw025": (5.84, 0.25, 450, 0.14, 0.17, 330),
    "M6Tw076": (5.86, 0.76, 453, 0.02, 0.13, 310),
    "M8Tw048": (7.87, 0.48, 480, 0.06, 0.15, 310),
    "M14Tw018": (13.64, 0.18, 646, 0.19, 0.19, 430),
}

pytestmark = pytest.mark.skipif(
    not FlatPlateDNS.is_available("M2p5"),
    reason="ZDC flat-plate DNS_data not present (bulk is local/gitignored)",
)


@pytest.fixture(scope="module")
def cases():
    return {name: FlatPlateDNS.load(name) for name in ZDC_CASES
            if FlatPlateDNS.is_available(name)}


@pytest.mark.parametrize("name", ZDC_CASES)
def test_header_conditions_match(cases, name):
    """The loaded wall parameters reproduce the file's own condition tables."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    m_inf, tw_tr, re_tau, neg_bq, m_tau, npts = KNOWN[name]
    assert c.wall["m_inf"] == pytest.approx(m_inf, abs=0.01)
    assert c.wall["tw_tr"] == pytest.approx(tw_tr, abs=0.01)
    assert c.re_tau == pytest.approx(re_tau, abs=1.0)
    assert c.wall["b_q"] == pytest.approx(-neg_bq, abs=5e-3)
    assert c.wall["m_tau"] == pytest.approx(m_tau, abs=5e-3)
    assert c.n == npts


@pytest.mark.parametrize("name", ZDC_CASES)
def test_wall_row_and_coordinates(cases, name):
    """No-slip isothermal wall row; monotone wall-normal grid; the profile
    extends past the boundary-layer edge (y+ beyond Re_tau)."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    assert c.yplus[0] == 0.0 and c.U[0] == 0.0
    assert c.T[0] == 1.0
    assert c.rho[0] == pytest.approx(1.0, abs=1e-12)
    # the header's tabulated Tw matches the file's own wall row to its
    # printed rounding for four cases; M6Tw076's wall row is 295.7 K against
    # the nominal 300.0 (a 1.4 percent source quirk, recorded here), which is
    # why the record normalizes by the file's own wall row
    wall_T_kelvin = c.T_wall_over_Tinf * c.free["T_inf"]
    assert wall_T_kelvin == pytest.approx(c.free["T_w"], rel=2e-2)
    assert np.all(np.diff(c.yplus) > 0)
    assert c.yplus[-1] > c.re_tau
    # cooled or adiabatic-like wall per the header table
    assert c.wall["b_q"] <= 0.0


@pytest.mark.parametrize("name", ZDC_CASES)
def test_favre_anisotropy_record(cases, name):
    """R = 2k(b + I/3) reproduces the file's Favre anisotropy exactly, the
    trace identity holds to file precision (parse guard), the shear anisotropy
    is negative in the log layer, and the record is realizable off the wall."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    from UQ import realizability
    from UQ import discrepancy as dq
    c = cases[name]
    assert np.abs(c.anisotropy_trace()).max() < 1e-7
    b = dq.reynolds_anisotropy(c.R[1:])
    log_layer = (c.yplus[1:] > 30) & (c.y_outer[1:] < 0.5)
    assert np.all(b[log_layer, 0, 1] < 0.0)
    assert np.all(realizability.is_realizable(c.R[1:]))


@pytest.mark.parametrize("name", ZDC_CASES)
def test_derived_heat_flux(cases, name):
    """The derived wall-normal flux is masked to well-defined gradients, has
    the cold-wall sign structure (negative below the temperature peak,
    positive above), and only its wall-normal component is populated."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    assert c.q_hat.shape == (c.n, 3)
    # measured valid fractions run 27 to 38 percent of the stations
    assert c.q_hat_valid.sum() > 0.2 * c.n
    assert np.all(c.q_hat[~c.q_hat_valid] == 0.0)
    assert np.all(c.q_hat[:, 0] == 0.0) and np.all(c.q_hat[:, 2] == 0.0)
    peak = np.abs(c.q_hat[:, 1]).max()
    assert 5e-3 < peak < 0.5
    if c.wall["tw_tr"] < 0.8:
        # cooled wall: the friction-heated temperature maximum sits inside
        # the buffer layer, and the turbulent flux carries heat toward the
        # wall below it and away from it above (the valid mask already
        # excludes the gradient zero-crossing band around the peak)
        y_peak = c.yplus[int(np.argmax(c.T))]
        # the flux zero crossing sits up to ~10 percent above the T-maximum
        # (the turbulent flux is not gradient-aligned through the peak, a
        # real countergradient band), so the sign split uses y+ margins and
        # a one-percent amplitude tolerance for the crossing neighbourhood
        below = c.q_hat_valid & (c.yplus < 0.8 * y_peak)
        above = c.q_hat_valid & (c.yplus > 1.3 * y_peak)
        assert above.any()
        assert np.all(c.q_hat[above, 1] > -0.01 * peak)
        assert np.max(c.q_hat[above, 1]) > 0.5 * peak
        if below.any():
            assert np.all(c.q_hat[below, 1] < 0.01 * peak)


@pytest.mark.parametrize("name", ZDC_CASES)
def test_measured_prandtl_reference(cases, name):
    """The measured Pr_t profile is sane in the log layer (the reference the
    calibrated Pr_t posterior is compared against)."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    log_layer = (c.yplus > 30) & (c.y_outer < 0.5)
    med = float(np.median(c.pr_t[log_layer]))
    assert 0.6 < med < 1.2


@pytest.mark.parametrize("name", ZDC_CASES)
def test_van_driest_anchor(cases, name):
    """The van-Driest reconstruction residual (the modeled-sigma anchor) sits
    at the measured sub-percent level."""
    if name not in cases:
        pytest.skip(f"{name} not present")
    rep = observation_sigma.report(cases[name])
    assert rep["anchor_kind"] == "van_driest_reconstruction"
    # measured across the five cases: 0.0006 to 0.008
    assert rep["anchor_rms"] < 1.5e-2


def test_turbulent_mach_tracks_the_axis(cases):
    """M_t grows along the Mach axis (0.26 at M2.5 to 0.66 at M14, the
    Morkovin-breakdown conditioning feature) and tracks the file's own rms
    Mach-fluctuation column at the peak within the expected cousin margin."""
    if "M2p5" not in cases or "M14Tw018" not in cases:
        pytest.skip("axis endpoints not present")
    low, high = cases["M2p5"], cases["M14Tw018"]
    assert high.turbulent_mach().max() > 2.0 * low.turbulent_mach().max()
    mt = low.turbulent_mach()
    i = int(np.argmax(mt))
    assert mt[i] == pytest.approx(low.m_rms[i], rel=0.15)


def test_builds_dnsfield_with_both_discrepancy_legs(cases):
    """The record drives DNSField.extract end to end on the derived flux."""
    name = "M6Tw025"
    if name not in cases:
        pytest.skip(f"{name} not present")
    c = cases[name]
    fd = c.to_dnsfield(timescale_plus=np.ones(c.n), nu_t_plus=np.ones(c.n))
    out = fd.extract()
    assert out["features"].shape == (c.n, 5)
    assert out["reynolds_discrepancy"].shape == (c.n, 3, 3)
    assert out["heatflux_discrepancy"].shape == (c.n, 3)
    assert np.all(np.isfinite(out["heatflux_discrepancy"][c.q_hat_valid]))
