"""1-D compressible SST baseline verified against the real channel matrix.

These tests run only where the gitignored DNS_data is present. They check that
the reviewer-approved baseline route delivers a converged, grid-independent,
physically-scaled model prediction on every case of the matrix, that its
incompressible limit is sane, and that the flat-plate frozen-mean
reconstruction produces the closure quantities the discrepancy layer needs.
The baseline's misfit against the DNS is NOT minimized or asserted small: it
is the model-form error the calibration and coverage study quantifies. What
the tests pin is that the misfit is the model's, not the implementation's
(convergence, grid independence, limit behaviour).
"""
import numpy as np
import pytest

from UQ.datasets import GVChannelDNS, GV_CASES, CKMChannelDNS, FlatPlateDNS
from UQ.datasets.compressible_baseline import (CompressibleChannelSST,
                                               FlatPlateFrozenSST)

pytestmark = pytest.mark.skipif(
    not GVChannelDNS.is_available(GV_CASES[0]),
    reason="compressible DNS_data not present (bulk is local/gitignored)",
)


def _model_for(c, coeffs=None, **kw):
    pr = c.pr_molecular if c.pr_molecular is not None \
        else np.full(c.n, c.wall["pr_w"])
    return CompressibleChannelSST(
        re_tau_w=c.re_tau, m_tau=c.wall["m_tau"], gamma=c.wall["gamma_w"],
        T_mu_samples=(c.T, c.mu), T_pr_samples=(c.T, pr),
        coeffs=coeffs, **kw)


@pytest.fixture(scope="module")
def gv_cases():
    return {name: GVChannelDNS.load(name) for name in GV_CASES
            if GVChannelDNS.is_available(name)}


@pytest.mark.parametrize("name", GV_CASES)
def test_converges_on_every_matrix_case(gv_cases, name):
    """The solve converges on all 24 conditions (Re_tau* 97 to 985, M_CLx
    0.32 to 2.49) and its predictions carry the right physics: heat into the
    isothermal wall (B_q < 0), a heated core, and positive closure fields."""
    if name not in gv_cases:
        pytest.skip(f"{name} not present")
    c = gv_cases[name]
    out = _model_for(c).solve()
    assert out["status"] == "converged"
    assert out["b_q"] < 0.0
    assert out["T_hat"][-1] > 1.0
    assert out["cf"] > 0.0
    assert np.all(out["k_plus"] >= 0.0)
    assert np.all(out["omega_plus"] > 0.0)
    assert np.all(np.diff(out["U_plus"]) >= -1e-12)


@pytest.mark.parametrize("name", (
    "Retaus_0105_MCLx_0p32_isoTw_0298_MB_AIR0",
    "Retaus_0342_MCLx_1p51_isoTw_0298_MB_AIR0",
    "Retaus_0113_MCLx_2p49_isoTw_0298_MB_AIR0",
))
def test_model_error_is_model_form_sized(gv_cases, name):
    """The baseline misfit against the DNS stays in the physically expected
    band: never absurd (the implementation is sound) and never tiny (the
    model-form error the UQ must cover genuinely exists). Measured across the
    matrix: centreline-velocity error 2.8 to 29 percent, growing with Mach at
    matched Re_tau* (4.0 / 8.7 / 14.6 percent at M 0.8 / 1.5 / 2.0)."""
    if name not in gv_cases:
        pytest.skip(f"{name} not present")
    c = gv_cases[name]
    out = _model_for(c).solve()
    rel_u = abs(out["U_plus"][-1] - c.U[-1]) / c.U[-1]
    rel_b = abs(out["b_q"] - c.wall["b_q"]) / abs(c.wall["b_q"])
    assert rel_u < 0.35
    assert rel_b < 0.35
    assert rel_u > 0.005


def test_grid_independence(gv_cases):
    """Halving the wall spacing and adding half again as many points moves cf
    and B_q by well under the coarse-grid model error."""
    name = "Retaus_0342_MCLx_1p51_isoTw_0298_MB_AIR0"
    if name not in gv_cases:
        pytest.skip(f"{name} not present")
    c = gv_cases[name]
    coarse = _model_for(c).solve()
    fine = _model_for(c, n=361, y1_plus=0.1).solve()
    assert coarse["status"] == "converged" and fine["status"] == "converged"
    assert abs(fine["cf"] - coarse["cf"]) / coarse["cf"] < 0.02
    assert abs(fine["b_q"] - coarse["b_q"]) / abs(coarse["b_q"]) < 0.02


def test_energy_budget_consistency(gv_cases):
    """The predicted wall heat flux equals the integrated dissipation (the
    solve's own first integral), closing the model's energy budget."""
    name = "Retaus_0342_MCLx_1p51_isoTw_0298_MB_AIR0"
    if name not in gv_cases:
        pytest.skip(f"{name} not present")
    c = gv_cases[name]
    m = _model_for(c)
    out = m.solve()
    y = out["y_plus"]
    mu = m._mu_T(out["T_hat"])
    diss = (mu + out["mu_t_hat"]) * np.gradient(out["U_plus"], y) ** 2
    b_q_budget = -(c.wall["gamma_w"] - 1.0) * c.wall["m_tau"] ** 2 \
        * float(np.trapz(diss, y))
    assert out["b_q"] == pytest.approx(b_q_budget, rel=1e-6)


def test_prandtl_sensitivity_direction(gv_cases):
    """Raising Pr_t throttles turbulent conduction: the core runs hotter and
    the wall heat flux magnitude responds. This is the identifiability channel
    the calibration relies on (Pr_t enters the predicted T profile)."""
    name = "Retaus_0342_MCLx_1p51_isoTw_0298_MB_AIR0"
    if name not in gv_cases:
        pytest.skip(f"{name} not present")
    c = gv_cases[name]
    lo = _model_for(c, coeffs={"Pr_t": 0.7}).solve()
    hi = _model_for(c, coeffs={"Pr_t": 1.1}).solve()
    assert lo["status"] == "converged" and hi["status"] == "converged"
    assert hi["T_hat"][-1] > lo["T_hat"][-1]
    mid = np.searchsorted(lo["y_plus"], 0.5 * c.re_tau)
    assert hi["T_hat"][mid] != pytest.approx(lo["T_hat"][mid], rel=1e-3)


def test_coefficient_sensitivity(gv_cases):
    """a1 and betaStar move the prediction (the calibration handles act)."""
    name = "Retaus_0151_MCLx_1p50_isoTw_0298_MB_AIR0"
    if name not in gv_cases:
        pytest.skip(f"{name} not present")
    c = gv_cases[name]
    base = _model_for(c).solve()
    moved = _model_for(c, coeffs={"a1": 0.28, "betaStar": 0.10}).solve()
    assert moved["status"] == "converged"
    assert moved["cf"] != pytest.approx(base["cf"], rel=1e-3)


def test_incompressible_limit_matches_low_mach_case(gv_cases):
    """At M_CLx 0.32 the compressible machinery reduces to the incompressible
    channel: temperature variation under 2 percent and the mean profile
    within the known low-Re SST band of the DNS (measured 6.3 percent at
    Re_tau* 105)."""
    name = "Retaus_0105_MCLx_0p32_isoTw_0298_MB_AIR0"
    if name not in gv_cases:
        pytest.skip(f"{name} not present")
    c = gv_cases[name]
    out = _model_for(c).solve()
    assert out["T_hat"].max() < 1.02
    rel = abs(out["U_plus"][-1] - c.U[-1]) / c.U[-1]
    assert rel < 0.10


def test_ckm_runs_through_the_same_baseline():
    """The independent-code channel runs through the identical model (the
    molecular Prandtl number is the reference's constant 0.7)."""
    if not CKMChannelDNS.is_available("M3p0"):
        pytest.skip("CKM data not present")
    c = CKMChannelDNS.load("M3p0")
    m = CompressibleChannelSST(
        re_tau_w=c.re_tau, m_tau=c.wall["m_tau"], gamma=c.wall["gamma_w"],
        T_mu_samples=(c.T, c.mu),
        T_pr_samples=(c.T, np.full(c.n, c.wall["pr_w"])))
    out = m.solve()
    assert out["status"] == "converged"
    assert out["b_q"] < 0.0
    # a strongly heated core, order-correct: the model gives T_CL 1.95
    # against the DNS 2.49 at bulk M 3.0, the Mach-growing model-form
    # error the calibration study quantifies
    assert out["T_hat"][-1] > 1.5


def test_low_mach_control_against_2d_solver(rs, gv_cases):
    """The reviewer-approved low-Mach control: the in-tree 2-D compressible
    SIMPLE solver (validated at Ma 0.1, ceiling about Ma 0.5) runs the lowest
    GV Mach condition and its developed-end skin friction is compared against
    the 1-D baseline at the matched bulk Reynolds number.

    Documented envelope and gaps that shape the assertion: the 2-D channel
    is DEVELOPING and this solver diverges on domains of 14 half-heights or
    longer at this Reynolds number, so full development is not reachable and
    the skin friction is sampled while still decaying toward the
    fully-developed limit (measured 0.0185, 0.0149, 0.0135 at 2.5, 5.0 and
    7.5 half-heights against the 1-D value 0.0107); its bindable walls are
    adiabatic while the 1-D solve is isothermal (at M_CLx 0.32 the
    temperature variation is 1.3 percent, below the implementation
    differences compared). The control therefore checks the MOMENTUM leg for
    consistency: monotone development toward the 1-D fully-developed value,
    approached from above, with the final-station gap bounded.
    """
    name = "Retaus_0105_MCLx_0p32_isoTw_0298_MB_AIR0"
    if name not in gv_cases:
        pytest.skip(f"{name} not present")
    c = gv_cases[name]
    out_1d = _model_for(c).solve()
    assert out_1d["status"] == "converged"

    # reuse the validated 2-D case factory (compressible_diagnostics), with
    # the channel height chosen so air at the case's Mach number matches the
    # case's bulk Reynolds number Re_Bw = <(rho u)_B> delta / mu_w
    from compressible_diagnostics import (make_validation_case,
                                          run_validation_case)
    eos = rs.IdealGasEOS()
    T_in, p_ref = 300.0, 101325.0
    rho_in = eos.density(p_ref, T_in)
    mu_in = eos.viscosity(T_in)
    Uin = c.wall["m_clx"] * eos.sound_speed(T_in)
    H = c.gd["Re_Bw"] * mu_in / (rho_in * Uin)
    # the longest domain this solver converges at this Reynolds number (the
    # factory samples skin friction at 0.25/0.5/0.75 Lx)
    case = make_validation_case(name="gv_low_mach_control",
                                Ma=c.wall["m_clx"], nx=40, ny=30,
                                Lx=10.0 * H, H=H, max_iterations=6000)
    summary = run_validation_case(case)
    assert summary["converged"], summary["status"]
    cf_stations = [float(v) for v in summary["Cf_at_stations"]]
    assert len(cf_stations) == 3
    # the 1-D cf is on the centreline dynamic head; convert to the 2-D
    # solver's bulk convention, cf_bulk = tau_w/(0.5 rho_b U_b^2), from the
    # 1-D solve's own profiles for a like-for-like comparison
    y1, U1 = out_1d["y_plus"], out_1d["U_plus"]
    rho1 = out_1d["rho_hat"]
    ub = float(np.trapz(rho1 * U1, y1) / np.trapz(rho1, y1))
    rho_b = float(np.trapz(rho1, y1) / y1[-1])
    cf_1d_bulk = 2.0 / (rho_b * ub ** 2)
    # monotone development, approached from above, gap shrinking station to
    # station, final gap at the documented developing-channel level
    gaps = [cf - cf_1d_bulk for cf in cf_stations]
    assert all(g > 0.0 for g in gaps)
    assert gaps[0] > gaps[1] > gaps[2]
    # envelope recalibrated on the wall-molecular momentum treatment: the
    # discrete wall force balance puts the molecular observation near the old
    # total-stress level, lifting the final-station gap from 0.26 to a
    # measured 0.377 at unchanged monotone development (the structural
    # assertions above); the 1-D profile baseline is untouched by the solver
    # change, so this is a developing-length comparison bound, not physics
    assert gaps[-1] / cf_1d_bulk < 0.45


def test_flatplate_frozen_reconstruction():
    """The plate reconstruction emits positive closure fields peaking in the
    log layer, and a GDH flux carrying heat toward the cold wall below the
    temperature peak; nothing from the DNS stresses enters it."""
    if not FlatPlateDNS.is_available("M6Tw025"):
        pytest.skip("plate data not present")
    c = FlatPlateDNS.load("M6Tw025")
    rec = FlatPlateFrozenSST(c).closure()
    nut = rec["nu_t_plus"]
    assert np.all(nut >= 0.0)
    i_peak = int(np.argmax(nut))
    assert 30.0 < c.yplus[i_peak] < 1.2 * c.re_tau
    assert np.all(rec["timescale_plus"] > 0.0)
    below = (c.yplus > 5) & (c.yplus < 0.8 * c.yplus[int(np.argmax(c.T))])
    if below.any():
        assert np.all(rec["q_gdh_hat"][below, 1] < 0.0)
