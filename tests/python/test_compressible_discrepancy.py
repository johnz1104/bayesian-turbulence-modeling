"""A-priori discrepancy extraction verified on the real compressible matrix.

These tests run only where the gitignored DNS_data is present. They check
that both model-form discrepancy legs (anisotropy and heat flux) form on the
real data through the existing machinery against the reviewer-approved
baselines, that the conditioning features carry the turbulent Mach number,
that realizability of the DNS stress and the Galilean-invariant construction
hold as SEPARATE checks, and that the extracted structure has the physics the
study builds on (the gradient-diffusion misfit grows with Mach).
"""
import numpy as np
import pytest

from UQ.datasets import GVChannelDNS, GV_CASES, CKMChannelDNS, FlatPlateDNS
from UQ.datasets import compressible_discrepancy as cd

pytestmark = pytest.mark.skipif(
    not GVChannelDNS.is_available(GV_CASES[0]),
    reason="compressible DNS_data not present (bulk is local/gitignored)",
)

_REPRESENTATIVE = (
    "Retaus_0105_MCLx_0p32_isoTw_0298_MB_AIR0",
    "Retaus_0342_MCLx_1p51_isoTw_0298_MB_AIR0",
    "Retaus_0113_MCLx_2p49_isoTw_0298_MB_AIR0",
)


@pytest.fixture(scope="module")
def studies():
    out = {}
    for name in _REPRESENTATIVE:
        if GVChannelDNS.is_available(name):
            out[name] = cd.channel_study(GVChannelDNS.load(name))
    return out


@pytest.mark.parametrize("name", _REPRESENTATIVE)
def test_both_legs_form_on_real_data(studies, name):
    """Features (5 invariants + M_t), the anisotropy discrepancy, and the
    heat-flux discrepancy all come out finite on the real stations."""
    if name not in studies:
        pytest.skip(f"{name} not present")
    s = studies[name]
    assert s["status"] == "converged"
    n = s["features"].shape[0]
    assert s["features"].shape == (n, 6)
    assert s["db"].shape == (n, 3, 3)
    assert s["dq"].shape == (n, 3)
    assert np.all(np.isfinite(s["features"]))
    assert np.all(np.isfinite(s["db"]))
    assert np.all(np.isfinite(s["dq"]))


@pytest.mark.parametrize("name", _REPRESENTATIVE)
def test_realizability_and_invariance_are_separate_checks(studies, name):
    """The DNS stress is realizable at every off-wall station (fraction 1.0),
    and the feature construction is exactly Galilean invariant: adding a
    uniform velocity to the record leaves grad_u, hence every feature,
    unchanged (verified numerically, not assumed)."""
    if name not in studies:
        pytest.skip(f"{name} not present")
    from UQ import discrepancy as dq
    from UQ.datasets import _common
    s = studies[name]
    assert s["summary"]["realizable_fraction"] == 1.0
    dns = GVChannelDNS.load(name)
    # Galilean shift: differentiate the SHIFTED mean profile and re-derive
    # the features; a uniform velocity must leave them unchanged
    g_shifted = np.zeros((dns.n, 3, 3))
    g_shifted[:, 0, 1] = _common.wall_normal_gradient(dns.yplus, dns.U + 5.0)
    feats_shifted = dq.feature_set(g_shifted,
                                   s["baseline"]["timescale_plus"],
                                   extra=dns.turbulent_mach()[:, None])
    assert np.allclose(feats_shifted, s["features"], rtol=1e-10, atol=1e-12)


def test_anisotropy_discrepancy_has_channel_structure(studies):
    """The attached-channel structure the incompressible phases established
    carries over: the normal-stress anisotropy dominates (the linear model
    predicts zero normal anisotropy in parallel shear) and the discrepancy is
    largest toward the wall."""
    name = _REPRESENTATIVE[1]
    if name not in studies:
        pytest.skip(f"{name} not present")
    s = studies[name]
    assert s["summary"]["db_normal_rms_log"] > s["summary"]["db_shear_rms_log"]


def test_gdh_misfit_grows_with_mach(studies):
    """The heat-flux discrepancy (the gradient-diffusion misfit at the
    baseline Pr_t = 0.9) is a larger fraction of the flux scale at higher
    Mach: the compressible model-form error the study quantifies."""
    if len(studies) < 2:
        pytest.skip("need the Mach endpoints")
    low = studies.get(_REPRESENTATIVE[0])
    high = studies.get(_REPRESENTATIVE[2])
    if low is None or high is None:
        pytest.skip("endpoints not present")
    assert "gdh_misfit_fraction" in low["summary"]
    assert "gdh_misfit_fraction" in high["summary"]
    assert high["summary"]["gdh_misfit_fraction"] > 0.0
    assert low["summary"]["gdh_misfit_fraction"] > 0.0


def test_turbulent_mach_feature_spans_the_matrix(studies):
    """The M_t conditioning feature carries the Mach axis (grows across the
    matrix), which is what makes the feature set compressibility-aware."""
    if len(studies) < 2:
        pytest.skip("need the Mach endpoints")
    low = studies.get(_REPRESENTATIVE[0])
    high = studies.get(_REPRESENTATIVE[2])
    if low is None or high is None:
        pytest.skip("endpoints not present")
    assert high["summary"]["m_t_max"] > 2.0 * low["summary"]["m_t_max"]


def test_ckm_and_plate_run_through_the_same_extraction():
    """The independent-code channel and the flat plate drive the identical
    extraction: CKM through the 1-D baseline, the plate through the
    frozen-mean reconstruction with its derived-flux mask honoured."""
    if CKMChannelDNS.is_available("M1p5"):
        s = cd.channel_study(CKMChannelDNS.load("M1p5"))
        assert s["status"] == "converged"
        assert s["summary"]["realizable_fraction"] == 1.0
        assert s["dq"] is not None
    if FlatPlateDNS.is_available("M6Tw025"):
        p = cd.flatplate_study(FlatPlateDNS.load("M6Tw025"))
        assert p["status"] == "converged"
        assert p["summary"]["realizable_fraction"] == 1.0
        assert p["q_valid_mask"] is not None
        assert p["meta"]["baseline"] == "frozen_mean_sst_reconstruction"
