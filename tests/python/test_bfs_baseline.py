"""Le-Moin BFS SST baseline and the field-to-field discrepancy, on the real data.

Runs a real (coarse, fast) SST solve, so it skips where the binding or the
gitignored DNS_data is absent. The baseline matches the Le-Moin boundary
conditions verified from the data (free-slip top, prescribed inflow boundary
layer at x/h = -10): it checks the solve converges, the outer flow tracks the
DNS instead of growing a spurious top-wall layer, an inflow boundary layer of
the right character reaches the step, and the discrepancy assembly returns a
finite, traceless anisotropy discrepancy against the limiter-consistent
Boussinesq baseline through the canonical DNSField interface.
"""
import numpy as np
import pytest

pytest.importorskip("rans_sst_py", reason="C++ binding not built")
from UQ.datasets.backward_facing_step import BackwardFacingStepDNS
from UQ.datasets.bfs_baseline import BFSBaselineRANS
from UQ.datasets.separated_discrepancy import BFSDiscrepancy
from UQ import realizability as rz

pytestmark = pytest.mark.skipif(
    not BackwardFacingStepDNS.is_available(),
    reason="BFS DNS_data not present (bulk data is local/gitignored)",
)

# coarse, fast solve config for the test (the science runs a finer grid);
# slip_top and the matched inlet_delta come from DEFAULT_CONFIG
FAST_CFG = {"nx_up": 20, "nx_down": 26, "ny_up": 14, "ny_down": 10,
            "Lu": 10.0, "Ld": 22.0, "max_iter": 8000, "conv_tol": 1.0e-4}


@pytest.fixture(scope="module")
def baseline():
    return BFSBaselineRANS.solve(FAST_CFG)


@pytest.fixture(scope="module")
def disc(baseline):
    return BFSDiscrepancy.build(baseline=baseline)


def test_baseline_converges_with_physical_reattachment(baseline):
    # the corrected-BC baseline converges the separated field; its reattachment
    # is in the physical range around the DNS 6.28 h (the residual gap is the
    # model-form error the discrepancy quantifies, reported not asserted)
    assert baseline.status == "Converged"
    assert 4.0 < baseline.reattachment < 8.5


def test_baseline_field_is_finite(baseline):
    for f in (baseline.U, baseline.V, baseline.k, baseline.omega, baseline.nu_t):
        assert np.all(np.isfinite(f))
    assert np.all(baseline.k >= 0.0)


def test_slip_top_tracks_dns_outer_flow(baseline):
    # with the free-slip top (as the data shows: zero mean shear at the top
    # edge) the outer flow decelerates only through the expansion and stays at
    # DNS level; the old no-slip top grew a spurious layer (U ~ 0.5 at y = 5.9)
    s = baseline.sample_at(np.array([10.0, 19.0]), np.array([5.9, 5.9]))
    assert np.all(s["U"] > 0.8)


def test_inflow_boundary_layer_reaches_step(baseline):
    # the prescribed inlet layer develops to a DNS-like thickness by x/h = -3
    # (measured delta_999 = 1.158 h; a uniform inlet gives ~0.15 h) and the
    # near-wall profile is inside the layer, not free stream
    assert baseline.delta_999_at(-3.0) > 0.6
    u = baseline.sample_at(np.array([-3.0]), np.array([1.08]))["U"][0]
    assert u < 0.8


def test_legacy_config_still_solves():
    # the legacy uniform-inlet no-slip-top path is kept for regression
    # comparison and must still converge
    leg = BFSBaselineRANS.solve({**FAST_CFG, "Lu": 4.0, "nx_up": 12,
                                 "slip_top": False, "inlet_delta": None})
    assert leg.status == "Converged"


def test_sample_and_gradient_in_domain(baseline):
    # sample a clearly-interior downstream point (recirculation region)
    xq = np.array([6.0, 8.0])
    yq = np.array([0.5, 1.0])
    s = baseline.sample_at(xq, yq)
    assert np.all(np.isfinite(s["U"])) and np.all(np.isfinite(s["nu_t"]))
    g = baseline.velocity_gradient_at(xq, yq)
    assert g.shape == (2, 3, 3)
    # the shear dU/dy is the dominant gradient and is nonzero in the shear layer
    assert np.any(np.abs(g[:, 0, 1]) > 1e-6)
    # per-point wall-adaptive steps are accepted and give finite gradients
    g2 = baseline.velocity_gradient_at(xq, yq, dx=0.03, dy=np.array([0.01, 0.03]))
    assert np.all(np.isfinite(g2))
    tau = baseline.timescale_at(xq, yq)
    assert np.all(tau > 0.0)


def test_coordinate_mapping(baseline):
    # upstream (x < 0) y is shifted up by the step height; downstream unchanged
    x_h = np.array([-3.0, 6.0])
    y_h = np.array([0.5, 0.5])
    xm, ym = baseline.map_dns_points(x_h, y_h)
    assert np.allclose(xm, x_h)
    assert ym[0] == pytest.approx(0.5 + BFSBaselineRANS.STEP_H)   # upstream + h
    assert ym[1] == pytest.approx(0.5)                            # downstream


def test_discrepancy_is_wellformed(disc):
    m = disc.mask
    assert m.sum() > 0.9 * m.size          # most profile points are valid
    feats, db = disc.training_pairs()
    assert feats.shape[1] == 5 and np.all(np.isfinite(feats))
    assert db.shape[1:] == (3, 3) and np.all(np.isfinite(db))
    # the anisotropy discrepancy is traceless where turbulence is resolved
    tr = np.trace(db, axis1=1, axis2=2)
    assert np.max(np.abs(tr)) < 1e-9
    # the discrepancy is a real, nonzero correction (baseline is not the DNS)
    assert np.mean(disc.db_magnitude()[m]) > 0.05
    # and bounded: b lives in the realizable ball, so db magnitudes are O(1)
    assert np.max(disc.db_magnitude()[m]) < 2.5


def test_baseline_anisotropy_realizable(disc):
    # the limiter-consistent SST baseline anisotropy is itself realizable, so
    # the discrepancy is measured against a physically admissible reference
    R_base = 2.0 * (disc.b_baseline + np.eye(3) / 3.0)
    assert np.mean(rz.is_realizable(R_base[disc.mask], tol=1e-6)) == 1.0


def test_attached_station_has_channel_structure(disc):
    # at the attached upstream station the discrepancy has the canonical
    # attached-flow structure (the channel finding): the near-wall
    # normal-stress anisotropy error dominates and the outer layer is milder.
    # The old thin-inlet baseline inverted this by comparing free stream
    # against boundary-layer turbulence in the outer region.
    mag = disc.db_magnitude()
    sel = (disc.station == 0) & disc.mask
    near = sel & (disc.y_h < 0.15)
    outer = sel & (disc.y_h > 0.3) & (disc.y_h < 1.2)
    assert np.mean(mag[outer]) < np.mean(mag[near])


def test_reattachment_error_is_real(disc):
    # the baseline does not reproduce the DNS reattachment exactly; the signed
    # gap (grid- and BC-sensitive in magnitude) is the model-form signal
    assert np.isfinite(disc.reattachment_error)
    assert abs(disc.reattachment_error) > 0.02
