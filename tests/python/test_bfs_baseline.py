"""Le-Moin BFS SST baseline and the field-to-field discrepancy, on the real data.

Runs a real (coarse, fast) SST solve, so it skips where the binding or the
gitignored DNS_data is absent. It checks that the baseline converges and
under-predicts the reattachment (the known SST model-form error), that the field
interpolation, velocity gradient, and timescale are well-formed, and that the
discrepancy assembly returns a finite, traceless anisotropy discrepancy conditioned
on realizable-baseline features through the canonical DNSField interface.
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

# coarse, fast solve config for the test (the science runs a finer grid)
FAST_CFG = {"nx_up": 12, "nx_down": 26, "ny_up": 14, "ny_down": 10,
            "Lu": 4.0, "Ld": 22.0, "max_iter": 6000, "conv_tol": 1.0e-4}


@pytest.fixture(scope="module")
def baseline():
    return BFSBaselineRANS.solve(FAST_CFG)


@pytest.fixture(scope="module")
def disc(baseline):
    return BFSDiscrepancy.build(baseline=baseline)


def test_baseline_converges_and_underpredicts(baseline):
    # the channel-calibrated SST converges the separated field and under-predicts
    # the reattachment (5100-Re Le-Moin DNS truth is 6.28 h) - the model-form error
    assert baseline.status == "Converged"
    assert 3.0 < baseline.reattachment < 6.28


def test_baseline_field_is_finite(baseline):
    for f in (baseline.U, baseline.V, baseline.k, baseline.omega, baseline.nu_t):
        assert np.all(np.isfinite(f))
    assert np.all(baseline.k >= 0.0)


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


def test_baseline_anisotropy_realizable(disc):
    # the SST (Boussinesq) baseline anisotropy is itself realizable, so the
    # discrepancy is measured against a physically admissible reference
    from UQ import discrepancy as dq
    xq, yq = disc.baseline.map_dns_points(disc.dns.x_h, disc.dns.y_h)
    g = disc.baseline.velocity_gradient_at(xq, yq)
    tau = disc.baseline.timescale_at(xq, yq)
    S, _ = dq.strain_rotation(np.where(np.isfinite(g), g, 0.0),
                              np.where(np.isfinite(tau), tau, 1.0))
    R_base = 2.0 * (dq.boussinesq_anisotropy(S) + np.eye(3) / 3.0)
    assert np.mean(rz.is_realizable(R_base[disc.mask], tol=1e-6)) == 1.0


def test_reattachment_error_is_negative(disc):
    # SST under-predicts the reattachment length, so the error (RANS - DNS) < 0
    assert disc.reattachment_error < 0.0
