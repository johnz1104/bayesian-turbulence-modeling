"""Backward-facing step DNS loader (Le, Moin and Kim 1997) on the real raw files.

These tests run only where the gitignored DNS_data is present (locally); on CI
without the bulk data they skip. They check the loader against the published case
parameters (the six station x/h, Re_h, the reattachment length) and the internal
consistency of the canonical record (tensor assembly from the rms columns, k, the
vanishing spanwise off-diagonals), plus two data-only physics anchors that mirror
the modeled-sigma anchors of the wall-bounded loaders: the DNS Reynolds stress is
realizable at every resolved station point, and the wall Cf changes sign across the
reattachment (negative inside the recirculation, positive in recovery), consistent
with the published x_r/h.
"""
import numpy as np
import pytest

from UQ.datasets.backward_facing_step import (
    BackwardFacingStepDNS, BFS_STATIONS, REATTACHMENT_XR_H, RE_H)

pytestmark = pytest.mark.skipif(
    not BackwardFacingStepDNS.is_available(),
    reason="BFS DNS_data not present (bulk data is local/gitignored)",
)


@pytest.fixture(scope="module")
def bfs():
    return BackwardFacingStepDNS.load()


def test_six_stations_with_expected_xh(bfs):
    # the six streamwise stations map to the readme x/h and are ordered
    assert len(bfs.stations) == 6
    expected = [x for _, x in BFS_STATIONS]
    assert bfs.station_xh.tolist() == expected
    assert np.all(np.diff(bfs.station_xh) > 0)


def test_case_parameters(bfs):
    assert bfs.meta["re_h"] == RE_H == 5100
    assert bfs.reattachment_truth() == REATTACHMENT_XR_H == pytest.approx(6.28)
    assert bfs.meta["case"] == "backward_facing_step"


def test_reynolds_tensor_assembly(bfs):
    # symmetric, nonnegative normal stresses, k = 0.5 tr(R)
    assert np.max(np.abs(bfs.R - np.transpose(bfs.R, (0, 2, 1)))) == 0.0
    assert np.all(bfs.uu >= 0) and np.all(bfs.vv >= 0) and np.all(bfs.ww >= 0)
    assert np.allclose(bfs.k, 0.5 * np.trace(bfs.R, axis1=1, axis2=2))


def test_spanwise_offdiagonals_vanish(bfs):
    # the mean is spanwise-homogeneous, so R_xz and R_yz are identically zero
    assert np.all(bfs.R[:, 0, 2] == 0.0)
    assert np.all(bfs.R[:, 1, 2] == 0.0)


def test_realizability_anchor(bfs):
    # data-only physics anchor: DNS stress realizable at every resolved point
    assert bfs.realizable_fraction() == 1.0


def test_anisotropy_target_traceless_on_resolved(bfs):
    # b_DNS = R/(2k) - I/3 is traceless where turbulence is resolved (k > 0)
    b = bfs.b_dns()
    mask = bfs.valid_mask()
    tr = np.trace(b[mask], axis1=1, axis2=2)
    assert np.max(np.abs(tr)) < 1e-9


def test_cf_reattachment_signature(bfs):
    # Cf < 0 inside the recirculation (x/h = 4), Cf > 0 in the recovery region,
    # so the reattachment is the downstream Cf sign change, near the published x_r/h.
    # (Upstream of the step Cf is also positive, so the full profile has a second,
    # separation sign change at the step; the reattachment is the downstream one.)
    wall = bfs.wall
    assert wall[4.0]["Cf"] < 0.0
    assert wall[10.0]["Cf"] > 0.0 and wall[15.0]["Cf"] > 0.0
    xs, cf = bfs.cf_stations()
    down = xs > 0.0                          # stations downstream of the step
    signs = np.sign(cf[down])
    n_changes = int(np.sum(np.abs(np.diff(signs)) > 1))
    assert n_changes == 1                    # exactly the reattachment
    # the sign change brackets the published reattachment length
    xd = xs[down]
    last_neg = xd[np.where(cf[down] < 0)[0].max()]
    first_pos = xd[np.where(cf[down] > 0)[0].min()]
    assert last_neg <= REATTACHMENT_XR_H <= first_pos


def test_dnsfield_wiring(bfs):
    # the canonical record drives the discrepancy/feature interface unchanged
    field = bfs.to_dnsfield()
    out = field.extract()
    assert out["features"].shape == (bfs.n, 5)
    assert out["reynolds_discrepancy"].shape == (bfs.n, 3, 3)


def test_wall_table_complete(bfs):
    # every station carries its U_e, U_tau, Cf, Cp and boundary-layer thicknesses
    for _, x in BFS_STATIONS:
        entry = bfs.wall[x]
        for key in ("U_e", "U_tau", "Cf", "Cp", "Delta_999", "Theta"):
            assert key in entry and np.isfinite(entry[key])
