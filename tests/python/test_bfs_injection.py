"""A-posteriori Reynolds-stress injection on the real Le-Moin BFS.

Runs real (coarse, fast) coupled solves, so it skips where the binding or the
gitignored DNS_data is absent. Verifies the pre-registered injection gate on
the real case: the injected solve CONVERGES with realizability re-asserted
every outer iteration; db = 0 reproduces the baseline solve (the deferred
correction adds exactly -div(2 k db) relative to baseline); a genuine
anisotropy shift produces a physical reattachment response; targets are
projected into the realizable set; and injected runs are deterministic in
isolation (no warm-start-order dependence).
"""
import numpy as np
import pytest

pytest.importorskip("rans_sst_py", reason="C++ binding not built")
from UQ.datasets.backward_facing_step import BackwardFacingStepDNS
from UQ.datasets.bfs_baseline import BFSBaselineRANS
from UQ.datasets.bfs_injection import BFSInjection
from UQ import realizability as rz

pytestmark = pytest.mark.skipif(
    not BackwardFacingStepDNS.is_available(),
    reason="BFS DNS_data not present (bulk data is local/gitignored)",
)

FAST_CFG = {"nx_up": 20, "nx_down": 26, "ny_up": 14, "ny_down": 10,
            "Lu": 10.0, "Ld": 22.0, "max_iter": 8000, "conv_tol": 1.0e-4}


@pytest.fixture(scope="module")
def setup():
    dns = BackwardFacingStepDNS.load()
    baseline = BFSBaselineRANS.solve(FAST_CFG, dns=dns)
    inj = BFSInjection(cfg=FAST_CFG, dns=dns, baseline=baseline)
    return {"dns": dns, "baseline": baseline, "inj": inj, "runs": {}}


def _run_db0(setup):
    if "db0" not in setup["runs"]:
        inj = setup["inj"]
        bt, _ = inj.target_from_db(np.zeros((inj.n_cells, 3, 3)))
        setup["runs"]["db0"] = inj.run(bt)
    return setup["runs"]["db0"]


def test_db0_injection_reproduces_baseline(setup):
    # with db = 0 the injected force is -div(2 k (b_B_frozen - b_B_running)),
    # only interpolation/freezing error away from zero: the coupled solve must
    # converge and land on the baseline reattachment
    r = _run_db0(setup)
    assert r["status"] == "Converged"
    assert abs(r["reattachment"] - setup["baseline"].reattachment) < 0.15


def test_realizability_asserted_every_iteration(setup):
    # the barycentric check runs each outer iteration of the coupled solve and
    # the projected target passes it always (separate from the invariant
    # feature construction)
    r = _run_db0(setup)
    d = r["diagnostics"]
    assert d["active"]
    assert d["checked_iters"] >= r["iterations"]
    assert d["all_realizable"]
    assert d["max_violation"] == 0.0


def test_anisotropy_shift_moves_reattachment(setup):
    # a one-component-ward shift of the bubble anisotropy is a genuine closure
    # perturbation: the coupled solve converges and the reattachment responds
    inj = setup["inj"]
    in_bubble = ((inj.cc_x > 0.0) & (inj.cc_x < 7.0) & (inj.cc_y < 1.2)
                 & inj.cell_mask)
    shift = np.zeros((inj.n_cells, 3, 3))
    shift[:, 0, 0], shift[:, 1, 1], shift[:, 2, 2] = 0.30, -0.15, -0.15
    db = np.where(in_bubble[:, None, None], shift, 0.0)
    bt, _ = inj.target_from_db(db)
    r = inj.run(bt)
    setup["runs"]["shift"] = r
    assert r["status"] == "Converged"
    assert r["diagnostics"]["all_realizable"]
    assert abs(r["reattachment"] - setup["baseline"].reattachment) > 0.2


def test_targets_are_projected_realizable(setup):
    # an intentionally unrealizable db comes back inside the barycentric set
    inj = setup["inj"]
    db = np.zeros((inj.n_cells, 3, 3))
    db[:, 0, 0], db[:, 1, 1], db[:, 2, 2] = 2.0, -1.0, -1.0   # far outside
    bt, dist = inj.target_from_db(db)
    R = 2.0 * (bt + np.eye(3) / 3.0)
    assert np.all(rz.is_realizable(R, tol=1e-8))
    assert np.max(dist) > 0.5          # the projection genuinely moved points


def test_injected_runs_are_order_independent(setup):
    # injected solves bypass the warm-start cache entirely, so re-running the
    # same target after other runs reproduces the identical result
    inj = setup["inj"]
    r_first = _run_db0(setup)
    bt, _ = inj.target_from_db(np.zeros((inj.n_cells, 3, 3)))
    r_again = inj.run(bt)
    assert r_again["reattachment"] == pytest.approx(r_first["reattachment"],
                                                    abs=1e-12)
    assert r_again["iterations"] == r_first["iterations"]


def test_wall_cf_shows_recirculation(setup):
    # the baseline separated field has negative Cf inside the bubble and
    # positive Cf in recovery along the downstream bottom wall
    r = _run_db0(setup)
    xw, cf = setup["inj"].wall_cf(r["fields"])
    assert np.all(np.isfinite(cf))
    assert np.min(cf[xw < 4.0]) < 0.0
    assert np.max(cf[xw > 10.0]) > 0.0
