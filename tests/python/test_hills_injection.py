"""A-posteriori Reynolds-stress injection on the real periodic hills.

Runs real (coarse, fast) coupled solves on the streamwise-periodic
curved-bottom mesh, so it skips where the binding or the gitignored DNS_data
is absent. Verifies the injection gate on the second geometry: the injected
solve CONVERGES with realizability re-asserted every outer iteration; db = 0
reproduces the baseline solve; a genuine anisotropy shift converges; and the
pinned quantity extraction (wall-shear bubble geometry, mean-velocity probes)
is consistent between the solver record and the DNS reading.
"""
import numpy as np
import pytest

pytest.importorskip("rans_sst_py", reason="C++ binding not built")
from UQ.datasets.periodic_hills import PeriodicHillsDNS
from UQ.datasets.hills_baseline import HillsBaselineRANS
from UQ.datasets.hills_injection import HillsInjection
from UQ.datasets.separated_aposteriori import HillsAPosteriori

pytestmark = pytest.mark.skipif(
    not PeriodicHillsDNS.is_available("1p0"),
    reason="periodic-hills DNS_data not present (bulk data is local/gitignored)",
)

# coarse, fast solve config (the science runs the pinned production grid);
# body_force is the coarse-grid matched value (crest bulk 0.995)
FAST_CFG = {"nx": 48, "ny": 32, "max_iter": 15000, "conv_tol": 1.0e-4,
            "body_force": 0.0095}


@pytest.fixture(scope="module")
def setup():
    dns = PeriodicHillsDNS.load("1p0")
    baseline = HillsBaselineRANS.solve(FAST_CFG, dns=dns)
    inj = HillsInjection(cfg=FAST_CFG, dns=dns, baseline=baseline)
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
    assert abs(r["reattachment"] - setup["baseline"].reattachment) < 0.2


def test_realizability_asserted_every_iteration(setup):
    r = _run_db0(setup)
    d = r["diagnostics"]
    assert d["active"]
    assert d["checked_iters"] >= r["iterations"]
    assert d["all_realizable"]
    assert d["max_violation"] == 0.0


def test_bubble_geometry_matches_baseline_record(setup):
    # the pinned crossing rules on the db0 fields agree with the baseline
    # record's reattachment, and the separation onset sits upstream of it
    r = _run_db0(setup)
    x_s, x_r = setup["inj"].bubble_geometry(r["fields"])
    assert np.isfinite(x_s) and np.isfinite(x_r)
    assert x_s < x_r
    assert abs(x_r - r["reattachment"]) < 1e-12   # run() uses the same rule


def test_dns_bubble_reading_consistent(setup):
    # the DNS-side bubble helper shares the reattachment reading of the loader
    dns = setup["dns"]
    x_s, x_r, length = dns.bottom_wall_bubble()
    assert abs(x_r - dns.bottom_wall_reattachment()) < 1e-12
    assert 0.0 <= x_s < x_r
    assert abs(length - (x_r - x_s)) < 1e-12


def test_probe_extraction(setup):
    # the pinned admissible probes exist, and both the member field and the
    # DNS supply finite mean velocity at every one of them
    inj = setup["inj"]
    assert inj.probe_x.size >= 12                 # most of the 8x3 grid stays
    r = _run_db0(setup)
    u_member = inj.probe_velocity(r["fields"])
    u_truth = inj.probe_truth()
    assert np.all(np.isfinite(u_member))
    assert np.all(np.isfinite(u_truth))
    assert u_member.shape == u_truth.shape


def test_anisotropy_shift_converges(setup):
    # a one-component-ward shift of the bubble anisotropy is a genuine closure
    # perturbation on the curved periodic mesh: the coupled solve converges
    # with realizability enforced
    inj = setup["inj"]
    in_bubble = ((inj.cc_x > 0.5) & (inj.cc_x < 6.0) & (inj.cc_y < 1.5)
                 & inj.cell_mask)
    shift = np.zeros((inj.n_cells, 3, 3))
    shift[:, 0, 0], shift[:, 1, 1], shift[:, 2, 2] = 0.30, -0.15, -0.15
    db = np.where(in_bubble[:, None, None], shift, 0.0)
    bt, _ = inj.target_from_db(db)
    r = inj.run(bt)
    assert r["status"] == "Converged"
    assert r["diagnostics"]["all_realizable"]
    assert np.isfinite(r["reattachment"])


def test_scalar_scoring_handles_missing_crossings():
    # scoring is exclusion-aware: non-converged members and members whose
    # bubble vanished (no crossing) are counted, the rest are scored
    members = [
        {"reattachment": 4.0, "status": "Converged", "all_realizable": True},
        {"reattachment": 5.0, "status": "Converged", "all_realizable": True},
        {"reattachment": float("nan"), "status": "Converged",
         "all_realizable": True},
        {"reattachment": 9.9, "status": "Unconverged", "all_realizable": True},
    ]
    rec = HillsAPosteriori.score_scalar(members, "reattachment", 4.5,
                                        level=0.9)
    assert rec["n_members"] == 4
    assert rec["n_nonconverged"] == 1
    assert rec["n_no_crossing"] == 1
    assert rec["contains_truth"]
    assert abs(rec["mean"] - 4.5) < 1e-12


# ---- status-gated QoI extraction (audit regression) -------------------------
# Two committed hills member records were bit-identical to their predecessors
# because a diverged solve left the previous run's fields visible and the
# wrapper extracted bubble geometry status-blind. The gate must (a) never read
# held fields for a non-converged member (NaN QoIs), and (b) read them for a
# converged one. Driven through stubs so the contract is pinned without a
# solve; the C++ side additionally invalidates held fields on divergence.

class _StubResult:
    def __init__(self, status):
        self.status = status
        self.simple_iters = 7


class _StubFM:
    def __init__(self, status):
        self._status = status
        self.last_fields_calls = 0

    def set_target_anisotropy(self, b6):
        pass

    def clear_target_anisotropy(self):
        pass

    def evaluate(self, theta):
        return _StubResult(self._status)

    def has_last_fields(self):
        return True   # fields LOOK available either way; the gate must decide

    def last_fields(self):
        self.last_fields_calls += 1
        return "FIELDS"

    def injection_diagnostics(self):
        return {}


def _stub_wrapper(status):
    from UQ.datasets.hills_injection import HillsInjection
    w = object.__new__(HillsInjection)
    w._fwd = {"fm": _StubFM(status)}
    w._theta = [0.31, 0.09]
    w.bubble_geometry = lambda fields: (1.25, 4.75)
    return w


def test_nonconverged_member_records_nan_and_never_reads_fields():
    from UQ.datasets.hills_injection import HillsInjection
    for status in ("DivergenceDetected", "Unconverged", "InvalidParameters"):
        w = _stub_wrapper(status)
        rec = HillsInjection.run(w, np.zeros((4, 3, 3)))
        assert rec["status"] == status
        assert np.isnan(rec["separation"]) and np.isnan(rec["reattachment"])
        assert np.isnan(rec["bubble_length"])
        assert rec["fields"] is None
        assert w._fwd["fm"].last_fields_calls == 0, \
            "non-converged member must never read held fields"


def test_converged_member_extracts_qois():
    from UQ.datasets.hills_injection import HillsInjection
    w = _stub_wrapper("Converged")
    rec = HillsInjection.run(w, np.zeros((4, 3, 3)))
    assert rec["status"] == "Converged"
    assert rec["separation"] == 1.25 and rec["reattachment"] == 4.75
    assert abs(rec["bubble_length"] - 3.5) < 1e-15
    assert w._fwd["fm"].last_fields_calls == 1
