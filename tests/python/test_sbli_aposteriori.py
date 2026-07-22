"""A-posteriori ensemble machinery: the pure construction and scoring
pieces, hermetic (no DNS data, no solver)."""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "python"))

from UQ import realizability
from UQ.datasets import sbli_aposteriori as apo
from UQ.datasets.sbli_interaction import SBLIWallSeries


class _FakeRecord:
    """The scoring surface of an interaction record: a wall series and the
    field-row shock landmark."""

    def __init__(self, series, shock):
        self.series = series
        self._shock = shock

    def shock_position(self):
        return self._shock


def _synthetic_wall(x, sep=(-2.0, 3.0), cp_rise=0.5, cf_level=2.5e-3):
    cf = np.full_like(x, cf_level)
    inside = (x > sep[0]) & (x < sep[1])
    cf[inside] = -0.5e-3
    cp = cp_rise / (1.0 + np.exp(-(x - 0.5)))
    st = 1.0e-3 * (1.0 + 0.3 * np.tanh(x))
    return {"x_star": x, "Cf": cf, "Cp": cp, "qw": st.copy(), "St": st}


def test_db3_tensor_structure():
    rng = np.random.default_rng(0)
    draws = rng.normal(0.0, 0.05, size=(7, 4, 3))
    db = apo._db3_to_tensor(draws)
    assert db.shape == (7, 4, 3, 3)
    # trace-free, symmetric, span off-diagonals zero
    tr = np.trace(db, axis1=2, axis2=3)
    assert np.max(np.abs(tr)) < 1e-14
    assert np.allclose(db, np.swapaxes(db, 2, 3))
    assert np.max(np.abs(db[:, :, 0, 2])) == 0.0
    assert np.max(np.abs(db[:, :, 1, 2])) == 0.0


def test_corner_targets_realizable_and_masked():
    rng = np.random.default_rng(1)
    # start from realizable anisotropies (project random symmetric batches)
    raw = rng.normal(0.0, 0.2, size=(50, 3, 3))
    raw = 0.5 * (raw + np.swapaxes(raw, 1, 2))
    b_base, _ = realizability.project_anisotropy(raw)
    mask = np.ones(50, dtype=bool)
    mask[::5] = False
    fam = apo.corner_targets(b_base, mask)
    assert set(fam) == {"1C_d1", "2C_d1", "3C_d1",
                        "1C_d0.5", "2C_d0.5", "3C_d0.5"}
    for lab, b in fam.items():
        _, dist = realizability.project_anisotropy(b)
        assert np.max(dist) < 1e-8, lab
        # masked cells revert exactly to the baseline
        assert np.allclose(b[~mask], b_base[~mask])
    # the full 1C corner actually moves the unmasked eigenvalues
    moved = np.abs(fam["1C_d1"][mask] - b_base[mask]).max()
    assert moved > 1e-3


def test_landmarks_from_wall():
    x = np.linspace(-12.0, 12.0, 241)
    w = _synthetic_wall(x)
    lm = apo.landmarks_from_wall(w)
    assert lm["x_s"] is not None and abs(lm["x_s"] - (-2.0)) < 0.2
    assert lm["x_r"] is not None and abs(lm["x_r"] - 3.0) < 0.2
    # half-rise of the logistic sits at its midpoint
    assert lm["shock"] is not None and abs(lm["shock"] - 0.5) < 0.3
    # attached wall: no separation landmarks, half-rise still defined
    wa = _synthetic_wall(x, sep=(50.0, 60.0))
    lma = apo.landmarks_from_wall(wa)
    assert lma["x_s"] is None and lma["x_r"] is None


def test_score_ensemble_coverage_and_landmarks():
    x = np.linspace(-12.0, 12.0, 241)
    truth = _synthetic_wall(x)
    series = SBLIWallSeries(x, truth["Cf"],
                            pw=1.0 + truth["Cp"], pwrms=0.01 * np.ones_like(x),
                            St=truth["St"])
    # the series normalizes cp by its own plateau and q_dyn; the synthetic
    # pw above is not in file units, so pin the truth to the member
    # convention directly (production consistency is the observation
    # operator's documented (p - p_inf)/q_dyn against the series' plateau
    # form of the same quantity)
    series.cp = truth["Cp"]
    record = _FakeRecord(series, shock=0.5)

    rng = np.random.default_rng(2)
    members = []
    for i in range(12):
        w = _synthetic_wall(x)
        # scale AND shift: a pure scaling leaves the logistic midpoint
        # invariant, and under the unified sub-grid landmark rule twelve
        # identical half-rises form a zero-width band (the old grid-index
        # rule masked this by quantizing onto the truth's grid point); the
        # streamwise shift gives the landmark ensemble genuine scatter
        # around the truth, which is what the containment assertion tests
        dxs = 0.15 * rng.normal()
        w["Cf"] = np.interp(x - dxs, x, w["Cf"]) * (1.0 + 0.05 * rng.normal())
        w["Cp"] = np.interp(x - dxs, x, w["Cp"]) * (1.0 + 0.05 * rng.normal())
        w["St"] = np.interp(x - dxs, x, w["St"]) * (1.0 + 0.05 * rng.normal())
        members.append({"status": "EvaluationStatus.Converged",
                        "wall": w,
                        "landmarks": apo.landmarks_from_wall(w)})
    out = apo.score_ensemble(members, record)
    assert out["scored"] is True
    assert out["n_converged"] == 12
    # an ensemble scattered around the truth covers it at most stations
    assert out["Cf_station_coverage_0.9"] > 0.5
    assert out["Cp_station_coverage_0.9"] > 0.5
    assert out["shock_contained"] is True
    # too few converged members: honestly unscored
    few = apo.score_ensemble(members[:1], record)
    assert few["scored"] is False


def test_dq_solver_unit_scale():
    class _U:
        U_inf = 500.0
        T_inf = 100.0

    class _R:
        reference = {"u_tau_over_uinf": 0.05, "T_w": 1.9}

    dq = np.ones((2, 4, 2))
    out = apo.dq_to_solver_units(dq, _R(), _U())
    assert np.allclose(out, 0.05 * 500.0 * 1.9 * 100.0)


def test_job_output_mapping(tmp_path):
    from UQ import reproduce_sbli_aposteriori as rep
    d = str(tmp_path)
    j1 = ["--stage", "member", "--fold", "s1.0", "--kind", "flow",
          "--index", "3", "--model-seed", "1", "--results", d]
    assert rep._job_output(d, "s1.0", j1).endswith(
        "member_s1.0_flow_ms1_3.npz")
    j2 = j1 + ["--attached"]
    assert rep._job_output(d, "s1.0", j2).endswith(
        "member_attached_s1.0_flow_ms1_3.npz")
    j3 = ["--stage", "member", "--fold", "s1.0", "--corner", "2C_d0.5"]
    assert rep._job_output(d, "s1.0", j3).endswith(
        "member_s1.0_corner_2C_d0.5.npz")
    # the exploratory fold's outputs land in the exploratory namespace
    j4 = ["--stage", "member", "--fold", "faradiab", "--kind", "flow",
          "--index", "0", "--model-seed", "0", "--results", d]
    p4 = rep._job_output(d, "faradiab", j4)
    assert os.sep + "exploratory" + os.sep in p4
