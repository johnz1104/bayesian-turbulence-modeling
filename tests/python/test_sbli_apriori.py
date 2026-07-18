"""Tests for the a-priori shock-interaction study glue: the baseline
configuration physics (oblique-shock relations, units, inflow construction),
the split and scoring plumbing on synthetic rows, and a smoke-epoch training
pass. No production training, no solver convergence runs (those live in the
reproduce driver and are recorded in the numbers JSON).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "build"))

rans = pytest.importorskip("rans_sst_py")

from UQ.datasets.sbli_baseline import (SBLIUnits, oblique_shock_state,
                                       inflow_state_table, SBLIBaseline,
                                       RE_INLET)
from UQ.datasets import sbli_apriori
from UQ.datasets.sbli_apriori import SBLIAPriori, seed_mean
from UQ.datasets.sbli_interaction import SBLIInteractionDNS


def test_units_recover_dataset_reynolds():
    u = SBLIUnits()
    re = u.rho_inf * u.U_inf * u.delta0 / u.mu_inf
    assert abs(re / RE_INLET - 1.0) < 1e-12
    assert abs(u.U_inf / u.a_inf - 2.28) < 1e-12


def test_oblique_shock_relations():
    u = SBLIUnits()
    post, beta = oblique_shock_state(u, deflection_deg=8.0)
    # the returned beta satisfies the theta-beta-M identity for theta = 8 deg
    M1 = u.mach
    g = 1.4
    rhs = (2.0 / np.tan(beta) * (M1 ** 2 * np.sin(beta) ** 2 - 1.0)
           / (M1 ** 2 * (g + np.cos(2 * beta)) + 2.0))
    assert abs(rhs - np.tan(np.deg2rad(8.0))) < 1e-3
    # weak solution: beta between the Mach angle and 45 degrees at M 2.28
    assert np.rad2deg(beta) > np.rad2deg(np.arcsin(1 / M1))
    assert np.rad2deg(beta) < 45.0
    # post-shock: compression, downward deflection, still supersonic
    assert post.p > u.p_inf and post.rho > u.rho_inf
    assert post.v < 0.0
    a2 = np.sqrt(g * post.p / post.rho)
    assert np.hypot(post.u, post.v) / a2 > 1.0


_DATA = SBLIInteractionDNS.is_available("1.0")


@pytest.mark.skipif(not _DATA, reason="SBLI interaction data not present")
def test_inflow_state_table_from_record():
    record = SBLIInteractionDNS.wall_thermal("1.0")
    units = SBLIUnits()
    y, rho, u, v, p, k, om = inflow_state_table(record, units, 0)
    assert y[0] == 0.0 and np.all(np.diff(y) > 0)
    assert np.all(rho > 0) and np.all(k >= 0) and np.all(om > 0)
    # the layer accelerates from the wall to the free stream
    assert u[0] == pytest.approx(0.0, abs=1e-6)
    assert abs(u[-1] / units.U_inf - 1.0) < 0.05
    assert np.all(p == units.p_inf)


@pytest.mark.skipif(not _DATA, reason="SBLI interaction data not present")
def test_configure_smoke_no_solve():
    record = SBLIInteractionDNS.wall_thermal("1.0")
    base = SBLIBaseline.configure(record, with_shock=True, nx=40, ny=24,
                                  x_hi=4.0, height=6.0,
                                  max_iterations=1)
    # shock geometry recorded and inside the domain
    assert 30.0 < base.meta["beta_deg"] < 45.0
    assert base.meta["x_lo"] < base.meta["x_entry_star"] < 0.0
    # the wall-temperature profile carries the measured recovery value
    # upstream and the s-condition downstream (s = 1.0: both at recovery)
    Tw = base.wall_temps_dim / base.units.T_inf
    assert abs(Tw[0] - 1.932) < 0.02
    assert abs(Tw[-1] - 1.932) < 0.02


def _synthetic_extraction(n=60, with_dq=True, history=True):
    rng = np.random.default_rng(0)
    nf = 7 if history else 6
    return {
        "features": rng.standard_normal((n, nf)),
        "db": rng.standard_normal((n, 3, 3)) * 0.05,
        "db_free": rng.standard_normal((n, 3)) * 0.05,
        "dq": rng.standard_normal((n, 3)) * 0.01 if with_dq else None,
        "x": np.linspace(-10, 8, n),
        "y": np.full(n, 0.3),
        "region": np.array(["upstream"] * (n // 2)
                           + ["interaction"] * (n - n // 2), dtype=object),
        "realizable_fraction": 1.0,
        "g_basis": rng.normal(size=(n, 3)),
        # identity coefficient-to-component maps: the synthetic fixture's
        # basis is orthonormal by construction, so g and db_free coincide up
        # to the draw and the reconstruction path is exercised end to end
        "basis_M": np.tile(np.eye(3), (n, 1, 1)),
        "meta": {"n_points": n},
    }


def test_target_and_feature_views():
    ext = _synthetic_extraction()
    assert SBLIAPriori._target(ext, "dq_y").shape == (60, 1)
    assert SBLIAPriori._target(ext, "dq_joint").shape == (60, 2)
    assert SBLIAPriori._target(ext, "db").shape == (60, 3)
    assert SBLIAPriori._features(ext, history=False).shape == (60, 6)
    assert SBLIAPriori._features(ext, history=True).shape == (60, 7)
    with pytest.raises(ValueError):
        SBLIAPriori._target(ext, "nope")


def test_loso_smoke_epoch():
    torch = pytest.importorskip("torch")
    cases = {f"s{i}": _synthetic_extraction() for i in range(3)}
    study = SBLIAPriori(test_sets=cases, train_sets=cases, attached=None,
                        results_dir=".")
    res = study.loso("dq_y", history=False, seeds=(0,), epochs=1)
    assert set(res) == set(cases)
    one = res["s0"]["models"]
    assert set(one) == {"flow", "gauss", "pooled"}
    scores = one["flow"][0]
    assert len(scores["coverage_0.9"]) == 1
    assert "region_coverage_0.9" in scores
    for reg in scores["region_coverage_0.9"].values():
        assert 0.0 <= reg[0] <= 1.0
    # the joint leg carries the energy score
    res_j = study.insample("dq_joint", history=False, seeds=(0,), epochs=1)
    assert "energy_score" in res_j["models"]["flow"][0]


def test_seed_mean():
    per_seed = [{"coverage_0.9": [0.8, 0.6]}, {"coverage_0.9": [0.9, 0.7]}]
    assert seed_mean(per_seed, "coverage_0.9") == [
        pytest.approx(0.85), pytest.approx(0.65)]
    per_scalar = [{"energy_score": 1.0}, {"energy_score": 2.0}]
    assert seed_mean(per_scalar, "energy_score") == pytest.approx(1.5)


def test_strides_cover_every_case():
    assert set(sbli_apriori.TEST_STRIDE) == {
        "adiabatic", "s0.5", "s0.75", "s1.0", "s1.4", "s1.9"}
    for case, (sx, sy) in sbli_apriori.TEST_STRIDE.items():
        tx, ty = sbli_apriori._train_stride(case)
        assert (tx, ty) == (2 * sx, 2 * sy)
