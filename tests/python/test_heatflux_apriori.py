"""Machinery tests for the heat-flux model-form study module.

Asserts the ASSEMBLY, SPLIT and SCORE-WEIGHT logic is sound (the
pre-registered case lists and families, the interior mask, well-formed
(features, dq) records on real data, both model kinds through the identical
fit / sample / score path, and the pinned conformal weight definitions); the
study NUMBERS (full-epoch coverage and scores per held-out case) belong to
the evidence package, not to tests, and no quality threshold is asserted
here. Real-data tests skip where the gitignored DNS_data is absent; model
tests skip where torch is absent.
"""
import numpy as np
import pytest

from UQ.datasets.gv_channel import GVChannelDNS, GV_CASES
from UQ.datasets.heatflux_apriori import (HeatFluxAPriori,
                                          PooledGaussianDiagnostic,
                                          LOW_MACH, HIGH_MACH,
                                          MACH_FAMILY_EDGES, TARGETS,
                                          mach_family, thermal_score_weights,
                                          baseline_wall_flux)

# one low-Mach and one high-Mach case are enough for the machinery checks
CASE_LOW = GV_CASES[6]     # nominal Re_tau* 105, M_CLx 0.32
CASE_HIGH = GV_CASES[21]   # nominal Re_tau* 342, M_CLx 1.51

data_present = GVChannelDNS.is_available(CASE_LOW) \
    and GVChannelDNS.is_available(CASE_HIGH)


# ---- split constants (pure logic, no data) -----------------------------------

def test_cross_mach_split_is_the_committed_one():
    assert len(LOW_MACH) == 8
    assert len(HIGH_MACH) == 16
    assert set(LOW_MACH).isdisjoint(HIGH_MACH)
    assert set(LOW_MACH) | set(HIGH_MACH) == set(GV_CASES)
    assert all(GVChannelDNS.parse_tag(c)[1] < 1.0 for c in LOW_MACH)
    assert all(GVChannelDNS.parse_tag(c)[1] >= 1.47 for c in HIGH_MACH)


def test_mach_families_partition_the_matrix():
    fams = [mach_family(c) for c in GV_CASES]
    counts = [fams.count(i) for i in range(len(MACH_FAMILY_EDGES))]
    # the pinned family sizes of the pre-registration
    assert counts == [3, 5, 6, 8, 2]
    # every case falls in exactly one family (edges are half-open)
    for c in GV_CASES:
        m = GVChannelDNS.parse_tag(c)[1]
        hits = [lo <= m < hi for lo, hi in MACH_FAMILY_EDGES]
        assert sum(hits) == 1


def test_target_definitions():
    # wall-normal primary; joint (x, y) family leg; spanwise never modeled
    assert TARGETS["wall_normal"] == (1,)
    assert TARGETS["joint_xy"] == (0, 1)
    assert all(2 not in cols for cols in TARGETS.values())


def test_interior_mask_logic():
    class Stub:
        yplus = np.array([1.0, 31.0, 50.0, 200.0])
        y_outer = np.array([0.01, 0.3, 0.5, 0.95])
    mask = HeatFluxAPriori.interior_mask(Stub())
    # y+ > 30 AND y/delta < 0.9: only the two middle stations survive
    assert mask.tolist() == [False, True, True, False]


# ---- pinned conformal score weights (stubs, no data) --------------------------

class _StubCal:
    """Just enough of CompressibleCalibration for the weight definitions."""
    def __init__(self, rho, q_idx, n_thermal):
        class _D:
            pass
        self.dns = _D()
        self.dns.rho = np.asarray(rho)
        self.q_idx = np.asarray(q_idx)
        self.heldout_index = np.arange(33, 33 + n_thermal)


def test_thermal_score_weights_definitions():
    rho = np.linspace(1.0, 0.25, 12)
    cal = _StubCal(rho, q_idx=[2, 5, 9], n_thermal=4)

    w_abs = thermal_score_weights(cal, "absolute")
    assert np.allclose(w_abs, 1.0) and w_abs.size == 4

    w_bq = thermal_score_weights(cal, "bq_base", b_q_base=-0.05)
    assert np.allclose(w_bq, 1.0 / 0.05) and w_bq.size == 4
    # the quiescent-limit floor: |B_q| below 1e-4 never divides by less
    w_floor = thermal_score_weights(cal, "bq_base", b_q_base=-1e-6)
    assert np.allclose(w_floor, 1.0 / 1e-4)

    w_sl = thermal_score_weights(cal, "semilocal")
    # wall-flux entry at the wall (rho+ = 1), then sqrt(rho+) per q station
    assert w_sl.size == 4
    assert w_sl[0] == 1.0
    assert np.allclose(w_sl[1:], np.sqrt(rho[[2, 5, 9]]))

    with pytest.raises(ValueError):
        thermal_score_weights(cal, "no_such_score")


def test_weighted_conformal_coverage_math():
    # the weighted score s = w |resid| with interval m +/- q / w: coverage
    # is exact on a fabricated set where the quantile is known
    from UQ import conformal as cf
    scores = np.arange(1.0, 100.0)     # 99 calibration scores
    q = cf.conformal_quantile(scores, alpha=0.1)
    # level = ceil(100 * 0.9)/99; np.quantile(method="higher") lands on the
    # next score above level * (n - 1) = 89.09, which is scores[90] = 91
    assert q == 91.0
    resid = np.array([10.0, 89.9, 91.0, 95.0])
    w = np.ones(4)
    cov = float(np.mean(resid * w <= q))
    assert cov == 0.75
    # halving the weights doubles the effective interval in raw units
    cov_wide = float(np.mean(resid * (w / 2.0) <= q))
    assert cov_wide == 1.0


# ---- the pooled diagnostic (pure numpy) ---------------------------------------

def test_pooled_gaussian_diagnostic():
    rng = np.random.default_rng(3)
    Y = rng.normal(loc=[1.0, -2.0], scale=[0.5, 2.0], size=(4000, 2))
    X = rng.normal(size=(4000, 6))
    model = PooledGaussianDiagnostic(seed=0).fit(X, Y, epochs=1)
    assert np.allclose(model.mu, Y.mean(0))
    assert np.allclose(model.sd, Y.std(0) + 1e-12)
    s = model.sample(X[:100], n_per=64)
    assert s.shape == (100, 64, 2)
    # features do not condition the pooled law; same seed, same draws
    s2 = model.sample(X[100:200], n_per=64)
    assert np.allclose(s, s2)
    # an explicit draw seed changes the draws deterministically
    s3 = model.sample(X[:100], n_per=64, seed=7)
    s4 = model.sample(X[:100], n_per=64, seed=7)
    assert not np.allclose(s, s3)
    assert np.allclose(s3, s4)


# ---- assembly on real data ----------------------------------------------------

@pytest.mark.skipif(not data_present,
                    reason="GV DNS_data not present (bulk is local/gitignored)")
class TestAssembly:

    @pytest.fixture(scope="class")
    def study(self):
        return HeatFluxAPriori.build(gv_cases=(CASE_LOW, CASE_HIGH),
                                     ckm_cases=(), zdc_cases=())

    def test_records_are_wellformed(self, study):
        assert list(study.gv) == [CASE_LOW, CASE_HIGH]
        assert not study.skipped
        for tag in study.gv:
            rec = study.cases[tag]
            assert rec["features"].shape == (rec["n"], 6)
            assert rec["dq"].shape == (rec["n"], 3)
            assert rec["n"] > 10
            assert np.all(np.isfinite(rec["features"]))
            assert np.all(np.isfinite(rec["dq"]))
            # DNS-stress realizability holds on the attached data (the
            # standing separate check, carried through the extraction)
            assert rec["realizable_fraction"] == 1.0
            # the baseline wall flux is a cooled-wall prediction (negative)
            assert rec["b_q_base"] < 0.0
            assert rec["m_t_max"] > 0.0

    def test_mach_axis_is_monotone_in_m_t(self, study):
        # the conditioning premise: the higher-Mach case reaches higher M_t
        assert study.cases[CASE_HIGH]["m_t_max"] \
            > study.cases[CASE_LOW]["m_t_max"]

    def test_rows_pool_and_split_no_leak(self, study):
        X, Y = study.rows(study.gv, "joint_xy")
        n_total = sum(study.cases[t]["n"] for t in study.gv)
        assert X.shape == (n_total, 6)
        assert Y.shape == (n_total, 2)
        # wall-normal rows are column 1 of the joint rows
        Xw, Yw = study.rows(study.gv, "wall_normal")
        assert np.allclose(Yw[:, 0], Y[:, 1])
        assert np.allclose(Xw, X)

    def test_baseline_wall_flux_matches_assembly(self, study):
        # the standalone scale used by the conformal leg is the same solve
        # the assembly recorded (fixed nominal coefficients)
        b = baseline_wall_flux(GVChannelDNS.load(CASE_LOW))
        assert b is not None
        assert np.isclose(b, study.cases[CASE_LOW]["b_q_base"],
                          rtol=1e-10, atol=0.0)


# ---- both model kinds through the identical path (torch) ----------------------

@pytest.mark.skipif(not data_present,
                    reason="GV DNS_data not present (bulk is local/gitignored)")
def test_fit_and_evaluate_smoke():
    torch = pytest.importorskip("torch")
    study = HeatFluxAPriori.build(gv_cases=(CASE_LOW, CASE_HIGH),
                                  ckm_cases=(), zdc_cases=())
    for target, d in (("wall_normal", 1), ("joint_xy", 2)):
        for kind in ("flow", "gauss", "pooled"):
            model = study.fit(kind, (CASE_LOW,), target, seed=0, epochs=1)
            out = study.evaluate(model, CASE_HIGH, target, n_samples=16,
                                 sample_seed=11)
            for lv in (0.9, 0.5):
                cov = out[f"coverage_y_{lv:g}"]
                assert 0.0 <= cov <= 1.0
            assert out["crps_y"] >= 0.0
            assert out["arrays"]["pit_y"].shape == \
                (study.cases[CASE_HIGH]["n"],)
            assert np.all((out["arrays"]["pit_y"] >= 0.0)
                          & (out["arrays"]["pit_y"] <= 1.0))
            if d == 2:
                assert out["energy_score"] >= 0.0
                assert out["crps_x"] >= 0.0
            # the pinned draw seed makes the score call-order independent
            out2 = study.evaluate(model, CASE_HIGH, target, n_samples=16,
                                  sample_seed=11)
            assert out2["crps_y"] == out["crps_y"]
            # the default (sample_seed=None) derives a stable seed from the
            # evaluation identity, so scores stay call-order independent even
            # when the caller omits the seed: two calls agree, and an
            # intervening draw on the shared global RNG does not perturb them
            d1 = study.evaluate(model, CASE_HIGH, target, n_samples=16)
            model.sample(study.cases[CASE_LOW]["features"], n_per=8)
            d2 = study.evaluate(model, CASE_HIGH, target, n_samples=16)
            assert d1["crps_y"] == d2["crps_y"]
