"""A-priori model-form checkpoint machinery on the real BFS discrepancy.

Runs a real (coarse, fast) baseline solve and short model fits, so it skips
where torch, the binding, or the gitignored DNS_data is absent. Asserts the
MACHINERY is sound (round-trip component maps, realizable predictive samples,
well-formed scores through the standard harness, both model kinds through the
identical path); the checkpoint NUMBERS (full-epoch coverage and scores per
held-out station) belong to the evidence package, not to tests, and no quality
threshold is asserted here.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rans_sst_py", reason="C++ binding not built")
from UQ.datasets.backward_facing_step import BackwardFacingStepDNS
from UQ.datasets.bfs_baseline import BFSBaselineRANS
from UQ.datasets.separated_apriori import (SeparatedAPriori,
                                           anisotropy_to_components,
                                           COMPONENT_NAMES)
from UQ.generative import GenerativeDiscrepancyModel

pytestmark = pytest.mark.skipif(
    not BackwardFacingStepDNS.is_available(),
    reason="BFS DNS_data not present (bulk data is local/gitignored)",
)

FAST_CFG = {"nx_up": 20, "nx_down": 26, "ny_up": 14, "ny_down": 10,
            "Lu": 10.0, "Ld": 22.0, "max_iter": 8000, "conv_tol": 1.0e-4}


def test_component_roundtrip():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 3, 3))
    b = 0.5 * (a + np.swapaxes(a, 1, 2))
    b -= np.trace(b, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3.0
    comp = anisotropy_to_components(b)
    back = GenerativeDiscrepancyModel.components_to_anisotropy(comp)
    assert np.allclose(back, b, atol=1e-12)


@pytest.fixture(scope="module")
def apriori():
    dns = BackwardFacingStepDNS.load()
    baseline = BFSBaselineRANS.solve(FAST_CFG, dns=dns)
    return SeparatedAPriori.build(dns=dns, baseline=baseline)


def test_training_set_is_wellformed(apriori):
    assert apriori.n > 500
    assert apriori.features.shape == (apriori.n, 5)
    assert apriori.db_comp.shape == (apriori.n, 5)
    assert np.all(np.isfinite(apriori.features))
    assert np.all(np.isfinite(apriori.db_comp))
    # b_DNS = b_baseline + db reproduces the record identity
    assert np.allclose(apriori.b_dns - apriori.b_base,
                       GenerativeDiscrepancyModel.components_to_anisotropy(
                           apriori.db_comp), atol=1e-10)


def test_station_split_partitions(apriori):
    tr, te = apriori.station_split(0)
    assert te.sum() > 0 and tr.sum() > 0
    assert int(te.sum() + tr.sum()) == apriori.n
    assert np.all(apriori.station[te] == 0)


@pytest.mark.parametrize("kind", ["flow", "gauss"])
def test_fit_evaluate_path_is_wellformed(apriori, kind):
    # short fit: the machinery check, not the checkpoint numbers
    tr, te = apriori.station_split(2)
    model = apriori.fit(kind, tr, seed=0, epochs=60)
    out = apriori.evaluate(model, te, n_samples=48, level=0.9)
    assert out["n_test"] == int(te.sum())
    assert out["realizable_fraction"] == 1.0     # every projected draw realizable
    for name in COMPONENT_NAMES:
        assert 0.0 <= out["coverage"][name] <= 1.0
        assert out["sharpness"][name] > 0.0
        assert np.isfinite(out["crps"][name])
    assert np.isfinite(out["energy_score"]) and out["energy_score"] > 0.0
