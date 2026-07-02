"""Dense-field periodic-hills discrepancy and the generalized a-priori driver.

Runs a real (coarse, fast) baseline solve, so it skips where torch, the
binding, or the gitignored DNS_data is absent. Asserts the machinery of the
pinned hills protocol: the stride-subsampled interior point set, a traceless
and bounded discrepancy against the limiter-consistent baseline, realizable
baseline anisotropy, the band grouping through the shared a-priori driver, and
the cross-geometry scoring path. Study numbers belong to the evidence package.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rans_sst_py", reason="C++ binding not built")
from UQ.datasets.periodic_hills import PeriodicHillsDNS
from UQ.datasets.hills_baseline import HillsBaselineRANS
from UQ.datasets.hills_discrepancy import HillsDiscrepancy
from UQ.datasets.separated_apriori import SeparatedAPriori
from UQ import realizability as rz

pytestmark = pytest.mark.skipif(
    not PeriodicHillsDNS.is_available("1p0"),
    reason="periodic-hills DNS_data not present (bulk data is local/gitignored)",
)

FAST_CFG = {"nx": 48, "ny": 32, "max_iter": 15000, "conv_tol": 1.0e-4,
            "body_force": 0.0095}


@pytest.fixture(scope="module")
def disc():
    dns = PeriodicHillsDNS.load("1p0")
    baseline = HillsBaselineRANS.solve(FAST_CFG, dns=dns)
    return HillsDiscrepancy.build(dns=dns, baseline=baseline)


def test_discrepancy_is_wellformed(disc):
    assert disc.mask.sum() > 5000            # the dense field supplies bulk data
    feats, db = disc.training_pairs()
    assert feats.shape[1] == 5 and np.all(np.isfinite(feats))
    assert db.shape[1:] == (3, 3) and np.all(np.isfinite(db))
    tr = np.trace(db, axis1=1, axis2=2)
    assert np.max(np.abs(tr)) < 1e-9
    mag = disc.db_magnitude()[disc.mask]
    assert np.mean(mag) > 0.05               # a genuine model-form signal
    assert np.max(mag) < 2.5                 # bounded (no near-wall pathology)


def test_baseline_anisotropy_realizable(disc):
    R_base = 2.0 * (disc.b_baseline[disc.mask] + np.eye(3) / 3.0)
    assert np.mean(rz.is_realizable(R_base, tol=1e-6)) == 1.0


def test_reattachment_signal_present(disc):
    # SST massively over-predicts the hills reattachment; the signed error is
    # the model-form signal (magnitude reported by the study, not asserted)
    assert np.isfinite(disc.reattachment_error)
    assert abs(disc.reattachment_error) > 0.5


def test_band_grouping_through_shared_driver(disc):
    ap = SeparatedAPriori(disc)
    assert ap.n == int(disc.mask.sum())
    assert len(ap.station_xh) == HillsDiscrepancy.N_BANDS
    tr, te = ap.station_split(2)
    assert te.sum() > 0 and int(tr.sum() + te.sum()) == ap.n


def test_cross_geometry_scoring_path(disc):
    # the external-evaluation path is exercised against the same record: it
    # must agree with the internal all-points evaluation (identical inputs)
    ap = SeparatedAPriori(disc)
    model = ap.fit("gauss", np.ones(ap.n, dtype=bool), seed=0, epochs=40)
    torch.manual_seed(0)
    internal = ap.evaluate(model, np.ones(ap.n, dtype=bool), n_samples=24)
    torch.manual_seed(0)
    external = ap.evaluate_external(model, ap, n_samples=24)
    assert external["n_test"] == internal["n_test"]
    assert external["coverage"] == internal["coverage"]
    assert external["realizable_fraction"] == 1.0
