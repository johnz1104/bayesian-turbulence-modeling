"""A-posteriori ensemble machinery on the real BFS (coarse, tiny ensembles).

Runs real coupled solves, so it skips where torch, the binding, or the
gitignored DNS_data is absent. Asserts the MACHINERY: coherent shared-latent
members produce distinct, converging, realizable coupled solutions with a
genuine reattachment spread; the eigenspace corner family runs through the
same injection; and the scoring records are well-formed. The study NUMBERS
(24-member production ensembles) belong to the evidence package, and no
quality threshold is asserted here.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rans_sst_py", reason="C++ binding not built")
from UQ.datasets.backward_facing_step import BackwardFacingStepDNS
from UQ.datasets.bfs_baseline import BFSBaselineRANS
from UQ.datasets.separated_aposteriori import BFSAPosteriori

pytestmark = pytest.mark.skipif(
    not BackwardFacingStepDNS.is_available(),
    reason="BFS DNS_data not present (bulk data is local/gitignored)",
)

FAST_CFG = {"nx_up": 20, "nx_down": 26, "ny_up": 14, "ny_down": 10,
            "Lu": 10.0, "Ld": 22.0, "max_iter": 8000, "conv_tol": 1.0e-4}


@pytest.fixture(scope="module")
def study():
    dns = BackwardFacingStepDNS.load()
    baseline = BFSBaselineRANS.solve(FAST_CFG, dns=dns)
    return BFSAPosteriori.build(cfg=FAST_CFG, dns=dns, baseline=baseline)


def test_shared_latent_members_are_coherent_and_distinct(study):
    # Some sampled closures genuinely admit no steady solution under the
    # coupled iteration (separated flows are marginal; the methods note pins
    # the honest treatment: count and exclude). The machinery contract is that
    # every member reports status and realizability, converged members exist
    # and carry physical, distinct reattachments.
    model = study.train("flow", seed=0, epochs=60)
    torch.manual_seed(0)
    members = study.run_ensemble(model, n_members=3, shared_latent=True)
    assert all(m["status"] in ("Converged", "Unconverged") for m in members)
    assert all(m["all_realizable"] for m in members)
    xr_ok = [m["reattachment"] for m in members if m["status"] == "Converged"]
    assert len(xr_ok) >= 1
    assert np.all(np.isfinite([m["reattachment"] for m in members]))
    assert all(2.0 < v < 12.0 for v in xr_ok)
    # coherent draws are different closure hypotheses: the full member set
    # (converged or not) spans a genuine spread
    xr_all = [m["reattachment"] for m in members]
    assert max(xr_all) - min(xr_all) > 1e-3


def test_reattachment_scoring_record(study):
    model = study.train("gauss", seed=0, epochs=60)
    torch.manual_seed(0)
    members = study.run_ensemble(model, n_members=3, shared_latent=True)
    rec = BFSAPosteriori.score_reattachment(members, truth=6.28, level=0.9)
    assert rec["n_members"] == 3
    assert rec["n_nonconverged"] + len(
        [m for m in members if m["status"] == "Converged"]) == 3
    assert np.isfinite(rec["crps"])
    assert rec["band"][0] <= rec["band"][1]
    assert isinstance(rec["contains_truth"], bool)


def test_eigenspace_family_and_cf_scoring(study):
    corners = study.run_eigenspace(delta_b=0.5)
    env = BFSAPosteriori.score_envelope(corners, truth=6.28)
    assert set(env["corners"]) <= {"1C", "2C", "3C"}
    assert env["envelope"][0] <= env["envelope"][1]
    assert np.isfinite(env["crps_uniform_reading"])
    cf = study.score_cf(list(corners.values()), level=0.9)
    assert cf["n_used"] >= 2
    assert 0.0 <= cf["coverage"] <= 1.0
    assert np.isfinite(cf["crps"]) and np.isfinite(cf["energy_score"])
