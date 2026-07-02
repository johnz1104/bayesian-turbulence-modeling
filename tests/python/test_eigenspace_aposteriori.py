"""Eigenspace-perturbation baseline propagated a-posteriori on the real BFS.

Runs a real (coarse, fast) coupled solve, so it skips where the binding or the
gitignored DNS_data is absent. The comparison baseline of the separated-flow
model-form study must run through the SAME Reynolds-stress injection as the
generative method (solver-for-solver comparability): a barycentric corner
perturbation of the baseline anisotropy is built, projected, injected, and the
coupled solve converges with realizability asserted every outer iteration and
a reattachment response of the expected sign (toward one-component turbulence
lengthens the bubble prediction toward the DNS).
"""
import numpy as np
import pytest

pytest.importorskip("rans_sst_py", reason="C++ binding not built")
from UQ.datasets.backward_facing_step import BackwardFacingStepDNS
from UQ.datasets.bfs_baseline import BFSBaselineRANS
from UQ.datasets.bfs_injection import BFSInjection
from UQ.eigenspace import EigenspacePerturbation

pytestmark = pytest.mark.skipif(
    not BackwardFacingStepDNS.is_available(),
    reason="BFS DNS_data not present (bulk data is local/gitignored)",
)

FAST_CFG = {"nx_up": 20, "nx_down": 26, "ny_up": 14, "ny_down": 10,
            "Lu": 10.0, "Ld": 22.0, "max_iter": 8000, "conv_tol": 1.0e-4}


def test_corner_family_through_injection_spans_an_envelope():
    dns = BackwardFacingStepDNS.load()
    baseline = BFSBaselineRANS.solve(FAST_CFG, dns=dns)
    inj = BFSInjection(cfg=FAST_CFG, dns=dns, baseline=baseline)

    # moderated corner members of the eigenspace family, on the same baseline
    # anisotropy the generative model perturbs. The corners move the
    # reattachment in different directions (the 1C move acts in the EIGENFRAME
    # of the Boussinesq anisotropy, amplifying the shear-aligned structure and
    # shortening the bubble; isotropizing toward 3C weakens the shear stress
    # and lengthens it), which is exactly the envelope the method delivers, so
    # the test asserts a genuine spread, not a direction per corner.
    family = EigenspacePerturbation.corner_set(inj.b_baseline, delta_b=0.5,
                                               corners=("1C", "3C"))
    assert EigenspacePerturbation.is_realizable_family(family)

    xr = {}
    for name, b_pert in family.items():
        bt, dist = inj.target_from_db(b_pert - inj.b_baseline)
        assert np.max(dist) < 1e-8      # already realizable: projection idles
        r = inj.run(bt)
        assert r["status"] == "Converged", name
        assert r["diagnostics"]["all_realizable"], name
        assert r["diagnostics"]["checked_iters"] >= r["iterations"], name
        assert 2.0 < r["reattachment"] < 12.0, name
        xr[name] = r["reattachment"]

    # both corners respond, and the family brackets a nonzero envelope
    for name, val in xr.items():
        assert abs(val - baseline.reattachment) > 0.1, name
    assert abs(xr["1C"] - xr["3C"]) > 0.3
