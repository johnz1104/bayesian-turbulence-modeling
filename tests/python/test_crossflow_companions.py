"""A-priori Step-2 companions: pipe (cross-geometry) and rotating channel.

Verifies the a-priori discrepancy diagnostics on the real companion data. The pipe
reproduces the channel/Couette discrepancy structure across geometry; the rotating
channel exposes the <uw> structural model-form failure (a linear eddy-viscosity
gives b_13 = 0 exactly here) and the diffuse-inflation cost of a global correction.
Skips where the gitignored DNS_data is absent.
"""
import numpy as np
import pytest

from UQ.datasets import PipeDNS, RotatingChannelDNS, PIPE_CASES, ROTATING_CASES
from UQ.datasets.crossflow_companions import pipe_discrepancy, rotating_diagnostic

pytestmark = pytest.mark.skipif(
    not RotatingChannelDNS.is_available(15),
    reason="Step-2 companion DNS_data not present (bulk data is local/gitignored)",
)


@pytest.mark.parametrize("n", [500, 2000])
def test_pipe_discrepancy_structure(n):
    """Cross-geometry: the pipe Boussinesq discrepancy is shear-dominated,
    normal-anisotropy-dominated, and realizable, like the channel and Couette."""
    if not PipeDNS.is_available(n):
        pytest.skip(f"pipe Re_tau={n} not present")
    d = pipe_discrepancy(PipeDNS.load(n))
    assert d["uv_dominates_offdiagonal"]
    assert d["normal_dominates_discrepancy"]
    assert d["dns_stress_realizable"]


def test_rotating_uw_is_structural_failure():
    """A linear eddy-viscosity gives b_13 = -C_mu S_13 = 0 exactly in this mean flow
    (S_13 = 0), yet the DNS <uw> anisotropy is nonzero: the entire <uw> stress is
    structural model-form error no Boussinesq closure can represent."""
    for ro in ROTATING_CASES:
        if not RotatingChannelDNS.is_available(ro):
            continue
        r = rotating_diagnostic(RotatingChannelDNS.load(ro))
        assert r["b13_boussinesq_max"] < 1e-12          # exactly zero by construction
        assert r["b13_dns_rms"] > 1e-2                  # DNS <uw> anisotropy is real
        assert r["dns_realizable"]


def test_rotating_global_correction_is_diffuse():
    """The discrepancy is uneven across components, so a single global band must
    over-inflate the median component to cover the worst (localisation cost > 1) and
    cannot equalise the per-component coverage (a nonzero spread). This is exactly
    the diffuse inflation the Step-3 feature-conditioned model is built to remove."""
    r = rotating_diagnostic(RotatingChannelDNS.load(15))
    assert r["localisation_cost"] > 1.5                 # global band over-inflates
    assert r["coverage_spread_pooled_band"] > 0.1       # cannot equalise coverage
    # the per-component bands genuinely differ (the closure fails unevenly)
    bands = list(r["band_per_component"].values())
    assert max(bands) > 1.5 * min(bands)
