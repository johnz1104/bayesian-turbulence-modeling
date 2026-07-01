"""Periodic hills DNS loader (Xiao et al. 2020) on the real raw files.

These tests run only where the gitignored DNS_data is present (locally); on CI
without the bulk data they skip. They exercise both on-disk formats (VTK for
alpha = 0.5, ASCII for alpha = 1.0) through the same canonical record, and check
the case parameters plus two data-only physics anchors for the dense field: the
DNS Reynolds stress is realizable across the fluid, and the interpolated mean
satisfies continuity (du/dx + dv/dy small relative to the strain rate). The
blanked solid interior is masked and the spatial velocity gradient is formed by
differencing on the tensor grid.
"""
import numpy as np
import pytest

from UQ.datasets.periodic_hills import PeriodicHillsDNS, PEHILL_CASES, RE_B

pytestmark = pytest.mark.skipif(
    not PeriodicHillsDNS.is_available("1p0"),
    reason="periodic-hills DNS_data not present (bulk data is local/gitignored)",
)


@pytest.fixture(scope="module")
def ascii_case():
    return PeriodicHillsDNS.load("1p0")        # ASCII .dat format


@pytest.fixture(scope="module")
def vtk_case():
    return PeriodicHillsDNS.load("0p5")        # VTK .vtr format


def test_ascii_case_parameters(ascii_case):
    d = ascii_case
    assert d.meta["case"] == "periodic_hills"
    assert d.meta["re_b"] == RE_B == 5600
    assert d.alpha == 1.0
    # tensor grid: N == nX * nY
    nY, nX = d.shape
    assert d.n == nX * nY


def test_vtk_case_parameters(vtk_case):
    d = vtk_case
    assert d.alpha == 0.5
    assert d.meta["re_b"] == RE_B
    nY, nX = d.shape
    assert d.n == nX * nY == 736 * 385


@pytest.mark.parametrize("fx", ["ascii_case", "vtk_case"])
def test_tensor_assembly_and_energy(fx, request):
    d = request.getfixturevalue(fx)
    # symmetric Reynolds stress; nonnegative energy on the fluid
    assert np.max(np.abs(d.R - np.transpose(d.R, (0, 2, 1)))) == 0.0
    assert np.all(d.k[d.fluid_mask] >= 0.0)


@pytest.mark.parametrize("fx", ["ascii_case", "vtk_case"])
def test_blanking_masks(fx, request):
    d = request.getfixturevalue(fx)
    # the blanked solid interior is a real fraction of the bounding grid, and the
    # interior (clean-stencil) mask is a subset of the fluid mask
    assert 0.0 < d.fluid_mask.mean() < 1.0
    assert np.all(d.interior_mask <= d.fluid_mask)
    assert d.interior_mask.sum() > 0


@pytest.mark.parametrize("fx", ["ascii_case", "vtk_case"])
def test_realizability_anchor(fx, request):
    d = request.getfixturevalue(fx)
    # data-only physics anchor: DNS stress realizable across the fluid
    assert d.realizable_fraction() == 1.0


@pytest.mark.parametrize("fx", ["ascii_case", "vtk_case"])
def test_continuity_anchor(fx, request):
    d = request.getfixturevalue(fx)
    # data-only physics anchor: the interpolated DNS mean is divergence-free, so
    # the RMS continuity residual is a small fraction of the RMS strain rate
    assert d.continuity_anchor() < 0.05


@pytest.mark.parametrize("fx", ["ascii_case", "vtk_case"])
def test_velocity_gradient_is_planar(fx, request):
    d = request.getfixturevalue(fx)
    g = d.velocity_gradient()
    assert g.shape == (d.n, 3, 3)
    # the mean is 2D and spanwise-homogeneous: no z-derivatives, no W-gradients
    assert np.all(g[:, 2, :] == 0.0)
    assert np.all(g[:, :, 2] == 0.0)


def test_reattachment_is_physical(ascii_case):
    # the standard periodic hill (alpha = 1) reattaches near x/h ~ 4.5 to 5
    x_r = ascii_case.bottom_wall_reattachment()
    assert x_r is not None
    assert 3.0 < x_r < 6.0


def test_dnsfield_wiring(ascii_case):
    # the canonical record drives the discrepancy/feature interface on interior pts
    field = ascii_case.to_dnsfield(timescale=np.ones(ascii_case.n))
    out = field.extract()
    n_int = int(ascii_case.interior_mask.sum())
    assert out["features"].shape == (n_int, 5)
    assert out["reynolds_discrepancy"].shape == (n_int, 3, 3)


def test_all_available_cases_load():
    # every compiled case parses and is realizable, across both formats
    for case in PEHILL_CASES:
        if not PeriodicHillsDNS.is_available(case):
            continue
        d = PeriodicHillsDNS.load(case)
        assert d.n == d.shape[0] * d.shape[1]
        assert d.realizable_fraction() == 1.0
