"""Python-binding tests for the density-based shock-capturing solver (dbns).

These exercise the same path a future a-posteriori coupling would use: build a
mesh, configure boundary conditions and settings, run the solver, and read the
fields and the realizability projection back into numpy. No real DNS is needed;
the shock tube is checked against the analytic Riemann structure and the
realizability projection against its defining property.
"""
import os
import numpy as np
import pytest

rans = pytest.importorskip("rans_sst_py")


def _has_dbns():
    return hasattr(rans, "DBNSSolver")


pytestmark = pytest.mark.skipif(not _has_dbns(), reason="dbns bindings not built")


def test_sod_shock_tube_from_python():
    """Run the Sod problem through the bound solver and check the structure."""
    eos = rans.IdealGasEOS()
    eos.gamma = 1.4
    nx = 100
    mesh = rans.Mesh.make_channel_2d(nx, 1, 1.0, 0.02)

    bcs = rans.DBNSBoundaryConditions()
    ext = rans.DBNSBoundarySpec(); ext.kind = rans.DBNSBoundaryKind.Extrapolate
    slip = rans.DBNSBoundarySpec(); slip.kind = rans.DBNSBoundaryKind.SlipWall
    bcs.set("inlet", ext); bcs.set("outlet", ext)
    bcs.set("top", slip); bcs.set("bottom", slip)

    st = rans.DBNSSettings()
    st.time_mode = rans.TimeMode.Unsteady
    st.t_end = 0.2
    st.cfl = 0.4
    st.viscous = False
    st.turbulent = False
    st.reconstruct_order = 2

    sst = rans.SSTCoefficients()
    solver = rans.DBNSSolver(mesh, eos, sst, bcs, st)

    centers = mesh.cell_centers()
    init = np.zeros((mesh.n_cells(), 4))
    for i in range(mesh.n_cells()):
        x = centers[i, 0]
        if x < 0.5:
            init[i] = [1.0, 0.0, 0.0, 1.0]
        else:
            init[i] = [0.125, 0.0, 0.0, 0.1]
    solver.init_field(init)

    rep = solver.solve()
    assert rep.status == rans.EvaluationStatus.Converged
    f = solver.fields()
    rho = f["rho"]
    assert np.all(rho > 0.0) and np.all(f["p"] > 0.0)
    # left undisturbed state stays ~1.0, far-right undisturbed stays ~0.125
    assert abs(rho[2] - 1.0) < 0.02
    assert abs(rho[-2] - 0.125) < 0.02
    # contact/shock region density should sit between the two states
    assert 0.12 < rho[nx // 2] < 1.0


def test_realizability_projection_property():
    """An unrealizable Reynolds stress projects into the realizable set with 2k
    preserved; an isotropic stress is unchanged."""
    # isotropic stress (k = 1.5 -> 2k = 3, each normal = 1) is realizable
    p = rans.project_reynolds_stress(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert abs((p[0] + p[1] + p[2]) - 3.0) < 1e-9
    assert abs(p[0] - 1.0) < 1e-9 and abs(p[3]) < 1e-12

    # strongly one-component stress: almost all energy in xx
    xx, yy, zz = 2.9, 0.05, 0.05
    tr = xx + yy + zz
    q = rans.project_reynolds_stress(xx, yy, zz, 0.0, 0.0, 0.0)
    # trace (2k) preserved
    assert abs((q[0] + q[1] + q[2]) - tr) < 1e-7
    # projected anisotropy lies in the barycentric triangle: rebuild b and check
    twoK = q[0] + q[1] + q[2]
    b = np.array([q[0], q[1], q[2]]) / twoK - 1.0 / 3.0
    ev = np.sort(b)[::-1]
    c1 = ev[0] - ev[1]
    c2 = 2.0 * (ev[1] - ev[2])
    c3 = 3.0 * ev[2] + 1.0
    assert c1 > -1e-7 and c2 > -1e-7 and c3 > -1e-7
    assert abs(c1 + c2 + c3 - 1.0) < 1e-7


def test_fields_shapes():
    """fields() returns per-cell numpy arrays of consistent length."""
    eos = rans.IdealGasEOS()
    mesh = rans.Mesh.make_channel_2d(20, 10, 1.0, 0.2)
    bcs = rans.DBNSBoundaryConditions()
    ext = rans.DBNSBoundarySpec(); ext.kind = rans.DBNSBoundaryKind.Extrapolate
    for pch in ("inlet", "outlet", "top", "bottom"):
        bcs.set(pch, ext)
    st = rans.DBNSSettings()
    sst = rans.SSTCoefficients()
    solver = rans.DBNSSolver(mesh, eos, sst, bcs, st)
    solver.init_uniform(rans.Primitive(1.0, 100.0, 0.0, 1.0e5))
    solver.prepare_properties()
    f = solver.fields()
    n = mesh.n_cells()
    for key in ("rho", "u", "v", "p", "T", "mach", "muT"):
        assert f[key].shape == (n,)
    assert np.all(f["T"] > 0.0)
