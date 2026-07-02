"""Streamwise-periodic curved-channel mesh through the Python binding.

The C++ test carries the numerics verification (exact geometry, checkerboard
guard, momentum balance); this exercises the binding surface the periodic-hills
baseline will drive: the mesh factory from numpy arrays, the wall patches, the
body-force setting, and a short body-force-driven coupled solve.
"""
import numpy as np
import pytest

pytest.importorskip("rans_sst_py", reason="C++ binding not built")
import rans_sst_py as rs


def test_periodic_curved_mesh_solves_with_body_force():
    nx, ny, Lx, yTop = 30, 20, 4.5, 3.0
    xN = np.linspace(0.0, Lx, nx + 1)
    yB = 0.5 * (1.0 + np.cos(2.0 * np.pi * xN / Lx))   # periodic bump, h = 1
    mesh = rs.Mesh.make_curved_channel_periodic_2d(xN, yB, yTop, ny,
                                                   Re=2800.0, yPlusTarget=1.0)
    mesh.compute_wall_distance()

    assert set(mesh.patch_names()) == {"bottom_wall", "top_wall"}
    assert set(mesh.patch_types()) == {"wall"}
    assert mesh.n_cells() == nx * ny

    kIn, omIn, nu = 1e-4, 10.0, 3.0e-4
    bcs = rs.FlowBoundaryConditions.channel_defaults(mesh, 1.0, kIn, omIn)
    s = rs.SolverSettings()
    s.max_iterations = 15000
    s.convergence_tol = 1.0e-4
    s.alpha_u, s.alpha_p = 0.4, 0.25
    s.verbose = False
    s.body_force = rs.Vec3(4.0e-3, 0.0, 0.0)

    obs = rs.ObservationOperator()
    obs.add_reattachment_length("bottom_wall", xr_obs=2.0, sigma=1.0)
    ps = rs.InferenceParameterSet.a1_betaStar()
    fm = rs.ForwardModel(mesh, ps, obs, bcs, nu, s,
                         rs.Vec3(0.3, 0.0, 0.0), 0.0, kIn, omIn)
    r = fm.evaluate(list(ps.pack(rs.SSTCoefficients())))
    status = str(r.status).split(".")[-1]
    assert status in ("Converged", "Unconverged")
    assert fm.has_last_fields()

    ff = fm.last_fields()
    U = np.asarray(ff["U"])
    if U.ndim == 1:
        U = U.reshape(-1, 3)
    vols = np.asarray(mesh.cell_volumes())
    u_bulk = float(np.sum(U[:, 0] * vols) / np.sum(vols))
    assert np.all(np.isfinite(U))
    assert u_bulk > 0.05          # the body force established a bulk flow
