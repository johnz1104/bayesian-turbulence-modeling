"""Python-binding tests for the density-based model-form injection surface.

The a-posteriori coupled leg drives the solver from Python: set a sampled
closure correction (target anisotropy plus heat-flux correction), solve, read
the diagnostics. These tests exercise exactly that surface on a small laminar
case: value-copy semantics, shape validation, the zero-correction identity,
the diagnostics dictionary, and the per-face boundary profile that the
interaction baseline's measured-inflow configuration uses.
"""
import numpy as np
import pytest

rans = pytest.importorskip("rans_sst_py")

pytestmark = pytest.mark.skipif(
    not hasattr(rans, "DBNSSolver"), reason="dbns bindings not built")


def _laminar_channel(nx=20, ny=10, implicit=False):
    eos = rans.IdealGasEOS()
    mesh = rans.Mesh.make_channel_2d(nx, ny, 1.0, 0.2)
    bcs = rans.DBNSBoundaryConditions()
    ext = rans.DBNSBoundarySpec(); ext.kind = rans.DBNSBoundaryKind.Extrapolate
    slip = rans.DBNSBoundarySpec(); slip.kind = rans.DBNSBoundaryKind.SlipWall
    bcs.set("inlet", ext); bcs.set("outlet", ext)
    bcs.set("top", slip); bcs.set("bottom", slip)
    st = rans.DBNSSettings()
    st.viscous = False
    st.turbulent = False
    st.max_iterations = 50
    if implicit:
        st.implicit_steady = True
        st.cfl_implicit = 10.0
        st.cfl_ramp_start = 1.0
        st.cfl_ramp_iters = 10
    sst = rans.SSTCoefficients()
    solver = rans.DBNSSolver(mesh, eos, sst, bcs, st)
    solver.init_uniform(rans.Primitive(1.0, 100.0, 0.0, 1.0e5))
    return mesh, solver


def test_set_target_correction_shapes():
    mesh, solver = _laminar_channel()
    n = mesh.n_cells()
    b = np.zeros((n, 3, 3))
    dq = np.zeros((n, 2))
    solver.set_target_correction(b, dq)
    d = solver.injection_diagnostics()
    assert d["active"] is True
    assert d["all_realizable"] is True
    assert d["max_db"] == 0.0 and d["max_dq"] == 0.0

    with pytest.raises(RuntimeError):
        solver.set_target_correction(np.zeros((n, 2, 3)), dq)
    with pytest.raises(RuntimeError):
        solver.set_target_correction(b, np.zeros((n, 3)))

    solver.clear_target_correction()
    assert solver.injection_diagnostics()["active"] is False


def test_zero_correction_identity_smoke():
    mesh, base = _laminar_channel()
    rep_b = base.solve()
    mesh2, inj = _laminar_channel()
    n = mesh2.n_cells()
    inj.set_target_correction(np.zeros((n, 3, 3)), np.zeros((n, 2)))
    rep_i = inj.solve()
    fb, fi = base.fields(), inj.fields()
    for key in ("rho", "u", "p"):
        assert np.allclose(fb[key], fi[key], rtol=0, atol=0), key
    d = inj.injection_diagnostics()
    assert d["checked_iters"] > 0


def test_diagnostics_record_violation():
    mesh, solver = _laminar_channel()
    n = mesh.n_cells()
    b = np.zeros((n, 3, 3))
    b[:, 0, 0] = 0.9          # beyond the one-component corner (2/3)
    b[:, 1, 1] = -0.45
    b[:, 2, 2] = -0.45
    solver.set_target_correction(b, np.zeros((0,)))
    solver.solve()
    d = solver.injection_diagnostics()
    assert d["all_realizable"] is False
    assert d["max_violation"] > 0.0
    assert d["max_db"] == pytest.approx(0.9)


def test_boundary_profile_constant_matches_uniform():
    """A per-face profile equal to the uniform freestream must reproduce the
    uniform-inflow solve exactly (the profile plumbing adds nothing when it
    should add nothing)."""
    eos = rans.IdealGasEOS()

    def build(with_profile):
        mesh = rans.Mesh.make_channel_2d(30, 8, 1.0, 0.2)
        bcs = rans.DBNSBoundaryConditions()
        inflow = rans.DBNSBoundarySpec()
        inflow.kind = rans.DBNSBoundaryKind.SupersonicInflow
        inflow.freestream = rans.Primitive(1.0, 700.0, 0.0, 1.0e5)
        if with_profile:
            prof = np.tile([1.0, 700.0, 0.0, 1.0e5, 0.0, 0.0], (8, 1))
            inflow.set_profile(prof)
        ext = rans.DBNSBoundarySpec()
        ext.kind = rans.DBNSBoundaryKind.Extrapolate
        slip = rans.DBNSBoundarySpec()
        slip.kind = rans.DBNSBoundaryKind.SlipWall
        bcs.set("inlet", inflow); bcs.set("outlet", ext)
        bcs.set("top", slip); bcs.set("bottom", slip)
        st = rans.DBNSSettings()
        st.viscous = False; st.turbulent = False
        st.max_iterations = 60
        sst = rans.SSTCoefficients()
        solver = rans.DBNSSolver(mesh, eos, sst, bcs, st)
        solver.init_uniform(rans.Primitive(1.0, 700.0, 0.0, 1.0e5))
        solver.solve()
        return solver.fields()

    fu = build(False)
    fp = build(True)
    for key in ("rho", "u", "p"):
        assert np.allclose(fu[key], fp[key], rtol=0, atol=0), key


def test_boundary_profile_shape_validation():
    spec = rans.DBNSBoundarySpec()
    with pytest.raises(RuntimeError):
        spec.set_profile(np.zeros((4, 5)))
    spec.set_profile(np.zeros((4, 6)))
    spec.clear_profile()


def test_wall_temperature_profile_constant_matches_uniform():
    """A per-face wall-temperature profile equal to the uniform value must
    reproduce the uniform-wall solve exactly, and the observation operator's
    profile variant must agree with its uniform form."""
    eos = rans.IdealGasEOS()

    def build(with_profile):
        mesh = rans.Mesh.make_channel_2d(16, 10, 0.02, 0.01)
        bcs = rans.DBNSBoundaryConditions()
        wallT = 320.0
        wall = rans.DBNSBoundarySpec()
        wall.kind = rans.DBNSBoundaryKind.NoSlipIsothermal
        wall.wall_temp = wallT
        if with_profile:
            wall.set_wall_temp_profile(np.full(16, wallT))
        slip = rans.DBNSBoundarySpec()
        slip.kind = rans.DBNSBoundaryKind.SlipWall
        ext = rans.DBNSBoundarySpec()
        ext.kind = rans.DBNSBoundaryKind.Extrapolate
        anchor = rans.DBNSBoundarySpec()
        anchor.kind = rans.DBNSBoundaryKind.SubsonicOutflow
        anchor.back_pressure = 1.0e5
        bcs.set("bottom", wall); bcs.set("top", slip)
        bcs.set("inlet", ext); bcs.set("outlet", anchor)
        st = rans.DBNSSettings()
        st.viscous = True
        st.turbulent = False
        st.const_mu = 3.0e-4
        st.implicit_steady = True
        st.cfl_implicit = 100.0
        st.cfl_ramp_start = 1.0
        st.cfl_ramp_iters = 20
        st.max_iterations = 400
        st.convergence_tol = 1e-30
        sst = rans.SSTCoefficients()
        solver = rans.DBNSSolver(mesh, eos, sst, bcs, st)
        solver.init_uniform(rans.Primitive(1.16, 50.0, 0.0, 1.0e5))
        solver.solve()
        ref = rans.ReferenceState()
        ref.rho = 1.16; ref.U = 50.0; ref.T = 300.0; ref.p = 1.0e5
        obs = rans.DBNSObservation(solver, ref)
        uniform = obs.wall("bottom", wallT)
        profile = obs.wall_profile("bottom", np.full(16, wallT), wallT)
        return solver.fields(), uniform, profile

    fu, obs_u_uni, obs_u_prof = build(False)
    fp, _, _ = build(True)
    for key in ("rho", "u", "p", "T"):
        assert np.allclose(fu[key], fp[key], rtol=0, atol=0), key
    for key in ("Cf", "qw", "St"):
        assert np.allclose(obs_u_uni[key], obs_u_prof[key], rtol=0, atol=0), key
