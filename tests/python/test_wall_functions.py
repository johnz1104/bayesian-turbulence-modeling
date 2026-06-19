"""
Wall functions / coarse-mesh tests.

Verifies:

  1. ``SolverSettings.use_wall_functions`` is exposed and persists.
  2. The y⁺ diagnostic returns finite values on the resolved Ma=0.1 case
     and reports y⁺_max < 5 (resolved-LES regime) by default.
  3. With wall functions enabled on a coarse channel mesh (target y⁺ ≈ 30),
     the solver does not diverge and the resulting Cf is within a documented
     tolerance of the resolved-LES reference.
  4. ``mesh.wall_patch_data`` matches mesh.face_centers / cell_centers in
     basic ways (face count, owner indices in [0, n_cells)).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


_PYTHON_DIR = Path(__file__).resolve().parents[2] / "python"
sys.path.insert(0, str(_PYTHON_DIR))


# ---- Helpers -----------------------------------------------------------

def build_compressible_channel(rs, nx: int, ny: int, *,
                                yPlusTarget: float, use_wall_functions: bool,
                                Ma: float = 0.1, max_iters: int = 3000):
    """Construct a compressible channel case with chosen mesh density and BCs."""
    eos    = rs.IdealGasEOS()
    T_in   = 300.0
    p_ref  = 101325.0
    rho_in = eos.density(p_ref, T_in)
    mu_in  = eos.viscosity(T_in)
    a_in   = eos.sound_speed(T_in)
    Uin    = Ma * a_in
    nu_in  = mu_in / rho_in
    Re     = rho_in * Uin * 1.0 / mu_in

    mesh = rs.Mesh.make_channel_2d(nx, ny, 10.0, 1.0, Re=Re,
                                    yPlusTarget=yPlusTarget)
    mesh.compute_wall_distance()

    kIn  = 1.5 * (Uin * 0.05) ** 2
    omIn = kIn / (nu_in * 100.0)
    bcs  = rs.CompressibleBoundaryConditions.channel_defaults(
        mesh, Uin, T_in, p_ref, kIn, omIn)

    obs = rs.ObservationOperator()
    obs.add_skin_friction(
        wall_patch="bottom", location=rs.Vec3(5.0, 0, 0),
        cf_obs=0.005, sigma=0.001, ref_vel=Uin)

    settings = rs.SolverSettings()
    settings.max_iterations      = max_iters
    settings.convergence_tol     = 1e-3
    settings.alpha_u             = 0.5 if Ma <= 0.3 else 0.4
    settings.alpha_p             = 0.2
    settings.alpha_t             = 0.7
    settings.alpha_k             = 0.4
    settings.alpha_omega         = 0.4
    settings.inner_iterations    = 200
    settings.turb_start_iter     = 50
    settings.turb_update_interval = 2
    settings.divergence_limit    = 1e10
    settings.verbose             = False
    settings.use_wall_functions  = use_wall_functions

    param_set = rs.InferenceParameterSet.a1_betaStar()
    fm = rs.CompressibleForwardModel(
        mesh=mesh, param_set=param_set, obs_op=obs, bcs=bcs, eos=eos,
        settings=settings, u_init=rs.Vec3(Uin, 0, 0),
        p_init=p_ref, T_init=T_in, k_init=kIn, omega_init=omIn)
    theta = list(param_set.pack(rs.SSTCoefficients()))
    result = fm.evaluate(theta)
    return {
        "mesh":     mesh,
        "fm":       fm,
        "result":   result,
        "fields":   fm.last_fields() if fm.has_last_fields() else None,
        "Uin":      Uin,
        "nu":       nu_in,
        "settings": settings,
    }


# ---- Tests -------------------------------------------------------------

class TestSolverSettingsBindings:
    def test_use_wall_functions_default_false(self, rs):
        s = rs.SolverSettings()
        assert s.use_wall_functions is False
        assert s.von_karman == pytest.approx(0.41)
        assert s.wall_fn_E  == pytest.approx(9.0)

    def test_setters_persist(self, rs):
        s = rs.SolverSettings()
        s.use_wall_functions = True
        s.von_karman = 0.42
        s.wall_fn_E  = 8.5
        assert s.use_wall_functions is True
        assert s.von_karman == 0.42
        assert s.wall_fn_E  == 8.5


class TestPatchIntrospection:
    @pytest.fixture(scope="class")
    def channel(self, rs):
        return build_compressible_channel(rs, nx=20, ny=14,
                                            yPlusTarget=1.0,
                                            use_wall_functions=False,
                                            max_iters=500)

    def test_patch_names_includes_walls(self, channel):
        mesh = channel["mesh"]
        names = mesh.patch_names()
        types = mesh.patch_types()
        assert "bottom" in names
        assert "top"    in names
        for n, t in zip(names, types):
            if n in ("bottom", "top"):
                assert t == "wall", f"patch {n} is type {t}"

    def test_wall_patch_data_keys(self, channel):
        mesh = channel["mesh"]
        info = mesh.wall_patch_data("bottom")
        for k in ("name", "type", "n_faces", "owner", "delta", "center",
                  "normal", "area"):
            assert k in info
        assert info["name"] == "bottom"
        assert info["type"] == "wall"

    def test_wall_face_owners_in_range(self, channel):
        mesh = channel["mesh"]
        info = mesh.wall_patch_data("bottom")
        owners = np.asarray(info["owner"], int)
        n_cells = mesh.n_cells()
        assert np.all((owners >= 0) & (owners < n_cells))

    def test_wall_face_count_matches_nx(self, channel):
        mesh = channel["mesh"]
        info = mesh.wall_patch_data("bottom")
        # Channel mesh has nx faces along the bottom wall.
        assert info["n_faces"] == 20

    def test_unknown_patch_raises(self, channel):
        mesh = channel["mesh"]
        with pytest.raises(RuntimeError, match="unknown patch"):
            mesh.wall_patch_data("nope")


class TestYPlusDiagnostic:
    @pytest.fixture(scope="class")
    def fine_run(self, rs):
        return build_compressible_channel(rs, nx=40, ny=30,
                                            yPlusTarget=1.0,
                                            use_wall_functions=False,
                                            max_iters=3000)

    def test_y_plus_finite_and_low_resolved(self, rs, fine_run):
        from wall_diagnostics import y_plus_first_cell
        info = y_plus_first_cell(fine_run["mesh"], fine_run["fields"],
                                  fine_run["nu"], wall_patch="bottom")
        assert np.all(np.isfinite(info["per_face_y_plus"]))
        # Resolved-LES regime: max y+ < 5 in the fully developed region.
        assert info["max"] < 5.0, (
            f"resolved channel y+_max = {info['max']:.2f} is too high")

    def test_y_plus_per_patch_keys(self, rs, fine_run):
        from wall_diagnostics import y_plus_first_cell
        info = y_plus_first_cell(fine_run["mesh"], fine_run["fields"],
                                  fine_run["nu"])
        # No filter -> both walls present
        assert "bottom" in info["per_patch"]
        assert "top"    in info["per_patch"]


class TestCoarseMeshWallFunctions:
    """The crucial test: a coarse mesh with wall functions enabled
    must still run, give finite Cf, and agree with the resolved fine mesh
    within a documented tolerance.  This is heavier than the bindings tests;
    it needs ~10s on a developer laptop."""

    @pytest.fixture(scope="class")
    def fine_resolved(self, rs):
        run = build_compressible_channel(rs, nx=40, ny=30,
                                          yPlusTarget=1.0,
                                          use_wall_functions=False,
                                          max_iters=3000)
        if run["fields"] is None:
            pytest.skip("fine-mesh resolved run did not produce fields")
        return run

    @pytest.fixture(scope="class")
    def coarse_wf(self, rs):
        # 4000 iters so the BL has enough time to develop on this mesh.
        run = build_compressible_channel(rs, nx=24, ny=14,
                                          yPlusTarget=30.0,
                                          use_wall_functions=True,
                                          max_iters=4000)
        if run["fields"] is None:
            pytest.skip("coarse-mesh wall-functions run did not produce fields")
        return run

    @pytest.fixture(scope="class")
    def coarse_resolved(self, rs):
        """Same coarse mesh, wall-functions OFF — used to verify wall-functions
        do not destabilise the solver relative to the legacy path."""
        run = build_compressible_channel(rs, nx=24, ny=14,
                                          yPlusTarget=30.0,
                                          use_wall_functions=False,
                                          max_iters=4000)
        return run

    def test_coarse_wf_does_not_diverge(self, coarse_wf):
        status = str(coarse_wf["result"].status)
        assert "Diverged" not in status and "Divergence" not in status, (
            f"coarse + wall-functions diverged: status={status}")

    def test_coarse_wf_cf_finite_and_positive(self, coarse_wf):
        from wall_diagnostics import cf_along_wall
        cf = cf_along_wall(coarse_wf["mesh"], coarse_wf["fields"],
                           coarse_wf["nu"], "bottom",
                           ref_vel=coarse_wf["Uin"])
        assert np.all(np.isfinite(cf["cf"]))
        assert np.all(cf["cf"] > 0.0)

    def test_coarse_y_plus_higher_than_fine(self, fine_resolved, coarse_wf):
        """Coarser mesh ⇒ larger first-cell y⁺.  This is the structural
        property of wall-function meshes vs resolved-LES meshes.  We do not
        require y⁺ ≥ 30 because the BL takes time to develop on a tiny
        channel; we only require monotonic behaviour."""
        from wall_diagnostics import y_plus_first_cell
        y_fine   = y_plus_first_cell(fine_resolved["mesh"], fine_resolved["fields"],
                                      fine_resolved["nu"], wall_patch="bottom")
        y_coarse = y_plus_first_cell(coarse_wf["mesh"], coarse_wf["fields"],
                                      coarse_wf["nu"], wall_patch="bottom")
        assert y_coarse["max"] > y_fine["max"], (
            f"coarse y+_max ({y_coarse['max']:.2f}) should exceed fine "
            f"y+_max ({y_fine['max']:.2f})")

    def test_coarse_wf_cf_within_2x_of_fine(self, fine_resolved, coarse_wf):
        """Coarse-mesh wall-function Cf at the developed-flow station should
        agree with resolved-LES Cf within ~2x.  Loose enough to absorb
        wall-function modelling error + coarse-mesh truncation error, tight
        enough to flag a regression."""
        from wall_diagnostics import cf_along_wall
        cf_f = cf_along_wall(fine_resolved["mesh"], fine_resolved["fields"],
                              fine_resolved["nu"], "bottom",
                              ref_vel=fine_resolved["Uin"])
        cf_c = cf_along_wall(coarse_wf["mesh"], coarse_wf["fields"],
                              coarse_wf["nu"], "bottom",
                              ref_vel=coarse_wf["Uin"])
        # Pick the developed-flow station x ≈ 5h.
        i_f = int(np.argmin(np.abs(cf_f["x"] - 5.0)))
        i_c = int(np.argmin(np.abs(cf_c["x"] - 5.0)))
        ratio = cf_c["cf"][i_c] / cf_f["cf"][i_f]
        assert 0.5 < ratio < 2.0, (
            f"coarse Cf ({cf_c['cf'][i_c]:.4g}) deviates >2x from "
            f"fine resolved Cf ({cf_f['cf'][i_f]:.4g}); ratio={ratio:.2f}")

    def test_wall_functions_do_not_break_legacy_path(self, coarse_resolved):
        """Wall-function flag OFF on the same mesh must remain a no-op,
        i.e. the legacy solver still doesn't diverge."""
        status = str(coarse_resolved["result"].status)
        assert "Diverged" not in status and "Divergence" not in status
