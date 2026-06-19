"""
Smoke tests for the compressible Python bindings.

Validates that the new ``CompressibleForwardModel`` and ``IdealGasEOS``
bindings behave consistently with the C++ test suite (which already runs a
full Ma=0.1 channel solve under CTest).  These Python tests focus on:

  - Symbol availability (so ``from rans_sst_py import CompressibleForwardModel``
    keeps working).
  - EOS sanity (gamma=1.4, R=287, density round-trip).
  - End-to-end evaluate() on a tiny channel case: returns an EvaluationResult
    with the same shape as the incompressible ForwardModel.
  - last_fields() exposes density and temperature alongside the existing
    velocity/pressure/turbulence fields.

Each test runs a small mesh (12x10) and a short solve so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_compressible_symbols_present(rs):
    for name in (
        "ForwardModel",
        "CompressibleForwardModel",
        "IdealGasEOS",
        "CompressibleBoundaryConditions",
    ):
        assert hasattr(rs, name), f"rans_sst_py missing {name!r}"


class TestIdealGasEOS:
    def test_default_constants(self, rs):
        eos = rs.IdealGasEOS()
        assert eos.gamma == pytest.approx(1.4)
        assert eos.R == pytest.approx(287.0)
        assert eos.Cp() == pytest.approx(1004.5, rel=1e-3)
        assert eos.Cv() == pytest.approx(717.5, rel=1e-3)

    def test_density_round_trip(self, rs):
        eos = rs.IdealGasEOS()
        p, T = 101325.0, 300.0
        rho = eos.density(p, T)
        assert eos.temperature(p, rho) == pytest.approx(T, rel=1e-12)
        assert eos.pressure(rho, T)    == pytest.approx(p, rel=1e-12)

    def test_sound_speed_air(self, rs):
        eos = rs.IdealGasEOS()
        assert eos.sound_speed(300.0) == pytest.approx(347.2, rel=1e-2)
        # Mach number = U / a
        assert eos.mach_number(34.72, 300.0) == pytest.approx(0.1, rel=1e-2)

    def test_viscosity_increases_with_T(self, rs):
        eos = rs.IdealGasEOS()
        mus = [eos.viscosity(T) for T in (250.0, 300.0, 400.0, 600.0)]
        assert all(m1 < m2 for m1, m2 in zip(mus, mus[1:]))


@pytest.fixture(scope="module")
def compressible_case(rs):
    """Build a tiny Ma=0.1 channel for the bindings tests.  Module-scoped so
    we run the solver only once across the test class below."""
    eos    = rs.IdealGasEOS()
    T_in   = 300.0
    p_ref  = 101325.0
    rho_in = eos.density(p_ref, T_in)
    mu_in  = eos.viscosity(T_in)
    Uin    = 0.1 * eos.sound_speed(T_in)
    Re     = rho_in * Uin * 1.0 / mu_in
    nu_in  = mu_in / rho_in

    mesh = rs.Mesh.make_channel_2d(12, 10, 10.0, 1.0, Re=Re, yPlusTarget=1.0)
    mesh.compute_wall_distance()
    bcs = rs.CompressibleBoundaryConditions.channel_defaults(
        mesh, Uin, T_in, p_ref,
        kIn=1.5 * (Uin * 0.05) ** 2,
        omIn=(1.5 * (Uin * 0.05) ** 2) / (nu_in * 100.0))

    obs = rs.ObservationOperator()
    obs.add_skin_friction(
        wall_patch="bottom", location=rs.Vec3(5.0, 0.0, 0.0),
        cf_obs=0.005, sigma=0.001, ref_vel=Uin)

    settings = rs.SolverSettings()
    settings.max_iterations      = 800
    settings.convergence_tol     = 1e-3
    settings.divergence_limit    = 1e10
    settings.alpha_u             = 0.5
    settings.alpha_p             = 0.2
    settings.alpha_t             = 0.7
    settings.alpha_k             = 0.4
    settings.alpha_omega         = 0.4
    settings.inner_iterations    = 200
    settings.turb_start_iter     = 50
    settings.turb_update_interval = 2
    settings.verbose             = False

    param_set = rs.InferenceParameterSet.a1_betaStar()
    fm = rs.CompressibleForwardModel(
        mesh=mesh, param_set=param_set, obs_op=obs, bcs=bcs, eos=eos,
        settings=settings, u_init=rs.Vec3(Uin, 0, 0),
        p_init=p_ref, T_init=T_in,
        k_init=1.5 * (Uin * 0.05) ** 2,
        omega_init=(1.5 * (Uin * 0.05) ** 2) / (nu_in * 100.0))

    theta_def = list(param_set.pack(rs.SSTCoefficients()))
    result    = fm.evaluate(theta_def)
    return {"fm": fm, "result": result, "Uin": Uin, "T_in": T_in,
            "p_ref": p_ref, "mesh": mesh, "eos": eos}


class TestCompressibleForwardModel:
    def test_evaluate_returns_result(self, rs, compressible_case):
        result = compressible_case["result"]
        # Status should be one of the documented enum values; either
        # Converged or Unconverged is acceptable for an 800-iter run.
        assert result.status in (
            rs.EvaluationStatus.Converged,
            rs.EvaluationStatus.Unconverged,
        )
        assert result.simple_iters > 0
        assert np.isfinite(result.log_lik)

    def test_predictions_match_observation_count(self, compressible_case):
        result = compressible_case["result"]
        assert len(result.predictions) == 1   # one Cf observation
        assert np.isfinite(result.predictions[0])

    def test_last_fields_keys(self, compressible_case):
        fm = compressible_case["fm"]
        assert fm.has_last_fields()
        f = fm.last_fields()
        # Compressible bindings must expose density and temperature on top of
        # the standard incompressible field set.
        for k in ("U", "p", "T", "rho", "k", "omega", "nuT", "F1", "F2", "Pk"):
            assert k in f, f"compressible last_fields() missing {k!r}"

    def test_field_shapes_and_positivity(self, compressible_case):
        fm   = compressible_case["fm"]
        mesh = compressible_case["mesh"]
        f    = fm.last_fields()
        n    = mesh.n_cells()
        assert f["U"].shape   == (n, 3)
        for k in ("p", "T", "rho", "k", "omega", "nuT"):
            assert f[k].shape == (n,), f"{k} has unexpected shape {f[k].shape}"
        # Solver must not produce nonpositive density or temperature.
        assert np.all(f["rho"] > 0.0)
        assert np.all(f["T"]   > 0.0)
        # Pressures are absolute Pascals: must stay positive too.
        assert np.all(f["p"]   > 0.0)

    def test_max_mach_stays_subsonic(self, compressible_case):
        fm  = compressible_case["fm"]
        eos = compressible_case["eos"]
        f   = fm.last_fields()
        Umag    = np.linalg.norm(f["U"], axis=1)
        Ma_max  = float(np.max(Umag / np.sqrt(eos.gamma * eos.R * f["T"])))
        # Inlet is Ma=0.1 in a developing channel; max should stay subsonic.
        assert Ma_max < 0.8, f"unexpectedly high Ma_max={Ma_max:.3f}"

    def test_temperature_stays_in_finite_band(self, compressible_case):
        # At Ma=0.1 the adiabatic temperature rise across the channel is small
        # (Tt/T = 1 + (gamma-1)/2 * Ma^2 ≈ 1.002).  At this small mesh the
        # solver may not be fully converged within 800 iters, so we use a
        # generous band that only catches genuinely runaway temperatures.
        fm = compressible_case["fm"]
        f  = fm.last_fields()
        T_min, T_max = float(np.min(f["T"])), float(np.max(f["T"]))
        # Inlet T = 300; allow up to ±100 K transient swings before flagging.
        assert 200.0 < T_min < 400.0, f"T_min outside sane band: {T_min} K"
        assert 200.0 < T_max < 400.0, f"T_max outside sane band: {T_max} K"


class TestObservationCompatibility:
    """The compressible model must produce predictions that ObservationOperator
    can consume without changes — i.e. the existing incompressible-shaped
    interfaces should accept the same evaluate() output."""

    def test_predictions_are_python_floats(self, compressible_case):
        result = compressible_case["result"]
        for v in result.predictions:
            assert isinstance(v, float)
            assert np.isfinite(v)

    def test_log_lik_finite_for_default_theta(self, compressible_case):
        # Default theta + observations chosen near a typical Cf must give a
        # finite log-likelihood; -inf would indicate the binding broke the
        # Gaussian likelihood path.
        result = compressible_case["result"]
        assert np.isfinite(result.log_lik)
