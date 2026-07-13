"""
Regression test for the compressible Ma=0.1 channel.

Loads ``tests/regression/compressible_channel_ma01.json``, re-runs the case
via ``compressible_diagnostics.run_validation_case``, and asserts the live
output stays within the documented tolerances.

A failure here indicates either:
  * an intentional change to the compressible solver (in which case the
    baseline JSON must be regenerated and committed in the same PR), or
  * an unintended regression worth investigating before any other PHASE.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT       = Path(__file__).resolve().parents[2]
BASELINE_PATH   = REPO_ROOT / "tests" / "regression" / "compressible_channel_ma01.json"


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="module")
def actual_summary(rs):  # rs fixture ensures the bindings build is available
    from compressible_diagnostics import (
        make_validation_case, run_validation_case,
    )
    case = make_validation_case(name="ma_0_1", Ma=0.1, nx=40, ny=30,
                                 max_iterations=3000)
    return run_validation_case(case)


class TestMa01Baseline:
    def test_converged(self, actual_summary):
        assert actual_summary["converged"]
        assert not actual_summary["diverged"]

    def test_positivity(self, actual_summary):
        assert actual_summary["positivity_ok"], (
            f"positivity violated: {actual_summary['positivity']}"
        )
        assert actual_summary["rho_min"] > 0.0
        assert actual_summary["p_min"]   > 0.0
        assert actual_summary["T_min"]   > 0.0

    def test_iter_count_within_tolerance(self, baseline, actual_summary):
        tol  = baseline["tolerances"]["simple_iters_abs"]
        ref  = baseline["expected"]["simple_iters"]
        live = actual_summary["simple_iters"]
        assert abs(live - ref) <= tol, (
            f"SIMPLE iteration count drifted: live={live} ref={ref} (tol ±{tol})"
        )

    def test_Ma_max_within_tolerance(self, baseline, actual_summary):
        ref  = baseline["expected"]["Ma_max"]
        live = actual_summary["Ma_max"]
        rel  = baseline["tolerances"]["Ma_max_rel"]
        assert abs(live - ref) / abs(ref) < rel, (
            f"Ma_max drift: live={live} ref={ref} rel={abs(live-ref)/abs(ref)} "
            f"vs tol {rel}"
        )

    def test_temperature_extrema(self, baseline, actual_summary):
        tol = baseline["tolerances"]["T_extrema_abs_K"]
        for key in ("T_min", "T_max"):
            ref = baseline["expected"][key]
            live = actual_summary[key]
            assert abs(live - ref) <= tol, (
                f"{key} drift: live={live} ref={ref} tol ±{tol}"
            )

    def test_density_extrema(self, baseline, actual_summary):
        rel = baseline["tolerances"]["rho_extrema_rel"]
        for key in ("rho_min", "rho_max"):
            ref = baseline["expected"][key]
            live = actual_summary[key]
            assert abs(live - ref) / abs(ref) < rel, (
                f"{key} drift: live={live} ref={ref} rel={abs(live-ref)/abs(ref)}"
            )

    def test_U_max(self, baseline, actual_summary):
        rel = baseline["tolerances"]["U_max_rel"]
        ref = baseline["expected"]["U_max"]
        live = actual_summary["U_max"]
        assert abs(live - ref) / abs(ref) < rel

    def test_Cf_stations(self, baseline, actual_summary):
        rel = baseline["tolerances"]["Cf_rel"]
        ref = baseline["expected"]["Cf_at_stations"]
        live = actual_summary["Cf_at_stations"]
        assert len(live) == len(ref)
        for i, (l, r) in enumerate(zip(live, ref)):
            assert abs(l - r) / abs(r) < rel, (
                f"Cf station {i} drift: live={l} ref={r}"
            )

    def test_mass_flux_imbalance(self, baseline, actual_summary):
        cap = baseline["tolerances"]["mass_flux_rel_imbalance_max"]
        live = actual_summary["mass_flux"]["rel_imbalance"]
        assert live < cap, (
            f"mass-flux imbalance {live:.3e} exceeds cap {cap:.3e}"
        )


class TestPositivityInvariants:
    """No baseline comparison — these must hold for every Ma=0.1 run, ever."""

    def test_no_negative_rho_p_T(self, actual_summary):
        assert actual_summary["rho_min"] > 0.0
        assert actual_summary["p_min"]   > 0.0
        assert actual_summary["T_min"]   > 0.0

    def test_max_Mach_subsonic(self, actual_summary):
        assert actual_summary["Ma_max"] < 0.8


def test_last_fields_pressure_is_thermodynamic(rs):
    """The exported "p" is the thermodynamic static pressure; the mechanical
    working variable is exposed separately and differs by exactly
    (2/3) rho k, the Favre-trace absorption of the two-pressure convention."""
    import numpy as np
    from compressible_diagnostics import make_validation_case
    case = make_validation_case(name="ma_0_1_export", Ma=0.1, nx=16, ny=12,
                                max_iterations=800)
    fwd = case["forward"]
    theta = list(case["param_set"].pack(rs.SSTCoefficients()))
    result = fwd.evaluate(theta)
    # the export identity is state-independent, so any evaluated (non-diverged)
    # state discriminates it; full convergence is the smoke test's job
    assert str(result.status).split(".")[-1] in ("Converged", "Unconverged")
    assert fwd.has_last_fields()
    ff = fwd.last_fields()
    p_t = np.asarray(ff["p"]); p_m = np.asarray(ff["p_mechanical"])
    rho = np.asarray(ff["rho"]); k = np.asarray(ff["k"])
    assert np.allclose(p_m - p_t, (2.0 / 3.0) * rho * np.maximum(k, 0.0),
                       rtol=0, atol=1e-9 * np.max(p_m))
    assert np.all(p_t > 0.0)
    # the shim is not a no-op: turbulence is live somewhere in the channel
    assert np.max(p_m - p_t) > 0.0
