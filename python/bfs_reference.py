"""
Backward-facing step reference material: Driver–Seegmiller (1985) data and
the conservative SIMPLE solver settings the BFS case requires.

Extracted from bayesian_inference.py (2026-06-12 modularization).  Used by
the universal case library (case_library.build_bfs) and by the research
synthetic-calibration harnesses.  Zero dependencies by design.
"""


def _bfs_solver_settings(rs):
    """Conservative SIMPLE settings for the backward-facing step.

    BFS has strong initial transients due to the recirculation zone: aggressive
    relaxation diverges in ~10 iterations.  These settings use low alpha_u/alpha_p
    and delay turbulence activation to let the mean-flow pattern establish first.
    """
    s = rs.SolverSettings()
    s.max_iterations     = 12000
    s.convergence_tol    = 1e-4
    s.divergence_limit   = 1e8     # looser limit for the initial BFS transient
    s.alpha_u            = 0.3
    s.alpha_p            = 0.2
    s.alpha_k            = 0.4
    s.alpha_omega        = 0.4
    s.inner_iterations   = 300
    s.inner_tolerance    = 1e-4
    s.turb_start_iter    = 30      # let mean flow establish before activating k-omega
    s.turb_update_interval = 2
    s.verbose            = False
    s.report_interval    = 2000
    return s


# Driver-Seegmiller (1985) backward-facing step reference data
# Re_h = 37,600 (step height h=1, U_ref=1).  Geometry: h_s=1, H_up=2, H=3.
# Cf and U values are approximate literature values for validation comparisons only.
DS1985 = {
    'Re_h':   37600.0,
    'h_s':    1.0,      # step height (reference length)
    'H_up':   2.0,      # upstream channel height above step
    'H':      3.0,      # total downstream height
    'Lu':     6.0,      # upstream domain length (in h units)
    'Ld':     20.0,     # downstream domain length (in h units)
    'xr_h':   6.26,     # reattachment length / h  (Menter SST prediction range: 5.5–7)
    # Cf at x/h stations on bottom_wall_down  [station, Cf_mean, Cf_std]
    'Cf_stations': [
        (1.0,  -0.00120, 0.00020),
        (2.0,  -0.00090, 0.00020),
        (3.0,  -0.00050, 0.00020),
        (4.0,   0.00020, 0.00015),
        (5.0,   0.00080, 0.00015),
        (6.0,   0.00140, 0.00015),
        (7.0,   0.00180, 0.00015),
    ],
    # U/U_ref profiles at x/h stations, y/h values, U_mean (normalised)
    # Abbreviated: 4 stations x 3 y-points each
    'U_profiles': [
        (1.0,  0.5, -0.10, 0.04),
        (1.0,  1.5,  0.70, 0.04),
        (4.0,  0.5,  0.20, 0.04),
        (4.0,  1.5,  0.85, 0.04),
        (6.0,  0.5,  0.55, 0.04),
        (6.0,  1.5,  0.90, 0.04),
        (10.0, 0.5,  0.75, 0.04),
        (10.0, 1.5,  0.92, 0.04),
    ],
}
