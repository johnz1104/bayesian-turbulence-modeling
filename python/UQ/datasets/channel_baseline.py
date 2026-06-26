"""Boussinesq SST baseline channel field, matched in Re_tau to a DNS case.

This is the baseline the model-form discrepancy is formed against, and the
forward map the calibration runs. It resolves the open question of where the
baseline eddy viscosity and turbulence timescale come from for real data: from a
converged SST solve at the matched friction Reynolds number, not from a
correlation.

Solver scope. The incompressible solver provides a developing inlet/outlet
channel only (no streamwise-periodic / fully-developed mode), so the
fully-developed profile is taken from the column nearest the outlet, where the
flow is most developed. Re_tau is matched to the DNS case by a secant iteration
on the molecular viscosity nu (bulk velocity and half-height fixed at one). The
developing channel is known to sit somewhat below Dean's fully-developed Cf;
that gap is reported, not hidden, and is the solver's documented behaviour
(validate against the solver's correlation check, not against a perfect Dean
match).

Wall units. Profiles are returned in wall units (^+) using the solve's own
friction velocity u_tau (from the near-outlet wall shear), so they are directly
comparable, station by station, with the wall-unit DNS profiles at matched
Re_tau.
"""
import numpy as np

# the inference layer is solver-coupled; import lazily so importing this module
# (and the UQ.datasets package) never forces the built extension to be present.
from solver_bindings import _rs

_CMU = 0.09  # Boussinesq C_mu, for the turbulence timescale tau = nu_t/(C_mu k)


class ChannelBaselineRANS:
    """One converged SST channel solve, matched in Re_tau, in wall units.

    Public profiles (1-D, wall to centerline, wall units):
      yplus, U (U^+), k (k^+), nu_t (nu_t/nu), omega (omega nu/u_tau^2)
    Scalars: nu, u_tau, re_tau (achieved), cf, cf_dean, status, iterations.
    """

    # one channel configuration, sized for ensemble tractability (~20-40 s/solve);
    # tests override it with a tiny, fast config to exercise the machinery only
    HALF_HEIGHT = 1.0
    U_BULK = 1.0
    DEFAULT_CONFIG = {"nx": 48, "ny": 64, "Lx": 20.0,
                      "max_iter": 6000, "conv_tol": 1.0e-4, "yplus_target": 0.5}

    def __init__(self, nu, u_tau, re_tau, yplus, U, k, nu_t, omega,
                 cf, status, iterations, meta):
        self.nu = float(nu)
        self.u_tau = float(u_tau)
        self.re_tau = float(re_tau)
        self.yplus = np.asarray(yplus, dtype=float)
        self.U = np.asarray(U, dtype=float)            # U^+
        self.k = np.asarray(k, dtype=float)            # k^+
        self.nu_t = np.asarray(nu_t, dtype=float)      # nu_t/nu
        self.omega = np.asarray(omega, dtype=float)    # omega^+
        self.cf = float(cf)
        self.cf_dean = 0.073 * self.reynolds_bulk() ** (-0.25)
        self.status = status
        self.iterations = int(iterations)
        self.meta = meta

    # ---- solving and Re_tau matching --------------------------------------

    @staticmethod
    def _build_case(nu, cfg):
        """Build mesh, BCs, and a ForwardModel for a developing channel at nu."""
        rs = _rs()
        h = ChannelBaselineRANS.HALF_HEIGHT
        Ub = ChannelBaselineRANS.U_BULK
        Lx = cfg["Lx"]
        re_b_half = Ub * h / nu  # the solver's half-height bulk Reynolds number

        mesh = rs.Mesh.make_channel_2d(
            nx=cfg["nx"], ny=cfg["ny"],
            Lx=Lx, Ly=2.0 * h, Re=re_b_half, yPlusTarget=cfg["yplus_target"])
        mesh.compute_wall_distance()

        Tu = 0.05
        kIn = 1.5 * (Ub * Tu) ** 2
        omIn = kIn / (nu * 100.0)
        bcs = rs.FlowBoundaryConditions.channel_defaults(mesh, Ub, kIn, omIn)

        # a trivial observation keeps the ForwardModel well-formed; the baseline
        # only reads last_fields(), so the observed value is irrelevant here
        obs = rs.ObservationOperator()
        obs.add_skin_friction(wall_patch="bottom",
                              location=rs.Vec3(0.9 * Lx, 0.0, 0.0),
                              cf_obs=6.0e-3, sigma=1.0e-3, ref_vel=Ub)
        param_set = rs.InferenceParameterSet.a1_betaStar()

        settings = rs.SolverSettings()
        settings.max_iterations = cfg["max_iter"]
        settings.convergence_tol = cfg["conv_tol"]
        settings.verbose = False
        settings.alpha_u = 0.7
        settings.alpha_p = 0.5

        fm = rs.ForwardModel(mesh, param_set, obs, bcs, nu, settings,
                             rs.Vec3(Ub, 0.0, 0.0), 0.0, kIn, omIn)
        return rs, mesh, fm, param_set

    @staticmethod
    def _solve(nu, cfg):
        """Solve at nu (default SST), extract the near-outlet developed profile.

        Returns a dict of wall-unit profiles plus u_tau, Cf, Re_tau, status. The
        column nearest the outlet is the most-developed station; only the bottom
        half (wall to centerline) is kept, matching the half-channel DNS record.
        """
        from wall_diagnostics import cf_along_wall

        rs, mesh, fm, param_set = ChannelBaselineRANS._build_case(nu, cfg)
        theta = list(param_set.pack(rs.SSTCoefficients()))
        result = fm.evaluate(theta)
        status = str(result.status).split(".")[-1]
        if not fm.has_last_fields():
            return {"status": status, "ok": False}

        ff = fm.last_fields()
        cc = np.asarray(mesh.cell_centers())
        x = cc[:, 0]
        y = cc[:, 1]
        U = np.asarray(ff["U"])[:, 0]
        k = np.asarray(ff["k"])
        nuT = np.asarray(ff["nuT"])
        omega = np.asarray(ff["omega"])

        Lx = cfg["Lx"]
        h = ChannelBaselineRANS.HALF_HEIGHT
        Ub = ChannelBaselineRANS.U_BULK

        # u_tau from the wall shear averaged over the developed (near-outlet) wall
        cf = cf_along_wall(mesh, ff, nu, "bottom", Ub)
        dev = cf["x"] > 0.85 * Lx
        cf_dev = float(np.mean(cf["cf"][dev]))
        u_tau = Ub * np.sqrt(cf_dev / 2.0)
        re_tau = u_tau * h / nu

        # near-outlet column, bottom half only (wall to centerline)
        xs = np.unique(x)
        x_out = xs[np.argmin(np.abs(xs - 0.97 * Lx))]
        col = np.abs(x - x_out) < 1e-9
        yb = y[col]
        keep = yb <= h + 1e-12
        order = np.argsort(yb[keep])
        ysel = yb[keep][order]

        prof = {
            "status": status,
            "ok": True,
            "nu": nu,
            "u_tau": u_tau,
            "re_tau": re_tau,
            "cf": cf_dev,
            "iterations": cfg["max_iter"],
            # wall units
            "yplus": ysel * u_tau / nu,
            "U": U[col][keep][order] / u_tau,
            "k": k[col][keep][order] / u_tau ** 2,
            "nu_t": nuT[col][keep][order] / nu,
            "omega": omega[col][keep][order] * nu / u_tau ** 2,
        }
        return prof

    @staticmethod
    def match(re_tau_target, tol=0.03, max_solves=6, re_b_init=None, verbose=False,
              cfg=None):
        """Match Re_tau by a secant iteration on nu; return a ChannelBaselineRANS.

        re_b_init: optional (low, high) bracket for the solver half-height bulk
        Reynolds number Re_b = 1/nu; defaults to a power-law guess calibrated on
        the solver (Re_tau ~ 520 at Re_b 1e4). cfg overrides the solve resolution
        (tests pass a tiny, fast config).
        """
        cfg = dict(ChannelBaselineRANS.DEFAULT_CONFIG if cfg is None else cfg)
        # initial Re_b guesses from the solver's own Re_b -> Re_tau power law
        if re_b_init is None:
            g = 1.0e4 * (re_tau_target / 520.0) ** (1.0 / 0.88)
            re_b_lo, re_b_hi = 0.85 * g, 1.20 * g
        else:
            re_b_lo, re_b_hi = re_b_init

        def f(re_b):
            prof = ChannelBaselineRANS._solve(1.0 / re_b, cfg)
            if not prof["ok"]:
                return None, prof
            return prof["re_tau"] - re_tau_target, prof

        r_lo, p_lo = f(re_b_lo)
        r_hi, p_hi = f(re_b_hi)
        if r_lo is None or r_hi is None:
            raise ValueError(f"baseline solve failed bracketing Re_tau="
                             f"{re_tau_target} (status "
                             f"{p_lo.get('status')} / {p_hi.get('status')})")

        best = p_hi if abs(r_hi) < abs(r_lo) else p_lo
        for it in range(max_solves):
            if verbose:
                print(f"  match Re_tau={re_tau_target}: Re_b={1.0/best['nu']:.0f} "
                      f"-> Re_tau={best['re_tau']:.1f}")
            if abs(best["re_tau"] - re_tau_target) / re_tau_target <= tol:
                break
            # secant update on Re_b
            denom = (r_hi - r_lo)
            if abs(denom) < 1e-9:
                break
            re_b_new = re_b_hi - r_hi * (re_b_hi - re_b_lo) / denom
            re_b_new = float(np.clip(re_b_new, 0.3 * re_b_lo, 3.0 * re_b_hi))
            r_new, p_new = f(re_b_new)
            if r_new is None:
                break
            re_b_lo, r_lo = re_b_hi, r_hi
            re_b_hi, r_hi = re_b_new, r_new
            if abs(p_new["re_tau"] - re_tau_target) < abs(best["re_tau"] - re_tau_target):
                best = p_new

        meta = {
            "config": dict(cfg),
            "re_tau_target": float(re_tau_target),
            "note": "developing inlet/outlet channel; profile taken near outlet",
        }
        return ChannelBaselineRANS(
            nu=best["nu"], u_tau=best["u_tau"], re_tau=best["re_tau"],
            yplus=best["yplus"], U=best["U"], k=best["k"], nu_t=best["nu_t"],
            omega=best["omega"], cf=best["cf"], status=best["status"],
            iterations=best["iterations"], meta=meta)

    # ---- derived quantities and interpolation -----------------------------

    def reynolds_bulk(self):
        """Bulk Reynolds number Re_b = U_b (2 h) / nu (Dean convention)."""
        return self.U_BULK * (2.0 * self.HALF_HEIGHT) / self.nu

    def _interp(self, field, yplus_query):
        """Interpolate a wall-unit profile onto the DNS y^+ stations.

        DNS stations beyond the baseline's outermost point (near the centerline)
        are clamped to the edge value; the discrepancy work uses the wall side.
        """
        yq = np.asarray(yplus_query, dtype=float)
        return np.interp(yq, self.yplus, field,
                         left=field[0], right=field[-1])

    def profiles_at(self, yplus_query):
        """All baseline wall-unit profiles interpolated to the DNS y^+ stations."""
        return {
            "U": self._interp(self.U, yplus_query),
            "k": self._interp(self.k, yplus_query),
            "nu_t": self._interp(self.nu_t, yplus_query),
            "omega": self._interp(self.omega, yplus_query),
        }

    def timescale_plus_at(self, yplus_query, k_floor=1e-8):
        """Baseline turbulence timescale tau^+ = nu_t^+ / (C_mu k^+) at stations.

        This is the timescale the discrepancy non-dimensionalisation expects, so
        that the linear-eddy-viscosity anisotropy b = -C_mu dev(S) reproduces the
        actual baseline Boussinesq anisotropy -nu_t/k * dev(strain). Equivalently
        tau^+ = k^+/epsilon^+ = 1/(C_mu omega^+).
        """
        p = self.profiles_at(yplus_query)
        k = np.maximum(p["k"], k_floor)
        return p["nu_t"] / (_CMU * k)
