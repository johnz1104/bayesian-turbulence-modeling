"""A-posteriori plane-Couette forward model and calibration (DNS_plan.md Step 2).

The moving-wall incompressible SST solver (FlowBoundaryConditions.couette_defaults,
the streamwise-invariant Couette setup) is wrapped as a theta -> Couette-QoI
forward map, so the cross-flow coverage test predicts the held-out Couette truth
THROUGH the same SST closure that was calibrated on the channel, not through an
analytic surrogate. This is the a-posteriori cross-flow generalization: calibrate
on channel (Step 1), predict Couette here, and ask whether the calibrated UQ
covers it.

Units. The solver runs with the moving wall at Uwall = 2 and the stationary wall
at 0, so the Couette bulk velocity U_b = Uwall/2 = 1 and the solved streamwise
velocity is already in outer (U/U_b) units, exactly as the channel forward model
runs at U_b = 1. The friction Reynolds number is matched to each DNS Couette case
by a secant on the molecular viscosity nu (the same matching the channel baseline
uses). The QoI is the half-gap mean-velocity profile U/U_b versus y/h and the wall
skin friction Cf, mirroring the channel QoI.

Reuse. CouetteCalibration subclasses ChannelCalibration, inheriting the ensemble,
surrogate, tempered-posterior, posterior-predictive, coverage, conformal, and
learning-rate machinery unchanged; only the forward model (moving-wall BCs), the
QoI selection (Couette half gap), and the modeled observation band are Couette
specific. The channel calibration class is consumed, not modified.
"""
import numpy as np

from solver_bindings import _rs
from priors import make_sampling_prior
from gp_surrogate import GPSurrogate, MultiOutputSurrogate

from .channel_calibration import ChannelCalibration, PARAM_SETS, DEFAULT_CFG
from . import observation_sigma as obs_sigma

# Couette forward-model resolution. Few streamwise cells (the fully-developed flow
# is x-invariant, so a short, narrow domain converges fast); fine wall-normal grid.
COUETTE_CFG = {"nx": 4, "ny": 96, "Lx": 1.5,
               "max_iter": 12000, "conv_tol": 1.0e-4, "yplus_target": 0.3}

_UWALL = 2.0   # moving-wall speed so the Couette bulk U_b = Uwall/2 = 1 (outer units)


class CouetteForwardRANS:
    """Matched-Re_tau moving-wall Couette SST solve, in wall units.

    A thin standalone wrapper used to match nu to a target Re_tau and to inspect
    a single solved Couette profile. The calibration ensemble itself runs through
    CouetteCalibration (which reuses the channel ensemble machinery).
    """

    @staticmethod
    def _build(nu, cfg, theta=None):
        rs = _rs()
        Uwall = _UWALL
        Ub = 0.5 * Uwall
        Lx = cfg["Lx"]
        re_b = Ub * 1.0 / nu                      # half-height bulk Reynolds number
        mesh = rs.Mesh.make_channel_2d(nx=cfg["nx"], ny=cfg["ny"], Lx=Lx, Ly=2.0,
                                       Re=Uwall * 1.0 / nu,
                                       yPlusTarget=cfg["yplus_target"])
        mesh.compute_wall_distance()
        kIn = 1.5 * (Ub * 0.05) ** 2
        omIn = kIn / (nu * 100.0)
        bcs = rs.FlowBoundaryConditions.couette_defaults(mesh, Uwall, kIn, omIn)
        obs = rs.ObservationOperator()
        obs.add_skin_friction("bottom", rs.Vec3(0.5 * Lx, 0.0, 0.0), 6.0e-3, 1.0e-3, Ub)
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
        """Solve one Couette case at default SST; return the half-gap wall-unit profile."""
        from wall_diagnostics import cf_along_wall
        rs, mesh, fm, param_set = CouetteForwardRANS._build(nu, cfg)
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
        Lx = cfg["Lx"]
        Ub = 0.5 * _UWALL
        cf = cf_along_wall(mesh, ff, nu, "bottom", Ub)
        cf_w = float(np.mean(cf["cf"]))
        u_tau = Ub * np.sqrt(cf_w / 2.0)
        re_tau = u_tau * 1.0 / nu
        # x-invariant: take the mid-domain column, bottom half (wall to centerline)
        xs = np.unique(x)
        x_mid = xs[len(xs) // 2]
        col = np.abs(x - x_mid) < 1e-9
        yb = y[col]
        keep = yb <= 1.0 + 1e-12
        order = np.argsort(yb[keep])
        ys = yb[keep][order]
        return {
            "status": status, "ok": True, "nu": nu, "u_tau": u_tau, "re_tau": re_tau,
            "cf": cf_w, "y_h": ys, "yplus": ys * u_tau / nu,
            "U_over_Ub": U[col][keep][order] / Ub,
        }

    @staticmethod
    def match(re_tau_target, tol=0.04, max_solves=8, cfg=None, verbose=False):
        """Match the solved Re_tau to a target by a secant on nu; return the profile.

        Returns the solved-profile dict plus the matched nu. Mirrors the channel
        baseline's Re_tau matching (the Couette Re_tau / nu relation has no closed
        form in the solver, so it is found numerically once per case).
        """
        cfg = dict(COUETTE_CFG if cfg is None else cfg)
        # initial nu bracket from the solver's own Re_b -> Re_tau scaling, calibrated
        # on this moving-wall setup (Re_tau ~ 0.083 Re_b^0.88, Re_b = 1/nu)
        g = (re_tau_target / 0.083) ** (1.0 / 0.88)    # Re_b ~ guess for this Re_tau
        nu_lo, nu_hi = 1.0 / (1.4 * g), 1.0 / (0.7 * g)

        def f(nu):
            prof = CouetteForwardRANS._solve(nu, cfg)
            if not prof["ok"]:
                return None, prof
            return prof["re_tau"] - re_tau_target, prof

        r_lo, p_lo = f(nu_lo)
        r_hi, p_hi = f(nu_hi)
        if r_lo is None or r_hi is None:
            raise ValueError(f"Couette baseline solve failed bracketing Re_tau="
                             f"{re_tau_target}")
        best = p_hi if abs(r_hi) < abs(r_lo) else p_lo
        for _ in range(max_solves):
            if verbose:
                print(f"  match Re_tau={re_tau_target}: nu=1/{1.0/best['nu']:.0f} "
                      f"-> Re_tau={best['re_tau']:.1f}")
            if abs(best["re_tau"] - re_tau_target) / re_tau_target <= tol:
                break
            denom = r_hi - r_lo
            if abs(denom) < 1e-12:
                break
            nu_new = nu_hi - r_hi * (nu_hi - nu_lo) / denom
            nu_new = float(np.clip(nu_new, 0.2 * nu_hi, 5.0 * nu_lo))
            r_new, p_new = f(nu_new)
            if r_new is None:
                break
            nu_lo, r_lo = nu_hi, r_hi
            nu_hi, r_hi = nu_new, r_new
            if abs(p_new["re_tau"] - re_tau_target) < abs(best["re_tau"] - re_tau_target):
                best = p_new
        best["matched_nu"] = best["nu"]
        return best


class CouetteCalibration(ChannelCalibration):
    """A-posteriori Couette forward map theta -> Couette QoI, with calibrated UQ.

    Subclasses ChannelCalibration to inherit the ensemble / surrogate / tempered-
    posterior / posterior-predictive / coverage / conformal machinery; overrides
    only the Couette-specific forward model (moving-wall BCs), the QoI selection
    (half-gap U/U_b plus Cf from the Couette DNS), and the modeled observation
    band (the files carry no DNS _stdev, so observation uncertainty is the modeled
    relative value of UQ.datasets.observation_sigma).
    """

    def __init__(self, dns, nu, param_set="a1_betaStar", n_stations=20,
                 yplus_lo=1.0, rel=0.005, cfg=None):
        self.dns = dns
        self.param_set_name = param_set
        self.cfg = dict(COUETTE_CFG if cfg is None else cfg)
        self.rs = _rs()
        self.param_set = PARAM_SETS[param_set](self.rs)
        self.prior = make_sampling_prior(self.param_set)
        self.ndim = self.prior.ndim
        self.nu = float(nu)
        self.re_b = 0.5 * _UWALL / self.nu
        self.rel = float(rel)
        self._select_stations(n_stations, yplus_lo)
        self.X = self.loglik = self.preds = None
        self.gp = self.mo = None

    # ---- Couette QoI: half-gap U/U_b profile + Cf -------------------------

    def _select_stations(self, n_stations, yplus_lo):
        d = self.dns
        ub_plus = d.wall_velocity()                       # U_b^+ = centerline U^+
        targets = np.geomspace(yplus_lo, 0.95 * d.re_tau, n_stations)
        idx = sorted(set(int(np.argmin(np.abs(d.yplus - t))) for t in targets))
        self.idx = np.array(idx)
        self.y_h = d.y_outer[self.idx]                    # y/h station coordinates
        u_out = d.U[self.idx] / ub_plus                   # U/U_b (outer, friction-free)
        self.cf_obs = 2.0 / ub_plus ** 2                  # Cf = 2/(U_b^+)^2
        self.cf_sig = 0.05 * self.cf_obs                  # 5% modeled on the Cf QoI
        self.qoi_truth = np.concatenate([[self.cf_obs], u_out])
        self.qoi_names = ["Cf"] + [f"U/Ub@y+={d.yplus[i]:.0f}" for i in self.idx]
        self.n_qoi = self.qoi_truth.size
        self.set_observation_band(self.rel)

    def set_observation_band(self, rel):
        """Set the modeled observation sigma to rel * |local scale| (the Step-2 band).

        Velocity QoIs take the modeled relative band (0.5 or 1 percent of U/U_b);
        the Cf QoI keeps its 5 percent modeled value. Labeled modeled, never a DNS
        _stdev (the Couette files carry none). Changing the band re-runs coverage at
        a different observation level for the sensitivity report.
        """
        self.rel = float(rel)
        u_sig = obs_sigma.relative(self.qoi_truth[1:], rel=rel)
        self.qoi_sigma = np.concatenate([[self.cf_sig], u_sig])
        return self.qoi_sigma

    def _observation_operator(self):
        rs = self.rs
        Lx = self.cfg["Lx"]
        obs = rs.ObservationOperator()
        obs.add_skin_friction("bottom", rs.Vec3(0.5 * Lx, 0.0, 0.0),
                              self.cf_obs, self.cf_sig, 0.5 * _UWALL)
        for yh, u, s in zip(self.y_h, self.qoi_truth[1:], self.qoi_sigma[1:]):
            obs.add_velocity_profile(rs.Vec3(0.5 * Lx, float(yh), 0.0), 0,
                                     float(u), float(s))
        return obs

    def _forward_model(self):
        rs = self.rs
        Uwall = _UWALL
        Ub = 0.5 * Uwall
        mesh = rs.Mesh.make_channel_2d(nx=self.cfg["nx"], ny=self.cfg["ny"],
                                       Lx=self.cfg["Lx"], Ly=2.0,
                                       Re=Uwall / self.nu,
                                       yPlusTarget=self.cfg["yplus_target"])
        mesh.compute_wall_distance()
        kIn = 1.5 * (Ub * 0.05) ** 2
        omIn = kIn / (self.nu * 100.0)
        bcs = rs.FlowBoundaryConditions.couette_defaults(mesh, Uwall, kIn, omIn)
        obs = self._observation_operator()
        settings = rs.SolverSettings()
        settings.max_iterations = self.cfg["max_iter"]
        settings.convergence_tol = self.cfg["conv_tol"]
        settings.verbose = False
        settings.alpha_u = 0.7
        settings.alpha_p = 0.5
        fm = rs.ForwardModel(mesh, self.param_set, obs, bcs, self.nu, settings,
                             rs.Vec3(Ub, 0.0, 0.0), 0.0, kIn, omIn)
        self._alive = (mesh, obs, bcs, settings)        # pybind11 lifetime
        return fm
