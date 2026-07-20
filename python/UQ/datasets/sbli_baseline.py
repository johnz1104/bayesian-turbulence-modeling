"""Density-based SST baseline configurations for the impinging-shock interaction.

The pre-registered baseline route for both interaction legs: the shock-capturing
density-based SST solve serves the discrepancy extraction (a-priori) and the
coupled ensembles (a-posteriori), one solve per wall-thermal configuration,
with

- the inflow anchored to the data's own upstream state: the profile boundary
  takes the record's first field column (mean velocity, temperature, density)
  with the turbulence energy from the record's stresses and the specific rate
  from the shear-consistent eddy viscosity omega = k / nu_t,
  nu_t = -R_xy / (dU/dy), floored and capped where the construction degenerates
  (freestream, wall row);
- the incident shock imposed through the top boundary: faces upstream of the
  shock's top-boundary entry hold the freestream, faces downstream hold the
  post-shock state of the oblique-shock relations at the dataset's incidence
  (theta-beta-M at Mach 2.28, deflection 8 degrees), positioned so the
  inviscid impingement lands at the interaction origin;
- the wall isothermal with the record's own measured wall-temperature row
  (recovery upstream of the thermal switch, the imposed s-condition
  downstream), the per-face wall-temperature profile of the pre-registration
  addendum; the adiabatic 2011 case runs at the recovery value everywhere,
  which is the mean-equivalent adiabatic state its own wall row measures;
- supersonic outflow (extrapolation) and a domain tall and long enough that
  the reflected wave leaves through the outflow upstream of the top boundary's
  influence returning to the scored wall region.

Units: the DNS records are nondimensional (free-stream density, velocity and
temperature one, lengths in incoming-layer thicknesses). The solver runs
DIMENSIONAL air (Sutherland viscosity), with the reference length chosen so
the incoming-layer Reynolds number matches the dataset (Re = 16750 on the
inlet thickness at Mach 2.28 over a 300 K free stream). SBLIUnits converts
both ways; the Sutherland-versus-power-law viscosity difference over the
interaction's temperature range is a documented baseline-model choice of a
few percent in mu.

Gate A (pre-registered): the attached configuration (no imposed shock) must
reproduce the measured incoming layer, skin friction within ten percent of
the dataset's 2.56e-3 at the incoming-profile station. Gate B: the coupled
interaction configuration converges (solver-classified) for every
wall-thermal case with the impingement within one reference length of the
DNS half-rise point, the offset reported. Both gates are measured by the
reproduce driver and recorded in the numbers JSON, never assumed.
"""
import os
import sys

import numpy as np

_BUILD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "..", "build")
if os.path.isdir(_BUILD) and _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)

import rans_sst_py as rans

MACH = 2.28
INCIDENCE_DEG = 8.0
GAMMA = 1.4
RE_INLET = 16750.0          # dataset Reynolds number on the inlet thickness
T_INF_K = 300.0             # dimensional free-stream temperature (air)


class SBLIUnits:
    """Conversion between the records' free-stream units and the dimensional
    solver configuration (air at 300 K, Sutherland viscosity), with the
    reference length set by the dataset Reynolds number."""

    def __init__(self, mach=MACH, re_inlet=RE_INLET, T_inf=T_INF_K,
                 p_inf=101325.0):
        self.eos = rans.IdealGasEOS()
        self.eos.gamma = GAMMA
        self.mach = float(mach)
        self.T_inf = float(T_inf)
        self.p_inf = float(p_inf)
        self.rho_inf = self.p_inf / (self.eos.R * self.T_inf)
        self.a_inf = np.sqrt(GAMMA * self.eos.R * self.T_inf)
        self.U_inf = self.mach * self.a_inf
        self.mu_inf = self.eos.viscosity(self.T_inf)
        # reference length from Re = rho U delta0 / mu
        self.delta0 = re_inlet * self.mu_inf / (self.rho_inf * self.U_inf)

    def length(self, x_star):
        return np.asarray(x_star, dtype=float) * self.delta0

    def velocity(self, u_hat):
        return np.asarray(u_hat, dtype=float) * self.U_inf

    def temperature(self, T_hat):
        return np.asarray(T_hat, dtype=float) * self.T_inf

    def density(self, rho_hat):
        return np.asarray(rho_hat, dtype=float) * self.rho_inf

    def pressure_from(self, rho_hat, T_hat):
        return self.density(rho_hat) * self.eos.R * self.temperature(T_hat)

    def freestream(self):
        V = rans.Primitive(self.rho_inf, self.U_inf, 0.0, self.p_inf)
        return V


def oblique_shock_state(units, deflection_deg=INCIDENCE_DEG):
    """Post-shock primitive state of the weak oblique shock deflecting the
    free stream DOWNWARD by the generator incidence (theta-beta-M).

    Returns (primitive, beta_rad): the uniform state the top boundary holds
    downstream of the shock's entry point, and the shock angle.
    """
    M1 = units.mach
    theta = np.deg2rad(deflection_deg)
    # weak-branch solve of tan(theta) = 2 cot(beta) (M1^2 sin^2 b - 1) /
    #                                   (M1^2 (gamma + cos 2b) + 2)
    betas = np.linspace(np.arcsin(1.0 / M1) + 1e-6, np.pi / 2 - 1e-6, 20000)
    lhs = np.tan(theta)
    rhs = (2.0 / np.tan(betas)
           * (M1 ** 2 * np.sin(betas) ** 2 - 1.0)
           / (M1 ** 2 * (GAMMA + np.cos(2.0 * betas)) + 2.0))
    idx = np.argmax(rhs >= lhs)     # first crossing = the weak solution
    beta = float(betas[idx])

    M1n = M1 * np.sin(beta)
    p_ratio = 1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (M1n ** 2 - 1.0)
    rho_ratio = ((GAMMA + 1.0) * M1n ** 2) / ((GAMMA - 1.0) * M1n ** 2 + 2.0)
    T_ratio = p_ratio / rho_ratio
    M2n = np.sqrt((1.0 + 0.5 * (GAMMA - 1.0) * M1n ** 2)
                  / (GAMMA * M1n ** 2 - 0.5 * (GAMMA - 1.0)))
    M2 = M2n / np.sin(beta - theta)
    a2 = units.a_inf * np.sqrt(T_ratio)
    speed = M2 * a2

    V = rans.Primitive(units.rho_inf * rho_ratio,
                       speed * np.cos(theta),
                       -speed * np.sin(theta),
                       units.p_inf * p_ratio)
    return V, beta


def _nearest_index(grid, query):
    """True nearest-cell index (a bare searchsorted picks the upper
    neighbor, a half-cell systematic bias that matters most in the clustered
    wall-normal direction)."""
    idx = np.clip(np.searchsorted(grid, query), 1, grid.size - 1)
    lower = idx - 1
    pick_lower = np.abs(query - grid[lower]) <= np.abs(grid[idx] - query)
    return np.where(pick_lower, lower, idx)


def _record_column(record, x_star):
    """Nearest field column of a loader record to the given coordinate."""
    return int(np.argmin(np.abs(record.x - x_star)))


def inflow_state_table(record, units, i_col, k_floor_hat=1e-8,
                       nu_t_cap_hat=2e-3):
    """The dimensional inflow state as a function of wall distance, built from
    the record's own field column: (y_dim, rho, u, v, p, k, omega) arrays.

    k comes from the record's stresses; omega from the shear-consistent eddy
    viscosity nu_t = -R_xy / (dU/dy) (floored against its degenerate rows and
    capped at nu_t_cap_hat in free-stream units), so the imposed layer carries
    the data's own turbulence state. Above the layer the state relaxes to the
    laminar free stream with ambient SST values.
    """
    y_hat = record.y
    u_hat = record.U[i_col].copy()
    v_hat = record.V[i_col].copy()
    T_hat = record.T[i_col].copy()
    rho_hat = record.rho[i_col].copy()
    R = record.R[i_col]                          # (ny, 3, 3) free-stream units
    k_hat = 0.5 * (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])
    k_hat = np.maximum(k_hat, k_floor_hat)

    dUdy = np.gradient(u_hat, y_hat)
    nu_t_hat = -R[:, 0, 1] / np.where(np.abs(dUdy) > 1e-8, dUdy, 1e-8)
    nu_t_hat = np.clip(nu_t_hat, 1e-9, nu_t_cap_hat)
    omega_hat = k_hat / nu_t_hat

    # ambient free-stream turbulence where the layer's signal dies out
    ambient = k_hat < 10.0 * k_floor_hat
    k_hat[ambient] = k_floor_hat
    omega_hat[ambient] = 5.0                     # U_inf / delta0 units

    y = units.length(y_hat)
    rho = units.density(rho_hat)
    u = units.velocity(u_hat)
    v = units.velocity(v_hat)
    p = np.full_like(y, units.p_inf)             # zero-pressure-gradient layer
    k = k_hat * units.U_inf ** 2
    omega = omega_hat * units.U_inf / units.delta0
    return y, rho, u, v, p, k, omega


class SBLIBaseline:
    """One density-based SST baseline configuration of the interaction.

    Static factories build the attached (gate A) and interaction (gate B)
    configurations from a loader record; solve() runs the implicit march;
    sample_fields() interpolates the converged baseline onto the record's
    (x, y) points for the discrepancy extraction; wall() reads the observation
    operator with the same per-face wall temperatures as the solve.
    """

    def __init__(self, record, units, mesh, bcs, settings, wall_temps_dim,
                 with_shock, meta):
        self.record = record
        self.units = units
        self.mesh = mesh
        self.bcs = bcs
        self.settings = settings
        self.wall_temps_dim = wall_temps_dim
        self.with_shock = with_shock
        self.meta = dict(meta)
        sst = rans.SSTCoefficients()
        self.solver = rans.DBNSSolver(mesh, units.eos, sst, bcs, settings)
        self.solver.init_uniform(units.freestream())
        self.report = None

    # ---- construction --------------------------------------------------------

    @staticmethod
    def _mesh_and_faces(units, x_lo, x_hi, height, nx, ny, yplus_target=1.0):
        Lx = units.length(x_hi - x_lo)
        H = units.length(height)
        # one-sided plate clustering: the two-sided channel mapping wastes
        # half its refinement on the far-field top and leaves the buffer
        # layer too coarse for the transported omega (measured: the SST
        # settles over-mixed and the skin friction runs high, converging as
        # the near-wall growth ratio drops)
        mesh = rans.Mesh.make_plate_2d(nx, ny, Lx, H, RE_INLET, yplus_target)
        return mesh, Lx, H

    @staticmethod
    def configure(record, s_case=None, x_lo=None, x_hi=14.0, height=8.0,
                  nx=480, ny=96, with_shock=True, cfl=200.0,
                  max_iterations=60000, convergence_tol=1e-6,
                  yplus_target=1.0, verbose=False, report_interval=1000,
                  early_abort_iter=0, early_abort_rel_max=0.0):
        """Build the configuration for one record.

        The domain spans [x_lo, x_hi] x [0, height] reference lengths, x_lo
        defaulting to the record's own first column; the mesh x-origin sits at
        x_lo, so solver x = (x* - x_lo) delta0. Shock geometry: entry at the
        top such that the inviscid impingement lands at x* = 0.
        """
        units = SBLIUnits()
        if x_lo is None:
            x_lo = float(record.x[0])
        mesh, Lx, H = SBLIBaseline._mesh_and_faces(units, x_lo, x_hi, height,
                                                   nx, ny, yplus_target)

        bcs = rans.DBNSBoundaryConditions()

        # inflow: the record's own first column as a per-face profile
        i_col = _record_column(record, x_lo)
        y, rho, u, v, p, k, om = inflow_state_table(record, units, i_col)
        inflow = rans.DBNSBoundarySpec()
        inflow.kind = rans.DBNSBoundaryKind.SupersonicInflow
        inflow.freestream = units.freestream()
        centers = mesh.patch_face_centers("inlet") \
            if hasattr(mesh, "patch_face_centers") else None
        if centers is None:
            # inlet faces of the channel mesh run bottom to top at x = 0 with
            # ny faces; their centers follow the wall-clustered node spacing,
            # recovered from the cell centers' unique y values
            cc = mesh.cell_centers()
            ys = np.unique(np.round(cc[:, 1], 12))
        else:
            ys = centers[:, 1]
        prof = np.zeros((ys.size, 6))
        prof[:, 0] = np.interp(ys, y, rho, left=rho[0], right=units.rho_inf)
        prof[:, 1] = np.interp(ys, y, u, left=0.0, right=units.U_inf)
        prof[:, 2] = np.interp(ys, y, v, left=0.0, right=0.0)
        prof[:, 3] = np.interp(ys, y, p, left=p[0], right=units.p_inf)
        prof[:, 4] = np.interp(ys, y, k, left=k[0], right=k[-1])
        prof[:, 5] = np.interp(ys, y, om, left=om[0], right=om[-1])
        inflow.set_profile(prof)
        bcs.set("inlet", inflow)

        # top: freestream upstream of the shock entry, post-shock downstream
        top = rans.DBNSBoundarySpec()
        if with_shock:
            post, beta = oblique_shock_state(units)
            x_entry_star = 0.0 - height / np.tan(beta)
            top.kind = rans.DBNSBoundaryKind.FixedState
            top.freestream = units.freestream()
            xs_top = np.unique(np.round(mesh.cell_centers()[:, 0], 12))
            tprof = np.zeros((xs_top.size, 6))
            x_entry_dim = units.length(x_entry_star - x_lo)
            for j, xc in enumerate(xs_top):
                V = units.freestream() if xc < x_entry_dim else post
                tprof[j] = [V.rho, V.u, V.v, V.p, 0.0, 0.0]
            top.set_profile(tprof)
            meta_shock = {"beta_deg": float(np.rad2deg(beta)),
                          "x_entry_star": float(x_entry_star)}
        else:
            top.kind = rans.DBNSBoundaryKind.Extrapolate
            meta_shock = {}
        bcs.set("top", top)

        out = rans.DBNSBoundarySpec()
        out.kind = rans.DBNSBoundaryKind.Extrapolate
        bcs.set("outlet", out)

        # wall thermal condition, per the record's own physics:
        # - the wall-thermal (heated/cooled) campaigns are controlled
        #   isothermal experiments, so the wall prescribes the record's
        #   measured temperature row (recovery upstream of the thermal
        #   switch, the s-condition after);
        # - the 2011 adiabatic campaign is a genuinely adiabatic DNS, so the
        #   wall is the solver's zero-heat-flux condition. Prescribing the
        #   DNS recovery temperature there (the previous convention) forces
        #   the MODEL to a nonzero wall heat flux wherever its own recovery
        #   differs from the DNS value, which is a matched-thermal-state
        #   sensitivity, not an adiabatic wall; the review flagged it and the
        #   gate adjudication runs on the honest condition.
        wall = rans.DBNSBoundarySpec()
        if record.meta.get("wall_thermal") == "adiabatic":
            wall.kind = rans.DBNSBoundaryKind.NoSlipAdiabatic
            wall_temps_dim = None
        else:
            wall.kind = rans.DBNSBoundaryKind.NoSlipIsothermal
            Tw_row_hat = record.wall_temperature_row()
            xs_wall = np.unique(np.round(mesh.cell_centers()[:, 0], 12))
            xw_star = xs_wall / units.delta0 + x_lo
            Tw_faces_hat = np.interp(xw_star, record.x, Tw_row_hat,
                                     left=Tw_row_hat[0], right=Tw_row_hat[-1])
            wall_temps_dim = units.temperature(Tw_faces_hat)
            wall.wall_temp = float(wall_temps_dim[0])
            wall.set_wall_temp_profile(wall_temps_dim)
        bcs.set("bottom", wall)

        st = rans.DBNSSettings()
        st.verbose = verbose
        st.report_interval = report_interval
        st.early_abort_iter = early_abort_iter
        st.early_abort_rel_max = early_abort_rel_max
        st.time_mode = rans.TimeMode.Steady
        st.implicit_steady = True
        st.cfl_implicit = cfl
        st.cfl_ramp_start = 1.0
        st.cfl_ramp_iters = 500
        st.viscous = True
        st.turbulent = True
        st.reconstruct_order = 2
        st.limit_reconstruction = True
        st.convergence_tol = convergence_tol
        st.max_iterations = max_iterations

        meta = {"case": record.case, "x_lo": float(x_lo),
                "x_hi": float(x_hi), "height": float(height),
                "nx": nx, "ny": ny, **meta_shock}
        return SBLIBaseline(record, units, mesh, bcs, st, wall_temps_dim,
                            with_shock, meta)

    # ---- run and read ---------------------------------------------------------

    def solve(self):
        self.report = self.solver.solve()
        return self.report

    def wall(self):
        """Observation record on the wall with the solve's own per-face
        temperatures; x returned in interaction coordinates."""
        ref = rans.ReferenceState()
        ref.rho = self.units.rho_inf
        ref.U = self.units.U_inf
        ref.T = self.units.T_inf
        ref.p = self.units.p_inf
        # the data's own turbulent recovery (the measured wall temperature
        # 1.9318 T_inf implies r = 0.896); the operator's default is the
        # laminar sqrt(Pr), which would bias the heated cases' Stanton
        ref.recovery_factor = (1.9318 - 1.0) / (0.2 * MACH ** 2)
        obs = rans.DBNSObservation(self.solver, ref)
        if self.wall_temps_dim is None:
            # adiabatic wall: the wall temperature is the SOLVED near-wall
            # value (the zero-heat-flux condition makes the face temperature
            # the zero-gradient extrapolation), so the reported qw is the
            # discrete residual of the adiabatic condition (near zero) and
            # St reads against the solved recovery temperature
            cc = np.asarray(self.mesh.cell_centers())
            xs_wall = np.unique(np.round(cc[:, 0], 12))
            y1_star = float(cc[:, 1].min()) / self.units.delta0
            xw_star = xs_wall / self.units.delta0 + self.meta["x_lo"]
            s = self.sample_fields(xw_star, np.full(xs_wall.size, y1_star))
            temps = self.units.temperature(np.asarray(s["T"]))
        else:
            temps = self.wall_temps_dim
        w = obs.wall_profile("bottom", temps, float(temps[0]))
        x_star = np.asarray(w["x"]) / self.units.delta0 + self.meta["x_lo"]
        return {"x_star": x_star, "Cf": np.asarray(w["Cf"]),
                "Cp": np.asarray(w["Cp"]), "qw": np.asarray(w["qw"]),
                "St": np.asarray(w["St"])}

    def sample_fields(self, x_star, y_star):
        """Baseline fields at record points (nondimensionalized back to the
        record's free-stream units), nearest-cell sampled on the structured
        grid: dict of u, v, T, rho, k, omega, nu_t, plus the velocity-gradient
        tensor entries from the same grid.
        """
        u = self.units
        cc = self.mesh.cell_centers()
        xs = np.unique(np.round(cc[:, 0], 12))
        ys = np.unique(np.round(cc[:, 1], 12))
        nxm, nym = xs.size, ys.size
        f = self.solver.fields()

        # recover the structured (nx, ny) layout independent of the mesh's
        # own cell numbering: sort by y then x
        order = np.lexsort((cc[:, 0], cc[:, 1]))
        def as_grid(vals):
            g = np.asarray(vals)[order].reshape(nym, nxm).T   # (nx, ny)
            return g

        gu = as_grid(f["u"]); gv = as_grid(f["v"])
        gT = as_grid(f["T"]); grho = as_grid(f["rho"])
        gk = as_grid(f["k"]); gom = as_grid(f["omega"])
        gmuT = as_grid(f["muT"])

        xq = u.length(np.asarray(x_star) - self.meta["x_lo"])
        yq = u.length(np.asarray(y_star))
        ix = _nearest_index(xs, xq)
        iy = _nearest_index(ys, yq)

        dx = np.gradient(xs); dy = np.gradient(ys)
        dudx = np.gradient(gu, axis=0) / dx[:, None]
        dudy = np.gradient(gu, axis=1) / dy[None, :]
        dvdx = np.gradient(gv, axis=0) / dx[:, None]
        dvdy = np.gradient(gv, axis=1) / dy[None, :]

        pick = (ix, iy)
        out = {
            "u": gu[pick] / u.U_inf,
            "v": gv[pick] / u.U_inf,
            "T": gT[pick] / u.T_inf,
            "rho": grho[pick] / u.rho_inf,
            "k": gk[pick] / u.U_inf ** 2,
            "omega": gom[pick] * u.delta0 / u.U_inf,
            "nu_t": (gmuT[pick] / grho[pick])
                    / (u.U_inf * u.delta0),
            "dudx": dudx[pick] * u.delta0 / u.U_inf,
            "dudy": dudy[pick] * u.delta0 / u.U_inf,
            "dvdx": dvdx[pick] * u.delta0 / u.U_inf,
            "dvdy": dvdy[pick] * u.delta0 / u.U_inf,
        }
        return out
