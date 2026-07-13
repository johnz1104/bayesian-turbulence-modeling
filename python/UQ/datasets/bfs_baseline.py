"""Boussinesq SST baseline field for the Le-Moin backward-facing step.

This is the baseline the model-form discrepancy is formed against. Unlike the
wall-bounded channel baseline (a single developed column), the backward-facing step
is a genuine two-dimensional field, so the whole converged SST field (U, V, k,
omega, nu_t) is kept and interpolated to the DNS profile stations. The eddy
viscosity and the turbulence timescale come from the converged solve, not a
correlation.

Geometry (Le, Moin and Kim 1997): step height h = 1 (reference length), expansion
ratio 1.2, so the downstream channel height is 6 h and the inlet channel height is
5 h, at Re_h = U0 h / nu = 5100.

Boundary conditions match the DNS setup, both verified against the data:

- Top boundary: FREE-SLIP (symmetry), not a wall. The published profiles carry
  zero mean shear at the top edge at every station (|dU/dy| < 4e-4 over the top
  0.37 h) with U at free-stream level, which identifies a zero-stress boundary; a
  no-slip top would drive U to zero and grow a spurious top boundary layer (the
  earlier no-slip baseline had U = 0.48 against the DNS 0.94 at y/h = 5.9).
- Inflow: at the DNS inflow plane x/h = -10, a prescribed turbulent
  boundary-layer profile (the measured x/h = -3 station shape rescaled to an
  inlet thickness delta_in), not a uniform stream. The DNS boundary layer at
  x/h = -3 is delta_999 = 1.158 h (stat-inf.dat); a uniform inlet at x = -4
  develops only a thin layer by the step and mis-states the attached-flow
  discrepancy. delta_in is matched so the solved delta_999 at x/h = -3
  reproduces the measured one (match_inlet_delta); this matches the boundary
  CONDITIONS of the comparison to the data and touches no pre-registered
  quantity. The free stream is quiet (k from the measured top-row fluctuation
  level), as in the DNS.

The channel-calibrated SST still under-predicts the reattachment, which is the
model-form error the discrepancy quantifies.

Coordinate convention. The mesh places the step at x = 0, the downstream bottom
wall at y = 0, and the step-top (the upstream bottom wall) at y = h. The DNS profile
y is measured from the local wall (the downstream stations from y = 0, the upstream
station from the step-top), so the upstream station maps to the mesh by adding h to
its y (map_dns_points).
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# the inference layer is solver-coupled; import lazily so importing this module
# never forces the built extension to be present.
from solver_bindings import _rs
from bfs_reference import _bfs_solver_settings

_CMU = 0.09     # Boussinesq C_mu, for the turbulence timescale tau = 1/(C_mu omega)
_BETA1 = 0.075  # Menter beta1, for the viscous-sublayer omega ~ 6 nu/(beta1 y^2)
_KAPPA = 0.41   # von Karman, for the log-layer omega = u_tau/(sqrt(C_mu) kappa y)


class BFSBaselineRANS:
    """One converged SST backward-facing-step field (Le-Moin geometry).

    Public 2-D field (per cell): x, y, U, V, k, omega, nu_t. Scalars: nu,
    reattachment (the solver QoI), status, iterations. The field is interpolated
    to arbitrary query points (sample_at), and its velocity gradient and turbulence
    timescale are evaluated there (velocity_gradient_at, timescale_at) for the
    Boussinesq baseline anisotropy and the invariant conditioning features.
    """

    STEP_H = 1.0             # step height h (reference length)
    EXPANSION_H = 6.0        # downstream channel height (expansion ratio 1.2 -> 6 h)
    RE_H = 5100.0            # Re_h = U0 h / nu (Le, Moin and Kim 1997)
    U_BULK = 1.0
    REATTACH_XR_H = 6.28     # published DNS reattachment, for reference

    # measured inflow anchors at x/h = -3 (stat-inf.dat): the boundary-layer
    # thickness the solved inlet layer must reproduce there, and the friction
    # velocity used to synthesize the inlet omega profile.
    DNS_DELTA999 = 1.1583
    DNS_UTAU = 0.0485
    # quiet free stream, from the measured top-row fluctuation level at x/h = -3
    # (u' ~ 2.9e-4 U0): k_fs = 0.5 (u'^2 + v'^2 + w'^2) ~ 5e-8 U0^2.
    K_FREESTREAM = 5.0e-8

    # sized for a converged separated field in about a minute; tests override with
    # a coarse, fast config that only exercises the machinery. inlet_delta is the
    # prescribed inlet boundary-layer thickness at x/h = -10, matched by
    # match_inlet_delta on this grid so the solved delta_999 at x/h = -3
    # reproduces DNS_DELTA999 (0.6784 develops into 1.186 there, 2.4 percent
    # above the measured 1.1583). inlet_delta None gives the legacy uniform
    # inlet, slip_top False the legacy no-slip top (regression comparison only).
    DEFAULT_CONFIG = {"nx_up": 40, "nx_down": 48, "ny_up": 24, "ny_down": 18,
                      "Lu": 10.0, "Ld": 22.0, "max_iter": 12000, "conv_tol": 1.0e-4,
                      "slip_top": True, "inlet_delta": 0.6784}

    def __init__(self, x, y, U, V, k, omega, nu_t, nu, reattachment,
                 status, iterations, meta):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.U = np.asarray(U, float)
        self.V = np.asarray(V, float)
        self.k = np.asarray(k, float)
        self.omega = np.asarray(omega, float)
        self.nu_t = np.asarray(nu_t, float)
        self.nu = float(nu)
        self.reattachment = float(reattachment)
        self.status = status
        self.iterations = int(iterations)
        self.meta = meta or {}
        # one linear interpolator over the unstructured cell centers, all fields at
        # once (columns U, V, k, omega, nu_t); NaN is returned outside the domain.
        self._pts = np.column_stack([self.x, self.y])
        self._vals = np.column_stack([self.U, self.V, self.k, self.omega, self.nu_t])
        self._interp = LinearNDInterpolator(self._pts, self._vals)

    # ---- solving ----------------------------------------------------------

    @staticmethod
    def inlet_profiles(y_local, dns_station, delta_in, nu, u_tau,
                       k_fs=None, om_cap=None):
        """Inlet U, k, omega at wall-normal offsets y_local, for thickness delta_in.

        U and k take the measured x/h = -3 station shape rescaled from the
        measured delta_999 to delta_in (free stream exactly U0 = 1 and the quiet
        k_fs outside the layer). omega is synthesized (it is not measured) from
        the standard two-layer form the SST wall treatment itself uses,
          omega_vis = 6 nu / (beta1 y^2),  omega_log = u_tau / (sqrt(C_mu) kappa y),
          omega = sqrt(omega_vis^2 + omega_log^2),
        with y capped at delta_in so omega stays at its edge value outside the
        layer instead of decaying to zero or being pinned artificially high.
        """
        y_local = np.asarray(y_local, float)
        k_fs = BFSBaselineRANS.K_FREESTREAM if k_fs is None else float(k_fs)
        scale = BFSBaselineRANS.DNS_DELTA999 / float(delta_in)

        y_dns = dns_station["y"]
        # BL shape normalized so the free stream is exactly 1 (U_e at -3 is 0.9993)
        U_shape = dns_station["U"] / np.max(dns_station["U"])
        k_shape = dns_station["k"]

        y_eq = y_local * scale        # equivalent y in the measured profile
        U = np.interp(y_eq, y_dns, U_shape, right=1.0)
        k = np.interp(y_eq, y_dns, k_shape, right=k_fs)
        k = np.maximum(k, k_fs)

        y_eff = np.minimum(np.maximum(y_local, 1e-12), float(delta_in))
        om_vis = 6.0 * nu / (_BETA1 * y_eff ** 2)
        om_log = u_tau / (np.sqrt(_CMU) * _KAPPA * y_eff)
        omega = np.sqrt(om_vis ** 2 + om_log ** 2)
        if om_cap is not None:
            omega = np.minimum(omega, float(om_cap))
        return U, k, omega

    @staticmethod
    def build_forward(cfg=None, dns=None):
        """Build the Le-Moin BFS forward model without solving.

        Returns a dict holding the ForwardModel and every object it references
        (mesh, bcs, obs, ps, settings), which the caller must keep alive as one
        unit while the model is used (the model stores references, not copies).
        Shared by the baseline solve and the a-posteriori injection wrapper so
        both run the identical setup.
        """
        cfg = {**BFSBaselineRANS.DEFAULT_CONFIG, **(cfg or {})}
        rs = _rs()
        h, H, Re, Ub = (BFSBaselineRANS.STEP_H, BFSBaselineRANS.EXPANSION_H,
                        BFSBaselineRANS.RE_H, BFSBaselineRANS.U_BULK)
        nu = Ub * h / Re
        mesh = rs.Mesh.make_backward_facing_step_2d(
            cfg["nx_up"], cfg["nx_down"], cfg["ny_up"], cfg["ny_down"],
            cfg["Lu"], cfg["Ld"], h, H, Re=Re, yPlusTarget=1.0)
        if cfg.get("slip_top", False):
            # the Le-Moin top boundary is free-slip (zero measured shear at the
            # top edge of every station); retype it so the BC factory assigns the
            # symmetry BCs and the wall distance sees only the true walls
            mesh.set_patch_type("top_wall", "symmetry")
        mesh.compute_wall_distance()

        delta_in = cfg.get("inlet_delta", None)
        if delta_in is not None:
            # quiet free stream, as measured; the inlet layer carries the
            # turbulence. omega edge value at the layer edge anchors the ambient.
            kIn = BFSBaselineRANS.K_FREESTREAM
            omIn = BFSBaselineRANS.DNS_UTAU / (np.sqrt(_CMU) * _KAPPA * float(delta_in))
        else:
            Tu = 0.05
            kIn = 1.5 * (Ub * Tu) ** 2
            omIn = kIn / (nu * 100.0)
        bcs = rs.FlowBoundaryConditions.bfs_defaults(mesh, Ub, kIn, omIn)

        if delta_in is not None:
            if dns is None:
                from .backward_facing_step import BackwardFacingStepDNS
                dns = BackwardFacingStepDNS.load()
            station = dns.stations[0]           # the x/h = -3 profile shape
            inlet = mesh.wall_patch_data("inlet")
            y_face = np.asarray(inlet["center"])[:, 1] - h   # offset above step-top
            Uin, kin, omin = BFSBaselineRANS.inlet_profiles(
                y_face, station, delta_in, nu, BFSBaselineRANS.DNS_UTAU)
            vel = np.column_stack([Uin, np.zeros_like(Uin), np.zeros_like(Uin)])
            bcs.set_velocity_profile(mesh, "inlet", vel)
            bcs.set_k_profile(mesh, "inlet", kin)
            bcs.set_omega_profile(mesh, "inlet", omin)

        # the reattachment QoI keeps the ForwardModel well-formed; the baseline reads
        # last_fields(), and the reattachment prediction is a useful diagnostic.
        obs = rs.ObservationOperator()
        obs.add_reattachment_length("bottom_wall_down",
                                    xr_obs=BFSBaselineRANS.REATTACH_XR_H, sigma=0.5)
        ps = rs.InferenceParameterSet.a1_betaStar()
        settings = _bfs_solver_settings(rs)
        settings.max_iterations = cfg["max_iter"]
        settings.convergence_tol = cfg["conv_tol"]
        # under-relaxation of the explicit Reynolds-stress-injection source
        # (harmless when nothing is injected); strongly perturbed sampled
        # closures may need a smaller value than the 0.3 default
        settings.alpha_injection = cfg.get("alpha_injection", 0.3)
        # standing probe hook: Rhie-Chow face dissipation on all meshes
        # (default off; the separated-flow off/on adjudication sets it)
        settings.rhie_chow_all_meshes = cfg.get("rhie_chow_all_meshes", False)

        fm = rs.ForwardModel(mesh, ps, obs, bcs, nu, settings,
                             rs.Vec3(Ub, 0.0, 0.0), 0.0, kIn, omIn)
        return {"rs": rs, "fm": fm, "mesh": mesh, "bcs": bcs, "obs": obs,
                "ps": ps, "settings": settings, "nu": nu, "kIn": kIn,
                "omIn": omIn, "cfg": cfg}

    @staticmethod
    def solve(cfg=None, dns=None):
        """Solve the Le-Moin BFS at Menter defaults, return the baseline field."""
        fwd = BFSBaselineRANS.build_forward(cfg, dns=dns)
        rs, fm, mesh, cfg = fwd["rs"], fwd["fm"], fwd["mesh"], fwd["cfg"]
        nu = fwd["nu"]
        ps = fwd["ps"]
        theta = list(ps.pack(rs.SSTCoefficients()))
        result = fm.evaluate(theta)
        status = str(result.status).split(".")[-1]
        if not fm.has_last_fields():
            raise ValueError(f"BFS baseline solve produced no fields (status {status})")

        ff = fm.last_fields()
        cc = np.asarray(mesh.cell_centers())
        Uf = np.asarray(ff["U"])
        if Uf.ndim == 1:
            Uf = Uf.reshape(-1, 3)
        delta_in = cfg.get("inlet_delta", None)
        meta = {
            "config": dict(cfg),
            "case": "backward_facing_step",
            "geometry": "Le-Moin, Re_h=5100, expansion ratio 1.2",
            "note": ("free-slip top, prescribed inflow boundary layer"
                     if delta_in is not None and cfg.get("slip_top", False)
                     else "developing inlet/outlet BFS; channel-calibrated SST"),
        }
        return BFSBaselineRANS(
            x=cc[:, 0], y=cc[:, 1], U=Uf[:, 0], V=Uf[:, 1],
            k=np.asarray(ff["k"]), omega=np.asarray(ff["omega"]),
            nu_t=np.asarray(ff["nuT"]), nu=nu,
            reattachment=float(result.predictions[0]), status=status,
            iterations=cfg["max_iter"], meta=meta)

    def delta_999_at(self, x, n_y=400):
        """Solved boundary-layer thickness delta_999 above the local bottom wall.

        Samples U on a fine wall-normal grid at station x, takes the edge
        velocity as the maximum of the column (the free stream under a slip
        top), and returns the smallest wall offset where U first reaches
        0.999 U_e, by linear interpolation between samples.
        """
        y_wall = self.STEP_H if x < 0.0 else 0.0
        y_top = self.EXPANSION_H
        ys = np.linspace(y_wall + 1e-3, y_top - 1e-3, n_y)
        U = self.sample_at(np.full(n_y, float(x)), ys)["U"]
        good = np.isfinite(U)
        ys, U = ys[good], U[good]
        Ue = np.max(U)
        target = 0.999 * Ue
        above = np.nonzero(U >= target)[0]
        if above.size == 0:
            return float("nan")
        j = above[0]
        if j == 0:
            return ys[0] - y_wall
        # linear interpolation to the crossing
        f = (target - U[j - 1]) / max(U[j] - U[j - 1], 1e-30)
        return (ys[j - 1] + f * (ys[j] - ys[j - 1])) - y_wall

    @staticmethod
    def match_inlet_delta(cfg=None, target=None, tol=0.02, max_iter=4, dns=None):
        """Secant on the inlet thickness so delta_999 at x/h = -3 matches the DNS.

        Returns (delta_in, achieved_delta999, baseline). This matches the inflow
        boundary condition to the measured layer thickness (a data anchor), and
        touches no pre-registered quantity.
        """
        target = BFSBaselineRANS.DNS_DELTA999 if target is None else float(target)
        base_cfg = {**BFSBaselineRANS.DEFAULT_CONFIG, **(cfg or {})}

        def run(delta_in):
            b = BFSBaselineRANS.solve({**base_cfg, "inlet_delta": delta_in}, dns=dns)
            return b, b.delta_999_at(-3.0)

        d0 = float(base_cfg.get("inlet_delta") or 1.05)
        b0, m0 = run(d0)
        if abs(m0 - target) <= tol * target:
            return d0, m0, b0
        # the solved thickness grows nearly one-for-one with the inlet thickness,
        # so a secant from a shifted second point converges in one or two steps
        d1 = d0 * target / max(m0, 1e-9)
        b1, m1 = run(d1)
        for _ in range(max_iter):
            if abs(m1 - target) <= tol * target:
                break
            denom = (m1 - m0)
            if abs(denom) < 1e-9:
                break
            d2 = d1 + (target - m1) * (d1 - d0) / denom
            d2 = float(np.clip(d2, 0.2, 3.0))
            d0, m0 = d1, m1
            d1 = d2
            b1, m1 = run(d1)
        return d1, m1, b1

    # ---- interpolation to query points ------------------------------------

    def sample_at(self, xq, yq):
        """Interpolate U, V, k, omega, nu_t to the query points (NaN outside)."""
        q = np.column_stack([np.asarray(xq, float), np.asarray(yq, float)])
        v = self._interp(q)
        return {"U": v[:, 0], "V": v[:, 1], "k": v[:, 2],
                "omega": v[:, 3], "nu_t": v[:, 4]}

    def velocity_gradient_at(self, xq, yq, dx=0.03, dy=None):
        """grad_u[i, j] = d u_i / d x_j at the query points by central differences.

        The interpolant is evaluated at the four neighbours (x +/- dx, y +/- dy);
        where a neighbour falls outside the domain (NaN) a one-sided difference is
        used, and a fully surrounded-by-NaN point yields zero (it is masked out
        downstream). Only the in-plane derivatives of U and V are formed (the mean
        is two-dimensional and spanwise-homogeneous).

        dx and dy may be scalars or per-point arrays. The wall-normal step dy
        (default: dx) should shrink toward the wall: a fixed 0.03 step spans many
        near-wall cells of the stretched mesh and smears the sublayer shear the
        discrepancy features depend on, so callers pass dy of the order of the
        local wall distance there (see BFSDiscrepancy.build).
        """
        xq = np.asarray(xq, float)
        yq = np.asarray(yq, float)
        dx = np.broadcast_to(np.asarray(dx, float), xq.shape)
        dy = dx if dy is None else np.broadcast_to(np.asarray(dy, float), yq.shape)

        def uv(x, y):
            v = self._interp(np.column_stack([x, y]))
            return v[:, 0], v[:, 1]

        U0, V0 = uv(xq, yq)
        Uxp, Vxp = uv(xq + dx, yq)
        Uxm, Vxm = uv(xq - dx, yq)
        Uyp, Vyp = uv(xq, yq + dy)
        Uym, Vym = uv(xq, yq - dy)

        def deriv(fp, fm, f0, step):
            # central where both sides valid; one-sided if one side is NaN; else 0
            d = np.where(np.isfinite(fp) & np.isfinite(fm), (fp - fm) / (2.0 * step),
                         np.where(np.isfinite(fp) & np.isfinite(f0), (fp - f0) / step,
                                  np.where(np.isfinite(fm) & np.isfinite(f0),
                                           (f0 - fm) / step, 0.0)))
            return d

        g = np.zeros((xq.size, 3, 3))
        g[:, 0, 0] = deriv(Uxp, Uxm, U0, dx)      # dU/dx
        g[:, 0, 1] = deriv(Uyp, Uym, U0, dy)      # dU/dy
        g[:, 1, 0] = deriv(Vxp, Vxm, V0, dx)      # dV/dx
        g[:, 1, 1] = deriv(Vyp, Vym, V0, dy)      # dV/dy
        return g

    def timescale_at(self, xq, yq):
        """Baseline turbulence timescale tau = 1/(C_mu omega) at the query points.

        This is the timescale the discrepancy non-dimensionalisation expects, so the
        linear-eddy-viscosity anisotropy b = -C_mu dev(S) reproduces the k-omega
        Boussinesq anisotropy. The omega form is bounded through the near-wall region.
        """
        om = self.sample_at(xq, yq)["omega"]
        return 1.0 / (_CMU * np.maximum(om, 1e-30))

    def map_dns_points(self, x_h, y_h):
        """Map DNS profile coordinates (x/h, y/h) to the mesh frame.

        x is common (the step is at x = 0). The DNS y is measured from the local
        wall: the downstream stations from the bottom wall (y = 0), the upstream
        station from the step-top (y = h), so upstream points are shifted up by h.
        """
        x_h = np.asarray(x_h, float)
        y_h = np.asarray(y_h, float)
        y_mesh = y_h + np.where(x_h < 0.0, self.STEP_H, 0.0)
        return x_h.copy(), y_mesh

    def __repr__(self):
        return (f"BFSBaselineRANS(Re_h={self.RE_H:.0f}, n_cells={self.x.size}, "
                f"x_r/h={self.reattachment:.2f}, status={self.status})")
