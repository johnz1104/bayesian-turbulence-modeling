"""Boussinesq SST baseline field for the Le-Moin backward-facing step.

This is the baseline the model-form discrepancy is formed against. Unlike the
wall-bounded channel baseline (a single developed column), the backward-facing step
is a genuine two-dimensional field, so the whole converged SST field (U, V, k,
omega, nu_t) is kept and interpolated to the DNS profile stations. The eddy
viscosity and the turbulence timescale come from the converged solve, not a
correlation.

Geometry (Le, Moin and Kim 1997): step height h = 1 (reference length), expansion
ratio 1.2, so the downstream channel height is 6 h and the inlet channel height is
5 h, at Re_h = U0 h / nu = 5100. The incompressible solver runs a developing
inlet/outlet domain, and the channel-calibrated SST under-predicts the reattachment
(about 5.1 h here against the DNS 6.28 h), which is the model-form error the
discrepancy quantifies.

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

_CMU = 0.09  # Boussinesq C_mu, for the turbulence timescale tau = 1/(C_mu omega)


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

    # sized for a converged separated field in ~30 s; tests override with a coarse,
    # fast config that only exercises the machinery.
    DEFAULT_CONFIG = {"nx_up": 24, "nx_down": 48, "ny_up": 24, "ny_down": 18,
                      "Lu": 4.0, "Ld": 22.0, "max_iter": 12000, "conv_tol": 1.0e-4}

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
    def solve(cfg=None):
        """Solve the Le-Moin BFS at Menter defaults, return the baseline field."""
        cfg = {**BFSBaselineRANS.DEFAULT_CONFIG, **(cfg or {})}
        rs = _rs()
        h, H, Re, Ub = (BFSBaselineRANS.STEP_H, BFSBaselineRANS.EXPANSION_H,
                        BFSBaselineRANS.RE_H, BFSBaselineRANS.U_BULK)
        nu = Ub * h / Re
        mesh = rs.Mesh.make_backward_facing_step_2d(
            cfg["nx_up"], cfg["nx_down"], cfg["ny_up"], cfg["ny_down"],
            cfg["Lu"], cfg["Ld"], h, H, Re=Re, yPlusTarget=1.0)
        mesh.compute_wall_distance()

        Tu = 0.05
        kIn = 1.5 * (Ub * Tu) ** 2
        omIn = kIn / (nu * 100.0)
        bcs = rs.FlowBoundaryConditions.bfs_defaults(mesh, Ub, kIn, omIn)
        # the reattachment QoI keeps the ForwardModel well-formed; the baseline reads
        # last_fields(), and the reattachment prediction is a useful diagnostic.
        obs = rs.ObservationOperator()
        obs.add_reattachment_length("bottom_wall_down",
                                    xr_obs=BFSBaselineRANS.REATTACH_XR_H, sigma=0.5)
        ps = rs.InferenceParameterSet.a1_betaStar()
        settings = _bfs_solver_settings(rs)
        settings.max_iterations = cfg["max_iter"]
        settings.convergence_tol = cfg["conv_tol"]

        fm = rs.ForwardModel(mesh, ps, obs, bcs, nu, settings,
                             rs.Vec3(Ub, 0.0, 0.0), 0.0, kIn, omIn)
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
        meta = {
            "config": dict(cfg),
            "case": "backward_facing_step",
            "geometry": "Le-Moin, Re_h=5100, expansion ratio 1.2",
            "note": "developing inlet/outlet BFS; channel-calibrated SST",
        }
        return BFSBaselineRANS(
            x=cc[:, 0], y=cc[:, 1], U=Uf[:, 0], V=Uf[:, 1],
            k=np.asarray(ff["k"]), omega=np.asarray(ff["omega"]),
            nu_t=np.asarray(ff["nuT"]), nu=nu,
            reattachment=float(result.predictions[0]), status=status,
            iterations=cfg["max_iter"], meta=meta)

    # ---- interpolation to query points ------------------------------------

    def sample_at(self, xq, yq):
        """Interpolate U, V, k, omega, nu_t to the query points (NaN outside)."""
        q = np.column_stack([np.asarray(xq, float), np.asarray(yq, float)])
        v = self._interp(q)
        return {"U": v[:, 0], "V": v[:, 1], "k": v[:, 2],
                "omega": v[:, 3], "nu_t": v[:, 4]}

    def velocity_gradient_at(self, xq, yq, dx=0.03):
        """grad_u[i, j] = d u_i / d x_j at the query points by central differences.

        The interpolant is evaluated at the four neighbours (x +/- dx, y +/- dx);
        where a neighbour falls outside the domain (NaN) a one-sided difference is
        used, and a fully surrounded-by-NaN point yields zero (it is masked out
        downstream). Only the in-plane derivatives of U and V are formed (the mean
        is two-dimensional and spanwise-homogeneous).
        """
        xq = np.asarray(xq, float)
        yq = np.asarray(yq, float)

        def uv(x, y):
            v = self._interp(np.column_stack([x, y]))
            return v[:, 0], v[:, 1]

        U0, V0 = uv(xq, yq)
        Uxp, Vxp = uv(xq + dx, yq)
        Uxm, Vxm = uv(xq - dx, yq)
        Uyp, Vyp = uv(xq, yq + dx)
        Uym, Vym = uv(xq, yq - dx)

        def deriv(fp, fm, f0):
            # central where both sides valid; one-sided if one side is NaN; else 0
            d = np.where(np.isfinite(fp) & np.isfinite(fm), (fp - fm) / (2.0 * dx),
                         np.where(np.isfinite(fp) & np.isfinite(f0), (fp - f0) / dx,
                                  np.where(np.isfinite(fm) & np.isfinite(f0),
                                           (f0 - fm) / dx, 0.0)))
            return d

        g = np.zeros((xq.size, 3, 3))
        g[:, 0, 0] = deriv(Uxp, Uxm, U0)      # dU/dx
        g[:, 0, 1] = deriv(Uyp, Uym, U0)      # dU/dy
        g[:, 1, 0] = deriv(Vxp, Vxm, V0)      # dV/dx
        g[:, 1, 1] = deriv(Vyp, Vym, V0)      # dV/dy
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
