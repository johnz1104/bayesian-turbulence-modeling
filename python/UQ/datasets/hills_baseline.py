"""Boussinesq SST baseline field for the periodic hills (Xiao et al. 2020).

The dense-field companion of the backward-facing-step baseline: the same
converged-field record (x, y, U, V, k, omega, nu_t at the cell centers, with
sample_at / velocity_gradient_at / timescale_at), produced on the
streamwise-PERIODIC curved-bottom mesh, so downstream discrepancy and injection
layers drive both geometries identically.

Geometry provenance: the hill surface is EXTRACTED FROM THE DATASET ITSELF (the
first-fluid transition of the DNS blanking mask per grid column), never from
hardcoded shape coefficients, so the RANS boundary matches the data's geometry
for every steepness case by construction. The domain is streamwise periodic
(wrap-around internal faces; no inlet exists) and the flow is driven by a
constant body force f_x standing in for the mean pressure gradient.

Bulk-flow convention: Re_b = U_b h / nu with h the hill height (1) and U_b the
bulk velocity of the CREST column (the standard periodic-hills convention).
The solve fixes nu = 1 / Re_b so U_b = 1 is the target, and matches f_x by a
secant on the solved crest-column bulk velocity (match_body_force), mirroring
the viscosity and inlet-thickness matching precedents of the other baselines.
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# the inference layer is solver-coupled; import lazily so importing this module
# never forces the built extension to be present.
from solver_bindings import _rs

from .periodic_hills import PeriodicHillsDNS

_CMU = 0.09  # Boussinesq C_mu, for the turbulence timescale tau = 1/(C_mu omega)


class HillsBaselineRANS:
    """One converged SST periodic-hills field (one steepness case).

    Public 2-D field (per cell): x, y, U, V, k, omega, nu_t. Scalars: nu,
    body_force (the matched drive), bulk velocity at the crest column,
    reattachment (from the wall-Cf sign change on the lee side), status.
    """

    HILL_H = 1.0
    U_BULK = 1.0

    # sized for a converged separated field in a few minutes; tests override
    # with a coarse, fast config. body_force is the matched drive for THIS
    # config at alpha = 1.0 (match_body_force); other cases re-match. The
    # relaxation and the from-rest spin-up are the separated-transient
    # treatment: a uniform initial stream slamming into the hill constriction
    # blows up within a dozen iterations, while the body force accelerating
    # the flow from rest develops it smoothly.
    #
    # Grid ceiling, documented: 72x48 and finer diverge from the near-wall
    # cells of the steep windward slope, where the terrain-following mesh is
    # most sheared and the pressure-correction Laplacian's orthogonal
    # approximation is least consistent with the true face normals; the
    # inconsistency grows with slope resolution and no relaxation level damps
    # it (the pre-divergence field is smooth, not odd-even). 60x40 converges
    # with margin and the reattachment is grid-stable against 48x32 (7.62 vs
    # 7.66). A non-orthogonal correction of the pressure operator is the
    # identified upgrade path if finer hills grids are ever needed.
    DEFAULT_CONFIG = {"nx": 60, "ny": 40, "max_iter": 25000, "conv_tol": 1.0e-4,
                      "body_force": 0.00906, "alpha_u": 0.3, "alpha_p": 0.2}

    def __init__(self, x, y, U, V, k, omega, nu_t, nu, body_force,
                 bulk_crest, reattachment, status, meta):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.U = np.asarray(U, float)
        self.V = np.asarray(V, float)
        self.k = np.asarray(k, float)
        self.omega = np.asarray(omega, float)
        self.nu_t = np.asarray(nu_t, float)
        self.nu = float(nu)
        self.body_force = float(body_force)
        self.bulk_crest = float(bulk_crest)
        self.reattachment = float(reattachment)
        self.status = status
        self.meta = meta or {}
        self._pts = np.column_stack([self.x, self.y])
        self._vals = np.column_stack([self.U, self.V, self.k, self.omega,
                                      self.nu_t])
        self._interp = LinearNDInterpolator(self._pts, self._vals)

    # ---- geometry from the data -------------------------------------------

    @staticmethod
    def hill_curve(dns, x_nodes):
        """Hill surface y_b at the mesh x nodes, from the DNS blanking mask.

        Per DNS grid column the surface sits between the last blanked and the
        first fluid point (their midpoint; a fully fluid column is floor 0).
        The sampled curve is interpolated to the mesh nodes and the periodic
        end values are averaged so y_b(0) == y_b(Lx) exactly.
        """
        nY, nX = dns.shape
        xg = dns.x.reshape(dns.shape)
        yg = dns.y.reshape(dns.shape)
        fm = dns.fluid_mask.reshape(dns.shape)
        xs = xg[0, :]
        ys = np.zeros(nX)
        for i in range(nX):
            col = fm[:, i]
            j0 = int(np.argmax(col))          # first fluid row
            if j0 == 0:
                ys[i] = 0.0
            else:
                ys[i] = 0.5 * (yg[j0 - 1, i] + yg[j0, i])
        yb = np.interp(x_nodes, xs, ys)
        # the DNS grid ends one cell short of the period: close the curve
        ends = 0.5 * (yb[0] + yb[-1])
        yb[0] = yb[-1] = ends
        return yb

    @staticmethod
    def domain_extent(dns):
        """(Lx, yTop) of the periodic cell, from the data grid.

        The tensor grid stores cell-like sample points; the streamwise period
        is the sample spacing times the column count (the last column sits one
        spacing short of the wrap image of the first).
        """
        nY, nX = dns.shape
        xs = dns.x.reshape(dns.shape)[0, :]
        dx = (xs[-1] - xs[0]) / (nX - 1)
        Lx = xs[-1] + dx - xs[0]
        yTop = float(dns.y.max())
        return float(Lx), yTop

    # ---- solving ------------------------------------------------------------

    @staticmethod
    def build_forward(cfg=None, dns=None, case="1p0"):
        """Build the periodic-hills forward model without solving.

        Returns the dict of live objects (the model stores references), exactly
        like the backward-facing-step builder.
        """
        cfg = {**HillsBaselineRANS.DEFAULT_CONFIG, **(cfg or {})}
        rs = _rs()
        dns = dns if dns is not None else PeriodicHillsDNS.load(case)
        re_b = float(dns.meta["re_b"])
        nu = HillsBaselineRANS.U_BULK * HillsBaselineRANS.HILL_H / re_b

        Lx, yTop = HillsBaselineRANS.domain_extent(dns)
        x_nodes = np.linspace(0.0, Lx, cfg["nx"] + 1)
        y_bottom = HillsBaselineRANS.hill_curve(dns, x_nodes)
        mesh = rs.Mesh.make_curved_channel_periodic_2d(
            x_nodes, y_bottom, yTop, cfg["ny"], Re=re_b, yPlusTarget=1.0)
        mesh.compute_wall_distance()

        kIn = 1.0e-4
        omIn = 10.0
        bcs = rs.FlowBoundaryConditions.channel_defaults(mesh, 1.0, kIn, omIn)

        settings = rs.SolverSettings()
        settings.max_iterations = cfg["max_iter"]
        settings.convergence_tol = cfg["conv_tol"]
        settings.alpha_u = cfg["alpha_u"]
        settings.alpha_p = cfg["alpha_p"]
        # standing probe hook: Rhie-Chow face dissipation on all meshes
        # (default off; the separated-flow off/on adjudication sets it)
        settings.rhie_chow_all_meshes = cfg.get("rhie_chow_all_meshes", False)
        settings.alpha_k = 0.4
        settings.alpha_omega = 0.4
        settings.inner_iterations = 300
        settings.inner_tolerance = 1e-4
        settings.turb_start_iter = 30
        settings.turb_update_interval = 2
        settings.divergence_limit = 1e8   # loose limit for the spin-up transient
        settings.verbose = False
        settings.body_force = rs.Vec3(cfg["body_force"], 0.0, 0.0)

        # the reattachment QoI on the curved bottom keeps the model well-formed
        obs = rs.ObservationOperator()
        obs.add_reattachment_length("bottom_wall", xr_obs=4.7, sigma=1.0)
        ps = rs.InferenceParameterSet.a1_betaStar()
        # from-rest initial state: the body force spins the flow up smoothly
        fm = rs.ForwardModel(mesh, ps, obs, bcs, nu, settings,
                             rs.Vec3(0.0, 0.0, 0.0), 0.0, kIn, omIn)
        return {"rs": rs, "fm": fm, "mesh": mesh, "bcs": bcs, "obs": obs,
                "ps": ps, "settings": settings, "nu": nu, "re_b": re_b,
                "Lx": Lx, "yTop": yTop, "x_nodes": x_nodes,
                "y_bottom": y_bottom, "cfg": cfg, "dns": dns}

    @staticmethod
    def solve(cfg=None, dns=None, case="1p0"):
        """Solve one periodic-hills baseline; return the converged field record."""
        fwd = HillsBaselineRANS.build_forward(cfg, dns=dns, case=case)
        rs, fm, mesh = fwd["rs"], fwd["fm"], fwd["mesh"]
        theta = list(fwd["ps"].pack(rs.SSTCoefficients()))
        result = fm.evaluate(theta)
        status = str(result.status).split(".")[-1]
        if not fm.has_last_fields():
            raise ValueError(f"hills baseline produced no fields (status {status})")

        ff = fm.last_fields()
        cc = np.asarray(mesh.cell_centers())
        Uf = np.asarray(ff["U"])
        if Uf.ndim == 1:
            Uf = Uf.reshape(-1, 3)
        base = HillsBaselineRANS(
            x=cc[:, 0], y=cc[:, 1], U=Uf[:, 0], V=Uf[:, 1],
            k=np.asarray(ff["k"]), omega=np.asarray(ff["omega"]),
            nu_t=np.asarray(ff["nuT"]), nu=fwd["nu"],
            body_force=fwd["cfg"]["body_force"],
            bulk_crest=HillsBaselineRANS._bulk_crest(mesh, Uf,
                                                     fwd["cfg"]["nx"]),
            reattachment=HillsBaselineRANS._reattachment(mesh, Uf, fwd["nu"],
                                                         np.asarray(ff["nuT"])),
            status=status,
            meta={"config": dict(fwd["cfg"]), "case": fwd["dns"].meta,
                  "re_b": fwd["re_b"], "Lx": fwd["Lx"], "yTop": fwd["yTop"]})
        return base

    @staticmethod
    def _bulk_crest(mesh, Uf, nx):
        """Bulk velocity of the crest column (cells i = 0 of every row)."""
        cc = np.asarray(mesh.cell_centers())
        vols = np.asarray(mesh.cell_volumes())
        n = cc.shape[0]
        idx = np.arange(n)
        crest = idx % nx == 0
        return float(np.sum(Uf[crest, 0] * vols[crest]) / np.sum(vols[crest]))

    @staticmethod
    def _reattachment(mesh, Uf, nu, nuT):
        """Reattachment x from the wall-shear sign change on the lee side.

        Wall shear per bottom face from the owner-cell tangential velocity; the
        reattachment is the first negative-to-positive crossing downstream of
        the separation (the standard periodic-hills x_r on the flat valley).
        """
        wall = mesh.wall_patch_data("bottom_wall")
        owner = np.asarray(wall["owner"], int)
        delta = np.asarray(wall["delta"], float)
        xw = np.asarray(wall["center"])[:, 0]
        order = np.argsort(xw)
        xw, owner, delta = xw[order], owner[order], delta[order]
        tau = (nu + nuT[owner]) * Uf[owner, 0] / np.maximum(delta, 1e-30)
        neg = tau < 0.0
        if not np.any(neg):
            return float("nan")
        i0 = int(np.argmax(neg))                     # separation onset
        for i in range(i0, xw.size - 1):
            if tau[i] < 0.0 <= tau[i + 1]:
                # linear interpolation to the crossing
                f = -tau[i] / (tau[i + 1] - tau[i])
                return float(xw[i] + f * (xw[i + 1] - xw[i]))
        return float("nan")

    @staticmethod
    def match_body_force(cfg=None, dns=None, case="1p0", tol=0.01, max_iter=4):
        """Secant on f_x so the crest-column bulk velocity is U_b = 1.

        Returns (f_x, achieved_bulk, baseline). Matches the drive to the
        case's bulk-Reynolds convention; touches no pre-registered quantity.
        """
        base_cfg = {**HillsBaselineRANS.DEFAULT_CONFIG, **(cfg or {})}
        dns = dns if dns is not None else PeriodicHillsDNS.load(case)

        def run(fb):
            b = HillsBaselineRANS.solve({**base_cfg, "body_force": fb}, dns=dns)
            return b, b.bulk_crest

        f0 = float(base_cfg["body_force"])
        b0, m0 = run(f0)
        if abs(m0 - 1.0) <= tol:
            return f0, m0, b0
        # the bulk responds nearly linearly in sqrt(f) (turbulent drag law);
        # a proportional first step then a secant converges in a step or two
        f1 = f0 / max(m0, 1e-6) ** 2
        b1, m1 = run(f1)
        for _ in range(max_iter):
            if abs(m1 - 1.0) <= tol:
                break
            denom = m1 - m0
            if abs(denom) < 1e-9:
                break
            f2 = f1 + (1.0 - m1) * (f1 - f0) / denom
            f2 = float(np.clip(f2, 1e-5, 1.0))
            f0, m0 = f1, m1
            f1 = f2
            b1, m1 = run(f1)
        return f1, m1, b1

    # ---- interpolation to query points (same surface as the BFS baseline) --

    def sample_at(self, xq, yq):
        """Interpolate U, V, k, omega, nu_t to the query points (NaN outside)."""
        q = np.column_stack([np.asarray(xq, float), np.asarray(yq, float)])
        v = self._interp(q)
        return {"U": v[:, 0], "V": v[:, 1], "k": v[:, 2],
                "omega": v[:, 3], "nu_t": v[:, 4]}

    def velocity_gradient_at(self, xq, yq, dx=0.03, dy=None):
        """grad_u by differencing the field interpolant (wall-adaptive dy).

        Same contract as the backward-facing-step baseline: scalar or per-point
        steps, one-sided at domain edges, zero where fully outside.
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
            return np.where(np.isfinite(fp) & np.isfinite(fm),
                            (fp - fm) / (2.0 * step),
                            np.where(np.isfinite(fp) & np.isfinite(f0),
                                     (fp - f0) / step,
                                     np.where(np.isfinite(fm) & np.isfinite(f0),
                                              (f0 - fm) / step, 0.0)))

        g = np.zeros((xq.size, 3, 3))
        g[:, 0, 0] = deriv(Uxp, Uxm, U0, dx)
        g[:, 0, 1] = deriv(Uyp, Uym, U0, dy)
        g[:, 1, 0] = deriv(Vxp, Vxm, V0, dx)
        g[:, 1, 1] = deriv(Vyp, Vym, V0, dy)
        return g

    def timescale_at(self, xq, yq):
        """Baseline turbulence timescale tau = 1/(C_mu omega) at query points."""
        om = self.sample_at(xq, yq)["omega"]
        return 1.0 / (_CMU * np.maximum(om, 1e-30))

    def __repr__(self):
        return (f"HillsBaselineRANS(re_b={self.meta.get('re_b')}, "
                f"n_cells={self.x.size}, x_r/h={self.reattachment:.2f}, "
                f"U_b={self.bulk_crest:.3f}, status={self.status})")
