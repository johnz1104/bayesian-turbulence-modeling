"""A-posteriori Reynolds-stress injection for the periodic hills.

The second-geometry counterpart of the backward-facing-step injection wrapper:
a prescribed per-cell target anisotropy b_target = b_baseline + db enters the
momentum equation as the explicit deferred-correction body force
-div(2 k b_target - 2 nu_t S) with the eddy-viscosity diffusion kept implicit,
on the streamwise-periodic curved-bottom mesh. Every target is projected into
the barycentric realizable set BEFORE injection and the solver re-checks
realizability each outer iteration (a check separate from the
Galilean-invariant feature construction); the diagnostics are returned with
every run, never masked.

Quantities follow the pinned protocol (UQ-RANS_research/separated_modelform/
METHODS_OPERATIONALIZATION.md section 9): the bubble geometry from the
wall-shear sign structure with the SAME crossing rules as the baseline record,
and the mean streamwise velocity at the pinned probe points; the dataset
carries no measured wall skin friction or pressure, so no Cf/Cp station vector
exists on this geometry.

Lifetime note: the forward model stores references to the mesh and boundary
objects, so this wrapper keeps the whole build dict alive for its own lifetime.
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from .hills_baseline import HillsBaselineRANS
from .. import realizability as rz
from .. import discrepancy as dq

# pinned probe grid (methods note section 9): stations x heights, kept where
# at least 0.2 h above the local hill surface and 0.5 h below the top wall
PROBE_STATIONS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
PROBE_HEIGHTS = (0.5, 1.0, 2.0)


class HillsInjection:
    """Injected periodic-hills forward runs against one shared mesh and baseline.

    Attributes:
      baseline           the converged HillsBaselineRANS the targets build on
      cc_x, cc_y (N,)    solver cell centers
      b_baseline (N,3,3) limiter-consistent Boussinesq anisotropy at the cells
      features  (N,5)    Galilean-invariant conditioning features at the cells
      cell_mask (N,)     cells with finite baseline samples
    """

    def __init__(self, cfg=None, dns=None, baseline=None, case="1p0"):
        self._fwd = HillsBaselineRANS.build_forward(cfg, dns=dns, case=case)
        self._theta = list(self._fwd["ps"].pack(self._fwd["rs"].SSTCoefficients()))
        self.baseline = (baseline if baseline is not None
                         else HillsBaselineRANS.solve(cfg, dns=dns, case=case))

        cc = np.asarray(self._fwd["mesh"].cell_centers())
        self.cc_x = cc[:, 0]
        self.cc_y = cc[:, 1]

        # wall-adaptive gradient step from the local hill surface and the top
        # wall, the same recipe as the dense-field discrepancy
        Lx, yTop = self._fwd["Lx"], self._fwd["yTop"]
        xs = np.linspace(0.0, Lx, 257)
        yb = HillsBaselineRANS.hill_curve(self._fwd["dns"], xs)
        y_surf = np.interp(self.cc_x, xs, yb)
        wall_off = np.minimum(np.maximum(self.cc_y - y_surf, 0.0),
                              np.maximum(yTop - self.cc_y, 0.0))
        dy = np.clip(0.4 * wall_off, 2.0e-3, 0.05)

        grad_u = self.baseline.velocity_gradient_at(self.cc_x, self.cc_y,
                                                    dx=0.05, dy=dy)
        timescale = self.baseline.timescale_at(self.cc_x, self.cc_y)
        sampled = self.baseline.sample_at(self.cc_x, self.cc_y)

        finite = (np.all(np.isfinite(grad_u), axis=(1, 2))
                  & np.isfinite(timescale) & np.isfinite(sampled["nu_t"])
                  & np.isfinite(sampled["k"]) & (sampled["k"] > 0.0))
        grad_u = np.where(finite[:, None, None], grad_u, 0.0)
        timescale = np.where(finite, timescale, 1.0)
        nu_t = np.where(finite, sampled["nu_t"], 0.0)
        k_rans = np.where(finite, sampled["k"], 1.0)
        self.cell_mask = finite

        # the same limiter-consistent baseline anisotropy the training db is
        # formed against, and the same invariant features it is conditioned on
        S, _ = dq.strain_rotation(grad_u, timescale)
        self.S_baseline = S   # kept for the five-state eigenvector perturbations
        self.b_baseline = dq.boussinesq_anisotropy_actual(S, nu_t, k_rans,
                                                          timescale)
        self.features = dq.feature_set(grad_u, timescale)

        # pinned admissible probes for the mean-velocity quantity
        px, py = np.meshgrid(np.asarray(PROBE_STATIONS),
                             np.asarray(PROBE_HEIGHTS), indexing="ij")
        px, py = px.ravel(), py.ravel()
        surf = np.interp(px, xs, yb)
        keep = (py >= surf + 0.2) & (py <= yTop - 0.5)
        self.probe_x = px[keep]
        self.probe_y = py[keep]

    @property
    def n_cells(self):
        return self.cc_x.size

    def target_from_db(self, db):
        """b_target = b_baseline + db, projected into the realizable set."""
        db = np.asarray(db, float)
        b = self.b_baseline + np.where(self.cell_mask[:, None, None], db, 0.0)
        bp, dist = rz.project_anisotropy(b)
        return bp, dist

    def run(self, b_target):
        """Solve with the injected target anisotropy; return the QoI record.

        The scored reattachment and separation use the same wall-shear
        crossing rules as the baseline record (NOT the C++ observation
        operator's reading, which only keeps the forward model well-formed).
        """
        fm = self._fwd["fm"]
        fm.set_target_anisotropy(np.asarray(b_target, float))
        result = fm.evaluate(self._theta)
        fm.clear_target_anisotropy()
        status = str(result.status).split(".")[-1]
        fields = fm.last_fields() if fm.has_last_fields() else None
        x_s = x_r = float("nan")
        if fields is not None:
            x_s, x_r = self.bubble_geometry(fields)
        return {
            "separation": x_s,
            "reattachment": x_r,
            "bubble_length": x_r - x_s,
            "status": status,
            "iterations": int(result.simple_iters),
            "diagnostics": dict(fm.injection_diagnostics()),
            "fields": fields,
        }

    def run_baseline(self):
        """Plain (no-injection) solve on the same forward model, for reference."""
        fm = self._fwd["fm"]
        fm.clear_target_anisotropy()
        result = fm.evaluate(self._theta)
        status = str(result.status).split(".")[-1]
        fields = fm.last_fields() if fm.has_last_fields() else None
        x_s = x_r = float("nan")
        if fields is not None:
            x_s, x_r = self.bubble_geometry(fields)
        return {
            "separation": x_s,
            "reattachment": x_r,
            "bubble_length": x_r - x_s,
            "status": status,
            "iterations": int(result.simple_iters),
        }

    # ---- pinned quantities from a converged field ---------------------------

    def wall_shear(self, fields):
        """(x, tau) along the bottom wall from the owner-cell velocity."""
        mesh = self._fwd["mesh"]
        nu = self._fwd["nu"]
        wall = mesh.wall_patch_data("bottom_wall")
        owner = np.asarray(wall["owner"], int)
        delta = np.asarray(wall["delta"], float)
        xw = np.asarray(wall["center"])[:, 0]
        U = np.asarray(fields["U"])
        if U.ndim == 1:
            U = U.reshape(-1, 3)
        nuT = np.asarray(fields["nuT"])
        tau = (nu + nuT[owner]) * U[owner, 0] / np.maximum(delta, 1e-30)
        order = np.argsort(xw)
        return xw[order], tau[order]

    def bubble_geometry(self, fields):
        """(x_s, x_r) by the pinned crossing rules on the wall-shear curve.

        Separation onset x_s: the interpolated positive-to-negative crossing
        into the first negative face (the face itself when the curve starts
        negative). Reattachment x_r: the first negative-to-positive crossing
        downstream of the onset, interpolated, exactly as the baseline record.
        """
        xw, tau = self.wall_shear(fields)
        neg = tau < 0.0
        if not np.any(neg):
            return float("nan"), float("nan")
        i0 = int(np.argmax(neg))
        if i0 > 0 and tau[i0 - 1] > 0.0:
            f = tau[i0 - 1] / (tau[i0 - 1] - tau[i0])
            x_s = float(xw[i0 - 1] + f * (xw[i0] - xw[i0 - 1]))
        else:
            x_s = float(xw[i0])
        x_r = float("nan")
        for i in range(i0, xw.size - 1):
            if tau[i] < 0.0 <= tau[i + 1]:
                f = -tau[i] / (tau[i + 1] - tau[i])
                x_r = float(xw[i] + f * (xw[i + 1] - xw[i]))
                break
        return x_s, x_r

    def probe_velocity(self, fields):
        """Mean streamwise velocity at the pinned admissible probes."""
        mesh = self._fwd["mesh"]
        cc = np.asarray(mesh.cell_centers())
        U = np.asarray(fields["U"])
        if U.ndim == 1:
            U = U.reshape(-1, 3)
        interp = LinearNDInterpolator(cc[:, :2], U[:, 0])
        return interp(np.column_stack([self.probe_x, self.probe_y]))

    def probe_truth(self, dns=None):
        """The DNS mean streamwise velocity at the same admissible probes."""
        dns = dns if dns is not None else self._fwd["dns"]
        pts = np.column_stack([dns.x[dns.fluid_mask], dns.y[dns.fluid_mask]])
        interp = LinearNDInterpolator(pts, dns.U[dns.fluid_mask])
        return interp(np.column_stack([self.probe_x, self.probe_y]))

    def __repr__(self):
        return (f"HillsInjection(n_cells={self.n_cells}, "
                f"baseline x_r/h={self.baseline.reattachment:.2f})")
