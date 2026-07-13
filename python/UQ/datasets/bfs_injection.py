"""A-posteriori Reynolds-stress injection for the Le-Moin backward-facing step.

Wraps the injected BFS forward model: a prescribed per-cell target anisotropy
b_target = b_baseline + db enters the momentum equation as the explicit
deferred-correction body force -div(2 k b_target - 2 nu_t S) with the
eddy-viscosity diffusion kept implicit (the pre-registered injection scheme;
see UQ-RANS_research/separated_modelform/METHODS_OPERATIONALIZATION.md). Every
target is projected into the barycentric realizable set BEFORE injection, and
the solver re-checks realizability each outer iteration (a check separate from
the Galilean-invariant feature construction); the diagnostics are returned with
every run, never masked.

This is the single mechanism shared by the eigenspace-perturbation baseline,
the Gaussian model-form baseline, and the generative model-form, so their
a-posteriori comparison is solver-for-solver.

Lifetime note: the forward model stores references to the mesh and boundary
objects, so this wrapper keeps the whole build dict alive for its own lifetime.
"""
import numpy as np

from .bfs_baseline import BFSBaselineRANS
from .. import realizability as rz
from .. import discrepancy as dq


class BFSInjection:
    """Injected BFS forward runs against one shared mesh and baseline field.

    Attributes:
      baseline          the converged BFSBaselineRANS the targets are built on
      cc_x, cc_y (N,)   solver cell centers
      b_baseline (N,3,3) limiter-consistent Boussinesq anisotropy at the cells
      features  (N,5)   Galilean-invariant conditioning features at the cells
    """

    def __init__(self, cfg=None, dns=None, baseline=None):
        self._fwd = BFSBaselineRANS.build_forward(cfg, dns=dns)
        self._theta = list(self._fwd["ps"].pack(self._fwd["rs"].SSTCoefficients()))
        self.baseline = (baseline if baseline is not None
                         else BFSBaselineRANS.solve(cfg, dns=dns))

        cc = np.asarray(self._fwd["mesh"].cell_centers())
        self.cc_x = cc[:, 0]
        self.cc_y = cc[:, 1]

        # local wall offset of each cell (upstream cells sit above the step
        # top at y = h), for the wall-adaptive gradient step
        wall_off = np.where(self.cc_x < 0.0,
                            self.cc_y - BFSBaselineRANS.STEP_H, self.cc_y)
        dy = np.clip(0.4 * np.maximum(wall_off, 0.0), 2.0e-3, 0.03)

        grad_u = self.baseline.velocity_gradient_at(self.cc_x, self.cc_y, dy=dy)
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
        self.b_baseline = dq.boussinesq_anisotropy_actual(S, nu_t, k_rans, timescale)
        self.features = dq.feature_set(grad_u, timescale)

    @property
    def n_cells(self):
        return self.cc_x.size

    def target_from_db(self, db):
        """b_target = b_baseline + db, projected into the realizable set.

        db is (N, 3, 3) over the solver cells (zero outside the valid mask).
        Returns the projected target and the projection distance per cell.
        """
        db = np.asarray(db, float)
        b = self.b_baseline + np.where(self.cell_mask[:, None, None], db, 0.0)
        bp, dist = rz.project_anisotropy(b)
        return bp, dist

    def run(self, b_target):
        """Solve with the injected target anisotropy; return the QoI record.

        b_target is (N, 3, 3), already realizable (use target_from_db). The
        run record carries the predicted reattachment, convergence status and
        iterations, the realizability diagnostics from the running solve, and
        the converged fields.
        """
        fm = self._fwd["fm"]
        fm.set_target_anisotropy(np.asarray(b_target, float))
        result = fm.evaluate(self._theta)
        fm.clear_target_anisotropy()
        status = str(result.status).split(".")[-1]
        fields = fm.last_fields() if fm.has_last_fields() else None
        return {
            "reattachment": float(result.predictions[0]),
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
        return {
            "reattachment": float(result.predictions[0]),
            "status": status,
            "iterations": int(result.simple_iters),
        }

    def wall_cf(self, fields):
        """Skin-friction Cf(x) along the downstream bottom wall.

        Cf = 2 nu (dU/dy)_wall / U0^2 from the wall-adjacent cell velocity and
        its wall distance (U0 = 1), applied to a last_fields dict from a run.
        """
        mesh = self._fwd["mesh"]
        nu = self._fwd["nu"]
        wall = mesh.wall_patch_data("bottom_wall_down")
        owner = np.asarray(wall["owner"], int)
        delta = np.asarray(wall["delta"], float)
        xw = np.asarray(wall["center"])[:, 0]
        U = np.asarray(fields["U"])
        if U.ndim == 1:
            U = U.reshape(-1, 3)
        Uown = U[owner, 0]
        cf = 2.0 * nu * Uown / np.maximum(delta, 1e-30)
        order = np.argsort(xw)
        return xw[order], cf[order]

    def __repr__(self):
        return (f"BFSInjection(n_cells={self.n_cells}, "
                f"baseline x_r/h={self.baseline.reattachment:.2f})")
