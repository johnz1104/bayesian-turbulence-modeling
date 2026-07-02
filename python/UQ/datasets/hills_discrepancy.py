"""Field-to-field model-form discrepancy for the periodic hills (dense field).

Forms the anisotropy discrepancy db = b_DNS - b_baseline at the (subsampled)
interior fluid points of the dense DNS field, conditioned on the
Galilean-invariant features of the RANS baseline, with the same
limiter-consistent conventions as the backward-facing-step discrepancy, so the
two geometries feed the identical training, scoring and injection layers.

Protocol pinned in UQ-RANS_research/separated_modelform/
METHODS_OPERATIONALIZATION.md section 8 before any result: interior points at
a fixed stride of 3 per grid direction; b_DNS gradient-free from the DNS
stress; b_baseline, k_baseline, timescale and the five invariant features from
the converged RANS field interpolated to the points; the wall-adaptive
gradient step measured from the local hill surface and the top wall.
"""
import numpy as np

from .periodic_hills import PeriodicHillsDNS
from .hills_baseline import HillsBaselineRANS


class HillsDiscrepancy:
    """The (feature, db) discrepancy set on the dense periodic-hills field.

    Attributes (per kept DNS point, leading axis N):
      features (N,5)   Pope invariants of the RANS baseline strain and rotation
      db (N,3,3)       anisotropy discrepancy b_DNS - b_baseline
      b_baseline (N,3,3) the limiter-consistent Boussinesq baseline anisotropy
      x, y (N,)        point coordinates
      band (N,)        streamwise band index 0..5 (the held-out unit)
      mask (N,)        valid points (finite baseline samples)
    """

    N_BANDS = 6
    STRIDE = 3

    def __init__(self, features, db, b_baseline, x, y, band, mask,
                 reattachment_error, dns, baseline):
        self.features = np.asarray(features, float)
        self.db = np.asarray(db, float)
        self.b_baseline = np.asarray(b_baseline, float)
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.band = np.asarray(band, int)
        self.mask = np.asarray(mask, bool)
        self.reattachment_error = float(reattachment_error)
        self.dns = dns
        self.baseline = baseline

    @staticmethod
    def build(dns=None, baseline=None, cfg=None, case="1p0", dx=0.05):
        """Assemble the dense-field discrepancy per the pinned protocol."""
        dns = dns if dns is not None else PeriodicHillsDNS.load(case)
        baseline = (baseline if baseline is not None
                    else HillsBaselineRANS.solve(cfg, dns=dns, case=case))

        # subsampled interior fluid points (fixed stride, pinned)
        nY, nX = dns.shape
        keep2d = np.zeros(dns.shape, dtype=bool)
        keep2d[::HillsDiscrepancy.STRIDE, ::HillsDiscrepancy.STRIDE] = True
        keep = (keep2d.ravel() & dns.interior_mask & dns.fluid_mask)
        idx = np.nonzero(keep)[0]

        xq = dns.x[idx]
        yq = dns.y[idx]

        # wall-adaptive gradient step from the local hill surface and top wall
        Lx, yTop = HillsBaselineRANS.domain_extent(dns)
        yb = HillsBaselineRANS.hill_curve(dns, np.linspace(0.0, Lx, 257))
        y_surf = np.interp(xq, np.linspace(0.0, Lx, 257), yb)
        wall_off = np.minimum(np.maximum(yq - y_surf, 0.0),
                              np.maximum(yTop - yq, 0.0))
        dy = np.clip(0.4 * wall_off, 2.0e-3, dx)

        grad_u = baseline.velocity_gradient_at(xq, yq, dx=dx, dy=dy)
        timescale = baseline.timescale_at(xq, yq)
        sampled = baseline.sample_at(xq, yq)
        nu_t = sampled["nu_t"]
        k_rans = sampled["k"]

        finite = (np.all(np.isfinite(grad_u), axis=(1, 2))
                  & np.isfinite(timescale) & np.isfinite(nu_t)
                  & np.isfinite(k_rans) & (k_rans > 0.0))

        grad_safe = np.where(np.isfinite(grad_u), grad_u, 0.0)
        ts_safe = np.where(np.isfinite(timescale), timescale, 1.0)
        nut_safe = np.where(np.isfinite(nu_t), nu_t, 0.0)
        krans_safe = np.where(np.isfinite(k_rans) & (k_rans > 0.0), k_rans, 1.0)

        # the DNS Reynolds stress with the RANS-derived closure quantities,
        # through the same canonical interface as the first geometry
        field = dns.to_dnsfield_at(idx, grad_u=grad_safe, timescale=ts_safe,
                                   nu_t=nut_safe, k_baseline=krans_safe)
        out = field.extract()
        b_base = field.baseline_anisotropy()

        band = np.minimum((xq / Lx * HillsDiscrepancy.N_BANDS).astype(int),
                          HillsDiscrepancy.N_BANDS - 1)

        # reattachment model-form signal: RANS x_r against the DNS-field value
        xr_dns = dns.bottom_wall_reattachment()
        xr_err = (baseline.reattachment - xr_dns
                  if xr_dns is not None else float("nan"))

        return HillsDiscrepancy(
            features=out["features"], db=out["reynolds_discrepancy"],
            b_baseline=b_base, x=xq, y=yq, band=band, mask=finite,
            reattachment_error=xr_err, dns=dns, baseline=baseline)

    # ---- derived views ----------------------------------------------------

    def training_pairs(self):
        """(features, db) on the valid points, for the conditional models."""
        return self.features[self.mask], self.db[self.mask]

    def db_magnitude(self):
        """Frobenius norm of the anisotropy discrepancy per point."""
        return np.sqrt(np.sum(self.db ** 2, axis=(1, 2)))

    def magnitude_by_band(self):
        """Mean discrepancy magnitude per streamwise band, on valid points."""
        mag = self.db_magnitude()
        out = {}
        for b in range(self.N_BANDS):
            sel = (self.band == b) & self.mask
            out[b] = float(np.mean(mag[sel])) if np.any(sel) else float("nan")
        return out
