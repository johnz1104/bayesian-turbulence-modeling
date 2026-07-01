"""Field-to-field model-form discrepancy for the separated backward-facing step.

Forms the anisotropy discrepancy db = b_DNS - b_baseline at the DNS profile points,
conditioned on the Galilean-invariant features of the RANS baseline. This is the
training target for the generative model-form.

The backward-facing step is sparse in the streamwise direction (six stations), so
the recipe does not need a DNS velocity gradient: the target anisotropy b_DNS =
R/(2k) - I/3 comes from the DNS Reynolds stress at the profile points (gradient
free), while the Boussinesq baseline anisotropy b_baseline and the Pope-invariant
conditioning features come from the dense RANS field interpolated to those points.
The assembly reuses the canonical DNSField interface: the DNS record supplies the
Reynolds stress and the RANS baseline supplies the velocity gradient, timescale, and
eddy viscosity, so DNSField.extract() returns exactly the features (from the RANS
gradient) and reynolds_discrepancy = b_DNS - b_Boussinesq(RANS).
"""
import numpy as np

from .backward_facing_step import BackwardFacingStepDNS
from .bfs_baseline import BFSBaselineRANS


class BFSDiscrepancy:
    """The (feature, db) discrepancy set along the six BFS profile stations.

    Attributes (per DNS profile point, leading axis N):
      features (N,5)   Pope invariants of the RANS baseline strain and rotation
      db (N,3,3)       anisotropy discrepancy b_DNS - b_baseline
      x_h, y_h (N,)    DNS profile coordinates (x/h station, y/h)
      station (N,)     integer station index
      mask (N,)        valid points: resolved DNS turbulence and inside the RANS
                       domain (finite baseline gradient, timescale, eddy viscosity)
    """

    def __init__(self, features, db, x_h, y_h, station, mask, reattachment_error,
                 dns, baseline):
        self.features = np.asarray(features, float)
        self.db = np.asarray(db, float)
        self.x_h = np.asarray(x_h, float)
        self.y_h = np.asarray(y_h, float)
        self.station = np.asarray(station, int)
        self.mask = np.asarray(mask, bool)
        self.reattachment_error = float(reattachment_error)
        self.dns = dns
        self.baseline = baseline

    @staticmethod
    def build(dns=None, baseline=None, cfg=None, dx=0.03):
        """Assemble the discrepancy from the DNS loader and the RANS baseline.

        Loads the BFS DNS and solves the baseline if not supplied, maps each DNS
        profile point into the mesh frame, interpolates the baseline gradient,
        timescale, and eddy viscosity there, and forms the discrepancy through the
        DNSField interface.
        """
        dns = dns if dns is not None else BackwardFacingStepDNS.load()
        baseline = baseline if baseline is not None else BFSBaselineRANS.solve(cfg)

        xq, yq = baseline.map_dns_points(dns.x_h, dns.y_h)
        grad_u = baseline.velocity_gradient_at(xq, yq, dx=dx)
        timescale = baseline.timescale_at(xq, yq)
        nu_t = baseline.sample_at(xq, yq)["nu_t"]

        # valid points: DNS turbulence resolved (k > 0) AND the baseline sample is
        # inside the RANS domain (finite gradient / timescale / eddy viscosity)
        finite = (np.all(np.isfinite(grad_u), axis=(1, 2))
                  & np.isfinite(timescale) & np.isfinite(nu_t))
        mask = dns.valid_mask() & finite

        # the DNS Reynolds stress with the RANS baseline gradient/timescale/nu_t:
        # DNSField.extract() gives features from the RANS gradient and the anisotropy
        # discrepancy b_DNS - b_Boussinesq(RANS baseline) at each profile point.
        grad_safe = np.where(np.isfinite(grad_u), grad_u, 0.0)
        ts_safe = np.where(np.isfinite(timescale), timescale, 1.0)
        nut_safe = np.where(np.isfinite(nu_t), nu_t, 0.0)
        field = dns.to_dnsfield(grad_u=grad_safe, timescale=ts_safe, nu_t=nut_safe)
        out = field.extract()

        return BFSDiscrepancy(
            features=out["features"], db=out["reynolds_discrepancy"],
            x_h=dns.x_h, y_h=dns.y_h, station=dns.station, mask=mask,
            reattachment_error=baseline.reattachment - dns.reattachment_truth(),
            dns=dns, baseline=baseline)

    # ---- derived views ----------------------------------------------------

    def training_pairs(self):
        """(features, db) on the valid points, for the generative model-form."""
        return self.features[self.mask], self.db[self.mask]

    def db_magnitude(self):
        """Frobenius norm of the anisotropy discrepancy per point."""
        return np.sqrt(np.sum(self.db ** 2, axis=(1, 2)))

    def magnitude_by_station(self):
        """Mean discrepancy magnitude per station (x/h), on valid points.

        The discrepancy is expected to be largest in the separated shear layer and
        the reattachment region, where the Boussinesq alignment fails, and smallest
        at the attached upstream station.
        """
        mag = self.db_magnitude()
        out = {}
        for i, s in enumerate(self.dns.stations):
            sel = (self.station == i) & self.mask
            out[s["x_h"]] = float(np.mean(mag[sel])) if np.any(sel) else float("nan")
        return out
