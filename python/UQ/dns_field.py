"""The DNS-schema discrepancy-extraction interface: the boundary where real DNS
plugs in.

A ``DNSField`` holds the high-fidelity fields named mandatory in
DNS_data/README.md (mean velocity gradient, the full Reynolds-stress tensor, and
for compressible cases temperature, density, and the turbulent heat flux), plus
the baseline closure quantities (k, the turbulence timescale, the eddy viscosity)
needed to form the model-form discrepancy. From it the interface produces the two
discrepancies and the Galilean-invariant feature set the learned closure consumes.

Until real DNS is compiled this is exercised on synthetic fake-DNS fields with a
planted discrepancy (``UQ.synthetic``); the recovery test proves the interface
reconstructs the known answer. Nothing here requires real DNS.
"""
import numpy as np

from . import discrepancy as dq


class DNSField:
    """A high-fidelity field in the DNS_data schema, ready for discrepancy work.

    Arrays are per sample point (leading axis N):
      grad_u (N,3,3)  mean velocity gradient, d u_i / d x_j
      R      (N,3,3)  full Reynolds-stress tensor (mandatory; case is excluded
                      without it, per the DNS_data acceptance criteria)
      k      (N,)     turbulent kinetic energy (defaults to 0.5 tr(R))
      timescale (N,)  turbulence timescale (e.g. 1/omega or k/epsilon)
      grad_T (N,3)    mean temperature gradient        (compressible/heat-flux)
      heat_flux (N,3) turbulent heat-flux vector       (compressible/heat-flux)
      nu_t   (N,)     baseline eddy viscosity for the gradient-diffusion flux
      Pr_t   scalar   turbulent Prandtl number
      meta   dict     Re, Ma, wall thermal condition, provenance
    """

    def __init__(self, grad_u, R, k=None, timescale=None, grad_T=None,
                 heat_flux=None, nu_t=None, Pr_t=0.9, meta=None):
        self.grad_u = np.asarray(grad_u, dtype=float)
        self.R = np.asarray(R, dtype=float)
        n = self.grad_u.shape[0]
        self.k = np.asarray(k, dtype=float) if k is not None \
            else 0.5 * np.trace(self.R, axis1=-2, axis2=-1)
        self.timescale = np.asarray(timescale, dtype=float) if timescale is not None \
            else np.ones(n)
        self.grad_T = None if grad_T is None else np.asarray(grad_T, dtype=float)
        self.heat_flux = None if heat_flux is None else np.asarray(heat_flux, dtype=float)
        self.nu_t = None if nu_t is None else np.asarray(nu_t, dtype=float)
        self.Pr_t = Pr_t
        self.meta = meta or {}

    @staticmethod
    def from_dict(d):
        """Build a field from a schema dict (the form a DNS loader would emit)."""
        return DNSField(
            grad_u=d["grad_u"], R=d["R"], k=d.get("k"),
            timescale=d.get("timescale"), grad_T=d.get("grad_T"),
            heat_flux=d.get("heat_flux"), nu_t=d.get("nu_t"),
            Pr_t=d.get("Pr_t", 0.9), meta=d.get("meta"),
        )

    def has_heat_flux(self):
        """Heat-flux discrepancy is only defined when the schema supplies it.

        For compressible heat-flux claims the DNS_data acceptance criteria
        require both the turbulent heat flux and grad(T) to be present.
        """
        return self.heat_flux is not None and self.grad_T is not None \
            and self.nu_t is not None

    def strain_rotation(self):
        return dq.strain_rotation(self.grad_u, self.timescale)

    def features(self, extra=None):
        """Galilean-invariant conditioning features (Pope invariants + extras)."""
        return dq.feature_set(self.grad_u, self.timescale, extra=extra)

    def reynolds_discrepancy(self):
        """Boussinesq Reynolds-stress anisotropy discrepancy db = b_DNS - b_Bouss."""
        S, _ = self.strain_rotation()
        return dq.reynolds_discrepancy(self.R, S)

    def heatflux_discrepancy(self):
        """Gradient-diffusion turbulent-heat-flux discrepancy (requires heat flux)."""
        if not self.has_heat_flux():
            raise ValueError("DNSField has no turbulent heat flux / grad(T) / nu_t")
        return dq.heatflux_discrepancy(self.heat_flux, self.grad_T, self.nu_t, self.Pr_t)

    def extract(self):
        """Return a dict with the features and both discrepancies (where available).

        This is the single call a learned-closure training pipeline makes per
        DNS field: features to condition on, and the discrepancy targets to learn.
        """
        out = {
            "features": self.features(),
            "reynolds_discrepancy": self.reynolds_discrepancy(),
        }
        if self.has_heat_flux():
            out["heatflux_discrepancy"] = self.heatflux_discrepancy()
        return out


def fake_dns_field(n=400, seed=0):
    """A synthetic DNSField in the schema, with a planted, recoverable discrepancy.

    Combines the planted Reynolds-stress field and the planted heat-flux field so
    the adapter can be exercised end to end without real DNS.
    """
    from . import synthetic as syn
    fd = syn.make_fake_dns(n=n, seed=seed)
    fh = syn.make_fake_heatflux(n=n, seed=seed + 100)
    return DNSField(
        grad_u=fd["grad_u"], R=fd["R_dns"], k=fd["k"], timescale=fd["timescale"],
        grad_T=fh["grad_T"], heat_flux=fh["q_dns"], nu_t=fh["nu_t"], Pr_t=fh["Pr_t"],
        meta={"regime": "synthetic", "note": "planted discrepancy"},
    ), fd["db_true"], fh["dq_true"]
