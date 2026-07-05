"""Shared base for the compressible wall-bounded DNS profile loaders.

The compressible attached-flow study consumes three third-party datasets (the
Gerolymos-Vallet channel matrix, the Coleman-Kim-Moser supersonic channel, the
Zhang-Duan-Choudhari flat plates), each in its own raw format and native
averaging. Every loader converts to ONE convention, stated here once (the scope
decision confirmed at study start):

- FAVRE (density-weighted) moments are the discrepancy convention. The stress
  record is the density-scaled Favre tensor

      R_ij = <rho u_i" u_j"> / <rho>                     [velocity^2]

  because the compressible RANS closure models the Favre stress. The anisotropy
  b = R/(2k) - I/3 is identical whether formed from <rho u_i"u_j"> or from R
  (the local density cancels in the ratio), so the existing discrepancy and
  realizability machinery consumes this record unchanged.

- WALL (+) units are the common normalization: velocity by u_tau =
  sqrt(tau_w/rho_w), length by nu_w/u_tau, stress and k by u_tau^2, temperature
  by T_w, density by rho_w, viscosity by mu_w. The semi-local coordinate y*
  (local-property scaling) is carried alongside as a diagnostic and a
  conditioning feature, never as the record's primary units.

- The turbulent heat flux is the Favre temperature-flux vector in (u_tau, T_w)
  units,

      q_hat_i = <rho h" u_i"> / (<rho> cp_w u_tau T_w),

  finite for a thermally quiescent wall (B_q -> 0, unlike friction-temperature
  units), and consistent with the gradient-diffusion baseline evaluated in the
  same units,

      q_GDH_i = -(nu_t^+ / Pr_t) d(T/T_w)/dx_i^+ ,

  where nu_t^+ = (mu_t/<rho>)/(mu_w/rho_w) is the local kinematic eddy
  viscosity in wall-viscosity units. Reynolds-averaged moments are retained
  only where a file provides them, as cross-check views, never as the record.

Wall parameters carried per case (the `wall` dict): the wall-heat-flux
parameter B_q := <q_w>/(rho_w cp_w u_tau T_w) (negative when heat flows into a
cooled wall), the friction Mach number M_tau := u_tau/a_w, the skin-friction
coefficient cf where the source states it, the semi-local friction Reynolds
number Re_tau*, and the isentropic exponent gamma_w. These are what the wall
heat-flux QoI and the unit conversions need.
"""
import numpy as np

from . import _common
from ..dns_field import DNSField


class CompressibleProfileDNS(_common.WallBoundedProfileDNS):
    """Canonical compressible dns_field record for a 1-D wall-bounded profile.

    Extends the incompressible wall-bounded record with the thermal fields the
    compressible study needs, per wall-normal station and in the module-level
    convention (Favre moments, wall units):

      T     {T}/T_w         Favre mean temperature on wall temperature
      rho   <rho>/rho_w     mean density on wall density
      mu    <mu>/mu_w       mean dynamic viscosity on wall viscosity
      q_hat (N,3)           Favre temperature-flux vector in (u_tau, T_w) units
      ystar                 semi-local wall distance (diagnostic / feature)
      mach                  mean streamwise Mach-number profile where given
      pr_molecular          molecular Prandtl-number profile where given
      wall  dict            B_q, M_tau, cf, Re_tau*, gamma_w and source extras

    The base-record fields keep their incompressible meaning: yplus, U (the
    Favre mean streamwise velocity in wall units), R (here the density-scaled
    Favre stress, velocity^2 form), k = tr(R)/2, re_tau (the WALL-unit friction
    Reynolds number; the semi-local Re_tau* lives in `wall`).
    """

    def __init__(self, yplus, U, R, k, re_tau, meta, T, rho, mu, y_outer=None,
                 ystar=None, q_hat=None, mach=None, pr_molecular=None,
                 wall=None):
        super().__init__(yplus=yplus, U=U, R=R, k=k, re_tau=re_tau, meta=meta,
                         y=y_outer, y_outer=y_outer)
        self.T = np.asarray(T, dtype=float)
        self.rho = np.asarray(rho, dtype=float)
        # mu is None where the source gives no viscosity profile (the flat
        # plate); the total-stress identities then do not apply to that case
        self.mu = None if mu is None else np.asarray(mu, dtype=float)
        self.ystar = None if ystar is None else np.asarray(ystar, dtype=float)
        self.q_hat = None if q_hat is None else np.asarray(q_hat, dtype=float)
        self.mach = None if mach is None else np.asarray(mach, dtype=float)
        self.pr_molecular = None if pr_molecular is None \
            else np.asarray(pr_molecular, dtype=float)
        self.wall = dict(wall or {})

    # ---- thermal derived quantities ----------------------------------------

    def dTdy(self):
        """Wall-normal mean temperature gradient d(T/T_w)/dy^+ (finite-differenced)."""
        return _common.wall_normal_gradient(self.yplus, self.T)

    def temperature_gradient(self):
        """Mean temperature-gradient vector grad_T[i] = d(T/T_w)/dx_i^+.

        Only the wall-normal component is nonzero for these statistically 1-D
        profiles; this is the grad_T the gradient-diffusion baseline consumes.
        """
        g = np.zeros((self.n, 3))
        g[:, 1] = self.dTdy()
        return g

    def turbulent_mach(self):
        """Turbulent Mach number M_t = sqrt(2 k) / <a> from the record itself.

        With k in u_tau^2 units, a/a_w = sqrt(T/T_w) (constant gamma, ideal
        gas) and M_tau = u_tau/a_w:  M_t = sqrt(2 k^+) M_tau / sqrt(T/T_w).
        This is the first compressibility-aware conditioning feature, computed
        identically for every dataset in the family.
        """
        return np.sqrt(2.0 * self.k) * self.wall["m_tau"] / np.sqrt(self.T)

    # ---- physics-anchor identities -----------------------------------------

    def total_stress_plus(self):
        """Total shear stress in tau_w units: mu^+ dU^+/dy^+ - <rho u"v">^+.

        The Favre shear in momentum-flux form is recovered from the record's
        velocity^2 form by <rho u"v">^+ = rho^+ R_xy^+. The viscous term uses
        <mu> d{u}/dy, neglecting the viscosity-fluctuation correlation
        <mu' du'/dy> (small at these conditions); the residual against the
        exact balance is exactly what the physics anchor measures.
        """
        return self.mu * self.dUdy() - self.rho * self.uv

    def total_stress_target(self):
        """The exact fully-developed channel balance for a uniform driving
        force per unit VOLUME (equivalently a constant mean pressure
        gradient): tau_total/tau_w = 1 - y/delta. Subclasses whose DNS is
        driven per unit MASS override with the density-weighted form.
        """
        return 1.0 - self.y_outer

    def total_stress_target_mass_forced(self):
        """The balance for a uniform driving force per unit MASS (f ~ rho g):
        d(tau)/dy = -<rho> g  =>  tau/tau_w = 1 - int_0^y <rho> dy' /
        int_0^delta <rho> dy' (the zero-crossing at the centreline fixes g).
        """
        dy = np.diff(self.y_outer)
        cum = np.concatenate([[0.0],
                              np.cumsum(0.5 * (self.rho[1:] + self.rho[:-1]) * dy)])
        return 1.0 - cum / cum[-1]

    def total_stress_residual(self):
        """Deviation of the file's own total stress from the exact balance.

        A data-only identity of the converged DNS (plus the small differencing
        and viscosity-correlation terms); its interior rms anchors the modeled
        observation sigma (UQ.datasets.observation_sigma).
        """
        return self.total_stress_plus() - self.total_stress_target()

    # ---- DNSField adapter ---------------------------------------------------

    def to_dnsfield(self, timescale_plus, nu_t_plus=None, Pr_t=0.9,
                    k_baseline_plus=None):
        """Build a UQ DNSField carrying both discrepancy legs.

        The anisotropy leg needs only the record (b is unit-free); the
        heat-flux leg becomes active when the caller supplies the baseline
        nu_t^+ (heatflux_discrepancy then compares q_hat against the
        gradient-diffusion flux -(nu_t^+/Pr_t) grad(T/T_w), consistently in
        (u_tau, T_w) units per the module convention). Supplying the
        baseline's own k^+ as well switches the Boussinesq anisotropy to its
        limiter-consistent actual-eddy-viscosity form (DNSField.k_baseline).
        """
        return DNSField(
            grad_u=self.velocity_gradient(),
            R=self.R,
            k=self.k,
            timescale=np.asarray(timescale_plus, dtype=float),
            grad_T=self.temperature_gradient(),
            heat_flux=self.q_hat,
            nu_t=None if nu_t_plus is None else np.asarray(nu_t_plus, dtype=float),
            Pr_t=Pr_t,
            k_baseline=None if k_baseline_plus is None
            else np.asarray(k_baseline_plus, dtype=float),
            meta=dict(self.meta),
        )
