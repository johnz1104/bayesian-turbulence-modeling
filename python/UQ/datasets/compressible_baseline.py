"""One-dimensional fully-developed compressible SST baseline, and the
frozen-mean flat-plate reconstruction (the reviewer-approved baseline routes).

The in-tree compressible SIMPLE solver is low-Mach (validated at Ma 0.1,
ceiling about Ma 0.5), so the RANS baseline at the data's Mach numbers is
profile-based: a genuine one-dimensional fully-developed compressible SST
solve on each channel case, and an algebraic SST-consistent reconstruction at
the flat-plate analysis station. The 1-D solve is a real model prediction
whose misspecification the UQ then covers; nothing DNS-derived enters it
except the case's operating point (Re_tau_w, M_tau), its molecular-property
laws mu(T) and Pr(T) (interpolated from the case's own mean profiles, so no
gas-model assumption is stacked on the data's), and gamma.

Formulation (wall units: y+ = y u_tau/nu_w, U+ = U/u_tau, T_hat = T/T_w,
mu_hat = mu/mu_w, k+ = k/u_tau^2, omega+ = omega nu_w/u_tau^2; p is uniform
across the channel so rho_hat = 1/T_hat):

  momentum (first integral, per-unit-mass forcing as the data pins):
      (mu_hat + mu_t_hat) dU+/dy+ = 1 - W(y),
      W = int_0^y rho_hat dy' / int_0^delta rho_hat dy'
  energy (first integral, dissipation form; the only compressible parameter
  is (gamma - 1) M_tau^2 = u_tau^2/(cp T_w)):
      (mu_hat Pr_w/Pr + mu_t_hat Pr_w/Pr_t) dT_hat/dy+ = Pr_w C(y),
      C(y) = (gamma-1) M_tau^2 int_y^delta (mu_hat + mu_t_hat)(dU+/dy+)^2 dy'
      predicted wall-heat-flux parameter B_q = -C(0)     [negative: into wall]
  SST k and omega transport (Menter 2003 constants, F1/F2 blending on the
  wall distance, Bradshaw limiter mu_t = rho a1 k / max(a1 omega, S F2)),
  solved as damped tridiagonal sweeps alternating with the two quadratures.

Predicted QoIs per case: U+(y+), T_hat(y+), cf = 2/(rho_hat_CL U+_CL^2)
(the source's centreline-dynamic-head definition), B_q, and the
gradient-diffusion heat-flux profile q_GDH = -(nu_t+/Pr_t) dT_hat/dy+ in the
record's (u_tau, T_w) units. The baseline closure quantities the discrepancy
machinery consumes are nu_t+ = mu_t_hat/rho_hat, the timescale
tau+ = 1/(C_mu omega+) (the loader-side convention), and k+.

Failure handling follows the house forward-model pattern: a status string
(converged / unconverged / diverged / invalid_parameters), never an
exception for control flow.
"""
import numpy as np

# Menter (2003) SST constants; a1 and betaStar are the calibration handles
# shared with the C++ solver's parameter sets, Pr_t joins them here
SST_DEFAULTS = {
    "a1": 0.31, "betaStar": 0.09,
    "sigma_k1": 0.85, "sigma_k2": 1.0,
    "sigma_w1": 0.5, "sigma_w2": 0.856,
    "beta1": 0.075, "beta2": 0.0828,
    "kappa": 0.41,
    "Pr_t": 0.9,
}


class CompressibleChannelSST:
    """Fully-developed compressible SST channel in wall units.

    Inputs per case (all from the loader record): re_tau_w (domain height in
    y+), m_tau, gamma, and the molecular-property samples (T_hat, mu_hat) and
    (T_hat, Pr) from the case's own mean profiles. Coefficients override
    SST_DEFAULTS (a1, betaStar, Pr_t are the calibration handles).
    """

    def __init__(self, re_tau_w, m_tau, gamma, T_mu_samples, T_pr_samples,
                 coeffs=None, n=241, y1_plus=0.2):
        self.re_tau = float(re_tau_w)
        self.m_tau = float(m_tau)
        self.gamma = float(gamma)
        self._mu_T = _monotone_interp(T_mu_samples)
        self._pr_T = _monotone_interp(T_pr_samples)
        self.pr_w = float(self._pr_T(1.0))
        self.c = dict(SST_DEFAULTS)
        if coeffs:
            self.c.update(coeffs)
        # geometric wall-normal grid, y+ = 0 to the centreline
        r = _stretch_ratio(self.re_tau, y1_plus, n - 1)
        steps = y1_plus * r ** np.arange(n - 1)
        self.y = np.concatenate([[0.0], np.cumsum(steps)])
        self.y *= self.re_tau / self.y[-1]
        self.n = n

    # ---- SST blending -------------------------------------------------------

    def _blend(self, k, om, rho, mu, dkdy, domdy):
        """Menter F1/F2 on the wall distance (wall units)."""
        y = np.maximum(self.y, 1e-12)
        nu = mu / rho
        cd = np.maximum(2.0 * rho * self.c["sigma_w2"] / np.maximum(om, 1e-12)
                        * dkdy * domdy, 1e-10)
        arg1 = np.minimum(
            np.maximum(np.sqrt(np.maximum(k, 0.0))
                       / (self.c["betaStar"] * np.maximum(om, 1e-12) * y),
                       500.0 * nu / (y ** 2 * np.maximum(om, 1e-12))),
            4.0 * rho * self.c["sigma_w2"] * k / (cd * y ** 2))
        F1 = np.tanh(arg1 ** 4)
        arg2 = np.maximum(2.0 * np.sqrt(np.maximum(k, 0.0))
                          / (self.c["betaStar"] * np.maximum(om, 1e-12) * y),
                          500.0 * nu / (y ** 2 * np.maximum(om, 1e-12)))
        F2 = np.tanh(arg2 ** 2)
        return F1, F2

    def _gamma_coeff(self, F1):
        """SST omega-production coefficient gamma_i, blended."""
        c = self.c
        g1 = c["beta1"] / c["betaStar"] \
            - c["sigma_w1"] * c["kappa"] ** 2 / np.sqrt(c["betaStar"])
        g2 = c["beta2"] / c["betaStar"] \
            - c["sigma_w2"] * c["kappa"] ** 2 / np.sqrt(c["betaStar"])
        return F1 * g1 + (1.0 - F1) * g2

    # ---- the solve ----------------------------------------------------------

    def solve(self, max_iter=4000, tol=1e-9, relax=0.6, relax_T=0.3):
        """Alternate the two quadratures (U, T) with damped tridiagonal k and
        omega sweeps until the profiles stop changing. Returns a dict with
        status and, when converged, the profiles and QoIs.

        The temperature update is under-relaxed (relax_T): the T -> mu, rho
        -> dissipation -> T loop is the stiff coupling at strong wall cooling
        (the highest-B_q case runs away without it), and the mean-flow
        updates inherit the same damping through the relaxed T.
        """
        y, n, c = self.y, self.n, self.c
        T = np.ones(n)
        U = np.zeros(n)
        k = 0.01 * np.minimum(y, 10.0)
        om = np.maximum(6.0 / (0.075 * np.maximum(y, 0.3) ** 2), 1e-3)
        # log-layer eddy-viscosity initial guess, mu_t ~ kappa y (with van
        # Driest damping): starting from zero leaves a laminar first iterate
        # whose dissipation transient trips the divergence guard at the
        # largest domain heights
        mut = c["kappa"] * y * (1.0 - np.exp(-y / 26.0)) ** 2
        status = "unconverged"
        for it in range(max_iter):
            rho = 1.0 / T
            mu = self._mu_T(T)
            # momentum quadrature: mass-forced total stress
            Wm = _cumtrapz(rho, y)
            tau = 1.0 - Wm / Wm[-1]
            dUdy = tau / (mu + mut)
            U_new = _cumtrapz(dUdy, y)
            # energy quadrature: dissipation integral, centreline-adiabatic
            diss = (mu + mut) * dUdy ** 2
            Cint = (self.gamma - 1.0) * self.m_tau ** 2 \
                * (_trapz_tail(diss, y))
            G = mu / self._pr_T(T) + mut / c["Pr_t"]
            dTdy = Cint / G
            T_new = T + relax_T * (1.0 + _cumtrapz(dTdy, y) - T)
            # k / omega damped sweeps on the frozen mean field
            S = np.abs(dUdy)
            dk = np.gradient(k, y)
            dom = np.gradient(om, y)
            F1, F2 = self._blend(k, om, rho, mu, dk, dom)
            sig_k = F1 * c["sigma_k1"] + (1.0 - F1) * c["sigma_k2"]
            sig_w = F1 * c["sigma_w1"] + (1.0 - F1) * c["sigma_w2"]
            beta = F1 * c["beta1"] + (1.0 - F1) * c["beta2"]
            gam = self._gamma_coeff(F1)
            P = np.minimum(mut * S ** 2,
                           10.0 * c["betaStar"] * rho * k * om)
            k = _tridiag_sweep(y, mu + sig_k * mut, P,
                               c["betaStar"] * rho * om, k,
                               bc0=0.0, relax=relax)
            om_w = 60.0 * (mu[1] / rho[1]) / (c["beta1"] * y[1] ** 2)
            P_om = gam * rho / np.maximum(mut, 1e-12) * P
            cd_kw = 2.0 * (1.0 - F1) * rho * c["sigma_w2"] \
                / np.maximum(om, 1e-12) * dk * dom
            om = _tridiag_sweep(y, mu + sig_w * mut, P_om + cd_kw,
                                beta * rho * om, om,
                                bc0=om_w, relax=relax)
            k = np.maximum(k, 0.0)
            om = np.maximum(om, 1e-8)
            # Bradshaw-limited eddy viscosity
            mut_new = rho * c["a1"] * k \
                / np.maximum(c["a1"] * om, F2 * S)
            mut_new[0] = 0.0
            dU_change = np.max(np.abs(U_new - U)) / max(np.max(U_new), 1e-12)
            dT_change = np.max(np.abs(T_new - T))
            U, T = U_new, T_new
            mut = relax * mut_new + (1.0 - relax) * mut
            if not np.all(np.isfinite(U)) or not np.all(np.isfinite(T)) \
                    or (it > 10 and T.max() > 20.0):
                status = "diverged"
                break
            # clip the early-transient temperature so a start-up overshoot
            # cannot poison the property lookups (the converged states sit
            # well inside the cap)
            T = np.minimum(T, 20.0)
            if it > 20 and dU_change < tol and dT_change < tol:
                status = "converged"
                break
        rho = 1.0 / T
        nu_t = mut / rho
        dTdy = np.gradient(T, y)
        q_gdh = np.zeros((n, 3))
        q_gdh[:, 1] = -(nu_t / c["Pr_t"]) * dTdy
        out = {
            "status": status, "iterations": it + 1,
            "y_plus": y, "U_plus": U, "T_hat": T, "rho_hat": rho,
            "k_plus": k, "omega_plus": om, "mu_t_hat": mut,
            "nu_t_plus": nu_t,
            "timescale_plus": 1.0 / (0.09 * om),
            "q_gdh_hat": q_gdh,
            "cf": 2.0 / (rho[-1] * max(U[-1], 1e-12) ** 2),
            "b_q": -(self.gamma - 1.0) * self.m_tau ** 2
                   * float(np.trapz((self._mu_T(T) + mut)
                                    * (np.gradient(U, y)) ** 2, y)),
        }
        return out


class FlatPlateFrozenSST:
    """SST-consistent algebraic reconstruction at the flat-plate station.

    The plate has no fully-developed limit, so its baseline closure
    quantities come from the model constants evaluated on the case's own
    mean profiles (stated as such, per the confirmed scope decision): the
    damped mixing length l = kappa y D with the van Driest factor
    D = 1 - exp(-y*/A+), the eddy viscosity mu_t = rho l^2 |dU/dy|, the
    equilibrium specific dissipation omega = |dU/dy| / sqrt(betaStar), and
    the GDH heat flux from the mean temperature gradient at the chosen Pr_t.
    Nothing from the DNS Reynolds stress or heat flux enters, so both
    discrepancy legs remain genuine model-versus-data statements.
    """

    APLUS = 26.0

    def __init__(self, dns, coeffs=None):
        self.dns = dns
        self.c = dict(SST_DEFAULTS)
        if coeffs:
            self.c.update(coeffs)

    def closure(self):
        d = self.dns
        dUdy = np.gradient(d.U, d.yplus)
        S = np.abs(dUdy)
        ystar = d.ystar if d.ystar is not None else d.yplus
        damping = 1.0 - np.exp(-ystar / self.APLUS)
        ell = self.c["kappa"] * d.yplus * damping
        nu_t = ell ** 2 * S                      # kinematic, wall units
        omega = S / np.sqrt(self.c["betaStar"])
        omega = np.maximum(omega, 1e-8)
        q_gdh = np.zeros((d.n, 3))
        q_gdh[:, 1] = -(nu_t / self.c["Pr_t"]) * d.dTdy()
        return {
            "nu_t_plus": nu_t,
            "omega_plus": omega,
            "timescale_plus": 1.0 / (0.09 * omega),
            "q_gdh_hat": q_gdh,
        }


# ---- small numerics ---------------------------------------------------------

def _monotone_interp(samples):
    """Linear interpolant over (T_hat, value) samples with edge clamping."""
    T, v = np.asarray(samples[0], dtype=float), np.asarray(samples[1],
                                                           dtype=float)
    order = np.argsort(T)
    T, v = T[order], v[order]

    def f(x):
        return np.interp(np.asarray(x, dtype=float), T, v)
    return f


def _stretch_ratio(total, first, m):
    """Geometric ratio r with first step `first` summing to `total` over m
    steps. The sum is monotone increasing in r, so bisection on
    [1 + 1e-6, 1.5] is robust for every domain height used here."""
    lo, hi = 1.0 + 1e-6, 1.5
    for _ in range(200):
        r = 0.5 * (lo + hi)
        s = first * (r ** m - 1.0) / (r - 1.0)
        if s < total:
            lo = r
        else:
            hi = r
        if hi - lo < 1e-14:
            break
    return 0.5 * (lo + hi)


def _cumtrapz(f, y):
    return np.concatenate([[0.0],
                           np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(y))])


def _trapz_tail(f, y):
    """int_y^end f dy' at every node (the energy quadrature kernel)."""
    total = _cumtrapz(f, y)
    return total[-1] - total


def _tridiag_sweep(y, diff_coeff, source, sink_coeff, phi, bc0, relax):
    """One damped implicit sweep of d/dy(D dphi/dy) + source - sink*phi = 0.

    Dirichlet at the wall (bc0), zero-gradient at the centreline. The sink is
    linearised as sink_coeff * phi (Patankar-style positive treatment).
    """
    n = y.size
    a = np.zeros(n)
    b = np.zeros(n)
    cc = np.zeros(n)
    d = np.zeros(n)
    b[0] = 1.0
    d[0] = bc0
    Df = 0.5 * (diff_coeff[1:] + diff_coeff[:-1])   # face diffusivity
    dy = np.diff(y)
    for i in range(1, n - 1):
        w = Df[i - 1] / dy[i - 1]
        e = Df[i] / dy[i]
        vol = 0.5 * (y[i + 1] - y[i - 1])
        a[i] = -w
        cc[i] = -e
        b[i] = w + e + sink_coeff[i] * vol
        d[i] = source[i] * vol
    a[n - 1] = -1.0
    b[n - 1] = 1.0
    d[n - 1] = 0.0
    new = _thomas(a, b, cc, d)
    return relax * new + (1.0 - relax) * phi


def _thomas(a, b, c, d):
    n = b.size
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
