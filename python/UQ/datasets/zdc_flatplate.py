"""Supersonic and hypersonic flat-plate DNS loader (Zhang, Duan and Choudhari 2018).

Parses the Zhang-Duan-Choudhari zero-pressure-gradient flat-plate database
(AIAA Journal 56(11) 2018, 4297-4311, doi:10.2514/1.J057296; hosted by the
migrated NASA Turbulence Modeling Resource) into the canonical compressible
dns_field record (UQ.datasets._compressible). Five cases, nominal freestream
Mach 2.5 to 14 and wall-to-recovery temperature ratio Tw/Tr 0.18 to 1.0: the
hypersonic extension of the attached matrix, the wall-cooling axis, and the
MEASURED turbulent-Prandtl-number reference profiles the calibrated Pr_t
posterior is compared against.

Raw layout, per case, under DNS_data/sup_hypersonic_plate_flow/:
  <case>_Stat.dat        Tecplot POINT profiles at one streamwise station,
                         28 columns named on the file's own '#variables=' alias
                         line (parsed, never hardcoded), with an extensive
                         README header carrying the freestream conditions and
                         the station's boundary-layer parameters.
  <case>_TKEBudget.dat   TKE budgets (present, not consumed by this loader).

SOURCE COORDINATE CONVENTION (stated in the header): x streamwise, y SPANWISE,
z WALL-NORMAL, so v is the spanwise and w the wall-normal fluctuation. The
loader maps to the pipeline convention (x streamwise, y wall-normal,
z spanwise): z -> y, w -> v; the file's <u'w'> column is the
streamwise/wall-normal shear covariance R_xy, its b13 the corresponding
anisotropy component, its b22/b33 the spanwise/wall-normal normal anisotropies.

Averaging in the record (documented mixed construction): the anisotropy is the
file's own FAVRE anisotropy columns (b11, b22, b33, b13 =
<rho u_i"u_j">/(2<rho k>) - delta_ij/3); the magnitude k and the mean profiles
are formed from the Reynolds rms and mean columns (the file gives no Favre
means). The stress record is assembled as R = 2 k (b + I/3), so the
downstream anisotropy b = R/(2k) - I/3 recovers the file's Favre anisotropy
exactly, while k carries the Reynolds level. The Favre-vs-Reynolds gap is an
O(M^2) percent-level effect measured by the loader tests.

DERIVED QUANTITY (labeled derived, never measured): the wall-normal turbulent
heat flux is NOT a raw column. It is recovered by inverting the file's own
definition of the turbulent Prandtl number,

    Pr_t := (<rho u'w'> dT/dz) / (<rho w'T'> du/dz)
    =>  <rho w'T'> = <rho u'w'> (dT/dz) / (Pr_t du/dz),

with <rho u'w'> approximated by <rho><u'w'> (the density triple correlation is
neglected; part of the derived label). In the record's (u_tau, T_w) units:

    q_hat_y = <u'w'>^+ (d(T/T_w)/dy^+) / (Pr_t dU^+/dy^+)

computed only where both finite-differenced gradients are away from their
zero crossings (the q_hat_valid mask); elsewhere q_hat is zero and masked.
Only the wall-normal component is recoverable, so the flat-plate heat-flux
discrepancy is a wall-normal-component statement. The streamwise component is
left zero and excluded by the mask semantics.

The physics anchor is the van-Driest-transform reconstruction: u_VD+ =
int sqrt(<rho>/rho_w) du+ recomputed from the file's own u+ and density
columns and compared against its u_VD+ column (a data-only cross-column
identity; measured median residual 0.06 to 0.9 percent per case). The
anisotropy trace identity b11+b22+b33 = 0 (measured 4e-10) is a parse guard,
exact by construction.
"""
import os
import re

import numpy as np

from . import _common
from ._compressible import CompressibleProfileDNS

ZDC_CASES = ("M2p5", "M6Tw025", "M6Tw076", "M8Tw048", "M14Tw018")

_SUBDIR = "sup_hypersonic_plate_flow"

# air constants stated in the file header (R = 287 J/(K kg), Pr = 0.71,
# gamma = Cp/Cv for a perfect diatomic gas)
_GAMMA = 1.4

# the '#variables=' alias names this loader consumes, as printed in the file
_WANTED = {
    "y_dim": "z",            # wall-normal coordinate, m (source z)
    "y_outer": "zod",        # z/delta
    "yplus": "zplus",        # z+
    "ystar": "zstar",        # semi-local z*
    "u_over_uinf": "u_uinf",
    "p_over_pinf": "p_pinf",
    "t_over_tinf": "t_tinf",
    "rho": "r_rhow",         # <rho>/rho_w
    "urms": "urms_utau",
    "vrms_span": "vrms_utau",   # source v = spanwise
    "wrms_wn": "wrms_utau",     # source w = wall-normal
    "m_rms": "Mrms",
    "upwp": "upwp_utausq",   # <u'w'>/u_tau^2, streamwise/wall-normal shear
    "b11": "b11",
    "b22_span": "b22",       # spanwise normal anisotropy (source y)
    "b33_wn": "b33",         # wall-normal normal anisotropy (source z)
    "b13": "b13",            # streamwise/wall-normal shear anisotropy
    "uvd_plus": "Uvd",
    "utl_plus": "Uvd_Trettel",
    "pr_t": "Prt",
    "sra_huang": "SRA_Huang",
}

_FREESTREAM_KEYS = ("m_inf", "u_inf", "rho_inf", "T_inf", "T_w", "tw_tr",
                    "delta_i_mm")
_STATION_KEYS = ("x_a_over_delta_i", "re_theta", "re_tau", "re_delta2",
                 "re_tau_star", "theta_mm", "H", "delta_mm", "ztau_mm",
                 "u_tau", "neg_b_q", "m_tau")


class FlatPlateDNS(CompressibleProfileDNS):
    """One Zhang-Duan-Choudhari flat-plate case as the canonical record.

    In addition to the compressible base record:

      pr_t          the MEASURED turbulent-Prandtl-number profile (the
                    reference the calibrated Pr_t posterior is compared to)
      sra_huang     the file's strong-Reynolds-analogy check column
      m_rms         the file's rms Mach-fluctuation column (a close cousin of
                    the computed turbulent Mach number, kept as a view)
      uvd_plus      the file's van Driest transformed velocity (anchor column)
      utl_plus      the file's Trettel-Larsson transformed velocity
      q_hat_valid   mask of stations where the derived heat flux is defined
      free, station the header condition tables, by name
    """

    def __init__(self, case, cols, free, station, meta):
        n = cols["yplus"].size
        u_plus = cols["u_over_uinf"] * (free["u_inf"] / station["u_tau"])
        # normalize temperature by the file's own wall row (the isothermal
        # wall temperature by definition); the header's tabulated Tw agrees
        # with T_inf times this row to its printed rounding (~0.2 percent),
        # asserted by the loader tests via T_wall_over_Tinf
        T = cols["t_over_tinf"] / cols["t_over_tinf"][0]

        # Reynolds k from the rms columns (u_tau^2 units), Favre shape from the
        # anisotropy columns: R = 2 k (b + I/3) recovers the file's b exactly
        k = 0.5 * (cols["urms"] ** 2 + cols["vrms_span"] ** 2
                   + cols["wrms_wn"] ** 2)
        b = _common.assemble_tensor(
            cols["b11"], cols["b33_wn"], cols["b22_span"], cols["b13"])
        eye = np.eye(3)
        R = 2.0 * k[:, None, None] * (b + eye / 3.0)
        # the wall row's anisotropy is the file's 0/0 convention (-1/3 each);
        # k = 0 there makes the stress exactly zero regardless
        R[0] = 0.0

        # wall parameters; the header tabulates -B_q, so B_q = -(listed): zero
        # for the Tw = Tr control, negative (heat into the wall) when cooled.
        # cf is derived from the header values for reference: tau_w =
        # rho_w u_tau^2 and rho_w = rho_inf T_inf/T_w at the plate's uniform
        # pressure, so cf = 2 (u_tau/U_inf)^2 (T_inf/T_w).
        wall = {
            "b_q": -station["neg_b_q"],
            "m_tau": station["m_tau"],
            "re_tau_star": station["re_tau_star"],
            "gamma_w": _GAMMA,
            "m_inf": free["m_inf"],
            "tw_tr": free["tw_tr"],
            "T_w": free["T_w"],
            "T_inf": free["T_inf"],
            "u_tau_dim": station["u_tau"],
            "cf_derived": 2.0 * (station["u_tau"] / free["u_inf"]) ** 2
                          * (free["T_inf"] / free["T_w"]),
        }
        super().__init__(
            yplus=cols["yplus"], U=u_plus, R=R, k=k, re_tau=station["re_tau"],
            meta=meta, T=T, rho=cols["rho"], mu=None,
            y_outer=cols["y_outer"], ystar=cols["ystar"], q_hat=None,
            mach=None, pr_molecular=None, wall=wall)
        self.case = case
        self.pr_t = cols["pr_t"]
        self.sra_huang = cols["sra_huang"]
        self.m_rms = cols["m_rms"]
        self.uvd_plus = cols["uvd_plus"]
        self.utl_plus = cols["utl_plus"]
        self.upwp = cols["upwp"]
        self.T_wall_over_Tinf = float(cols["t_over_tinf"][0])
        self.free = free
        self.station = station
        self.q_hat, self.q_hat_valid = self._derived_heat_flux()

    # ---- derived heat flux ----------------------------------------------------

    def _derived_heat_flux(self):
        """Wall-normal turbulent heat flux from the Pr_t definition (derived).

        q_hat_y = <u'w'>^+ (dT_hat/dy^+) / (Pr_t dU^+/dy^+), defined only away
        from the gradient zero crossings and off the wall; elsewhere zero and
        masked. See the module docstring for the derivation and the neglected
        density triple correlation.
        """
        dU = _common.wall_normal_gradient(self.yplus, self.U)
        dT = self.dTdy()
        valid = ((np.abs(dT) > 0.02 * np.abs(dT).max())
                 & (np.abs(dU) > 0.02 * np.abs(dU).max())
                 & (np.abs(self.pr_t) > 0.05)
                 & (self.yplus > 3.0) & (self.y_outer < 0.9))
        q = np.zeros((self.n, 3))
        q[valid, 1] = (self.upwp[valid] * dT[valid]
                       / (self.pr_t[valid] * dU[valid]))
        return q, valid

    # ---- physics anchor -------------------------------------------------------

    def van_driest_recomputed(self):
        """u_VD+ = int sqrt(<rho>/rho_w) du+ by trapezoid over the profile."""
        du = np.diff(self.U)
        s = np.sqrt(self.rho)
        return np.concatenate([[0.0],
                               np.cumsum(0.5 * (s[1:] + s[:-1]) * du)])

    def van_driest_residual(self):
        """Relative residual of the recomputed van Driest velocity against the
        file's own u_VD+ column (the data-only cross-column anchor identity)."""
        recomputed = self.van_driest_recomputed()
        return (recomputed - self.uvd_plus) / np.maximum(self.uvd_plus, 1e-9)

    def anisotropy_trace(self):
        """b11 + b22 + b33 from the file's own columns (exact-zero parse guard)."""
        R = self.R[1:]
        k = np.maximum(self.k[1:], 1e-30)
        return np.trace(R, axis1=1, axis2=2) / (2.0 * k) - 1.0

    # ---- location and parsing -------------------------------------------------

    @staticmethod
    def case_path(case, root=None):
        return os.path.join(_common.data_root(root), _SUBDIR,
                            f"{case}_Stat.dat")

    @staticmethod
    def is_available(case, root=None):
        return os.path.isfile(FlatPlateDNS.case_path(case, root))

    @staticmethod
    def _header_floats(lines, marker, keys):
        """The values line following the '# <marker> ...' names line."""
        for i, line in enumerate(lines):
            if re.match(rf"^#\s*{re.escape(marker)}\b", line):
                values = [float(tok) for tok in lines[i + 1].lstrip("# ").split()]
                if len(values) < len(keys):
                    raise ValueError(f"{marker}: {len(values)} values for "
                                     f"{len(keys)} keys")
                return dict(zip(keys, values))
        raise ValueError(f"header table '{marker}' not found")

    @staticmethod
    def load(case, root=None):
        """Parse one case into the canonical record."""
        path = FlatPlateDNS.case_path(case, root)
        with open(path) as fh:
            lines = fh.readlines()
        alias_idx = next((i for i, l in enumerate(lines)
                          if l.startswith("#variables=")), None)
        if alias_idx is None:
            raise ValueError(f"{path}: no '#variables=' alias line")
        names = [t.strip() for t in
                 lines[alias_idx].split("=", 1)[1].split(",")]
        data = np.loadtxt(path, skiprows=alias_idx + 1)
        if data.shape[1] != len(names):
            raise ValueError(f"{path}: {data.shape[1]} data columns, alias "
                             f"line names {len(names)}")
        m = re.search(r"I\s*=\s*(\d+)", "".join(lines[:alias_idx]))
        if m is None or int(m.group(1)) != data.shape[0]:
            raise ValueError(f"{path}: ZONE I does not match the data rows")
        index = {name: j for j, name in enumerate(names)}
        cols = {}
        for key, name in _WANTED.items():
            if name not in index:
                raise ValueError(f"{path}: column '{name}' not in alias line")
            cols[key] = data[:, index[name]]
        free = FlatPlateDNS._header_floats(lines, "Minf", _FREESTREAM_KEYS)
        station = FlatPlateDNS._header_floats(lines, "x_a/delta_i",
                                              _STATION_KEYS)
        meta = {
            "regime": "compressible",
            "case": "zdc_flatplate",
            "averaging": "favre anisotropy, reynolds magnitude and means "
                         "(documented mixed construction)",
            "source": "Zhang, Duan and Choudhari 2018 (AIAA J 56(11), "
                      "doi:10.2514/1.J057296)",
            "case_tag": case,
            "wall_thermal": f"isothermal, Tw/Tr = {free['tw_tr']:g}",
            "heat_flux_note": "wall-normal component DERIVED from the Pr_t "
                              "definition, not measured; see q_hat_valid",
            "sigma_note": "modeled observation uncertainty "
                          "(no per-point statistical uncertainty in file)",
        }
        return FlatPlateDNS(case=case, cols=cols, free=free, station=station,
                            meta=meta)

    @staticmethod
    def load_all(root=None):
        return [FlatPlateDNS.load(c, root) for c in ZDC_CASES
                if FlatPlateDNS.is_available(c, root)]

    def __repr__(self):
        return (f"FlatPlateDNS({self.case}, M_inf={self.wall['m_inf']:g}, "
                f"Tw/Tr={self.wall['tw_tr']:g}, Re_tau={self.re_tau:.0f}, "
                f"n={self.n})")
