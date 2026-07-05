"""Isothermal-wall supersonic channel DNS loader (Coleman, Kim and Moser 1995).

Parses the Coleman-Kim-Moser supersonic channel data (J. Fluid Mech. 305
(1995) 159-183; TSF9 1993; hosted by the migrated NASA Turbulence Modeling
Resource) into the canonical compressible dns_field record
(UQ.datasets._compressible). Two cases, bulk Mach 1.5 (Case A) and 3.0
(Case B), isothermal walls: the canonical named anchors and an
independent-code cross-check of the loader, baseline and discrepancy
conventions at conditions inside the Gerolymos-Vallet matrix span. These
cases are never calibrated on (the pre-registered role).

Raw layout, under DNS_data/isotherm-wall_supersonic_channel/: M=1x5.dist and
M=3.dist (grouped whitespace tables with prose headers; the _additional files
carry mass fluxes and budgets, not consumed here). Normalization as stated in
the headers (TSF9 / JFM 1995): velocities by the bulk velocity U_m, density
by the bulk density rho_m, temperature by the wall temperature T_w, pressure
by rho_m U_m^2, viscosity by its wall value; u_tau (in U_m units) and Re_tau
are printed in the header. The wall row of the pressure column encodes the
bulk Mach number (p_w = rho_w/(gamma M_B^2) in these units), which the loader
tests use as a data-only cross-check of the filename Mach.

The file spans the FULL channel (y in [-1, 1]); the record keeps the lower
half, wall to centreline (y_outer = 1 + y), and the loader tests report the
half-to-half asymmetry as a statistical-convergence diagnostic. Groups
consumed (by their own header-line names, never by position):

  y, <rho>, <p>, <T>, <u>, <v>, <w>, <mu>          Reynolds means
  y, <rho.T>/<rho>, <rho.u>/<rho>, ...             Favre means
  y, <u'u'>, <v'v'>, <w'w'>, <u'v'>, <T'T'>, <v'T'>          Reynolds moments
  y, <rho.u"u">/<rho>, ..., <rho.u"v">/<rho>, <rho.T"T">/<rho>,
     <rho.v"T">/<rho>                              Favre moments (the record)

Unit conversions to the canonical record (Favre, wall units): velocities and
u_tau-scaled moments divide by the header u_tau; the Favre temperature flux
is q_hat_y = [<rho.v"T">/<rho>] / (u_tau T_w-normalized T), which for the
file's units is the column divided by u_tau (T is already on T_w, and
constant cp makes <rho h"v">/(<rho> cp) = <rho T"v">/<rho> exactly). The
temperature-velocity correlation IS carried by these files (Reynolds and
Favre forms), so this set cross-checks the heat-flux leg as well as the
stress leg.

Derived (labeled derived): the wall-heat-flux parameter B_q =
-(d(T/T_w)/dy)_wall / (Pr Re_tau) from the file's own temperature profile
(one-sided second-order difference at the wall), with the molecular Prandtl
number Pr = 0.7 stated by the reference. The physics anchor is the
variable-density total-stress balance, with the forcing form (uniform per
volume versus per mass) pinned from the data exactly as for the
Gerolymos-Vallet matrix.
"""
import os
import re

import numpy as np

from . import _common
from ._compressible import CompressibleProfileDNS

CKM_CASES = ("M1p5", "M3p0")

_SUBDIR = "isotherm-wall_supersonic_channel"

_FILES = {"M1p5": "M=1x5.dist", "M3p0": "M=3.dist"}
_BULK_MACH = {"M1p5": 1.5, "M3p0": 3.0}

# molecular properties stated in the reference (CKM 1995: Pr = 0.7,
# mu ~ T^0.7, gamma = 1.4)
_PR = 0.7
_GAMMA = 1.4

_UTAU_LINE = re.compile(r"utau\s*=\s*([0-9.eE+-]+)\s+and\s+Re-tau\s*=\s*"
                        r"([0-9.eE+-]+)")

# the consumed groups, keyed by their full header-line name tuple
_G_MEAN = ("y", "<rho>", "<p>", "<T>", "<u>", "<v>", "<w>", "<mu>")
_G_FAVRE_MEAN = ("y", "<rho.T>/<rho>", "<rho.u>/<rho>", "<rho.v>/<rho>",
                 "<rho.w>/<rho>")
_G_REYNOLDS2 = ("y", "<u'u'>", "<v'v'>", "<w'w'>", "<u'v'>", "<T'T'>",
                "<v'T'>")
_G_FAVRE2 = ("y", '<rho.u"u">/<rho>', '<rho.v"v">/<rho>', '<rho.w"w">/<rho>',
             '<rho.u"v">/<rho>', '<rho.T"T">/<rho>', '<rho.v"T">/<rho>')


class CKMChannelDNS(CompressibleProfileDNS):
    """One Coleman-Kim-Moser case as the canonical record (lower half).

    In addition to the compressible base record:

      U_reynolds, T_reynolds   Reynolds-mean views (the record's U and T are
                               the Favre means, per the convention)
      reynolds                 the Reynolds second moments by name (views for
                               the Favre-versus-Reynolds cross-checks,
                               including <v'T'>)
      u_tau_bulk               u_tau in the file's bulk-velocity units
      asymmetry                half-to-half rms asymmetry of U+ (diagnostic)
      m_bulk                   bulk Mach number (filename, cross-checked
                               against the wall pressure)
    """

    def __init__(self, case, groups, u_tau, re_tau, meta):
        y = groups[_G_MEAN]["y"]
        lower = y <= 0.0
        n_low = int(lower.sum())

        def half(col):
            return col[lower]

        y_outer = 1.0 + half(y)
        yplus = y_outer * re_tau

        mean = groups[_G_MEAN]
        favre_mean = groups[_G_FAVRE_MEAN]
        favre2 = groups[_G_FAVRE2]

        U = half(favre_mean["<rho.u>/<rho>"]) / u_tau
        T = half(favre_mean["<rho.T>/<rho>"])
        rho_w = mean["<rho>"][0]
        rho = half(mean["<rho>"]) / rho_w
        mu = half(mean["<mu>"])

        # Favre stress in velocity^2 wall units: the file's <rho.ui"uj">/<rho>
        # columns are exactly R_ij in U_m^2 units, so divide by u_tau^2
        R = _common.assemble_tensor(
            half(favre2['<rho.u"u">/<rho>']) / u_tau ** 2,
            half(favre2['<rho.v"v">/<rho>']) / u_tau ** 2,
            half(favre2['<rho.w"w">/<rho>']) / u_tau ** 2,
            half(favre2['<rho.u"v">/<rho>']) / u_tau ** 2)
        k = 0.5 * np.trace(R, axis1=1, axis2=2)

        # Favre temperature flux in (u_tau, T_w) units: constant cp makes
        # <rho h"v">/(<rho> cp) = <rho T"v">/<rho> (the file's column, in
        # U_m T_w units), so the conversion is division by u_tau alone
        q_hat = np.zeros((n_low, 3))
        q_hat[:, 1] = half(favre2['<rho.v"T">/<rho>']) / u_tau

        # derived wall-heat-flux parameter (labeled derived): q_w = -lambda_w
        # (dT/dy)_w with lambda_w = mu_w cp / Pr gives, in the record's units,
        # B_q = -(d(T/T_w)/dy_outer)_wall / (Pr Re_tau); one-sided
        # second-order difference on the first three stations
        y0, y1, y2 = y_outer[0], y_outer[1], y_outer[2]
        T0, T1, T2 = T[0], T[1], T[2]
        h1, h2 = y1 - y0, y2 - y0
        dTdy_wall = (T1 * h2 ** 2 - T2 * h1 ** 2
                     - T0 * (h2 ** 2 - h1 ** 2)) / (h1 * h2 * (h2 - h1))
        b_q = -dTdy_wall / (_PR * re_tau)

        m_bulk = _BULK_MACH[case]
        wall = {
            "b_q": float(b_q),
            "b_q_note": "derived from the wall temperature gradient, Pr = 0.7",
            "m_tau": u_tau * m_bulk,   # a_w = U_m / M_B in the file's units
            "re_tau_star": None,
            "gamma_w": _GAMMA,
            "pr_w": _PR,
            "m_bulk": m_bulk,
        }
        super().__init__(
            yplus=yplus, U=U, R=R, k=k, re_tau=re_tau, meta=meta, T=T,
            rho=rho, mu=mu, y_outer=y_outer, ystar=None, q_hat=q_hat,
            mach=None, pr_molecular=None, wall=wall)
        self.case = case
        self.u_tau_bulk = u_tau
        self.m_bulk = m_bulk
        self.U_reynolds = half(mean["<u>"]) / u_tau
        self.T_reynolds = half(mean["<T>"])
        self.p_reynolds = half(mean["<p>"])
        self.reynolds = {name: half(col) for name, col
                         in groups[_G_REYNOLDS2].items() if name != "y"}
        # half-to-half asymmetry of the Favre mean velocity (statistical
        # convergence diagnostic; the record keeps the lower half). The upper
        # half is reversed to wall-to-centreline order so both halves align
        # station by station (the centreline node, y = 0, belongs to both).
        u_full = groups[_G_FAVRE_MEAN]["<rho.u>/<rho>"]
        u_lower = u_full[y <= 0.0]
        u_upper = u_full[y >= 0.0][::-1]
        n = min(u_lower.size, u_upper.size)
        scale = np.max(np.abs(u_full))
        self.asymmetry = float(np.sqrt(np.mean(
            (u_lower[:n] - u_upper[:n]) ** 2)) / scale)

    # ---- physics anchor -------------------------------------------------------

    def total_stress_target(self):
        """CKM forcing form, pinned from the data like the Gerolymos-Vallet
        matrix: at bulk M 3.0 the per-unit-mass target closes the outer
        balance five times better than the uniform-force target (0.5 versus
        2.8 percent rms); at bulk M 1.5 the closure is form-insensitive at
        that case's own ~1 percent convergence level (the three-digit header
        u_tau contributes), which is then its honest anchor level."""
        return self.total_stress_target_mass_forced()

    def bulk_mach_from_wall_pressure(self):
        """Data-only bulk Mach cross-check of the file's units: the ideal-gas
        wall state gives p_w = rho_w R T_w; with the file's normalization
        (rho on rho_m, p on rho_m U_m^2, T_w = 1) and a_w^2 = gamma R T_w,
        M_B = U_m/a_w = sqrt(rho_w / (gamma p_w))."""
        rho_w_bulk = self.meta["rho_wall_bulk"]
        p_w_bulk = float(self.p_reynolds[0])
        return float(np.sqrt(rho_w_bulk / (_GAMMA * p_w_bulk)))

    # ---- location and parsing -------------------------------------------------

    @staticmethod
    def case_path(case, root=None):
        return os.path.join(_common.data_root(root), _SUBDIR, _FILES[case])

    @staticmethod
    def is_available(case, root=None):
        return os.path.isfile(CKMChannelDNS.case_path(case, root))

    @staticmethod
    def _parse_groups(lines):
        """Parse every named column-group table in the file.

        A group starts at a header line whose tokens are non-numeric names
        beginning with 'y'; its data are the following lines that parse as
        exactly len(names) floats (separator/prose lines end it). Groups are
        keyed by their full name tuple, so a format change fails loudly.
        """
        groups = {}
        i = 0
        while i < len(lines):
            tokens = tuple(lines[i].split())
            if len(tokens) >= 4 and tokens[0] == "y" \
                    and not _is_float(tokens[1]):
                names = tokens
                rows = []
                j = i + 1
                while j < len(lines):
                    parts = lines[j].split()
                    if len(parts) == len(names) and all(_is_float(p)
                                                        for p in parts):
                        rows.append([float(p) for p in parts])
                    elif rows:
                        break
                    j += 1
                if rows:
                    data = np.asarray(rows)
                    groups[names] = {name: data[:, c]
                                     for c, name in enumerate(names)}
                i = j
            else:
                i += 1
        return groups

    @staticmethod
    def load(case, root=None):
        """Parse one case into the canonical record (lower half)."""
        path = CKMChannelDNS.case_path(case, root)
        with open(path) as fh:
            lines = fh.readlines()
        m = None
        for line in lines:
            m = _UTAU_LINE.search(line)
            if m is not None:
                break
        if m is None:
            raise ValueError(f"{path}: no 'utau = ... and Re-tau = ...' line")
        u_tau, re_tau = float(m.group(1)), float(m.group(2))
        groups = CKMChannelDNS._parse_groups(lines)
        for wanted in (_G_MEAN, _G_FAVRE_MEAN, _G_REYNOLDS2, _G_FAVRE2):
            if wanted not in groups:
                raise ValueError(f"{path}: group {wanted[1]}... not found")
        meta = {
            "regime": "compressible",
            "case": "ckm_supersonic_channel",
            "averaging": "favre",
            "source": "Coleman, Kim and Moser 1995 (JFM 305); TMR-hosted",
            "case_tag": case,
            "wall_thermal": "isothermal",
            "rho_wall_bulk": float(groups[_G_MEAN]["<rho>"][0]),
            "sigma_note": "modeled observation uncertainty "
                          "(no per-point statistical uncertainty in file)",
        }
        return CKMChannelDNS(case=case, groups=groups, u_tau=u_tau,
                             re_tau=re_tau, meta=meta)

    @staticmethod
    def load_all(root=None):
        return [CKMChannelDNS.load(c, root) for c in CKM_CASES
                if CKMChannelDNS.is_available(c, root)]

    def __repr__(self):
        return (f"CKMChannelDNS({self.case}, M_bulk={self.m_bulk:g}, "
                f"Re_tau={self.re_tau:.1f}, n={self.n})")


def _is_float(token):
    return re.fullmatch(r"[-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?", token) \
        is not None
