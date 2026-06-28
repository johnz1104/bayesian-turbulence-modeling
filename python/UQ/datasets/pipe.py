"""Turbulent pipe-flow DNS loader (Pirozzoli 2024).

Parses the Roma pipe-flow profiles into the canonical dns_field record
(DNS_data/README.md, "Standardized processing format"), the SAME record the
channel and Couette loaders emit, so the cross-geometry companion of DNS_plan.md
Step 2 reuses the Step-1 discrepancy / UQ machinery unchanged.

Raw layout, per friction Reynolds number, under DNS_data/pipe_flow/:

  Pipe_Re_tau<N>.txt
      header: clean '%' block carrying the bulk Reynolds number, the (actual)
              friction Reynolds number, and the friction factor
      cols (8): y^+, U_z^+, P^+, u_z^2, u_r^2, u_t^2, u_z u_r, p^2

The fluctuation columns are variances in cylindrical components, mapped to the
canonical Cartesian wall-bounded frame (x = streamwise, y = wall-normal, z =
spanwise):

  u_z^2  -> R_xx   (streamwise normal stress)
  u_r^2  -> R_yy   (radial = wall-normal normal stress)
  u_t^2  -> R_zz   (azimuthal = spanwise normal stress)
  u_z u_r -> R_xy  (the shear stress), WITH A SIGN FLIP (see below)

The radial coordinate r points from the axis toward the wall, so the wall-normal
direction into the fluid is -r and the wall-normal velocity fluctuation is
u_y = -u_r. Hence the canonical shear stress R_xy = <u_x u_y> = -<u_z u_r>: the
file's (positive) u_z u_r becomes a negative R_xy, exactly as the channel and
Couette shear stress is negative under a positive mean shear. This sign is not a
convention choice; it is fixed by the linear-total-stress identity dU^+/dy^+ -
R_xy = 1 - y^+/Re_tau, which the loader's total_stress_plus reproduces only with
the flip (UQ.datasets.observation_sigma anchors on it). P^+ and p^2 are pressure
statistics and are ignored. The off-diagonals u'w', v'w' vanish in this parallel
shear flow. Only y^+ is given; the outer coordinate is y/R = y^+/Re_tau, and k^+
= 0.5 (u_z^2 + u_r^2 + u_t^2). Everything is in wall units.
"""
import os
import re

import numpy as np

from . import _common

# nominal friction Reynolds numbers of the compiled cases (the filename labels;
# the actual value is read from each header, e.g. the "2000" file is Re_tau 1972)
PIPE_CASES = (500, 1140, 2000, 3000, 6000, 12000)

# the eight wall-unit columns of the raw file, in order
_COLS = ("yplus", "U", "P", "uu", "rr", "tt", "uzur", "pp")

_HEADER_PATTERNS = {
    "re_tau":          re.compile(r"Friction Reynolds number\s*=\s*([0-9.eE+-]+)"),
    "re_bulk":         re.compile(r"Bulk Reynolds number\s*=\s*([0-9.eE+-]+)"),
    "friction_factor": re.compile(r"Friction factor\s*=\s*([0-9.eE+-]+)"),
}
_NOMINAL_PATTERN = re.compile(r"Pipe_Re_tau([0-9]+)")


class PipeDNS(_common.WallBoundedProfileDNS):
    """A single Pirozzoli pipe-flow case as the canonical dns_field record.

    Attributes in addition to the base record (yplus, U, R, k, re_tau,
    uu/vv/ww/uv/uw/vw):
      re_tau_nominal   the filename label (the actual Re_tau is `re_tau`)
      re_bulk          bulk Reynolds number from the header
      friction_factor  Darcy friction factor (header value, or derived for the
                       one case whose header omits it)
    """

    def __init__(self, re_tau_nominal, re_tau, re_bulk, friction_factor,
                 yplus, U, uu, rr, tt, uzur, meta):
        # cylindrical -> Cartesian wall-bounded frame; R_xy carries the sign flip
        # R_xy = <u_x u_y> = -<u_z u_r> (wall-normal into-fluid direction is -r)
        R = _common.assemble_tensor(uu, rr, tt, -uzur)
        k = 0.5 * (np.asarray(uu, float) + np.asarray(rr, float) + np.asarray(tt, float))
        y_outer = np.asarray(yplus, float) / float(re_tau)        # y/R = y^+/Re_tau
        super().__init__(yplus=yplus, U=U, R=R, k=k, re_tau=re_tau, meta=meta,
                         y_outer=y_outer)
        self.re_tau_nominal = int(re_tau_nominal)
        self.re_bulk = float(re_bulk)
        self.friction_factor = float(friction_factor)
        self.uzur = np.asarray(uzur, dtype=float)                 # raw +ve column

    # ---- location and parsing ---------------------------------------------

    @staticmethod
    def case_path(re_tau_nominal, root=None):
        root = _common.data_root(root)
        return os.path.join(root, "pipe_flow", f"Pipe_Re_tau{int(re_tau_nominal)}.txt")

    @staticmethod
    def is_available(re_tau_nominal, root=None):
        return os.path.isfile(PipeDNS.case_path(re_tau_nominal, root))

    @staticmethod
    def _parse_header(path):
        """Read re_tau, re_bulk, and the friction factor from the '%' block.

        One compiled file (the lowest Reynolds number) omits the friction-factor
        line; there the Darcy factor is derived from f = 8 (u_tau/U_b)^2 with
        u_tau/U_b = 2 Re_tau / Re_bulk, the exact pipe relation.
        """
        params = {}
        with open(path) as fh:
            for line in fh:
                if not line.lstrip().startswith("%"):
                    break
                for key, pattern in _HEADER_PATTERNS.items():
                    if key in params:
                        continue
                    match = pattern.search(line)
                    if match is not None:
                        params[key] = float(match.group(1))
        if "re_tau" not in params or "re_bulk" not in params:
            raise ValueError(f"{path}: header missing friction/bulk Reynolds number")
        if "friction_factor" not in params:
            utau_over_ub = 2.0 * params["re_tau"] / params["re_bulk"]
            params["friction_factor"] = 8.0 * utau_over_ub ** 2
        return params

    @staticmethod
    def load(re_tau_nominal, root=None):
        """Parse one pipe case (by filename Reynolds number) into a record."""
        path = PipeDNS.case_path(re_tau_nominal, root)
        params = PipeDNS._parse_header(path)
        skip = _common.first_data_row(path, len(_COLS), comment_prefixes=("%",))
        data = np.loadtxt(path, comments="%", skiprows=skip)
        if data.shape[1] != len(_COLS):
            raise ValueError(f"{path}: expected {len(_COLS)} columns, "
                             f"got {data.shape[1]}")
        cols = {name: data[:, j] for j, name in enumerate(_COLS)}
        meta = {
            "regime": "incompressible",
            "case": "pipe_flow",
            "source": "Pirozzoli 2024 (JFM 989, A5)",
            "re_tau_nominal": int(re_tau_nominal),
            "file": os.path.relpath(path, _common.data_root(root)),
            "sigma_note": "modeled observation uncertainty (no DNS _stdev in file)",
        }
        return PipeDNS(
            re_tau_nominal=re_tau_nominal, re_tau=params["re_tau"],
            re_bulk=params["re_bulk"], friction_factor=params["friction_factor"],
            yplus=cols["yplus"], U=cols["U"], uu=cols["uu"], rr=cols["rr"],
            tt=cols["tt"], uzur=cols["uzur"], meta=meta)

    @staticmethod
    def load_all(root=None):
        """Load every available pipe case, ordered by friction Reynolds number."""
        return [PipeDNS.load(n, root) for n in PIPE_CASES
                if PipeDNS.is_available(n, root)]

    # ---- derived quantities -----------------------------------------------

    def total_stress_plus(self):
        """Total shear stress in wall units, dU^+/dy^+ - R_xy.

        In a pipe the total stress is linear, dU^+/dy^+ - R_xy = 1 - y^+/Re_tau
        (zero at the axis, one at the wall). With R_xy = -<u_z u_r> this equals
        dU^+/dy^+ + <u_z u_r>; the identity holding is the check that the shear
        sign flip is correct, and its deviation anchors the modeled observation
        sigma (UQ.datasets.observation_sigma).
        """
        return self.dUdy() - self.uv

    def total_stress_target(self):
        """The linear total-stress profile 1 - y^+/Re_tau the data should match."""
        return 1.0 - self.yplus / self.re_tau

    def __repr__(self):
        return (f"PipeDNS(Re_tau={self.re_tau:.0f} [nominal {self.re_tau_nominal}], "
                f"n={self.n}, Re_b={self.re_bulk:.3g})")
