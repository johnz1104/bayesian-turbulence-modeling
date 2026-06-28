"""Plane Couette DNS loader (Pirozzoli, Bernardini and Orlandi 2014).

Parses the raw Roma plane-Couette profiles into the canonical dns_field record
(DNS_data/README.md, "Standardized processing format"), the SAME record the
Lee-Moser channel loader emits, so the cross-flow generalization test (DNS_plan.md
Step 2) drives the Step-1 discrepancy / UQ / evaluation machinery unchanged.

Raw layout, per friction Reynolds number, under DNS_data/couette_flow/:

  roma_couette_<Re_tau>.txt
      header: '*' blocks and bare prose lines (NOT '%'); the friction Reynolds
              number is in the title line "... AT RETAU = <N>"
      cols (7): y, y^+, U^+, u'^+, v'^+, w'^+, u'v'^+

Note the velocity fluctuations u', v', w' are root-mean-square, so the normal
stresses are their squares (uu = u'^2, vv = v'^2, ww = w'^2); u'v'^+ is already
the (signed, negative) shear covariance and maps straight to R_xy. The spanwise
off-diagonals u'w' and v'w' vanish in this parallel shear flow. There is no
mean-gradient column, so dU^+/dy^+ is finite-differenced (UQ.datasets._common),
and no k column, so k^+ = 0.5 (uu + vv + ww). Everything is in wall units; there
is no separate nu or u_tau (the anisotropy and the discrepancy are dimensionless).

The profile runs from one wall (y^+ = 0, U^+ = 0) to the centerline (y/h = 1,
U^+ = U_wall^+ ~ 21.5 at Re_tau 986): the half-gap, like the half-channel record.
The Couette total stress is exactly constant, dU^+/dy^+ - u'v'^+ = 1 across the
gap, which the observation-sigma anchor (UQ.datasets.observation_sigma) uses as
the data-only physics residual.
"""
import os
import re

import numpy as np

from . import _common

# friction Reynolds numbers of the compiled cases (DNS_data/README.md). These are
# the values printed in each file's RETAU title line.
COUETTE_CASES = (171, 260, 507, 986)

# the seven wall-unit columns of the raw file, in order
_COLS = ("y", "yplus", "U", "u_rms", "v_rms", "w_rms", "uv")

# the friction Reynolds number lives only in the title line "... RETAU = <N>"
_RETAU_PATTERN = re.compile(r"RETAU\s*=\s*([0-9.eE+-]+)")


class CouetteDNS(_common.WallBoundedProfileDNS):
    """A single Pirozzoli plane-Couette case as the canonical dns_field record.

    Attributes (per wall-normal station, wall units), in addition to the base
    record (yplus, U, R, k, re_tau, uu/vv/ww/uv/uw/vw):
      y_outer       y/h, the outer wall-normal coordinate (0 at the wall, 1 at
                    the centerline)
      u_rms, v_rms, w_rms   the raw rms fluctuation columns (kept for provenance;
                    the normal stresses are their squares)
    """

    def __init__(self, re_tau_nominal, y_outer, yplus, U, u_rms, v_rms, w_rms, uv,
                 meta):
        # normal stresses are the squares of the rms columns; the shear covariance
        # u'v'^+ is signed (negative) and maps directly to R_xy
        R = _common.assemble_tensor(u_rms ** 2, v_rms ** 2, w_rms ** 2, uv)
        k = 0.5 * (u_rms ** 2 + v_rms ** 2 + w_rms ** 2)
        super().__init__(yplus=yplus, U=U, R=R, k=k, re_tau=re_tau_nominal,
                         meta=meta, y=y_outer, y_outer=y_outer)
        self.re_tau_nominal = int(re_tau_nominal)
        self.u_rms = np.asarray(u_rms, dtype=float)
        self.v_rms = np.asarray(v_rms, dtype=float)
        self.w_rms = np.asarray(w_rms, dtype=float)

    # ---- location and parsing ---------------------------------------------

    @staticmethod
    def case_path(re_tau_nominal, root=None):
        root = _common.data_root(root)
        return os.path.join(root, "couette_flow",
                            f"roma_couette_{int(re_tau_nominal)}.txt")

    @staticmethod
    def is_available(re_tau_nominal, root=None):
        return os.path.isfile(CouetteDNS.case_path(re_tau_nominal, root))

    @staticmethod
    def _parse_re_tau(path):
        """Read the friction Reynolds number from the RETAU title line."""
        with open(path) as fh:
            for line in fh:
                match = _RETAU_PATTERN.search(line)
                if match is not None:
                    return float(match.group(1))
        raise ValueError(f"{path}: no 'RETAU = <N>' title line found")

    @staticmethod
    def load(re_tau_nominal, root=None):
        """Parse one Couette case (by friction Reynolds number) into a record."""
        path = CouetteDNS.case_path(re_tau_nominal, root)
        re_tau = CouetteDNS._parse_re_tau(path)
        skip = _common.first_data_row(path, len(_COLS), comment_prefixes=("%", "*"))
        data = np.loadtxt(path, skiprows=skip)
        if data.shape[1] != len(_COLS):
            raise ValueError(f"{path}: expected {len(_COLS)} columns, "
                             f"got {data.shape[1]}")
        cols = {name: data[:, j] for j, name in enumerate(_COLS)}
        meta = {
            "regime": "incompressible",
            "case": "plane_couette",
            "source": "Pirozzoli, Bernardini and Orlandi 2014 (JFM 742)",
            "re_tau_nominal": int(re_tau_nominal),
            "file": os.path.relpath(path, _common.data_root(root)),
            "sigma_note": "modeled observation uncertainty (no DNS _stdev in file)",
        }
        return CouetteDNS(
            re_tau_nominal=re_tau, y_outer=cols["y"], yplus=cols["yplus"],
            U=cols["U"], u_rms=cols["u_rms"], v_rms=cols["v_rms"],
            w_rms=cols["w_rms"], uv=cols["uv"], meta=meta)

    @staticmethod
    def load_all(root=None):
        """Load every available Couette case, ordered by friction Reynolds number."""
        return [CouetteDNS.load(n, root) for n in COUETTE_CASES
                if CouetteDNS.is_available(n, root)]

    # ---- derived quantities -----------------------------------------------

    def wall_velocity(self):
        """The moving-wall speed in wall units, U_wall^+ = U^+ at the centerline.

        The half-gap profile runs from the stationary reference (U^+ = 0 at the
        wall) to the centerline, whose mean velocity is the average of the two
        wall speeds; in the frame here that centerline value is U_wall^+.
        """
        return float(self.U[-1])

    def total_stress_plus(self):
        """Total shear stress in wall units, dU^+/dy^+ - <u'v'>^+.

        For plane Couette this is exactly 1 across the gap (constant total stress,
        no pressure gradient), so its deviation from 1 is a data-only convergence
        and quality measure (UQ.datasets.observation_sigma anchors the modeled
        observation sigma on it). <u'v'>^+ is the signed (negative) covariance.
        """
        return self.dUdy() - self.uv

    def __repr__(self):
        return (f"CouetteDNS(Re_tau={self.re_tau:.0f}, n={self.n}, "
                f"U_wall+={self.wall_velocity():.2f})")
