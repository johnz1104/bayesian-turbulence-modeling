"""Streamwise-rotating turbulent channel DNS loader (University of Manitoba).

Parses the Yang and Wang (2018) streamwise-rotating channel profiles into the
canonical dns_field record (DNS_data/README.md, "Standardized processing
format"). This is the Step-2 model-stress-test companion (DNS_plan.md Step 2):
standard SST has no rotation correction, so the a-priori discrepancy is expected
to be large and structurally new, which is where calibrated uncertainty should
widen.

Raw layout, per rotation number, under DNS_data/rotating_channel_flow/:

  tu-darmstadt_re180_ro<Ro_tau>.txt   (the Ro_tau 7.5 file has no .txt suffix;
                                       all files use CRLF line endings)
      header: '%%' block; the flow conditions carry Re_tau = 180 and Ro_tau
      cols: 58, of which the first ten are used:
        y/h, U^+, W^+, <uu>^+, <vv>^+, <ww>^+, <uv>^+, <uw>^+, <vw>^+, TKE^+
      columns 11-58 are the Reynolds-stress-transport budget terms (produc, rot,
        p-strain, dissip, t-diff, p-diff, v-diff, res) for each of the six
        stresses; the residual columns res_*^+ are the DNS's own budget-closure
        check and anchor the modeled observation uncertainty.

Streamwise (x-axis) system rotation makes the mean spanwise velocity W^+ and the
out-of-plane shear stresses <uw>^+, <vw>^+ nonzero, so the mean velocity gradient
carries both dU^+/dy^+ and dW^+/dy^+ and the full anisotropy tensor is active
(the canonical 3x3 record already holds this). y/h is the outer coordinate; y^+ =
(y/h) Re_tau. The profile spans the full channel, y/h from +1 (one wall) through 0
(centerline) to -1 (the other wall), 129 stations; rotation breaks the
top/bottom symmetry of the standard channel, which the asymmetry of the loaded
profile shows directly. Everything is in wall units (Re_tau = 180 for every case).
"""
import os
import re

import numpy as np

from . import _common

# rotation numbers Ro_tau of the compiled cases (all at Re_tau = 180)
ROTATING_CASES = (7.5, 15, 30, 75, 150)
RE_TAU = 180.0

# the ten primary columns (indices 0..9); the rest are RSTE budget terms
_N_PRIMARY = 10
# 0-based indices of the per-stress budget residual columns res_*^+ (uu, vv, ww,
# uv, uw, vw): ten primary columns then eight budget terms per stress, residual
# last in each block of eight (produc, rot, p-strain, dissip, t-diff, p-diff,
# v-diff, res)
_RES_COLS = (17, 25, 33, 41, 49, 57)

_RE_TAU_PATTERN = re.compile(r"Re_tau\s*=\s*([0-9.eE+-]+)")
_RO_TAU_PATTERN = re.compile(r"Ro_tau\s*=\s*([0-9.eE+-]+)")


def _ro_label(ro_tau):
    """Filename label for a rotation number (7.5 -> '7.5', 15 -> '15')."""
    return f"{ro_tau:g}"


class RotatingChannelDNS(_common.WallBoundedProfileDNS):
    """A single streamwise-rotating channel case as the canonical dns_field record.

    Attributes in addition to the base record (yplus, U, R, k, re_tau, the full
    uu/vv/ww/uv/uw/vw and the mean W^+):
      ro_tau           rotation number Ro_tau = 2 Omega h / u_tau
      tke              the DNS turbulent kinetic energy column (k^+)
      budget_residual  (N, 6) the RSTE budget residuals res_*^+ for
                       (uu, vv, ww, uv, uw, vw), the data's own closure check
    """

    def __init__(self, ro_tau, re_tau, y_h, U, W, uu, vv, ww, uv, uw, vw, tke,
                 budget_residual, meta):
        R = _common.assemble_tensor(uu, vv, ww, uv, uw, vw)
        yplus = np.asarray(y_h, dtype=float) * re_tau            # y^+ = (y/h) Re_tau
        super().__init__(yplus=yplus, U=U, R=R, k=tke, re_tau=re_tau, meta=meta,
                         y_outer=y_h, W=W)
        self.ro_tau = float(ro_tau)
        self.tke = np.asarray(tke, dtype=float)
        self.budget_residual = np.asarray(budget_residual, dtype=float)

    # ---- location and parsing ---------------------------------------------

    @staticmethod
    def case_path(ro_tau, root=None):
        """Resolve the file for a rotation number, tolerating the missing .txt."""
        root = _common.data_root(root)
        base = os.path.join(root, "rotating_channel_flow",
                            f"tu-darmstadt_re180_ro{_ro_label(ro_tau)}")
        with_ext = base + ".txt"
        return with_ext if os.path.isfile(with_ext) else base

    @staticmethod
    def is_available(ro_tau, root=None):
        return os.path.isfile(RotatingChannelDNS.case_path(ro_tau, root))

    @staticmethod
    def _parse_header(path):
        """Read Re_tau and Ro_tau from the FLOW CONDITIONS block (CRLF-safe).

        Parsing is confined to the block introduced by "FLOW CONDITIONS"; the
        NOMENCLATURE block above it defines the symbols (e.g. "Ro_tau =
        2Omega*h/u_tau"), so a whole-header scan would misread the leading "2" as
        the rotation number.
        """
        re_tau = ro_tau = None
        in_block = False
        with open(path, newline="") as fh:
            for raw in fh:
                line = raw.replace("\r", "")
                stripped = line.strip()
                if not stripped:
                    continue                       # blank line inside the header
                if not stripped.startswith("%"):
                    break                          # first data row ends the header
                if "FLOW CONDITIONS" in line:
                    in_block = True
                if not in_block:
                    continue
                if re_tau is None:
                    m = _RE_TAU_PATTERN.search(line)
                    if m is not None:
                        re_tau = float(m.group(1))
                if ro_tau is None:
                    m = _RO_TAU_PATTERN.search(line)
                    if m is not None:
                        ro_tau = float(m.group(1))
        if re_tau is None or ro_tau is None:
            raise ValueError(f"{path}: header missing Re_tau / Ro_tau")
        return re_tau, ro_tau

    @staticmethod
    def load(ro_tau, root=None):
        """Parse one rotating-channel case (by rotation number) into a record."""
        path = RotatingChannelDNS.case_path(ro_tau, root)
        re_tau, ro_tau_hdr = RotatingChannelDNS._parse_header(path)
        # numpy.loadtxt strips the '\r' of CRLF and skips the '%' header lines
        data = np.loadtxt(path, comments="%")
        if data.shape[1] < max(_RES_COLS) + 1:
            raise ValueError(f"{path}: expected >= {max(_RES_COLS) + 1} columns, "
                             f"got {data.shape[1]}")
        col = {j: data[:, j] for j in range(_N_PRIMARY)}
        budget_residual = np.stack([data[:, j] for j in _RES_COLS], axis=1)
        meta = {
            "regime": "incompressible",
            "case": "streamwise_rotating_channel",
            "source": "University of Manitoba (Yang and Wang 2018, JFM 838)",
            "license": "all rights reserved by University of Manitoba; "
                       "may be used with reference",
            "re_tau": re_tau,
            "ro_tau": ro_tau_hdr,
            "file": os.path.relpath(path, _common.data_root(root)),
            "sigma_note": "modeled observation uncertainty (no DNS _stdev; budget "
                          "res_* columns are the data-only convergence check)",
        }
        return RotatingChannelDNS(
            ro_tau=ro_tau_hdr, re_tau=re_tau, y_h=col[0], U=col[1], W=col[2],
            uu=col[3], vv=col[4], ww=col[5], uv=col[6], uw=col[7], vw=col[8],
            tke=col[9], budget_residual=budget_residual, meta=meta)

    @staticmethod
    def load_all(root=None):
        """Load every available rotating-channel case, ordered by rotation number."""
        return [RotatingChannelDNS.load(n, root) for n in ROTATING_CASES
                if RotatingChannelDNS.is_available(n, root)]

    # ---- derived quantities -----------------------------------------------

    def out_of_plane_fraction(self):
        """Fraction of the shear-stress magnitude in the rotation-induced
        out-of-plane components, mean |<uw>,<vw>| / mean(|<uv>| + |<uw>| + |<vw>|).

        Zero for a non-rotating channel (only <uv> is nonzero); growing with it is
        the structural signature a Boussinesq SST baseline cannot represent.
        """
        uv = np.abs(self.uv)
        oop = np.abs(self.uw) + np.abs(self.vw)
        denom = float(np.mean(uv + oop)) + 1e-30
        return float(np.mean(oop)) / denom

    def budget_residual_level(self):
        """Representative RSTE budget-closure residual (rms over stations and
        components), the data's own convergence indicator."""
        return float(np.sqrt(np.mean(self.budget_residual ** 2)))

    def __repr__(self):
        return (f"RotatingChannelDNS(Re_tau={self.re_tau:.0f}, "
                f"Ro_tau={self.ro_tau:g}, n={self.n})")
