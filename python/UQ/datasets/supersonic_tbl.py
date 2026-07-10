"""Attached supersonic flat-plate boundary-layer loader (the dnsm2 family).

Parses the Rome supersonic boundary-layer database
(reynolds.dma.uniroma1.it/dnsm2; Pirozzoli and Bernardini, JFM 688 (2011)
120-168 and Phys. Fluids 25 (2013) 021704) into the canonical compressible
dns_field record (UQ.datasets._compressible). Twelve cases: Mach 2 at eight
friction Reynolds numbers (nominal Re_tau 200 to 1110), Mach 3 and Mach 4 at
Re_tau 400 and 500. The attached supersonic stress axis and the
preserve-attached control of the shock-interaction study.

Raw layout, per case, under DNS_data/supersonic_turbulent_BL/mach_<M>/
<case>.dat(.txt): a prose header carrying the skin friction, the friction Mach
number, four Reynolds numbers and the shape factors, then 20 whitespace
profile columns. The header's column-names line is NOT machine-splittable
(the thirteenth name, y+ du_vd+/dy+, contains internal whitespace), so this
loader consumes the columns BY POSITION in the documented order:

  0 y/delta99   1 y+       2 u+      3 u_vd+    4 urms+   5 vrms+   6 wrms+
  7 uv+         8 sqrt(rho/rho_w)    9 prms+   10 trms+  11 rhorms+
 12 y+ du_vd+/dy+   13 S(u)  14 S(T)  15 F(u)  16 F(T)  17 omx+  18 omy+
 19 omz+

and asserts exactly 20 data columns per row.

Verified limitations, load-bearing (also recorded in the manifest): these
files carry NO turbulent heat-flux vector, NO wall heat flux and NO mean
temperature column (nominally adiabatic walls, thermally quiescent, B_q taken
as zero). The record's mean density is the sqrt(rho/rho_w) column squared;
the mean temperature is the constant-pressure boundary-layer relation
T/T_w = rho_w/rho, stated as a RECONSTRUCTION wherever used. No viscosity
profile is given (mu = None, as for the flat plates); the semi-local
coordinate is not constructed. This set therefore never enters heat-flux
training and supports no heat-flux generalization claim.

The physics anchor is the van-Driest-transform reconstruction (identical in
kind to the flat-plate dataset): u_vd+ recomputed from the file's own u+ and
density-ratio columns against its u_vd+ column, a data-only cross-column
identity measured by the loader tests. The temperature-reconstruction sanity
(the implied wall-to-free-stream temperature ratio against the turbulent
recovery estimate) is a loader-test check, not an anchor.
"""
import os
import re

import numpy as np

from . import _common
from ._compressible import CompressibleProfileDNS

TBL_CASES = (
    "M2_Retau200", "M2_Retau250", "M2_Retau450", "M2_Retau580",
    "M2_Retau840", "M2_Retau900", "M2_Retau1000", "M2_Retau1110",
    "M3_Retau400", "M3_Retau500",
    "M4_Retau400", "M4_Retau500",
)

_SUBDIR = "supersonic_turbulent_BL"
_GAMMA = 1.4
_NCOLS = 20

# header keys parsed by regex from the prose header, in the file's own wording
_HEADER_KEYS = {
    "cf": "Skin friction",
    "m_tau": "Friction Mach number",
    "re_tau": "Retau",
    "re_delta": "Redelta",
    "re_theta": "Retheta",
    "re_delta2": "Redelta2",
    "dstar_over_delta": "dstar/delta99",
    "theta_over_delta": "theta/delta99",
    "H": "H",
    "H_inc": "Hinc",
}


class SupersonicTBLDNS(CompressibleProfileDNS):
    """One dnsm2 flat-plate case as the canonical compressible record.

    In addition to the compressible base record:

      uvd_plus     the file's van Driest transformed velocity (anchor column)
      m_inf        the nominal free-stream Mach number (from the case family)
      header       the parsed prose-header parameter dict, by name
    """

    def __init__(self, case, data, header, meta):
        yplus = data[:, 1]
        u_plus = data[:, 2]
        sqrt_rho = data[:, 8]
        rho = sqrt_rho ** 2                       # <rho>/rho_w
        # constant-pressure boundary-layer relation (reconstruction, stated):
        # p about constant across the layer, so T/T_w = rho_w/<rho>
        T = 1.0 / np.maximum(rho, 1e-12)

        k = 0.5 * (data[:, 4] ** 2 + data[:, 5] ** 2 + data[:, 6] ** 2)
        R = _common.assemble_tensor(data[:, 4] ** 2, data[:, 5] ** 2,
                                    data[:, 6] ** 2, data[:, 7])
        # the wall row (y+ = 0) carries exact zeros; keep the stress zero there
        R[0] = 0.0

        m_inf = float(case.split("_")[0][1:])
        wall = {
            "b_q": 0.0,                # nominally adiabatic (thermally quiescent)
            "m_tau": header["m_tau"],
            "cf": header["cf"],
            "gamma_w": _GAMMA,
            "m_inf": m_inf,
            "re_theta": header["re_theta"],
            "re_delta2": header["re_delta2"],
            "H": header["H"],
        }
        super().__init__(
            yplus=yplus, U=u_plus, R=R, k=k, re_tau=header["re_tau"],
            meta=meta, T=T, rho=rho, mu=None, y_outer=data[:, 0],
            ystar=None, q_hat=None, mach=None, pr_molecular=None, wall=wall)
        self.case = case
        self.uvd_plus = data[:, 3].copy()
        self.m_inf = m_inf
        self.header = dict(header)

    # ---- physics anchor -------------------------------------------------------

    def van_driest_recomputed(self):
        """u_vd+ = int sqrt(<rho>/rho_w) du+ by trapezoid over the profile."""
        du = np.diff(self.U)
        s = np.sqrt(self.rho)
        return np.concatenate([[0.0],
                               np.cumsum(0.5 * (s[1:] + s[:-1]) * du)])

    def van_driest_residual(self):
        """Relative residual of the recomputed van Driest velocity against the
        file's own u_vd+ column (the data-only cross-column anchor)."""
        recomputed = self.van_driest_recomputed()
        return (recomputed - self.uvd_plus) / np.maximum(self.uvd_plus, 1e-9)

    def implied_wall_temperature_ratio(self):
        """T_w/T_inf implied by the density reconstruction at the layer edge.

        With rho in rho_w units and constant pressure, the edge density is
        rho_inf/rho_w = T_w/T_inf, so the edge value itself is the implied
        ratio. Compared by the loader tests against the turbulent recovery
        estimate 1 + r (gamma-1)/2 M^2 with r = 0.89 (measured within one
        percent on every case)."""
        return float(self.rho[-1])

    # ---- location and parsing -------------------------------------------------

    @staticmethod
    def case_path(case, root=None):
        mach_dir = f"mach_{case.split('_')[0][1:]}"
        return os.path.join(_common.data_root(root), _SUBDIR, mach_dir,
                            f"{case}.dat")

    @staticmethod
    def is_available(case, root=None):
        base = SupersonicTBLDNS.case_path(case, root)
        return os.path.isfile(base) or os.path.isfile(base + ".txt")

    @staticmethod
    def load(case, root=None):
        """Parse one case into the canonical record."""
        if case not in TBL_CASES:
            raise ValueError(f"unknown dnsm2 case '{case}'")
        path = SupersonicTBLDNS.case_path(case, root)
        if not os.path.isfile(path):
            path = path + ".txt"
        with open(path) as fh:
            text = fh.read()

        header = {}
        for key, label in _HEADER_KEYS.items():
            m = re.search(re.escape(label) + r"\s*=\s*([0-9.eE+-]+)", text)
            if m is None:
                raise ValueError(f"{path}: header value '{label}' not found")
            header[key] = float(m.group(1))

        start = _common.first_data_row(path, _NCOLS)
        data = np.loadtxt(path, skiprows=start)
        if data.ndim != 2 or data.shape[1] != _NCOLS:
            raise ValueError(f"{path}: expected {_NCOLS} data columns, got "
                             f"{data.shape}")
        if abs(data[0, 1]) > 1e-12:
            raise ValueError(f"{path}: first row is not the wall (y+ = "
                             f"{data[0, 1]:g})")

        meta = {
            "regime": "compressible",
            "case": "supersonic_tbl",
            "case_tag": case,
            "source": "Pirozzoli and Bernardini (dnsm2, "
                      "reynolds.dma.uniroma1.it/dnsm2; JFM 688 (2011) and "
                      "Phys. Fluids 25 (2013) 021704)",
            "wall_thermal": "nominally adiabatic (thermally quiescent, "
                            "B_q taken 0)",
            "averaging": "reynolds moments in wall units; mean density from "
                         "the density-ratio column squared; mean temperature "
                         "by the constant-pressure relation (reconstruction)",
            "heat_flux_note": "no turbulent heat flux and no wall heat flux "
                              "in this dataset; never a dq training source",
            "sigma_note": "modeled observation uncertainty "
                          "(no per-point statistical uncertainty in file)",
        }
        return SupersonicTBLDNS(case=case, data=data, header=header, meta=meta)

    @staticmethod
    def load_all(root=None):
        return [SupersonicTBLDNS.load(c, root) for c in TBL_CASES
                if SupersonicTBLDNS.is_available(c, root)]

    def __repr__(self):
        return (f"SupersonicTBLDNS({self.case}, M_inf={self.m_inf:g}, "
                f"Re_tau={self.re_tau:.0f}, n={self.n})")
