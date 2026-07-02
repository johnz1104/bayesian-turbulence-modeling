"""Periodic hills DNS loader (Xiao, Wu, Laizet and Duan 2020).

Parses the parameterised-geometry periodic-hill DNS into the canonical dns_field
record (DNS_data/README.md, "Standardized processing format"). Unlike the
wall-bounded profile loaders (channel, Couette), this is a genuine two-dimensional
dense field, so the full mean velocity-gradient tensor is formed by differencing on
the grid and the record carries N flattened field points.

The slope family spans incipient to massive separation by a hill-steepness
parameter alpha (0.5 to 1.5) at a fixed bulk Reynolds number, so it is a
cross-geometry out-of-distribution axis at constant Re. Two on-disk formats appear
in the set and both reduce to the same tensor grid and the same record:

  VTK RectilinearGrid (.vtr)   alpha = 0.5 and 1.5. mean.vtr carries UMEAN, VMEAN,
                               WMEAN, PMEAN; rms_1.vtr the normal stresses UUMEAN,
                               VVMEAN, WWMEAN (and PPMEAN); rms_2.vtr the shear
                               stresses UVMEAN, UWMEAN, VWMEAN. Separable X/Y/Z
                               coordinate vectors.
  ASCII columnar (.dat)        the other cases. mean_files.dat columns x, y, U, V,
                               W, P; rms_files1.dat columns x, y, uu, vv, ww, pp;
                               rms_files2.dat columns x, y, uv, uw, vw. Same
                               x-fastest tensor ordering as the VTK point data.

The stress arrays are the Reynolds stresses (fluctuation covariances), so
R_xx = UUMEAN, R_yy = VVMEAN, R_zz = WWMEAN, R_xy = UVMEAN, and so on, directly.
The periodic-hill lower wall is a curved surface, and the data is interpolated onto
a rectilinear bounding grid with the solid interior blanked to exact zeros, so the
loader carries a fluid mask (nonzero turbulent energy) and an interior mask (a
fluid point whose four grid neighbours are all fluid, where the central-difference
gradient stencil is clean). The Reynolds number is fixed across the family at
Re_b = 5600 (crest bulk velocity and hill height), cross-checked against the
companion OpenFOAM drive Ubar = 0.020188 (volume averaged) / 0.7210 = 0.028 with
nu = 5e-6.
"""
import os
import re

import numpy as np

from . import _common
from ..dns_field import DNSField
from .. import discrepancy as dq
from .. import realizability as rz

# hill-steepness cases; alpha is the slope parameter (Xiao 2020), spanning
# incipient (0.5) to massive (1.5) separation. 1p0_refined is alpha = 1 on a finer
# mesh. The alpha = 0.5 and 1.5 cases are VTK, the rest ASCII.
PEHILL_CASES = ("0p5", "0p8", "1p0", "1p0_refined", "1p2", "1p5")
_ALPHA = {"0p5": 0.5, "0p8": 0.8, "1p0": 1.0, "1p0_refined": 1.0,
          "1p2": 1.2, "1p5": 1.5}
_VTK_CASES = frozenset(("0p5", "1p5"))

# fixed bulk Reynolds number of the slope family (crest bulk velocity, hill
# height); the geometry varies at constant Re (Xiao 2020).
RE_B = 5600


class PeriodicHillsDNS:
    """One periodic-hill case as the canonical dns_field record (dense 2D field).

    Attributes (flattened over the tensor grid, leading axis N, x-fastest):
      x, y         physical coordinates (hill height h = 1)
      U, V, W      mean velocity components
      P            mean pressure
      R (N,3,3)    full Reynolds-stress tensor from the second-order statistics
      k (N,)       turbulent kinetic energy 0.5 tr(R)
      shape        (nY, nX) tensor-grid shape
      fluid_mask   nonzero turbulent energy (the blanked solid interior is zero)
      interior_mask fluid point with four fluid neighbours (clean gradient stencil)
    """

    def __init__(self, case, x, y, U, V, W, P, R, shape, meta):
        self.case = case
        self.alpha = _ALPHA[case]
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.U = np.asarray(U, float)
        self.V = np.asarray(V, float)
        self.W = np.asarray(W, float)
        self.P = np.asarray(P, float)
        self.R = np.asarray(R, float)
        self.k = 0.5 * np.trace(self.R, axis1=1, axis2=2)
        self.shape = shape                          # (nY, nX)
        self.n = self.x.size
        self.meta = meta or {}

        # blanked solid interior is exactly zero; fluid has nonzero turbulent energy
        self.fluid_mask = (self.R[:, 0, 0] + self.R[:, 1, 1] + self.R[:, 2, 2]) > 0.0
        self._grad_u = self._velocity_gradient_grid()
        self.interior_mask = self._interior_mask()

    # ---- location and parsing ---------------------------------------------

    @staticmethod
    def case_dir(case, root=None):
        root = _common.data_root(root)
        return os.path.join(root, "periodic_hills", "pehill-5-cases-DNS",
                            f"case_{case}", "dns-data")

    @staticmethod
    def is_available(case, root=None):
        d = PeriodicHillsDNS.case_dir(case, root)
        probe = "mean.vtr" if case in _VTK_CASES else "mean_files.dat"
        return os.path.isfile(os.path.join(d, probe))

    @staticmethod
    def _read_vtr_arrays(path):
        """Extract every named DataArray from an ASCII VTK RectilinearGrid file."""
        with open(path, "rb") as fh:
            txt = fh.read().decode("latin-1")
        out = {}
        for m in re.finditer(r'<DataArray\b([^>]*)>(.*?)</DataArray>', txt, re.S):
            name = re.search(r'Name="([^"]+)"', m.group(1))
            if name is not None:
                out[name.group(1)] = np.fromstring(m.group(2), sep=" ")
        return out

    @staticmethod
    def _load_vtk(case, root=None):
        d = PeriodicHillsDNS.case_dir(case, root)
        mean = PeriodicHillsDNS._read_vtr_arrays(os.path.join(d, "mean.vtr"))
        r1 = PeriodicHillsDNS._read_vtr_arrays(os.path.join(d, "rms_1.vtr"))
        r2 = PeriodicHillsDNS._read_vtr_arrays(os.path.join(d, "rms_2.vtr"))
        xc, yc = mean["X_COORDINATES"], mean["Y_COORDINATES"]
        nX, nY = xc.size, yc.size
        # RectilinearGrid point data is x-fastest: build per-point coords by tiling
        x = np.tile(xc, nY)
        y = np.repeat(yc, nX)
        fields = dict(U=mean["UMEAN"], V=mean["VMEAN"], W=mean["WMEAN"],
                      P=mean["PMEAN"], uu=r1["UUMEAN"], vv=r1["VVMEAN"],
                      ww=r1["WWMEAN"], uv=r2["UVMEAN"], uw=r2["UWMEAN"],
                      vw=r2["VWMEAN"])
        return PeriodicHillsDNS._finish(case, x, y, fields, (nY, nX), root)

    @staticmethod
    def _load_ascii(case, root=None):
        d = PeriodicHillsDNS.case_dir(case, root)
        mean = np.loadtxt(os.path.join(d, "mean_files.dat"))       # x,y,U,V,W,P
        r1 = np.loadtxt(os.path.join(d, "rms_files1.dat"))         # x,y,uu,vv,ww,pp
        r2 = np.loadtxt(os.path.join(d, "rms_files2.dat"))         # x,y,uv,uw,vw
        x, y = mean[:, 0], mean[:, 1]
        nX = np.unique(np.round(x, 8)).size
        nY = x.size // nX
        fields = dict(U=mean[:, 2], V=mean[:, 3], W=mean[:, 4], P=mean[:, 5],
                      uu=r1[:, 2], vv=r1[:, 3], ww=r1[:, 4],
                      uv=r2[:, 2], uw=r2[:, 3], vw=r2[:, 4])
        return PeriodicHillsDNS._finish(case, x, y, fields, (nY, nX), root)

    @staticmethod
    def _finish(case, x, y, f, shape, root):
        R = _common.assemble_tensor(f["uu"], f["vv"], f["ww"], f["uv"],
                                    uw=f["uw"], vw=f["vw"])
        meta = {
            "regime": "incompressible",
            "case": "periodic_hills",
            "alpha": _ALPHA[case],
            "source": "Xiao, Wu, Laizet and Duan 2020 (Comput. Fluids 200, 104431)",
            "re_b": RE_B,
            "dir": os.path.relpath(PeriodicHillsDNS.case_dir(case, root),
                                   _common.data_root(root)),
            "sigma_note": "modeled observation uncertainty (no DNS _stdev in file)",
        }
        return PeriodicHillsDNS(case, x, y, f["U"], f["V"], f["W"], f["P"], R,
                                shape, meta)

    @staticmethod
    def load(case, root=None):
        """Parse one case (by alpha string, e.g. '1p0') into the canonical record."""
        if case in _VTK_CASES:
            return PeriodicHillsDNS._load_vtk(case, root)
        return PeriodicHillsDNS._load_ascii(case, root)

    @staticmethod
    def load_all(root=None):
        """Every available case, ordered by the slope parameter alpha."""
        return [PeriodicHillsDNS.load(c, root) for c in PEHILL_CASES
                if PeriodicHillsDNS.is_available(c, root)]

    # ---- grid geometry and gradients --------------------------------------

    def _grid(self, flat):
        return np.asarray(flat, float).reshape(self.shape)

    def _velocity_gradient_grid(self):
        """grad_u[i, j] = d u_i / d x_j by central differences on the tensor grid.

        The mean is two-dimensional and spanwise-homogeneous, so only the in-plane
        derivatives of U and V are formed (d/dz = 0, W ~ 0). numpy.gradient uses the
        (non-uniform) physical coordinate vectors for spacing. Gradients that
        straddle the blanked solid are cleaned by the interior mask, not here.
        """
        nY, nX = self.shape
        x1d = self.x.reshape(nY, nX)[0, :]        # x varies along axis 1
        y1d = self.y.reshape(nY, nX)[:, 0]        # y varies along axis 0
        g = np.zeros((self.n, 3, 3))
        for i, comp in ((0, self.U), (1, self.V)):
            fg = self._grid(comp)
            dfdy, dfdx = np.gradient(fg, y1d, x1d)     # axis0 = y, axis1 = x
            g[:, i, 0] = dfdx.reshape(-1)              # d u_i / dx
            g[:, i, 1] = dfdy.reshape(-1)              # d u_i / dy
        return g

    def _interior_mask(self):
        """Fluid points whose four grid neighbours are also fluid (clean stencil)."""
        nY, nX = self.shape
        fg = self.fluid_mask.reshape(nY, nX)
        interior = np.zeros_like(fg)
        interior[1:-1, 1:-1] = (fg[1:-1, 1:-1] & fg[:-2, 1:-1] & fg[2:, 1:-1] &
                                fg[1:-1, :-2] & fg[1:-1, 2:])
        return interior.reshape(-1)

    def velocity_gradient(self):
        """The full mean velocity-gradient tensor grad_u (N, 3, 3)."""
        return self._grad_u

    def continuity_residual(self):
        """Mean-continuity residual du/dx + dv/dy per point (incompressible)."""
        return self._grad_u[:, 0, 0] + self._grad_u[:, 1, 1]

    # ---- physics anchors and derived quantities ---------------------------

    def continuity_anchor(self):
        """Continuity residual on interior points, scaled by the strain magnitude.

        Data-only physics anchor for the dense field: the interpolated DNS mean
        satisfies du/dx + dv/dy = 0, so the RMS residual normalised by the RMS
        strain-rate magnitude is small. Returned as that dimensionless ratio.
        """
        m = self.interior_mask
        div = self.continuity_residual()[m]
        S = 0.5 * (self._grad_u + np.transpose(self._grad_u, (0, 2, 1)))
        s_mag = np.sqrt(2.0 * np.sum(S[m] ** 2, axis=(1, 2)))
        scale = np.sqrt(np.mean(s_mag ** 2))
        return float(np.sqrt(np.mean(div ** 2)) / max(scale, 1e-30))

    def realizable_fraction(self, tol=1e-9):
        """Fraction of fluid points whose DNS Reynolds stress is realizable.

        Data-only physics anchor (separate from the Galilean-invariant feature
        construction): the DNS stress lies in the barycentric realizable set.
        """
        m = self.fluid_mask & (self.k > 1e-12 * float(np.max(self.k)))
        return float(np.mean(rz.is_realizable(self.R[m], tol=tol)))

    def near_wall_streamwise_velocity(self):
        """Streamwise velocity in the lowest fluid cell of each column, versus x.

        Its sign is the sign of the bottom-wall shear (U = 0 at the no-slip wall),
        so negative marks the recirculation and the downstream sign change marks
        reattachment. Columns with no fluid cell are NaN.
        """
        nY, nX = self.shape
        Ug = self._grid(self.U)
        fg = self.fluid_mask.reshape(nY, nX)
        x1d = self.x.reshape(nY, nX)[0, :]
        u_wall = np.full(nX, np.nan)
        for i in range(nX):
            col = np.where(fg[:, i])[0]              # fluid rows in this column
            if col.size:
                u_wall[i] = Ug[col[0], i]            # lowest fluid cell
        return x1d, u_wall

    def bottom_wall_reattachment(self):
        """Reattachment x as the downstream end of the main recirculation bubble.

        The near-wall streamwise velocity is negative through the primary
        separation bubble behind the hill and turns positive at reattachment. The
        bubble is identified as the longest contiguous run of negative near-wall U
        (robust to the small crest and windward-foot sign changes), and x_r is the
        interpolated zero crossing at its downstream end. This is the DNS
        reattachment-length QoI truth. Returns None if the flow does not separate.
        """
        xi, ui = self.near_wall_streamwise_velocity()
        valid = ~np.isnan(ui)
        xi, ui = xi[valid], ui[valid]
        neg = ui < 0
        # longest contiguous run of negative near-wall U (the main bubble)
        best_len, best_end = 0, None
        run = 0
        for i in range(neg.size):
            run = run + 1 if neg[i] else 0
            if run > best_len and i + 1 < neg.size and ui[i + 1] > 0:
                best_len, best_end = run, i
        if best_end is None:
            return None
        j = best_end                                  # ui[j] < 0, ui[j+1] > 0
        x_r = xi[j] + (0.0 - ui[j]) * (xi[j + 1] - xi[j]) / (ui[j + 1] - ui[j])
        return float(x_r)

    def bottom_wall_bubble(self):
        """(x_s, x_r, length) of the main bubble, both ends interpolated.

        The same longest-negative-run rule as bottom_wall_reattachment; the
        separation point x_s is the interpolated zero crossing at the run's
        upstream end (the column itself when the run starts at the first valid
        column). Returns None if the flow does not separate.
        """
        xi, ui = self.near_wall_streamwise_velocity()
        valid = ~np.isnan(ui)
        xi, ui = xi[valid], ui[valid]
        neg = ui < 0
        best_len, best_end = 0, None
        run = 0
        for i in range(neg.size):
            run = run + 1 if neg[i] else 0
            if run > best_len and i + 1 < neg.size and ui[i + 1] > 0:
                best_len, best_end = run, i
        if best_end is None:
            return None
        j0 = best_end - best_len + 1                  # first negative column
        if j0 > 0:                                    # ui[j0-1] >= 0 > ui[j0]
            x_s = xi[j0 - 1] + (0.0 - ui[j0 - 1]) * (xi[j0] - xi[j0 - 1]) \
                / (ui[j0] - ui[j0 - 1])
        else:
            x_s = xi[0]
        j = best_end
        x_r = xi[j] + (0.0 - ui[j]) * (xi[j + 1] - xi[j]) / (ui[j + 1] - ui[j])
        return float(x_s), float(x_r), float(x_r - x_s)

    def to_dnsfield_at(self, idx, grad_u=None, timescale=None, nu_t=None,
                       k_baseline=None):
        """Build a UQ DNSField at an arbitrary point subset (flat indices).

        The production separated-flow recipe passes the RANS-derived grad_u
        (conditioning features), the baseline timescale, nu_t and k so the
        discrepancy is b_DNS minus the Boussinesq baseline the solver actually
        applies (limiter-consistent), exactly as the backward-facing-step
        record. With no arguments it falls back to the DNS gradient and a unit
        timescale (standalone inspection only).
        """
        idx = np.asarray(idx, int)
        g = self._grad_u[idx] if grad_u is None else np.asarray(grad_u, float)
        ts = (np.ones(idx.size) if timescale is None
              else np.asarray(timescale, float))
        return DNSField(
            grad_u=g, R=self.R[idx], k=self.k[idx], timescale=ts,
            nu_t=None if nu_t is None else np.asarray(nu_t, float),
            k_baseline=None if k_baseline is None
            else np.asarray(k_baseline, float),
            meta=dict(self.meta))

    def to_dnsfield(self, timescale, nu_t=None):
        """Build a UQ DNSField from the interior fluid points and a baseline timescale.

        Restricts to interior points (clean gradient stencil), and takes the
        baseline turbulence timescale (and optional nu_t) supplied by the RANS
        baseline so the discrepancy is b_DNS minus the Boussinesq baseline.
        """
        m = self.interior_mask
        ts = np.asarray(timescale, float)
        ts = ts[m] if ts.shape == (self.n,) else ts
        return DNSField(
            grad_u=self._grad_u[m], R=self.R[m], k=self.k[m], timescale=ts,
            nu_t=None if nu_t is None else np.asarray(nu_t, float)[m],
            meta=dict(self.meta))

    def __repr__(self):
        return (f"PeriodicHillsDNS(case={self.case}, alpha={self.alpha}, "
                f"Re_b={RE_B}, grid={self.shape[1]}x{self.shape[0]}, "
                f"fluid={int(self.fluid_mask.sum())}/{self.n})")
