"""Backward-facing step DNS loader (Le, Moin and Kim 1997).

Parses the Le-Moin backward-facing-step profiles into the canonical dns_field
record (DNS_data/README.md, "Standardized processing format"), the SAME record the
channel and Couette loaders emit, so the separated-flow discrepancy / UQ /
evaluation machinery drives it unchanged.

Raw layout, under DNS_data/backward_facing_step/bwst-allfiles/:

  x-nnn.dat     wall-normal profile at one streamwise station. The nodal index nnn
                maps to x/h (readme.txt): 181 -> -3, 360 -> 4, 411 -> 6, 513 -> 10,
                641 -> 15, 744 -> 19 (the six bracket the reattachment). '#' header
                lines, then seven columns: y/h, U/Uo, V/Uo, u'/Uo, v'/Uo, w'/Uo,
                u'v'/Uo^2. The mean is two-dimensional (both U and V).
  stat-inf.dat  per-station U_e, U_tau, Cf, Cp and boundary-layer thicknesses (two
                '#'-headed blocks). Cf here locates the reattachment (its sign
                changes between x/h = 4, inside the bubble, and the reattachment).

The velocity fluctuations u', v', w' are root-mean-square, so the normal stresses
are their squares (uu = u'^2, and so on); the u'v' column is already the signed
(negative in the shear layer) covariance and maps straight to R_xy. The spanwise
off-diagonals u'w' and v'w' vanish (the mean is spanwise-homogeneous). Everything
is normalised by the inlet free-stream U0 (velocities by U0, stresses by U0^2).

This dataset is sparse in x (six stations), so no dense DNS field and no resolved
streamwise gradient exist. The separated-flow recipe does not need one: the
anisotropy target b_DNS = R/(2k) - I/3 comes from the DNS Reynolds stress at the
profile points (gradient-free), while the Boussinesq baseline anisotropy and the
Galilean-invariant conditioning features come from the dense RANS baseline field
interpolated to these profile locations (formed in the baseline/discrepancy layer,
not here). The reattachment-length truth is the published x_r/h ~ 6.28, since six
Cf stations are too sparse to locate it directly. The wall-normal gradient here
(dU/dy, dV/dy) is well resolved and is provided for standalone inspection; the
streamwise gradient is left zero and is unused by the recipe.
"""
import os

import numpy as np

from . import _common
from ..dns_field import DNSField
from .. import discrepancy as dq
from .. import realizability as rz

# station nodal index -> nominal x/h (bwst-allfiles/readme.txt)
BFS_STATIONS = ((181, -3.0), (360, 4.0), (411, 6.0),
                (513, 10.0), (641, 15.0), (744, 19.0))
BFS_NNN = tuple(nnn for nnn, _ in BFS_STATIONS)

# published mean reattachment length and flow parameters (Le, Moin and Kim 1997):
# step height h, inlet free-stream U0, Re_h = U0 h / nu, expansion ratio 1.2.
REATTACHMENT_XR_H = 6.28
RE_H = 5100
EXPANSION_RATIO = 1.2

# the seven columns of x-nnn.dat, in order
_XCOLS = ("y", "U", "V", "u_rms", "v_rms", "w_rms", "uv")


class BackwardFacingStepDNS:
    """The Le-Moin backward-facing step as the canonical dns_field record.

    Holds the six station profiles concatenated into flattened N-point arrays
    (x_h, y_h, U, V, R, k, station index) plus the per-station wall quantities
    (U_e, U_tau, Cf, Cp) and the published reattachment length. One physical case
    (Re_h = 5100), so there is a single load(), not a per-condition family.

    Attributes (flattened over all station points, leading axis N):
      x_h, y_h     station x/h (broadcast per point) and profile y/h
      U, V         mean velocity components, normalised by U0
      R (N,3,3)    full Reynolds-stress tensor from the rms columns
      k (N,)       turbulent kinetic energy 0.5 tr(R)
      station      integer station index 0..5 for each point
      wall         dict x/h -> {U_e, U_tau, Cf, Cp, Delta_999, Delta_star, Theta}
    """

    def __init__(self, stations, wall, meta):
        self.stations = stations              # list of per-station dicts
        self.wall = wall                      # per-station wall quantities
        self.meta = meta or {}
        self.station_xh = np.array([s["x_h"] for s in stations], dtype=float)

        # flatten the station profiles into one N-point record
        self.y_h = np.concatenate([s["y"] for s in stations])
        self.U = np.concatenate([s["U"] for s in stations])
        self.V = np.concatenate([s["V"] for s in stations])
        self.R = np.concatenate([s["R"] for s in stations], axis=0)
        self.k = np.concatenate([s["k"] for s in stations])
        self.x_h = np.concatenate([np.full(s["y"].size, s["x_h"]) for s in stations])
        self.station = np.concatenate(
            [np.full(s["y"].size, i, dtype=int) for i, s in enumerate(stations)])
        self.n = self.y_h.size

        # component views (read-only convenience)
        self.uu = self.R[:, 0, 0]
        self.vv = self.R[:, 1, 1]
        self.ww = self.R[:, 2, 2]
        self.uv = self.R[:, 0, 1]

    # ---- location and parsing ---------------------------------------------

    @staticmethod
    def dir_path(root=None):
        root = _common.data_root(root)
        return os.path.join(root, "backward_facing_step", "bwst-allfiles")

    @staticmethod
    def station_path(nnn, root=None):
        return os.path.join(BackwardFacingStepDNS.dir_path(root), f"x-{nnn}.dat")

    @staticmethod
    def is_available(root=None):
        # keyed on the reattachment-bracketing station x/h = 6 (nnn = 411)
        return os.path.isfile(BackwardFacingStepDNS.station_path(411, root))

    @staticmethod
    def _parse_station(nnn, x_h, root=None):
        """Parse one x-nnn.dat profile into a station record.

        The normal stresses are the squares of the rms columns; the u'v' column is
        the signed shear covariance (R_xy). Spanwise off-diagonals vanish.
        """
        path = BackwardFacingStepDNS.station_path(nnn, root)
        data = np.loadtxt(path, comments="#")
        if data.ndim != 2 or data.shape[1] != len(_XCOLS):
            raise ValueError(f"{path}: expected {len(_XCOLS)} columns, "
                             f"got shape {data.shape}")
        cols = {name: data[:, j] for j, name in enumerate(_XCOLS)}
        # R_ij = <u_i' u_j'>: normals are rms^2, R_xy is the signed u'v' column
        R = _common.assemble_tensor(cols["u_rms"] ** 2, cols["v_rms"] ** 2,
                                    cols["w_rms"] ** 2, cols["uv"])
        k = 0.5 * (cols["u_rms"] ** 2 + cols["v_rms"] ** 2 + cols["w_rms"] ** 2)
        return {
            "nnn": nnn, "x_h": float(x_h), "y": cols["y"],
            "U": cols["U"], "V": cols["V"], "R": R, "k": k,
            "u_rms": cols["u_rms"], "v_rms": cols["v_rms"], "w_rms": cols["w_rms"],
        }

    @staticmethod
    def _parse_stat_inf(root=None):
        """Parse stat-inf.dat's two '#'-headed blocks into per-station wall data.

        Block 1: x/h, Delta_999, Delta_star, Theta, H, G.
        Block 2: x/h, U_e, U_tau, Cf, Cp. The blocks are keyed by x/h, which the
        loader matches to the nominal station x/h by nearest value (the file rounds
        -3 to -2.99 and 10 to 9.98).
        """
        path = os.path.join(BackwardFacingStepDNS.dir_path(root), "stat-inf.dat")
        rows6, rows5 = [], []
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                vals = [float(p) for p in parts]
                if len(vals) == 6:
                    rows6.append(vals)
                elif len(vals) == 5:
                    rows5.append(vals)
        bl = {r[0]: {"Delta_999": r[1], "Delta_star": r[2], "Theta": r[3],
                     "H": r[4], "G": r[5]} for r in rows6}
        cf = {r[0]: {"U_e": r[1], "U_tau": r[2], "Cf": r[3], "Cp": r[4]}
              for r in rows5}
        # merge the two blocks onto each station's nominal x/h by nearest match
        wall = {}
        for _, x_nom in BFS_STATIONS:
            entry = {}
            for table in (bl, cf):
                keys = list(table.keys())
                x_near = keys[int(np.argmin([abs(kk - x_nom) for kk in keys]))]
                entry.update(table[x_near])
            wall[x_nom] = entry
        return wall

    @staticmethod
    def load(root=None):
        """Parse the six station profiles and the wall table into one record."""
        stations = [BackwardFacingStepDNS._parse_station(nnn, x_h, root)
                    for nnn, x_h in BFS_STATIONS]
        wall = BackwardFacingStepDNS._parse_stat_inf(root)
        meta = {
            "regime": "incompressible",
            "case": "backward_facing_step",
            "source": "Le, Moin and Kim 1997 (JFM 330, 349-374)",
            "re_h": RE_H,
            "expansion_ratio": EXPANSION_RATIO,
            "reattachment_xr_h": REATTACHMENT_XR_H,
            "dir": os.path.relpath(BackwardFacingStepDNS.dir_path(root),
                                   _common.data_root(root)),
            "sigma_note": "modeled observation uncertainty (no DNS _stdev in file)",
        }
        return BackwardFacingStepDNS(stations, wall, meta)

    # ---- derived quantities -----------------------------------------------

    def b_dns(self):
        """The gradient-free anisotropy target b_DNS = R/(2k) - I/3 per point.

        This is the separated-flow model-form target on the sparse-in-x BFS: it
        needs the DNS Reynolds stress only, no velocity gradient. The Boussinesq
        baseline it is differenced against comes from the RANS field (elsewhere).
        """
        return dq.reynolds_anisotropy(self.R)

    def valid_mask(self, k_floor_rel=1e-6):
        """Points with resolved turbulence (k above a small fraction of its peak).

        The anisotropy b = R/(2k) - I/3 is degenerate where k -> 0 (the wall row is
        exactly zero), so the realizability anchor and any training exclude those.
        """
        return self.k > k_floor_rel * float(np.max(self.k))

    def realizable_fraction(self, tol=1e-9, k_floor_rel=1e-6):
        """Fraction of resolved-turbulence points whose DNS stress is realizable.

        Data-only physics anchor (separate from the Galilean-invariant feature
        construction): the DNS Reynolds stress must lie in the barycentric
        realizable set at every station where turbulence is resolved.
        """
        mask = self.valid_mask(k_floor_rel)
        return float(np.mean(rz.is_realizable(self.R[mask], tol=tol)))

    def velocity_gradient(self):
        """Wall-normal-resolved mean velocity gradient grad_u[i, j] = du_i/dx_j.

        Only the wall-normal derivatives dU/dy and dV/dy are formed (per station,
        by central differences on the profile's y grid); the streamwise gradient is
        left zero because the six stations are far too sparse in x to difference.
        The recipe conditions on RANS-derived features instead, so this is provided
        for standalone DNS inspection only.
        """
        g = np.zeros((self.n, 3, 3))
        dUdy = np.concatenate(
            [_common.wall_normal_gradient(s["y"], s["U"]) for s in self.stations])
        dVdy = np.concatenate(
            [_common.wall_normal_gradient(s["y"], s["V"]) for s in self.stations])
        g[:, 0, 1] = dUdy      # du_x/dy
        g[:, 1, 1] = dVdy      # du_y/dy
        return g

    def to_dnsfield(self, grad_u=None, timescale=None, nu_t=None, k_baseline=None):
        """Build a UQ DNSField from the flattened record.

        For the production recipe, pass the RANS-derived grad_u (features), the
        baseline turbulence timescale, nu_t, and the baseline k interpolated to
        the profile points, so the discrepancy is b_DNS (from the DNS stress
        here) minus the Boussinesq baseline the solver actually applies,
        b_B = -(nu_t/(k_baseline timescale)) S, limiter included. With no
        arguments it falls back to the wall-normal DNS gradient and a unit
        timescale, which is standalone inspection only and not a calibrated
        baseline.
        """
        g = self.velocity_gradient() if grad_u is None else np.asarray(grad_u, float)
        ts = np.ones(self.n) if timescale is None else np.asarray(timescale, float)
        return DNSField(
            grad_u=g, R=self.R, k=self.k, timescale=ts,
            nu_t=None if nu_t is None else np.asarray(nu_t, float),
            k_baseline=None if k_baseline is None else np.asarray(k_baseline, float),
            meta=dict(self.meta))

    def cf_stations(self):
        """(x/h, Cf) across the six stations, ordered by x/h.

        The reattachment sits where Cf changes sign; with these stations that is
        between x/h = 4 (Cf < 0, inside the recirculation) and the recovery region,
        consistent with the published x_r/h ~ 6.28.
        """
        xs = sorted(self.wall.keys())
        return np.array(xs), np.array([self.wall[x]["Cf"] for x in xs])

    def reattachment_truth(self):
        """Published mean reattachment length x_r/h (Le, Moin and Kim 1997)."""
        return REATTACHMENT_XR_H

    def __repr__(self):
        return (f"BackwardFacingStepDNS(Re_h={RE_H}, stations={len(self.stations)}, "
                f"n={self.n}, x_r/h={REATTACHMENT_XR_H})")
