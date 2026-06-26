"""Lee and Moser (2015) plane-channel DNS loader.

Parses the raw UT Austin / Oden Institute channel-flow profiles into the
canonical dns_field record (DNS_data/README.md, "Standardized processing
format"). One parser serves all five friction Reynolds numbers; it is the
loader pattern every later dataset (Couette, separated, compressible, SBLI)
reuses.

Raw layout, per Re_tau, under DNS_data/channel_flow/LM_Channel_Re_tau=<N>/:

  mean_velocity_pressure/*_mean_prof.dat.txt
      cols: y/delta, y^+, U^+, dU/dy^+, W^+, P^+                         (6)
  covariances_velocity/*_vel_fluc_prof.dat.txt
      cols: y/delta, y^+, u'u'^+, v'v'^+, w'w'^+, u'v'^+, u'w'^+,
            v'w'^+, k^+                                                  (9)

Each quantity has a paired *_stdev.dat.txt with the same columns giving the
statistical uncertainty (carried as observation uncertainty). Header comment
lines begin with %, and carry nu, delta (half height), u_tau, Re_tau. ALL
quantities are in wall units (signified ^+): velocities by u_tau, lengths by
nu/u_tau, stresses and k by u_tau^2, so the loader keeps the data in wall
units and records the friction scales (u_tau, nu) needed to redimensionalise.

Coordinate convention: x = streamwise (0), y = wall-normal (1), z = spanwise
(2). The only nonzero mean velocity gradient is dU/dy = grad_u[0, 1]; the DNS
supplies it directly, so no numerical differentiation is needed.
"""
import glob
import os
import re

import numpy as np

from ..dns_field import DNSField

# nominal friction Reynolds numbers of the compiled cases (DNS_data/README.md)
CHANNEL_CASES = (180, 550, 1000, 2000, 5200)

# wall-unit column order in the raw files (DNS_data/README.md raw layout)
_MEAN_COLS = ("y_delta", "yplus", "U", "dUdy", "W", "P")
_COV_COLS = ("y_delta", "yplus", "uu", "vv", "ww", "uv", "uw", "vw", "k")

# friction scales parsed from the "%" header block (key -> regex for the value)
_HEADER_PATTERNS = {
    "nu":     r"\bnu\s*=\s*([0-9.eE+-]+)",
    "u_tau":  r"\bu_tau\s*=\s*([0-9.eE+-]+)",
    "Re_tau": r"\bRe_tau\s*=\s*([0-9.eE+-]+)",
    "delta":  r"\bdelta\s*=\s*([0-9.eE+-]+)",
}

# Boussinesq C_mu, used only to turn the baseline (nu_t, k) into the turbulence
# timescale tau = nu_t / (C_mu k) that the discrepancy non-dimensionalisation
# expects (see discrepancy.boussinesq_anisotropy). Kept here as the single place
# the loader-side convention is documented.
_CMU = 0.09


class ChannelDNS:
    """A single Lee-Moser plane-channel case as the canonical dns_field record.

    Attributes are per wall-normal station (wall to centerline), in wall units:
      y, y_delta, yplus            wall-normal coordinate (y dimensional, y/delta, y^+)
      U, dUdy, W, P                mean profiles (U^+, dU/dy^+, W^+, P^+)
      uu, vv, ww, uv, uw, vw, k    Reynolds stresses and tke (^+)
      R                            (N, 3, 3) full Reynolds-stress tensor (^+)
      *_std                        paired statistical-uncertainty profiles
    Scalars: u_tau, nu, re_tau (actual), re_tau_nominal, delta (half height).
    """

    def __init__(self, re_tau_nominal, params, mean, mean_std, cov, cov_std, meta):
        self.re_tau_nominal = int(re_tau_nominal)
        self.u_tau = float(params["u_tau"])
        self.nu = float(params["nu"])
        self.re_tau = float(params["Re_tau"])
        self.delta = float(params["delta"])
        self.meta = meta

        # mean profiles (wall units); columns per _MEAN_COLS
        for j, name in enumerate(_MEAN_COLS):
            setattr(self, name, mean[:, j])
            setattr(self, name + "_std", mean_std[:, j])
        # covariance profiles (wall units); columns per _COV_COLS
        for j, name in enumerate(_COV_COLS):
            # y_delta / yplus repeat across files; keep the mean-file copy
            if name in ("y_delta", "yplus"):
                setattr(self, name + "_cov", cov[:, j])
                continue
            setattr(self, name, cov[:, j])
            setattr(self, name + "_std", cov_std[:, j])

        self.n = self.yplus.size
        # dimensional wall-normal coordinate y = (y/delta) * delta
        self.y = self.y_delta * self.delta

        # full Reynolds-stress tensor R_ij = <u_i' u_j'> in wall units, with
        # x=streamwise(0), y=wall-normal(1), z=spanwise(2)
        self.R = self._assemble_tensor(self.uu, self.vv, self.ww,
                                       self.uv, self.uw, self.vw)
        self.R_std = self._assemble_tensor(self.uu_std, self.vv_std, self.ww_std,
                                           self.uv_std, self.uw_std, self.vw_std)

    # ---- construction -----------------------------------------------------

    @staticmethod
    def _assemble_tensor(uu, vv, ww, uv, uw, vw):
        """Pack the six independent components into an (N, 3, 3) symmetric tensor."""
        n = uu.size
        T = np.zeros((n, 3, 3))
        T[:, 0, 0] = uu
        T[:, 1, 1] = vv
        T[:, 2, 2] = ww
        T[:, 0, 1] = T[:, 1, 0] = uv
        T[:, 0, 2] = T[:, 2, 0] = uw
        T[:, 1, 2] = T[:, 2, 1] = vw
        return T

    @staticmethod
    def _parse_header(path):
        """Parse the friction scales from the '%' parameter block.

        Parsing is confined to the block introduced by "The parameters defining
        the simulation ..."; the paper-title citation above it also contains the
        string "Re_tau = 5200", so a whole-header scan would misread every case
        as Re_tau 5200.
        """
        params = {}
        in_block = False
        with open(path) as fh:
            for line in fh:
                if not line.lstrip().startswith("%"):
                    break  # header ends at the first data row
                if "parameters defining the simulation" in line:
                    in_block = True
                    continue
                if not in_block:
                    continue
                if "normalized by u_tau" in line:
                    break  # parameter block ends here
                for key, pattern in _HEADER_PATTERNS.items():
                    if key in params:
                        continue
                    match = re.search(pattern, line)
                    if match is not None:
                        params[key] = float(match.group(1))
        missing = [k for k in _HEADER_PATTERNS if k not in params]
        if missing:
            raise ValueError(f"{path}: header missing {missing}")
        return params

    @staticmethod
    def _one(case_dir, subdir, pattern):
        """Resolve exactly one raw file under case_dir/subdir matching pattern."""
        hits = sorted(glob.glob(os.path.join(case_dir, subdir, pattern)))
        if len(hits) != 1:
            raise ValueError(f"expected 1 '{pattern}' in {case_dir}/{subdir}, "
                             f"found {len(hits)}")
        return hits[0]

    @staticmethod
    def data_root(root=None):
        """Locate DNS_data: explicit root, env override, or repo-relative default."""
        if root is not None:
            return root
        env = os.environ.get("QBTM_DNS_DATA")
        if env:
            return env
        # this file is python/UQ/datasets/channel.py -> repo root is 4 up
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, "..", "..", "..", "DNS_data")

    @staticmethod
    def case_dir(re_tau_nominal, root=None):
        root = ChannelDNS.data_root(root)
        return os.path.join(root, "channel_flow",
                            f"LM_Channel_Re_tau={int(re_tau_nominal)}")

    @staticmethod
    def is_available(re_tau_nominal, root=None):
        """True when the raw profiles for this case are present locally."""
        cdir = ChannelDNS.case_dir(re_tau_nominal, root)
        return (os.path.isdir(os.path.join(cdir, "mean_velocity_pressure")) and
                os.path.isdir(os.path.join(cdir, "covariances_velocity")))

    @staticmethod
    def load(re_tau_nominal, root=None):
        """Parse one case (by nominal Re_tau) into a ChannelDNS record."""
        cdir = ChannelDNS.case_dir(re_tau_nominal, root)
        mean_f = ChannelDNS._one(cdir, "mean_velocity_pressure", "*_mean_prof.dat.txt")
        means_f = ChannelDNS._one(cdir, "mean_velocity_pressure", "*_mean_stdev.dat.txt")
        cov_f = ChannelDNS._one(cdir, "covariances_velocity", "*_vel_fluc_prof.dat.txt")
        covs_f = ChannelDNS._one(cdir, "covariances_velocity", "*_vel_fluc_stdev.dat.txt")

        params = ChannelDNS._parse_header(mean_f)
        mean = np.loadtxt(mean_f, comments="%")
        mean_std = np.loadtxt(means_f, comments="%")
        cov = np.loadtxt(cov_f, comments="%")
        cov_std = np.loadtxt(covs_f, comments="%")

        if mean.shape[1] != len(_MEAN_COLS):
            raise ValueError(f"{mean_f}: expected {len(_MEAN_COLS)} columns, "
                             f"got {mean.shape[1]}")
        if cov.shape[1] != len(_COV_COLS):
            raise ValueError(f"{cov_f}: expected {len(_COV_COLS)} columns, "
                             f"got {cov.shape[1]}")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError(f"row mismatch: mean {mean.shape[0]} vs cov "
                             f"{cov.shape[0]} in {cdir}")

        meta = {
            "regime": "incompressible",
            "case": "plane_channel",
            "source": "Lee and Moser 2015 (UT Austin / Oden Institute)",
            "re_tau_nominal": int(re_tau_nominal),
            "files": {
                "mean": os.path.relpath(mean_f, ChannelDNS.data_root(root)),
                "cov": os.path.relpath(cov_f, ChannelDNS.data_root(root)),
            },
        }
        return ChannelDNS(re_tau_nominal, params, mean, mean_std, cov, cov_std, meta)

    @staticmethod
    def load_all(root=None):
        """Load every available case, ordered by Reynolds number."""
        return [ChannelDNS.load(n, root) for n in CHANNEL_CASES
                if ChannelDNS.is_available(n, root)]

    # ---- derived quantities ----------------------------------------------

    def velocity_gradient(self):
        """Mean velocity-gradient tensor (N, 3, 3), grad_u[i, j] = d u_i / d x_j.

        Channel flow is a parallel shear flow, so the only nonzero entry is
        d u_x / d y = dU/dy^+ (wall units). The DNS supplies it directly.
        """
        g = np.zeros((self.n, 3, 3))
        g[:, 0, 1] = self.dUdy
        return g

    def bulk_velocity(self):
        """Bulk (mean) velocity U_b^+ = (1/delta) integral_0^delta U^+ dy.

        The profile runs wall (y=0) to centerline (y=delta), so the half-channel
        average is the bulk velocity. In wall units U_b^+ = U_b / u_tau.
        """
        return float(np.trapz(self.U, self.y) / (self.y[-1] - self.y[0]))

    def skin_friction(self):
        """Skin-friction coefficient Cf = 2 (u_tau / U_b)^2 = 2 / (U_b^+)^2.

        Built from the friction velocity (tau_w = rho u_tau^2) and the bulk
        velocity, the standard channel definition. Dimensionless, so it needs no
        unit system: it is a pure ratio of the wall-unit profile.
        """
        ub_plus = self.bulk_velocity()
        return 2.0 / ub_plus ** 2

    def reynolds_bulk(self):
        """Bulk Reynolds number Re_b = U_b (2 delta) / nu (Dean convention).

        Uses the case friction scales: U_b = U_b^+ u_tau, channel height 2 delta.
        """
        ub = self.bulk_velocity() * self.u_tau
        return ub * (2.0 * self.delta) / self.nu

    def to_dnsfield(self, timescale_plus, nu_t_plus=None):
        """Build a UQ DNSField from this case and a baseline turbulence timescale.

        The Reynolds anisotropy b = R/(2k) - I/3 is a pure ratio, so wall units
        carry through unchanged. The Boussinesq baseline anisotropy needs the
        baseline turbulence timescale tau^+ = nu_t^+ / (C_mu k^+) at each DNS
        station (supplied by the baseline RANS field generator); it makes the
        non-dimensional strain S = strain * tau reproduce the actual baseline
        anisotropy (discrepancy.boussinesq_anisotropy).
        """
        return DNSField(
            grad_u=self.velocity_gradient(),
            R=self.R,
            k=self.k,
            timescale=np.asarray(timescale_plus, dtype=float),
            nu_t=None if nu_t_plus is None else np.asarray(nu_t_plus, dtype=float),
            meta=dict(self.meta, baseline="SST RANS at matched Re_tau"),
        )

    @staticmethod
    def cmu():
        """Boussinesq C_mu used for the loader-side timescale convention."""
        return _CMU

    def __repr__(self):
        return (f"ChannelDNS(Re_tau={self.re_tau:.1f} "
                f"[nominal {self.re_tau_nominal}], n={self.n}, "
                f"u_tau={self.u_tau:.6g}, nu={self.nu:.3g})")
