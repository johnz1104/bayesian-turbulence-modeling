"""Shared helpers for the wall-bounded DNS profile loaders (Couette, pipe,
rotating channel).

Each Step-2 dataset (DNS_plan.md Step 2) arrives in its own raw format from a
different group, but every loader emits the SAME canonical dns_field record
(DNS_data/README.md, "Standardized processing format"), so the discrepancy, UQ,
and evaluation layers downstream always see one uniform representation. The
Lee-Moser channel loader (channel.py) is the first of the family and established
the pattern (location, naming, provenance); this module factors the parts the
cross-flow loaders share so each new format is a thin parser on top of one base.

Coordinate convention (shared with channel.py): x = streamwise (0), y =
wall-normal (1), z = spanwise (2). Everything is in wall units (signified ^+):
velocities by u_tau, lengths by nu/u_tau, stresses and k by u_tau^2. The Reynolds
anisotropy b = R/(2k) - I/3 is a pure ratio, so the wall-unit normalisation
carries through the discrepancy and UQ unchanged.
"""
import os

import numpy as np

from ..dns_field import DNSField

# Boussinesq C_mu, the single place the loader-side timescale convention lives,
# matching channel.py: tau^+ = 1/(C_mu omega^+) = nu_t^+/(C_mu k^+).
_CMU = 0.09


def data_root(root=None):
    """Locate DNS_data: explicit root, QBTM_DNS_DATA env, or repo-relative default.

    Mirrors ChannelDNS.data_root so the cross-flow loaders honour the same
    centralized-data workflow (CLAUDE.md section 9): the bulk fields live once in
    QBTM/DNS_data and QBTM_DNS_DATA points every worktree at them.
    """
    if root is not None:
        return root
    env = os.environ.get("QBTM_DNS_DATA")
    if env:
        return env
    # this file is python/UQ/datasets/_common.py -> repo root is 3 up, DNS_data 1 more
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "..", "DNS_data")


def assemble_tensor(uu, vv, ww, uv, uw=None, vw=None):
    """Pack the independent components into an (N, 3, 3) symmetric Reynolds-stress
    tensor R_ij = <u_i' u_j'> with x=streamwise(0), y=wall-normal(1), z=spanwise(2).

    uw and vw default to zero: the spanwise off-diagonals vanish in a parallel
    shear flow (channel, Couette, pipe) and are supplied only by the
    streamwise-rotating channel, where rotation makes them nonzero.
    """
    uu = np.asarray(uu, dtype=float)
    n = uu.size
    zero = np.zeros(n)
    uw = zero if uw is None else np.asarray(uw, dtype=float)
    vw = zero if vw is None else np.asarray(vw, dtype=float)
    T = np.zeros((n, 3, 3))
    T[:, 0, 0] = uu
    T[:, 1, 1] = np.asarray(vv, dtype=float)
    T[:, 2, 2] = np.asarray(ww, dtype=float)
    T[:, 0, 1] = T[:, 1, 0] = np.asarray(uv, dtype=float)
    T[:, 0, 2] = T[:, 2, 0] = uw
    T[:, 1, 2] = T[:, 2, 1] = vw
    return T


def wall_normal_gradient(yplus, field):
    """d(field)/d(y^+) by second-order central differences on the (non-uniform)
    wall-normal grid (numpy.gradient's spacing-aware stencil).

    Used where the DNS supplies no mean-gradient column, unlike the Lee-Moser
    channel which gives dU/dy^+ directly. The Couette, pipe, and rotating-channel
    files all omit it, so the mean velocity gradient is finite-differenced here.
    """
    return np.gradient(np.asarray(field, dtype=float), np.asarray(yplus, dtype=float))


def first_data_row(path, ncols, comment_prefixes=("%", "*")):
    """Index of the first line that parses as exactly ncols floats.

    The Step-2 headers are heterogeneous (the pipe uses '%', the Couette uses '*'
    blocks plus bare prose lines, the rotating channel uses '%%'), so rather than
    trust a single comment character the loaders skip to the first line that reads
    as the expected number of floats. Returns the 0-based line index.
    """
    with open(path) as fh:
        for i, line in enumerate(fh):
            s = line.strip()
            if not s or s[0] in comment_prefixes:
                continue
            parts = s.split()
            if len(parts) != ncols:
                continue
            try:
                [float(p) for p in parts]
            except ValueError:
                continue
            return i
    raise ValueError(f"{path}: no data row with {ncols} float columns found")


class WallBoundedProfileDNS:
    """Canonical dns_field record for a 1-D wall-bounded profile in wall units.

    Holds, per wall-normal station: y^+ (and the outer coordinate where the file
    gives it), the mean velocity (U^+, and the spanwise W^+ where a second mean
    component is present), the full Reynolds-stress tensor R^+ and k^+, the
    friction Reynolds number, and metadata. The Boussinesq anisotropy and the
    model-form discrepancy consume exactly this record, identically to the channel
    case (UQ.datasets.channel_discrepancy.channel_discrepancy is generic over it).

    Subclasses (CouetteDNS, PipeDNS, RotatingChannelDNS) parse their own raw
    format and call __init__; the velocity gradient and the DNSField adapter are
    shared here.
    """

    def __init__(self, yplus, U, R, k, re_tau, meta, y=None, y_outer=None, W=None):
        self.yplus = np.asarray(yplus, dtype=float)
        self.U = np.asarray(U, dtype=float)
        self.W = None if W is None else np.asarray(W, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.k = np.asarray(k, dtype=float)
        self.re_tau = float(re_tau)
        self.y = None if y is None else np.asarray(y, dtype=float)
        self.y_outer = None if y_outer is None else np.asarray(y_outer, dtype=float)
        self.meta = meta or {}
        self.n = self.yplus.size

        # component views onto the assembled tensor (read-only convenience)
        self.uu = self.R[:, 0, 0]
        self.vv = self.R[:, 1, 1]
        self.ww = self.R[:, 2, 2]
        self.uv = self.R[:, 0, 1]
        self.uw = self.R[:, 0, 2]
        self.vw = self.R[:, 1, 2]

    # ---- derived quantities -----------------------------------------------

    def dUdy(self):
        """Streamwise mean velocity gradient dU^+/dy^+ (finite-differenced)."""
        return wall_normal_gradient(self.yplus, self.U)

    def velocity_gradient(self):
        """Mean velocity-gradient tensor grad_u[i, j] = d u_i / d x_j (wall units).

        The streamwise gradient dU^+/dy^+ = grad_u[0, 1] is always present. The
        spanwise gradient dW^+/dy^+ = grad_u[2, 1] is included only when a mean
        spanwise velocity exists (the streamwise-rotating channel), where rotation
        drives W^+ and the off-diagonal stresses and the full anisotropy is active.
        """
        g = np.zeros((self.n, 3, 3))
        g[:, 0, 1] = wall_normal_gradient(self.yplus, self.U)
        if self.W is not None:
            g[:, 2, 1] = wall_normal_gradient(self.yplus, self.W)
        return g

    def to_dnsfield(self, timescale_plus, nu_t_plus=None):
        """Build a UQ DNSField from this case and a baseline turbulence timescale.

        Identical contract to ChannelDNS.to_dnsfield: the anisotropy b = R/(2k) -
        I/3 is a pure ratio so wall units carry through, and the Boussinesq
        baseline needs the baseline timescale tau^+ at each station to reproduce
        the actual eddy-viscosity anisotropy (discrepancy.boussinesq_anisotropy).
        """
        return DNSField(
            grad_u=self.velocity_gradient(),
            R=self.R,
            k=self.k,
            timescale=np.asarray(timescale_plus, dtype=float),
            nu_t=None if nu_t_plus is None else np.asarray(nu_t_plus, dtype=float),
            meta=dict(self.meta),
        )

    @staticmethod
    def cmu():
        """Boussinesq C_mu used for the loader-side timescale convention."""
        return _CMU
