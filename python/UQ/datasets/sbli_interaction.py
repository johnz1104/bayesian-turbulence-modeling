"""Impinging oblique-shock interaction DNS loaders (adiabatic and wall-thermal).

Parses the two Rome impinging-shock interaction databases into one canonical
two-dimensional interaction record (free-stream Mach 2.28, shock-generator
incidence 8 degrees, gamma 1.4, statistically span-homogeneous fields, lengths
in incoming boundary-layer thicknesses):

- shock_wave_BLI (Pirozzoli and Bernardini, AIAA J. 49(6) 2011,
  doi:10.2514/1.J050901): the ADIABATIC interaction. 32 Tecplot blocks per
  averaging (favre_33..64 and reynolds_33..64, zones 61 x 344 sharing edge
  columns) tiling the downstream half of the 80.58 x 12.89 domain; a 13-column
  wall series (stat.dat) over the full domain; the incoming-layer profile
  blinc.dat at x = 43.6 (Re_tau 466, cf 2.56e-3); the turbulence-energy budget
  at three interaction stations. Carries the stress record, Cf and Cp, NO
  temperature-velocity covariance and NO wall heat flux.
- heat_transfer_SBLI (Bernardini, Asproulias, Larsson, Pirozzoli and Grasso,
  Phys. Rev. Fluids 1, 084403, 2016): the SAME interaction at five
  wall-to-recovery-temperature ratios s = 0.5, 0.75, 1.0, 1.4, 1.9. One
  13-column Tecplot zone per s (3001 x 284 for s = 0.5, 0.75; 1610 x 230 for
  the rest) carrying the thermodynamic mean state and the TURBULENT HEAT-FLUX
  VECTOR (u''T'', v''T''), the only interaction source of the dq targets; a
  5-column wall series per s (x*, cf, pw, pwrms, St). Verified quirks handled
  here: the s = 1.0 wall series has FOUR columns (no Stanton, adiabatic) and a
  different pw normalization, so wall pressure is reduced to Cp against each
  file's own upstream plateau; the adiabatic readme's 13-variable read loop
  against its files' 14 actual columns is resolved by counting columns from
  the data.

Averaging convention (documented; the loader tests measure the gap): the
tabulated double-prime covariances are taken as the density-weighted Favre
second moments, R_ij = <rho u_i''u_j''>/<rho> and q_i = <rho u_i''T''>/<rho>,
the standard tabulation for these databases; on the adiabatic set the
favre-versus-reynolds cross-check (favre_vs_reynolds_gap) measures the
outer-layer gap between the two averagings, which bounds the effect of either
interpretation.

Units: the stored field arrays keep the files' own free-stream
nondimensionalization (velocities on U_inf, temperature on T_inf, density on
rho_inf, lengths on the incoming-layer thickness). The pre-registered wall-unit
record for the discrepancy convention comes from the reference-scaled
accessors (k_wall_units, stress_wall_units, q_hat_wall_units), which use the
case's UPSTREAM reference friction state at x* = -7 (a local friction velocity
would vanish at separation and cannot scale an interaction target; the station
carries the dated pre-registration amendment noted at the constant below).

Interaction coordinate: the heated files come in x* with origin at the nominal
inviscid impingement point and are kept as-is. The adiabatic series comes in
raw domain x; its x* is re-originated at the measured half-rise point of its
smoothed wall pressure (the same landmark rule the pre-registration pins for
the shock-position QoI), stated in the record meta. Cross-campaign comparisons
align both datasets on their half-rise landmarks (cross_campaign_wall_residual).

Data-only physics anchors measured here (the modeled-sigma floors of the
pre-registration): the upstream momentum-integral residual of the adiabatic
wall series, the budget-closure residual of the three budget stations, and the
cross-campaign adiabatic wall residual between the two databases.
"""
import os
import re

import numpy as np

from . import _common
from .. import realizability

SBLI_S_CASES = ("0.5", "0.75", "1.0", "1.4", "1.9")

_ADIABATIC_SUBDIR = "shock_wave_BLI"
_THERMAL_SUBDIR = "heat_transfer_SBLI"
_ADIABATIC_BLOCKS = tuple(range(33, 65))

_GAMMA = 1.4
_MACH = 2.28

# pre-registered constants (fixed in the study pre-registration; the reference
# station carries the dated data-forced amendment recorded in the
# pre-registration addendum: the wall-thermal switch of the heated campaign
# ramps over x* in about [-10.5, -8], so the pre-registered -10 sits ON the
# ramp; -7 is post-switch for every s and upstream of the widest onset, -5.16
# at s = 1.9, measured before any discrepancy was extracted):
_XSTAR_REF = -7.0      # upstream reference station for the friction state
_ONSET_FACTOR = 1.05   # smoothed-wall-pressure rise factor defining onset
_SMOOTH_WIDTH = 0.5    # centered moving-average width, reference lengths
_YSTAR_MIN = 0.05      # interior mask, lower wall-normal bound
_YSTAR_MAX = 2.0       # interior mask, upper wall-normal bound
_K_FLOOR_FRACTION = 1e-3  # interior mask, turbulence-activity floor


def _find(path):
    """Return path or path + '.txt' (the compiled copies carry either)."""
    if os.path.isfile(path):
        return path
    alt = path + ".txt"
    if os.path.isfile(alt):
        return alt
    raise FileNotFoundError(path)


def _read_zone(path):
    """One Tecplot zone file: ' zone i= NX , j= NY' then NX*NY rows, i fastest.

    Returns (nx, ny, arr) with arr shaped (nx, ny, ncols): the FORTRAN read
    order is do j / do i, so the flat row index is (j-1)*nx + (i-1).
    """
    with open(path) as fh:
        header = fh.readline()
    m = re.search(r"i\s*=\s*(\d+)\s*,\s*j\s*=\s*(\d+)", header)
    if m is None:
        raise ValueError(f"{path}: no Tecplot zone header")
    nx, ny = int(m.group(1)), int(m.group(2))
    data = np.loadtxt(path, skiprows=1)
    if data.ndim != 2 or data.shape[0] != nx * ny:
        raise ValueError(f"{path}: {data.shape} does not match zone "
                         f"{nx} x {ny}")
    return nx, ny, data.reshape(ny, nx, data.shape[1]).transpose(1, 0, 2)


def _axes_from_grid(arr, path):
    """Extract separable x (col 0) and y (col 1) axes and assert separability."""
    x = arr[:, 0, 0].copy()
    y = arr[0, :, 1].copy()
    span = max(np.ptp(x), 1.0)
    if np.max(np.abs(arr[:, :, 0] - x[:, None])) > 1e-6 * span:
        raise ValueError(f"{path}: x coordinate varies along j")
    if np.max(np.abs(arr[:, :, 1] - y[None, :])) > 1e-6 * max(np.ptp(y), 1.0):
        raise ValueError(f"{path}: y coordinate varies along i")
    return x, y


def _tile_blocks(paths):
    """Concatenate single-row block zones along i, dropping shared edge columns.

    The last column of each block repeats the first of the next (verified
    format fact); the overlap agreement is asserted as a parse guard before
    the duplicate is dropped.
    """
    tiles = []
    prev_edge = None
    ny_ref = None
    for path in paths:
        nx, ny, arr = _read_zone(path)
        if ny_ref is None:
            ny_ref = ny
        elif ny != ny_ref:
            raise ValueError(f"{path}: ny {ny} differs from {ny_ref}")
        if prev_edge is not None:
            gap = np.max(np.abs(arr[0] - prev_edge))
            scale = max(np.max(np.abs(prev_edge)), 1.0)
            if gap > 1e-5 * scale:
                raise ValueError(f"{path}: shared edge column mismatch "
                                 f"({gap:.3e} against scale {scale:.3e})")
            tiles.append(arr[1:])
        else:
            tiles.append(arr)
        prev_edge = arr[-1]
    return np.concatenate(tiles, axis=0)


def _moving_average(values, x, width):
    """Centered moving average of pinned physical width on a near-uniform grid."""
    dx = np.median(np.diff(x))
    half = max(int(round(0.5 * width / max(dx, 1e-12))), 0)
    if half == 0:
        return np.asarray(values, dtype=float)
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    padded = np.concatenate([np.full(half, values[0]), values,
                             np.full(half, values[-1])])
    return np.convolve(padded, kernel, mode="valid")


class SBLIWallSeries:
    """Wall distributions against the streamwise coordinate, plus landmarks.

    Cp is formed against the file's own upstream plateau (the pinned rule that
    also absorbs the s = 1.0 normalization quirk): with the upstream wall
    pressure equal to the free-stream pressure on a zero-pressure-gradient
    layer, Cp = (pw - plateau) / (plateau * gamma/2 * M^2), since
    q_dyn = (gamma/2) M^2 p_inf.

    Landmarks (every rule fixed in the pre-registration): the shock position
    is the half-rise crossing of the smoothed wall pressure; the interaction
    onset is where the smoothed wall pressure first exceeds 1.05 times the
    upstream plateau; separation and reattachment bound the longest contiguous
    negative-cf run (linearly interpolated crossings), None when the series
    never separates.
    """

    def __init__(self, x, cf, pw, pwrms, St=None, extras=None, pw_valid=True):
        self.x = np.asarray(x, dtype=float)
        self.cf = np.asarray(cf, dtype=float)
        self.pw = np.asarray(pw, dtype=float)
        self.pwrms = np.asarray(pwrms, dtype=float)
        self.St = None if St is None else np.asarray(St, dtype=float)
        self.extras = dict(extras or {})
        # the s = 1.0 series' pressure column is integer-quantized and drifts
        # where a zero-pressure-gradient plateau belongs (verified quirk); its
        # loader marks it invalid, Cp is not formed, and the record's
        # field-row landmarks serve instead (pre-registration addendum)
        self.pw_valid = bool(pw_valid)

        if self.pw_valid:
            x0, x1 = self.x[0], self.x[-1]
            up = self.x < x0 + 0.25 * (x1 - x0)
            self.pw_plateau = float(np.median(self.pw[up]))
            qdyn_over_plateau = 0.5 * _GAMMA * _MACH ** 2
            self.cp = (self.pw - self.pw_plateau) / (self.pw_plateau
                                                     * qdyn_over_plateau)
            self._pw_smooth = _moving_average(self.pw, self.x, _SMOOTH_WIDTH)
            self.pw_downstream = float(np.median(self._pw_smooth[
                self.x > x1 - 0.10 * (x1 - x0)]))
        else:
            self.pw_plateau = None
            self.pw_downstream = None
            self.cp = None
            self._pw_smooth = None

    # ---- landmarks ----------------------------------------------------------

    @staticmethod
    def _first_crossing(x, values, level):
        above = values >= level
        idx = np.argmax(above)
        if not above.any() or idx == 0:
            return None
        x0, x1 = x[idx - 1], x[idx]
        v0, v1 = values[idx - 1], values[idx]
        return float(x0 + (level - v0) * (x1 - x0) / max(v1 - v0, 1e-30))

    def shock_position(self):
        """Half-rise point of the smoothed wall pressure (the pinned rule)."""
        if not self.pw_valid:
            raise ValueError("wall-pressure column marked invalid (the "
                             "documented s = 1.0 quirk); use the record's "
                             "field-row landmarks")
        level = 0.5 * (self.pw_plateau + self.pw_downstream)
        return self._first_crossing(self.x, self._pw_smooth, level)

    def onset(self):
        """First smoothed-pressure crossing of 1.05 times the upstream plateau."""
        if not self.pw_valid:
            raise ValueError("wall-pressure column marked invalid (the "
                             "documented s = 1.0 quirk); use the record's "
                             "field-row landmarks")
        return self._first_crossing(self.x, self._pw_smooth,
                                    _ONSET_FACTOR * self.pw_plateau)

    def separation_reattachment(self):
        """(x_s, x_r) bounding the longest negative-cf run; (None, None) if attached."""
        neg = self.cf < 0.0
        if not neg.any():
            return None, None
        # longest contiguous run of negative cf
        edges = np.diff(neg.astype(int))
        starts = list(np.where(edges == 1)[0] + 1)
        ends = list(np.where(edges == -1)[0] + 1)
        if neg[0]:
            starts.insert(0, 0)
        if neg[-1]:
            ends.append(neg.size)
        i0, i1 = max(zip(starts, ends), key=lambda se: se[1] - se[0])
        # interpolate the bounding zero crossings
        if i0 == 0:
            x_s = float(self.x[0])
        else:
            xa, xb = self.x[i0 - 1], self.x[i0]
            ca, cb = self.cf[i0 - 1], self.cf[i0]
            x_s = float(xa + ca * (xb - xa) / max(ca - cb, 1e-30))
        if i1 >= self.x.size:
            x_r = float(self.x[-1])
        else:
            xa, xb = self.x[i1 - 1], self.x[i1]
            ca, cb = self.cf[i1 - 1], self.cf[i1]
            x_r = float(xa - ca * (xb - xa) / max(cb - ca, 1e-30))
        return x_s, x_r

    def interpolate(self, name, x_star):
        """Linear interpolation of a series column at the given coordinate."""
        values = getattr(self, name)
        return float(np.interp(x_star, self.x, values))

    def momentum_integral_residual(self, x_lo=-25.0, margin=2.0):
        """Upstream attached-region momentum-integral residual (data-only).

        The compressible zero-pressure-gradient balance dtheta/dx = cf/2, read
        in integral form: the least-squares slope of the series' own momentum
        thickness over the pre-onset window against the window-mean cf/2 (a
        pointwise derivative of the tabulated theta is differencing-noise
        dominated and would overstate the data uncertainty). Applies only
        where the series carries the momentum thickness (the adiabatic
        stat.dat); the window is clipped at x_lo to stay clear of the inflow.
        Returns the relative gap |slope - mean(cf/2)| / mean(cf/2).
        """
        if "theta" not in self.extras:
            raise ValueError("series carries no momentum thickness")
        onset = self.onset()
        hi = (onset if onset is not None else self.x[-1]) - margin
        window = (self.x >= x_lo) & (self.x <= hi)
        if window.sum() < 10:
            raise ValueError("upstream window too short for the anchor")
        theta = np.asarray(self.extras["theta"], dtype=float)[window]
        slope = np.polyfit(self.x[window], theta, 1)[0]
        target = float(np.mean(0.5 * self.cf[window]))
        return float(abs(slope - target) / target)


class IncomingBoundaryLayer:
    """The adiabatic dataset's incoming-layer profile at x = 43.6 (blinc.dat)."""

    COLUMNS = ("y_outer", "yplus", "u_over_uinf", "uvd_plus", "urms", "vrms",
               "wrms", "uv", "urms_plus", "vrms_plus", "wrms_plus",
               "sqrt_rho_over_rhow")

    def __init__(self, data):
        if data.shape[1] != len(self.COLUMNS):
            raise ValueError(f"blinc: {data.shape[1]} columns, expected "
                             f"{len(self.COLUMNS)}")
        for j, name in enumerate(self.COLUMNS):
            setattr(self, name, data[:, j].copy())
        self.n = data.shape[0]
        # readme-stated station parameters carried for the baseline anchor
        self.station = {"x": 43.6, "re_tau": 466.0, "re_theta": 2344.0,
                        "cf": 2.56e-3, "H": 3.55, "mach": _MACH}

    def van_driest_residual(self, u_tau_over_uinf):
        """Recompute uvd+ from the profile's own u and density-ratio columns
        against its uvd+ column (the data-only cross-column identity), given
        the friction velocity that converts u/U_inf to u+."""
        u_plus = self.u_over_uinf / u_tau_over_uinf
        du = np.diff(u_plus)
        s = self.sqrt_rho_over_rhow
        recomputed = np.concatenate([[0.0],
                                     np.cumsum(0.5 * (s[1:] + s[:-1]) * du)])
        denom = np.maximum(self.uvd_plus, 1e-9)
        return (recomputed - self.uvd_plus) / denom


class SBLIInteractionDNS:
    """One interaction case: the dense 2-D Favre field, wall series, landmarks.

    Arrays are (nx, ny) on separable axes x (interaction coordinate) and y
    (wall-normal, wall at y = 0), in the files' free-stream units. The stress
    record R is (nx, ny, 3, 3) with x streamwise (0), y wall-normal (1),
    z spanwise (2) and the span off-diagonals zero by homogeneity; q is
    (nx, ny, 3) with the spanwise component zero, present only for the
    wall-thermal cases. rho, p, T are present for every case (the adiabatic
    record takes them from its reynolds files, a documented mixed
    construction). reynolds holds the adiabatic Reynolds-average views for
    the convention cross-check.
    """

    def __init__(self, case, x, y, U, V, R, series, meta, rho, p, T,
                 q=None, reynolds=None, blinc=None, budgets=None):
        self.case = case
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.U = np.asarray(U, dtype=float)
        self.V = np.asarray(V, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.rho = np.asarray(rho, dtype=float)
        self.p = np.asarray(p, dtype=float)
        self.T = np.asarray(T, dtype=float)
        self.q = None if q is None else np.asarray(q, dtype=float)
        self.series = series
        self.reynolds = reynolds
        self.blinc = blinc
        self.budgets = budgets
        self.meta = dict(meta or {})
        self.nx, self.ny = self.U.shape
        self.k = 0.5 * (self.R[:, :, 0, 0] + self.R[:, :, 1, 1]
                        + self.R[:, :, 2, 2])
        self.reference = self._reference_state()

    # ---- reference friction state (pre-registered upstream station) ---------

    def _reference_state(self):
        """The upstream reference friction state at the pinned x* = -10.

        u_tau/U_inf = sqrt((cf/2) rho_inf/rho_w) from the wall series' own cf
        and the field's own wall-row density; T_w from the field's wall row.
        The adiabatic series' utau column is carried alongside as a
        cross-check view where present.
        """
        i_ref = int(np.argmin(np.abs(self.x - _XSTAR_REF)))
        cf_ref = self.series.interpolate("cf", _XSTAR_REF)
        rho_w = float(self.rho[i_ref, 0])
        T_w = float(self.T[i_ref, 0])
        u_tau = float(np.sqrt(0.5 * cf_ref / rho_w))
        ref = {"x_star": _XSTAR_REF, "i_ref": i_ref, "cf": cf_ref,
               "rho_w": rho_w, "T_w": T_w, "u_tau_over_uinf": u_tau}
        if "utau" in self.series.extras:
            # the adiabatic series tabulates u_tau directly; carried as a
            # cross-check view against the cf-and-wall-density construction
            ref["u_tau_series"] = float(np.interp(
                _XSTAR_REF, self.series.x, self.series.extras["utau"]))
        return ref

    # ---- pre-registered wall-unit record ------------------------------------

    def k_wall_units(self):
        """Turbulence energy in u_tau_ref^2 units."""
        return self.k / self.reference["u_tau_over_uinf"] ** 2

    def stress_wall_units(self):
        """The Favre stress record in u_tau_ref^2 units (velocity^2 form)."""
        return self.R / self.reference["u_tau_over_uinf"] ** 2

    def q_hat_wall_units(self):
        """The heat-flux vector in (u_tau_ref, T_w) units, per the committed
        convention q_hat_i = <rho u_i''T''>/(<rho> u_tau T_w)."""
        if self.q is None:
            raise ValueError(f"{self.case}: no turbulent heat flux in this "
                             "dataset (adiabatic record)")
        scale = (self.reference["u_tau_over_uinf"] * self.reference["T_w"])
        return self.q / scale

    # ---- field-row landmarks (the pre-registration addendum's canonical route)

    def wall_pressure_row(self):
        """The field's own wall-row pressure (free-stream units), the landmark
        source that is healthy for every case (the s = 1.0 wall series' pw
        column is not)."""
        return self.p[:, 0]

    def wall_temperature_row(self):
        """The field's own wall-row temperature (T/T_inf)."""
        return self.T[:, 0]

    def _pressure_landmark(self, kind):
        pw = _moving_average(self.wall_pressure_row(), self.x, _SMOOTH_WIDTH)
        x0, x1 = self.x[0], self.x[-1]
        plateau = float(np.median(pw[self.x < x0 + 0.25 * (x1 - x0)]))
        downstream = float(np.median(pw[self.x > x1 - 0.10 * (x1 - x0)]))
        if kind == "onset":
            level = _ONSET_FACTOR * plateau
        else:
            level = 0.5 * (plateau + downstream)
        return SBLIWallSeries._first_crossing(self.x, pw, level)

    def onset(self):
        """Interaction onset from the field wall-pressure row (pinned rule;
        validated against the healthy wall series to 0.02 reference lengths)."""
        return self._pressure_landmark("onset")

    def shock_position(self):
        """Wall-pressure half-rise from the field wall-pressure row (pinned rule)."""
        return self._pressure_landmark("half")

    def cp_from_field(self):
        """(x, Cp) from the field wall-pressure row through the same
        upstream-plateau rule the wall series uses; the canonical Cp source
        where a series' pressure column is unusable (the s = 1.0 quirk)."""
        pw = self.wall_pressure_row()
        x0, x1 = self.x[0], self.x[-1]
        plateau = float(np.median(pw[self.x < x0 + 0.25 * (x1 - x0)]))
        cp = (pw - plateau) / (plateau * 0.5 * _GAMMA * _MACH ** 2)
        return self.x, cp

    def thermal_switch_position(self):
        """Where the wall temperature crosses halfway between its upstream
        (recovery) plateau and its downstream (imposed s-condition) plateau.

        The heated campaign develops its layer under an adiabatic wall and
        switches to the s-condition inside the saved window (the measured ramp
        spans about x* in [-10.5, -8]); this landmark splits the upstream
        region per the pre-registration addendum. Returns None when no switch
        exists (the adiabatic cases; step below two percent of the plateau).
        """
        Tw = _moving_average(self.wall_temperature_row(), self.x,
                             _SMOOTH_WIDTH)
        x0, x1 = self.x[0], self.x[-1]
        upstream = float(np.median(Tw[self.x < x0 + 0.10 * (x1 - x0)]))
        downstream = float(np.median(Tw[self.x > x0 + 0.50 * (x1 - x0)]))
        if abs(downstream - upstream) < 0.02 * upstream:
            return None
        level = 0.5 * (upstream + downstream)
        if downstream > upstream:
            return SBLIWallSeries._first_crossing(self.x, Tw, level)
        return SBLIWallSeries._first_crossing(self.x, -Tw, -level)

    # ---- masks and checks ----------------------------------------------------

    def interior_mask(self):
        """The pre-registered interior mask: y* in [0.05, 2.0] and turbulence
        energy above 1e-3 of the field maximum (excludes the laminar free
        stream, keeps the shock-amplified region)."""
        in_y = (self.y >= _YSTAR_MIN) & (self.y <= _YSTAR_MAX)
        active = self.k >= _K_FLOOR_FRACTION * self.k.max()
        return in_y[None, :] & active

    def realizable_fraction(self, tol=1e-9):
        """Fraction of interior stress records inside the realizable set."""
        m = self.interior_mask()
        R = self.R[m]
        ok = realizability.is_realizable(R, tol=tol)
        return float(np.mean(ok))

    def favre_vs_reynolds_gap(self):
        """Median relative gap between the Favre and Reynolds streamwise
        variances over the upstream attached outer layer (adiabatic only).

        Bounds the effect of the double-prime column interpretation; recorded
        by the loader tests as the convention finding.
        """
        if self.reynolds is None:
            raise ValueError("no Reynolds-average views on this record")
        onset = self.series.onset()
        cols = self.x < (onset - 2.0 if onset is not None else self.x[0])
        band = (self.y > 0.2) & (self.y < 0.8)
        sel = np.ix_(np.where(cols)[0], np.where(band)[0])
        favre = self.R[:, :, 0, 0][sel]
        reyn = self.reynolds["uu"][sel]
        denom = np.maximum(np.abs(reyn), 1e-12 * np.abs(reyn).max())
        return float(np.median(np.abs(favre - reyn) / denom))

    def budget_closure_residual(self):
        """Per-station closure of the tabulated budget terms (data-only anchor).

        The steady balance leaves the tabulated columns (convection,
        transport, production, viscous diffusion, dissipation, lumped
        pressure-dilatation and mass diffusion) summing to zero up to
        statistical convergence; returns {x_star: rms(sum)/max|production|}.
        """
        if not self.budgets:
            raise ValueError("no budget stations on this record")
        out = {}
        for x_star, table in self.budgets.items():
            terms = table[:, 1:]
            resid = terms.sum(axis=1)
            out[x_star] = float(np.sqrt(np.mean(resid ** 2))
                                / np.abs(terms[:, 2]).max())
        return out

    # ---- location and parsing -------------------------------------------------

    @staticmethod
    def adiabatic_dir(root=None):
        return os.path.join(_common.data_root(root), _ADIABATIC_SUBDIR)

    @staticmethod
    def thermal_dir(root=None):
        return os.path.join(_common.data_root(root), _THERMAL_SUBDIR)

    @staticmethod
    def is_available(case="adiabatic", root=None):
        if case == "adiabatic":
            probe = os.path.join(SBLIInteractionDNS.adiabatic_dir(root),
                                 "statistics", "favre_averages", "favre",
                                 "favre_33.dat")
        else:
            probe = os.path.join(SBLIInteractionDNS.thermal_dir(root),
                                 "favre_averages", f"favre_{case}.dat")
        for p in (probe, probe + ".txt"):
            if os.path.isfile(p):
                return True
        return False

    @staticmethod
    def adiabatic(root=None):
        """Load the adiabatic interaction (fields, wall series, blinc, budgets)."""
        base = os.path.join(SBLIInteractionDNS.adiabatic_dir(root),
                            "statistics")
        favre = _tile_blocks([
            _find(os.path.join(base, "favre_averages", "favre",
                               f"favre_{ib}.dat"))
            for ib in _ADIABATIC_BLOCKS])
        reyn = _tile_blocks([
            _find(os.path.join(base, "reynolds_averages", "reynolds",
                               f"reynolds_{ib}.dat"))
            for ib in _ADIABATIC_BLOCKS])
        if favre.shape[2] != 8:
            raise ValueError(f"adiabatic favre: {favre.shape[2]} columns, "
                             "expected 8")
        if reyn.shape[2] != 14:
            raise ValueError(f"adiabatic reynolds: {reyn.shape[2]} columns, "
                             "expected 14 (the readme loop says 13; the files "
                             "hold 14)")
        x_raw, y = _axes_from_grid(favre, "adiabatic favre tiles")
        x_check, _ = _axes_from_grid(reyn, "adiabatic reynolds tiles")
        if np.max(np.abs(x_raw - x_check)) > 1e-6 * np.ptp(x_raw):
            raise ValueError("favre and reynolds tiles disagree on x")

        # wall series over the full domain; block-edge x values duplicated
        stat = np.loadtxt(_find(os.path.join(base, "wall_stat", "stat.dat")))
        if stat.shape[1] != 13:
            raise ValueError(f"stat.dat: {stat.shape[1]} columns, expected 13")
        xs, keep = np.unique(stat[:, 0], return_index=True)
        stat = stat[keep]
        names = ("x", "cf", "pw", "pwrms", "utau", "deltav", "delta99",
                 "dstar", "theta", "dstarinc", "thetainc", "H", "Hinc")
        extras = {n: stat[:, j] for j, n in enumerate(names) if j >= 4}
        series_raw = SBLIWallSeries(stat[:, 0], stat[:, 1], stat[:, 2],
                                    stat[:, 3], St=None, extras=extras)
        # re-origin at the measured half-rise landmark (the pinned rule)
        x_shock_raw = series_raw.shock_position()
        if x_shock_raw is None:
            raise ValueError("adiabatic wall series shows no pressure rise")
        series = SBLIWallSeries(stat[:, 0] - x_shock_raw, stat[:, 1],
                                stat[:, 2], stat[:, 3], St=None,
                                extras=extras)
        x = x_raw - x_shock_raw

        blinc_data = np.loadtxt(_find(os.path.join(base, "bl_incoming",
                                                   "blinc.dat")))
        blinc = IncomingBoundaryLayer(blinc_data)

        budgets = {}
        for tag in ("-1.93", "-0.05", "2.10"):
            table = np.loadtxt(_find(os.path.join(
                base, "k-budget", f"tkebudget_xstar_{tag}.dat")))
            if table.shape[1] != 7:
                raise ValueError(f"budget {tag}: {table.shape[1]} columns")
            budgets[float(tag)] = table

        # Favre stress record from the favre tiles; thermodynamics and the
        # Reynolds views from the reynolds tiles (documented mixed construction)
        R = _assemble_stress(favre[:, :, 5], favre[:, :, 6], favre[:, :, 7],
                             favre[:, :, 4])
        reynolds_views = {"rho_rho": reyn[:, :, 7], "uu": reyn[:, :, 8],
                          "vv": reyn[:, :, 9], "ww": reyn[:, :, 10],
                          "uv": reyn[:, :, 11], "pp": reyn[:, :, 12],
                          "TT": reyn[:, :, 13]}
        meta = {
            "regime": "compressible",
            "case": "sbli_interaction",
            "case_tag": "adiabatic",
            "source": "Pirozzoli and Bernardini 2011 (AIAA J 49(6), "
                      "doi:10.2514/1.J050901)",
            "mach": _MACH, "incidence_deg": 8.0, "re_inlet": 16750.0,
            "wall_thermal": "adiabatic",
            "averaging": "favre stress tiles; thermodynamics and cross-check "
                         "views from the reynolds tiles (documented mixed "
                         "construction)",
            "x_origin": f"half-rise of the smoothed wall pressure at raw "
                        f"x = {x_shock_raw:.3f}",
            "heat_flux_note": "no temperature-velocity covariance in this "
                              "dataset (adiabatic); dq lives in the "
                              "wall-thermal sweep",
            "sigma_note": "modeled observation uncertainty (no per-point "
                          "statistical uncertainty in the files)",
        }
        return SBLIInteractionDNS(
            case="adiabatic", x=x, y=y, U=favre[:, :, 2], V=favre[:, :, 3],
            R=R, series=series, meta=meta, rho=reyn[:, :, 2], p=reyn[:, :, 5],
            T=reyn[:, :, 6], q=None, reynolds=reynolds_views, blinc=blinc,
            budgets=budgets)

    @staticmethod
    def wall_thermal(s, root=None):
        """Load one wall-to-recovery-ratio case of the heated sweep."""
        if s not in SBLI_S_CASES:
            raise ValueError(f"unknown wall-thermal case '{s}'")
        base = SBLIInteractionDNS.thermal_dir(root)
        nx, ny, arr = _read_zone(_find(os.path.join(
            base, "favre_averages", f"favre_{s}.dat")))
        if arr.shape[2] != 13:
            raise ValueError(f"favre_{s}: {arr.shape[2]} columns, expected 13")
        x, y = _axes_from_grid(arr, f"favre_{s}")

        wall = np.loadtxt(_find(os.path.join(base, "wall_stat",
                                             f"wallstat_s{s}.dat")))
        if s == "1.0":
            if wall.shape[1] != 4:
                raise ValueError(f"wallstat_s1.0: {wall.shape[1]} columns, "
                                 "expected 4 (no Stanton, adiabatic)")
            St = None
        else:
            if wall.shape[1] != 5:
                raise ValueError(f"wallstat_s{s}: {wall.shape[1]} columns, "
                                 "expected 5")
            St = wall[:, 4]
        series = SBLIWallSeries(wall[:, 0], wall[:, 1], wall[:, 2],
                                wall[:, 3], St=St, pw_valid=(s != "1.0"))

        R = _assemble_stress(arr[:, :, 7], arr[:, :, 8], arr[:, :, 9],
                             arr[:, :, 10])
        q = np.zeros((nx, ny, 3))
        q[:, :, 0] = arr[:, :, 11]
        q[:, :, 1] = arr[:, :, 12]
        meta = {
            "regime": "compressible",
            "case": "sbli_interaction",
            "case_tag": f"s{s}",
            "source": "Bernardini, Asproulias, Larsson, Pirozzoli and Grasso "
                      "2016 (Phys. Rev. Fluids 1, 084403)",
            "mach": _MACH, "incidence_deg": 8.0,
            "wall_thermal": f"isothermal, Tw/Tr = {s}"
                            if s != "1.0" else "adiabatic (s = 1.0)",
            "averaging": "favre field (density-weighted second moments)",
            "x_origin": "nominal inviscid impingement point (the file's own)",
            "heat_flux_note": "u''T'' and v''T'' carried; the only "
                              "interaction dq source",
            "sigma_note": "modeled observation uncertainty (no per-point "
                          "statistical uncertainty in the files)",
        }
        if s == "1.0":
            meta["stanton_note"] = ("no Stanton column (adiabatic); its wall "
                                    "series' pw column is integer-quantized "
                                    "and drifts upstream (unusable as "
                                    "pressure, marked pw_valid False): Cp and "
                                    "the landmarks come from the field "
                                    "wall-pressure row and the adiabatic "
                                    "campaign, per the pre-registration "
                                    "addendum")
        return SBLIInteractionDNS(
            case=f"s{s}", x=x, y=y, U=arr[:, :, 3], V=arr[:, :, 4], R=R,
            series=series, meta=meta, rho=arr[:, :, 2], p=arr[:, :, 5],
            T=arr[:, :, 6], q=q)

    @staticmethod
    def load_all_thermal(root=None):
        return [SBLIInteractionDNS.wall_thermal(s, root) for s in SBLI_S_CASES
                if SBLIInteractionDNS.is_available(s, root)]

    def __repr__(self):
        return (f"SBLIInteractionDNS({self.case}, {self.nx} x {self.ny}, "
                f"x* in [{self.x[0]:.1f}, {self.x[-1]:.1f}])")


def _assemble_stress(uu, vv, ww, uv):
    """Pack (nx, ny) Favre components into (nx, ny, 3, 3); span off-diagonals
    vanish by spanwise homogeneity."""
    nx, ny = uu.shape
    R = np.zeros((nx, ny, 3, 3))
    R[:, :, 0, 0] = uu
    R[:, :, 1, 1] = vv
    R[:, :, 2, 2] = ww
    R[:, :, 0, 1] = R[:, :, 1, 0] = uv
    return R


def cross_campaign_wall_residual(adiabatic, s10, x_window=(-8.0, 6.0)):
    """The cross-campaign data-only anchor: the two databases' adiabatic wall
    quantities compared over a shared interaction window, aligned on their
    half-rise landmarks (the 2011 record is already re-originated on its
    series half-rise; the 2016 s = 1.0 case is shifted by its FIELD half-rise,
    its series pressure column being the documented unusable one).

    cf comes from both wall series (healthy on both); Cp for the 2016 side
    comes from its field wall-pressure row through the same upstream-plateau
    rule. Returns the median relative cf residual and the median absolute Cp
    residual over the window: a between-campaigns consistency measure, the
    modeled-sigma floor the pre-registration names.
    """
    sa, sb = adiabatic.series, s10.series
    shift_b = s10.shock_position()
    if shift_b is None:
        raise ValueError("s = 1.0 field shows no pressure rise")
    grid = np.linspace(x_window[0], x_window[1], 141)
    cf_a = np.interp(grid, sa.x, sa.cf)
    cf_b = np.interp(grid, sb.x - shift_b, sb.cf)
    x_cp_b, cp_b_row = s10.cp_from_field()
    cp_a = np.interp(grid, sa.x, sa.cp)
    cp_b = np.interp(grid, x_cp_b - shift_b, cp_b_row)
    rel_cf = np.abs(cf_a - cf_b) / np.maximum(np.abs(cf_a), 1e-12)
    return {"cf_median_rel": float(np.median(rel_cf)),
            "cp_median_abs": float(np.median(np.abs(cp_a - cp_b)))}
