"""Discrepancy extraction for the impinging-shock interaction records.

Builds the (features, db, dq) arrays the model-form studies consume, from an
interaction loader record and its converged density-based SST baseline
(UQ.datasets.sbli_baseline), with the committed conventions:

- db = b_DNS - b_baseline: the DNS Favre anisotropy (unit-free) against the
  limiter-consistent Boussinesq anisotropy of the baseline's ACTUAL eddy
  viscosity (UQ.discrepancy.boussinesq_anisotropy_actual, the separated-study
  convention, so the a-priori training target and the a-posteriori injection
  add db to the same baseline object).
- dq = q_hat - q_GDH in the record's upstream-referenced wall units
  ((u_tau_ref, T_w) at the amended x* = -7 station): the record's turbulent
  heat-flux vector against the gradient-diffusion flux formed from the
  BASELINE eddy viscosity and the RECORD's own mean temperature gradients at
  fixed Pr_t = 0.9, the committed attached-extraction convention.
- features: the five Galilean-invariant Pope invariants of the BASELINE
  strain and rotation with the baseline SST timescale tau = 1/(C_mu omega),
  extended with the turbulent Mach number from the record itself
  (UQ.discrepancy.feature_set with the M_t extra), six features per point.
- the optional strain-history feature (the scoped-in ablation): the upstream
  mean-streamline exponentially-weighted average of the shear invariant, with
  the baseline timescale as the relaxation scale, integrated backward along
  the baseline velocity field. A seventh conditioning feature, used only by
  the pre-registered a-priori ablation leg.

Interior mask and strides are the pre-registered ones (the loader's interior
mask; the study applies the pinned strides). The DNS stress realizability
fraction is asserted at extraction; the feature construction is Galilean
invariant by construction (velocity gradients and a timescale only).
"""
import numpy as np

from .. import discrepancy, realizability
from ._common import _CMU


def objective_basis(S, W, db_free):
    """The deployed objective-basis representation of the db targets: per
    sample, coefficients on the first three integrity-basis tensors,
    normalized to unit Frobenius norm so the map is well-conditioned across
    the shock. Returns (g, basis_M): basis_M (N, 3, 3) takes coefficients to
    the free components (b11, b22, b12); g is exact via the pseudo-inverse
    (on a 2-D mean flow the rank-3 basis spans the achievable subspace, the
    pre-registered feasibility gates measured machine-zero residuals) and
    degenerate quiet-flow samples degrade gracefully instead of raising.
    One construction for every training pool (interaction extractions and
    the attached far-transfer db pool consume this same function)."""
    T = discrepancy.integrity_basis(S, W)[:, :3]
    Tn = np.linalg.norm(T.reshape(T.shape[0], 3, 9), axis=2)
    T = T / np.maximum(Tn, 1e-300)[:, :, None, None]
    basis_M = np.stack([T[:, :, 0, 0], T[:, :, 1, 1], T[:, :, 0, 1]], axis=1)
    g = np.einsum("nij,nj->ni", np.linalg.pinv(basis_M), db_free)
    return g, basis_M


def _gradients_tensor(sample):
    """(N, 3, 3) velocity-gradient tensor from the sampled baseline entries
    (free-stream units; spanwise row and column zero by homogeneity)."""
    n = sample["dudx"].size
    g = np.zeros((n, 3, 3))
    g[:, 0, 0] = sample["dudx"]
    g[:, 0, 1] = sample["dudy"]
    g[:, 1, 0] = sample["dvdx"]
    g[:, 1, 1] = sample["dvdy"]
    return g


def turbulent_mach(record, pts_x, pts_y):
    """M_t = sqrt(2 k) / a from the record itself at the flattened points:
    with k in free-stream velocity units and a/U_inf = sqrt(T_hat)/M_inf,
    M_t = sqrt(2 k_hat) M_inf / sqrt(T_hat)."""
    from .sbli_baseline import _nearest_index
    ix = _nearest_index(record.x, np.asarray(pts_x, dtype=float))
    iy = _nearest_index(record.y, np.asarray(pts_y, dtype=float))
    k_hat = record.k[ix, iy]
    T_hat = np.maximum(record.T[ix, iy], 1e-6)
    m_inf = record.meta["mach"]
    return np.sqrt(2.0 * np.maximum(k_hat, 0.0)) * m_inf / np.sqrt(T_hat)


def streamline_history_feature(baseline, x_star, y_star, n_steps=100,
                               horizon_taus=5.0):
    """The scoped-in history feature: the exponentially-weighted upstream
    average of the shear invariant along the mean streamline.

    For each sample point, trace backward through the baseline velocity field
    (midpoint steps of the local relaxation time over horizon_taus relaxation
    times) and average the first Pope invariant with weight exp(-s/tau); the
    relaxation scale is the local baseline timescale tau = 1/(C_mu omega).
    Points whose trace exits the domain hold their last in-domain value (the
    upstream state continues the incoming layer). Returns (N,) values, one
    added conditioning feature.
    """
    u = baseline.units
    cc = baseline.mesh.cell_centers()
    xs = np.unique(np.round(cc[:, 0], 12))
    ys = np.unique(np.round(cc[:, 1], 12))
    order = np.lexsort((cc[:, 0], cc[:, 1]))
    f = baseline.solver.fields()

    def as_grid(vals):
        return np.asarray(vals)[order].reshape(ys.size, xs.size).T

    gu = as_grid(f["u"]) / u.U_inf
    gv = as_grid(f["v"]) / u.U_inf
    gom = as_grid(f["omega"]) * u.delta0 / u.U_inf
    tau_grid = 1.0 / np.maximum(_CMU * gom, 1e-6)

    # the shear invariant on the grid: S* tau contracted, the first Pope
    # invariant of the nondimensionalized strain
    dx = np.gradient(xs) / u.delta0
    dy = np.gradient(ys) / u.delta0
    dudx = np.gradient(gu, axis=0) / dx[:, None]
    dudy = np.gradient(gu, axis=1) / dy[None, :]
    dvdx = np.gradient(gv, axis=0) / dx[:, None]
    dvdy = np.gradient(gv, axis=1) / dy[None, :]
    Sxx = dudx
    Syy = dvdy
    Sxy = 0.5 * (dudy + dvdx)
    div3 = (Sxx + Syy) / 3.0
    s1_grid = ((Sxx - div3) ** 2 + (Syy - div3) ** 2 + div3 ** 2
               + 2.0 * Sxy ** 2) * tau_grid ** 2

    xs_hat = xs / u.delta0 + baseline.meta["x_lo"]
    ys_hat = ys / u.delta0

    def interp(grid, px, py):
        ix = np.clip(np.searchsorted(xs_hat, px), 1, xs_hat.size - 1)
        iy = np.clip(np.searchsorted(ys_hat, py), 1, ys_hat.size - 1)
        x0, x1 = xs_hat[ix - 1], xs_hat[ix]
        y0, y1 = ys_hat[iy - 1], ys_hat[iy]
        wx = np.clip((px - x0) / np.maximum(x1 - x0, 1e-12), 0.0, 1.0)
        wy = np.clip((py - y0) / np.maximum(y1 - y0, 1e-12), 0.0, 1.0)
        g00 = grid[ix - 1, iy - 1]
        g10 = grid[ix, iy - 1]
        g01 = grid[ix - 1, iy]
        g11 = grid[ix, iy]
        return ((1 - wx) * (1 - wy) * g00 + wx * (1 - wy) * g10
                + (1 - wx) * wy * g01 + wx * wy * g11)

    px = np.asarray(x_star, dtype=float).copy()
    py = np.asarray(y_star, dtype=float).copy()
    tau0 = np.maximum(interp(tau_grid, px, py), 1e-6)
    dt = horizon_taus * tau0 / n_steps

    weight_sum = np.zeros_like(px)
    value_sum = np.zeros_like(px)
    s_elapsed = np.zeros_like(px)
    for _ in range(n_steps):
        s1 = interp(s1_grid, px, py)
        w = np.exp(-s_elapsed / tau0) * dt / tau0
        value_sum += w * s1
        weight_sum += w
        # midpoint backward step along the local velocity
        uu = interp(gu, px, py)
        vv = interp(gv, px, py)
        xm = px - 0.5 * dt * uu
        ym = py - 0.5 * dt * vv
        um = interp(gu, xm, ym)
        vm = interp(gv, xm, ym)
        px = np.clip(px - dt * um, xs_hat[0], xs_hat[-1])
        py = np.clip(py - dt * vm, ys_hat[0], ys_hat[-1])
        s_elapsed += dt
    return value_sum / np.maximum(weight_sum, 1e-12)


def interaction_study(record, baseline, stride=(8, 4), history=False):
    """The (features, db, dq) extraction at the record's strided interior
    points, everything in the committed conventions.

    Returns a dict: features (N, 6 or 7), db (N, 3, 3), db_free (N, 3) the
    independent components (b11, b22, b12; b33 follows from the trace and the
    span off-diagonals vanish), dq (N, 3) in (u_tau_ref, T_w) units or None
    for the adiabatic record, the point coordinates, region labels, masks and
    the realizability fraction of the DNS record.
    """
    m2 = record.interior_mask()
    sx, sy = stride
    strided = np.zeros_like(m2)
    strided[::sx, ::sy] = True
    sel = m2 & strided
    ii, jj = np.nonzero(sel)
    pts_x = record.x[ii]
    pts_y = record.y[jj]

    sample = baseline.sample_fields(pts_x, pts_y)

    grad_u = _gradients_tensor(sample)
    timescale = 1.0 / np.maximum(_CMU * sample["omega"], 1e-6)
    S, W = discrepancy.strain_rotation(grad_u, timescale)

    b_dns = realizability.anisotropy(record.R[ii, jj])
    b_base = discrepancy.boussinesq_anisotropy_actual(
        S, sample["nu_t"], np.maximum(sample["k"], 1e-12), timescale)
    db = b_dns - b_base

    m_t = turbulent_mach(record, pts_x, pts_y)
    extras = [m_t[:, None]]
    if history:
        h = streamline_history_feature(baseline, pts_x, pts_y)
        extras.append(h[:, None])
    features = discrepancy.feature_set(grad_u, timescale,
                                       extra=np.concatenate(extras, axis=1))

    dq = None
    if record.q is not None:
        # record heat flux in (u_tau_ref, T_w) units; the GDH flux from the
        # baseline eddy viscosity and the RECORD's own temperature gradients,
        # converted with the same reference scales
        q_hat = record.q_hat_wall_units()[ii, jj]
        dTdx = np.gradient(record.T, record.x, axis=0)[ii, jj]
        dTdy = np.gradient(record.T, record.y, axis=1)[ii, jj]
        grad_T = np.zeros((ii.size, 3))
        grad_T[:, 0] = dTdx
        grad_T[:, 1] = dTdy
        u_tau_hat = record.reference["u_tau_over_uinf"]
        T_w_hat = record.reference["T_w"]
        q_gdh_fs = discrepancy.gdh_heat_flux(grad_T, sample["nu_t"], Pr_t=0.9)
        q_gdh = q_gdh_fs / (u_tau_hat * T_w_hat)
        dq = q_hat - q_gdh

    # region labels from the record's own landmarks (the pinned rules)
    onset = record.onset()
    x_s, x_r = record.series.separation_reattachment()
    switch = record.thermal_switch_position()
    region = np.full(ii.size, "interaction", dtype=object)
    upstream_edge = onset if onset is not None else record.x[0]
    region[pts_x < upstream_edge] = "upstream"
    if switch is not None:
        region[pts_x < switch] = "pre_switch"
    if x_r is not None:
        region[pts_x > x_r + 2.0] = "relaxation"

    frac = record.realizable_fraction()

    # objective-basis representation of the db targets (the amendment's db
    # leg), through the shared deployed construction
    db_free = np.stack([db[:, 0, 0], db[:, 1, 1], db[:, 0, 1]], axis=1)
    g_basis, basis_M = objective_basis(S, W, db_free)

    return {
        "features": features,
        "db": db,
        "db_free": db_free,
        "g_basis": g_basis,
        "basis_M": basis_M,
        "dq": dq,
        "x": pts_x, "y": pts_y,
        "region": region,
        "realizable_fraction": frac,
        "meta": {"case": record.case, "stride": list(stride),
                 "n_points": int(ii.size), "history": bool(history),
                 "onset": onset, "x_s": x_s, "x_r": x_r, "switch": switch},
    }
