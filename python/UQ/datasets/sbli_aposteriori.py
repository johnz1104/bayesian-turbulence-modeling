"""Coupled a-posteriori ensembles for the impinging-shock interaction.

The pre-registered coupled-prediction leg: sampled realizable closures
propagated through the shock-capturing solve to the wall quantities, per
propagated fold (the held-out wall-thermal conditions s = 0.5, 1.0, 1.9),
against the eigenspace corner family and the Gaussian conditional through
the identical injection, plus the preserve-attached control.

Construction, exactly as pinned:

- fold models: the leave-that-s-out conditional flow and Gaussian, refit
  in-process from the cached extractions with the pinned settings
  (deterministic seeds, so no model serialization crosses processes);
- members: coherent field realizations (one shared latent per member); the
  anisotropy leg samples the three live components (b11, b22, b12; the trace
  fixes b33 and span symmetry zeroes the rest), adds the BASELINE Boussinesq
  anisotropy at the solver cells and projects every draw into the barycentric
  realizable set before injection; the heat-flux leg samples the (dq_x, dq_y)
  correction raw (no admissibility set exists and none is invented);
- conditioning at the solver cells: features from the converged BASELINE
  fields (strain, rotation, timescale, M_t) at cell centers, the
  separated-study pattern; outside the training-like region (the mirror of
  the extraction's interior mask) the target reverts to the baseline
  anisotropy and zero heat-flux correction, so the injection acts only where
  the model was trained to speak;
- injection: set_target_correction(b_target, dq_target, energy_reach) with
  dq converted from the record's (u_tau_ref, T_w) units to the solver's
  dimensional Favre flux; every member solve initializes FROM THE CONVERGED
  BASELINE FIELDS (init_field), a perturbation solve, not a cold start;
- the eigenspace corner family (1C, 2C, 3C at Delta_B 1.0 and 0.5) through
  the same injection with zero heat-flux correction (the anisotropy-only
  structural limitation the pre-registration names); non-converged members
  are counted and excluded with the exclusion stated;
- wall quantities at the pinned stations (x* = -12..12 step 1 intersected
  with the series range) plus the scalar landmarks (separation,
  reattachment, wall-pressure half-rise), one rule for every solve.
"""
import numpy as np

from .. import discrepancy
from .. import evaluation
from .. import realizability
from ..eigenspace import EigenspacePerturbation
from ._common import _CMU
from .sbli_apriori import SBLIAPriori, EPOCHS
from .sbli_discrepancy import _gradients_tensor, turbulent_mach

STATIONS = np.arange(-12.0, 12.5, 1.0)
N_MEMBERS = 24
MEMBER_SEED = 0
CORNER_DELTAS = (1.0, 0.5)
MASK_Y_MAX = 2.0          # the extraction interior mask's outer edge
MASK_K_FLOOR = 1e-3       # k > floor * max(k), the extraction's quiet cut


def fold_models(study, held, history=False, seed=MEMBER_SEED,
                epochs=EPOCHS, kinds=("flow", "gauss")):
    """The leave-`held`-out conditional models, refit in-process on the
    identical loso pools. BOTH legs train on the HEATED records only (the
    pre-registered split: the adiabatic campaign is an independent test
    surface, never a training member). Returns kind -> (db_model, dq_model)."""
    heated = [c for c in study.test_sets
              if study.test_sets[c]["dq"] is not None]
    db_cases = heated
    dq_cases = heated
    out = {}
    for kind in kinds:
        fitted = {}
        db_raw = not study.db_gate()["all_pass"]
        for leg, cases in (("db", db_cases), ("dq_joint", dq_cases)):
            train = [c for c in cases if c != held]
            X = np.concatenate([SBLIAPriori._features(study.train_sets[c],
                                                      history)
                                for c in train])
            Y = np.concatenate([SBLIAPriori._target(
                study.train_sets[c], leg,
                db_raw=(leg == "db" and db_raw)) for c in train])
            model = SBLIAPriori._make(kind, X.shape[1], Y.shape[1], seed)
            model.fit(X, Y, epochs=epochs, lr=1e-3, batch=256)
            fitted[leg] = model
        out[kind] = (fitted["db"], fitted["dq_joint"])
    return out


def cell_conditioning(baseline, record, m_t_from_fields=False):
    """Features, baseline Boussinesq anisotropy and the injection mask at
    every solver cell, from the converged baseline fields (record
    free-stream units throughout, the extraction's own conventions).

    m_t_from_fields switches the M_t extra from the record's own DNS value
    (the training convention, right for the interaction folds) to the same
    formula on the baseline's own k and T; the attached control has no DNS
    record for its configuration, so it conditions on the solve itself."""
    u = baseline.units
    cc = baseline.mesh.cell_centers()
    x_star = cc[:, 0] / u.delta0 + baseline.meta["x_lo"]
    y_star = cc[:, 1] / u.delta0
    sample = baseline.sample_fields(x_star, y_star)

    grad_u = _gradients_tensor(sample)
    timescale = 1.0 / np.maximum(_CMU * sample["omega"], 1e-6)
    S, _ = discrepancy.strain_rotation(grad_u, timescale)
    # contract: the conditioning state must carry the solver's own eddy
    # viscosity; a solver that was init_field'ed but never swept reports
    # muT = 0 everywhere and would zero the baseline anisotropy under
    # every injection target (run the derived-fields probe first)
    assert float(np.max(sample["nu_t"])) > 0.0, \
        "conditioning baseline has zero eddy viscosity everywhere"
    b_base = discrepancy.boussinesq_anisotropy_actual(
        S, sample["nu_t"], np.maximum(sample["k"], 1e-12), timescale)
    if m_t_from_fields:
        # M_t = sqrt(2 k_hat) M_inf / sqrt(T_hat), the extraction's formula
        # applied to the baseline state
        m_t = (np.sqrt(2.0 * np.maximum(sample["k"], 0.0))
               * record.meta["mach"]
               / np.sqrt(np.maximum(sample["T"], 1e-6)))
    else:
        # DNS-derived M_t is an A-PRIORI extraction convention; a held-out
        # coupled prediction must not read the held-out record's fields, so
        # every predictive caller passes m_t_from_fields=True (enforced at
        # the driver call sites; this branch remains for diagnostics)
        m_t = turbulent_mach(record,
                             np.clip(x_star, record.x[0], record.x[-1]),
                             np.clip(y_star, record.y[0], record.y[-1]))
    features = discrepancy.feature_set(grad_u, timescale,
                                       extra=m_t[:, None])

    # the per-cell coefficient-to-component map of the objective basis (the
    # SAME normalized first-three construction the extraction trains against),
    # so sampled basis coefficients reconstruct to tensors at the solver cells
    S_full, W_full = discrepancy.strain_rotation(grad_u, timescale)
    T = discrepancy.integrity_basis(S_full, W_full)[:, :3]
    Tn = np.linalg.norm(T.reshape(T.shape[0], 3, 9), axis=2)
    T = T / np.maximum(Tn, 1e-300)[:, :, None, None]
    basis_M = np.stack([T[:, :, 0, 0], T[:, :, 1, 1], T[:, :, 0, 1]], axis=1)

    # the training-like region: inside the layer band the extraction used,
    # turbulent by its quiet cut, and within the record's streamwise span
    k = sample["k"]
    # the training-support floor matches the extraction interior mask
    # (y* >= 0.05): wall-adjacent cells below it never appeared in training
    # and receive no injection
    mask = ((y_star >= 0.05) & (y_star <= MASK_Y_MAX)
            & (k > MASK_K_FLOOR * float(np.max(k)))
            & (x_star >= record.x[0]) & (x_star <= record.x[-1]))
    return features, b_base, mask, basis_M, S_full


def _db3_to_tensor(draws):
    """(n, m, 3) free components -> (n, m, 3, 3) span-symmetric db."""
    n, m, _ = draws.shape
    db = np.zeros((n, m, 3, 3))
    db[:, :, 0, 0] = draws[:, :, 0]
    db[:, :, 1, 1] = draws[:, :, 1]
    db[:, :, 2, 2] = -(draws[:, :, 0] + draws[:, :, 1])
    db[:, :, 0, 1] = draws[:, :, 2]
    db[:, :, 1, 0] = draws[:, :, 2]
    return db


def member_targets(models, features, b_base, mask, basis_M,
                   n_members=N_MEMBERS, seed=MEMBER_SEED, db_raw=False):
    """Coherent member targets for one model kind: (b_targets, dq_targets)
    with b (m, n, 3, 3) realizability-projected and dq (m, n, 2) in the
    record's (u_tau_ref, T_w) units; masked cells revert to the baseline
    anisotropy and zero correction. The db model emits objective-basis
    coefficients (the amendment's representation); the per-cell exact linear
    map reconstructs the free components before the tensor assembly."""
    import torch
    db_model, dq_model = models
    torch.manual_seed(seed)
    g_draws = np.asarray(db_model.sample(features, n_per=n_members,
                                         shared_latent=True))
    torch.manual_seed(seed + 1)
    dq_draws = np.asarray(dq_model.sample(features, n_per=n_members,
                                          shared_latent=True))

    # under the registered feasibility reversion the model emits raw free
    # components directly and no basis map applies
    db_draws = (g_draws if db_raw
                else np.einsum("nij,nmj->nmi", basis_M, g_draws))
    db = _db3_to_tensor(db_draws)
    b_t = b_base[:, None, :, :] + db
    proj, _ = realizability.project_anisotropy(b_t.reshape(-1, 3, 3))
    b_t = proj.reshape(b_t.shape)
    b_t[~mask] = b_base[~mask][:, None, :, :]
    dq = dq_draws[:, :, 0:2].copy()
    dq[~mask] = 0.0
    return (b_t.transpose(1, 0, 2, 3).copy(),
            dq.transpose(1, 0, 2).copy())


def corner_targets(b_base, mask, deltas=CORNER_DELTAS):
    """The eigenspace corner family: label -> (n, 3, 3) target anisotropy,
    masked cells reverting to the baseline; the heat-flux correction is
    identically zero for every corner (the anisotropy-only structural
    limitation the pre-registration names).

    The barycentric perturbation is defined on the realizable triangle, so
    the raw Boussinesq base (which excursions outside it, sharply at the
    shock) is projected first; masked cells revert to the RAW base, whose
    injection cancels identically against the running Boussinesq stress."""
    b_real, _ = realizability.project_anisotropy(b_base)
    out = {}
    for delta_b in deltas:
        fam = EigenspacePerturbation.corner_set(b_real, delta_b=delta_b)
        for corner, b_t in fam.items():
            b = np.array(b_t)
            b[~mask] = b_base[~mask]
            out[f"{corner}_d{delta_b:g}"] = b
    return out


def five_state_targets(b_base, mask, strain, deltas=CORNER_DELTAS):
    """The documented 2017 five-state family as injection targets, exactly
    the corner_targets conventions (projected base, masked reversion) with
    the production-extremal eigenvector alignments built from the baseline
    strain. Labels: {corner}_vmax/_vmin and 3C, suffixed _d<delta>."""
    b_real, _ = realizability.project_anisotropy(b_base)
    out = {}
    for delta_b in deltas:
        fam = EigenspacePerturbation.five_state_set(b_real, strain,
                                                    delta_b=delta_b)
        for label, b_t in fam.items():
            b = np.array(b_t)
            b[~mask] = b_base[~mask]
            out[f"{label}_d{delta_b:g}"] = b
    return out


def dq_to_solver_units(dq, record, units):
    """(m, n, 2) correction in (u_tau_ref, T_w) units -> the solver's
    dimensional Favre flux <rho u_i'' T''>/<rho> in m/s K."""
    scale = (record.reference["u_tau_over_uinf"] * units.U_inf
             * record.reference["T_w"] * units.T_inf)
    return dq * scale


def landmarks_from_wall(w):
    """Separation, reattachment and the wall-pressure half-rise of one wall
    record, with the SAME smoothing and interpolated-crossing rule the DNS
    records' landmark methods use (one rule for both sides, as registered):
    the pressure series is moving-averaged at the records' width before the
    half-rise level crossing, and every crossing (pressure half-rise, Cf
    zero crossings) is linearly interpolated rather than read at a grid
    index."""
    from .sbli_interaction import _moving_average, _SMOOTH_WIDTH
    x = np.asarray(w["x_star"], float)
    cf = np.asarray(w["Cf"], float)
    cp = _moving_average(np.asarray(w["Cp"], float), x, _SMOOTH_WIDTH)

    neg = cf < 0.0
    x_s = x_r = None
    if neg.any():
        edges = np.diff(neg.astype(int))
        starts = list(np.where(edges == 1)[0] + 1)
        ends = list(np.where(edges == -1)[0] + 1)
        if neg[0]:
            starts.insert(0, 0)
        if neg[-1]:
            ends.append(neg.size)
        i0, i1 = max(zip(starts, ends), key=lambda se: se[1] - se[0])
        if i0 > 0:
            f = cf[i0 - 1] / (cf[i0 - 1] - cf[i0])
            x_s = float(x[i0 - 1] + f * (x[i0] - x[i0 - 1]))
        else:
            x_s = float(x[i0])
        i1 = min(i1, x.size - 1)
        if 0 < i1 < x.size and cf[i1 - 1] < 0.0 <= cf[i1]:
            f = -cf[i1 - 1] / (cf[i1] - cf[i1 - 1])
            x_r = float(x[i1 - 1] + f * (x[i1] - x[i1 - 1]))
        else:
            x_r = float(x[i1])
    plat_up = float(np.median(cp[x < x[0] + 3.0]))
    plat_dn = float(np.median(cp[x > x[-1] - 2.0]))
    level = 0.5 * (plat_up + plat_dn)
    above = cp >= level
    idx = int(np.argmax(above))
    if not above.any() or idx == 0:
        x_half = None
    else:
        x0, x1v = x[idx - 1], x[idx]
        v0, v1 = cp[idx - 1], cp[idx]
        x_half = float(x0 + (level - v0) * (x1v - x0) / max(v1 - v0, 1e-30))
    return {"x_s": x_s, "x_r": x_r, "shock": x_half}


def region_masks(record, xs):
    """The pre-registered region split at the pinned stations: upstream of
    the interaction onset (the 1.05 factor rule), the interaction core
    (onset to reattachment plus two, the shock landmark standing in when
    the series never separates), and downstream."""
    series = record.series
    onset = series.onset() if series.pw_valid else record.onset()
    sep = series.separation_reattachment()
    edge_ref = sep[1] if sep[1] is not None else record.shock_position()
    if onset is None:
        onset = -4.0 if edge_ref is None else edge_ref - 6.0
    edge = (edge_ref if edge_ref is not None else onset + 4.0) + 2.0
    return {"upstream": xs < onset,
            "interaction": (xs >= onset) & (xs <= edge),
            "downstream": xs > edge}, float(onset), float(edge)


def score_ensemble(members, record, stations=STATIONS, baseline_wall=None):
    """Station coverage (0.90 and 0.50 bands), band sharpness and median
    errors of a converged ensemble's wall quantities against the record's
    own wall series, pooled and per pre-registered region, the scalar
    landmark containment, and where the baseline wall record is supplied
    the accuracy and sharpness comparisons against the baseline's own
    point error."""
    series = record.series
    xs = stations[(stations >= series.x[0]) & (stations <= series.x[-1])]
    truths = {"Cf": np.interp(xs, series.x, series.cf)}
    if series.cp is not None:
        truths["Cp"] = np.interp(xs, series.x, series.cp)
    elif hasattr(record, "cp_from_field"):
        # the s = 1.0 series' pressure column is the documented quantized
        # quirk; the record provides the field-row Cp exactly for this case
        xf, cpf = record.cp_from_field()
        keep = (xs >= xf[0]) & (xs <= xf[-1])
        if keep.any():
            truths["Cp"] = np.interp(xs, xf, cpf)
    if series.St is not None and not np.all(np.isnan(series.St)):
        truths["St"] = np.interp(xs, series.x, series.St)

    out = {"n_members": len(members),
           "n_converged": sum(1 for m in members
                              if "Converged" in m["status"])}
    # the pre-registered 18-of-24 labeling rule scaled to the member count
    out["propagation_unstable"] = bool(
        len(members) > 0 and out["n_converged"] < 0.75 * len(members))
    conv = [m for m in members if "Converged" in m["status"]]
    if len(conv) < 2:
        out["scored"] = False
        return out
    out["scored"] = True
    out["stations"] = [float(v) for v in xs]
    regions, onset, edge = region_masks(record, xs)
    out["region_edges"] = {"onset": onset, "interaction_edge": edge}

    for q, truth in truths.items():
        ens = np.stack([np.interp(xs, m["wall"]["x_star"], m["wall"][q])
                        for m in conv])
        med = np.median(ens, axis=0)
        for level, (ql, qh) in (("0.9", (0.05, 0.95)),
                                ("0.5", (0.25, 0.75))):
            lo = np.quantile(ens, ql, axis=0)
            hi = np.quantile(ens, qh, axis=0)
            cover = (truth >= lo) & (truth <= hi)
            out[f"{q}_station_coverage_{level}"] = float(np.mean(cover))
            out[f"{q}_band_halfwidth_median_{level}"] = float(
                np.median(0.5 * (hi - lo)))
            for rname, rmask in regions.items():
                if rmask.any():
                    out[f"{q}_coverage_{level}_{rname}"] = float(
                        np.mean(cover[rmask]))
        out[f"{q}_median_abs_error"] = float(np.median(np.abs(med - truth)))
        # proper scores and calibration diagnostics at the pinned stations
        # (fair estimators: the members are samples of the predictive)
        out[f"{q}_crps_stations"] = float(
            evaluation.crps_ensemble(truth, ens.T))
        out[f"{q}_pit_randomized"] = [float(v) for v in
                                      evaluation.pit_values_randomized(
                                          truth, ens.T, seed=0)]
        for rname, rmask in regions.items():
            if rmask.any():
                out[f"{q}_median_abs_error_{rname}"] = float(
                    np.median(np.abs(med - truth)[rmask]))
        if baseline_wall is not None:
            base = np.interp(xs, baseline_wall["x_star"], baseline_wall[q])
            berr = np.abs(base - truth)
            out[f"{q}_baseline_abs_error"] = float(np.median(berr))
            for rname, rmask in regions.items():
                if rmask.any():
                    out[f"{q}_baseline_abs_error_{rname}"] = float(
                        np.median(berr[rmask]))
            # the sharpness guard: in the interaction region a covering
            # band must be narrower than the baseline's own point error
            rmask = regions["interaction"]
            if rmask.any():
                lo = np.quantile(ens, 0.05, axis=0)
                hi = np.quantile(ens, 0.95, axis=0)
                half = float(np.median((0.5 * (hi - lo))[rmask]))
                out[f"{q}_sharpness_guard_interaction"] = bool(
                    half < float(np.median(berr[rmask])))

    sep = series.separation_reattachment()
    dns = {"x_s": sep[0], "x_r": sep[1], "shock": record.shock_position()}
    for name, truth in dns.items():
        vals = [m["landmarks"][name] for m in conv
                if m["landmarks"][name] is not None]
        out[f"{name}_n_defined"] = len(vals)
        if truth is None or len(vals) < 2:
            continue
        vals = np.asarray(vals, dtype=float)
        lo, hi = np.quantile(vals, [0.05, 0.95])
        out[f"{name}_contained"] = bool(lo <= truth <= hi)
        out[f"{name}_band"] = [float(lo), float(hi)]
        out[f"{name}_truth"] = float(truth)
    return out
