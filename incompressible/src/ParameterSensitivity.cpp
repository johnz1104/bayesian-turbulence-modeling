#include "ParameterSensitivity.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

// ADJOINT GROUNDWORK — see ParameterSensitivity.hpp.  Differentiates ONLY w.r.t. the 11
// SST coefficients at a FIXED primary state; never forms ∂R/∂U (the held adjoint core).

ParameterSensitivity::ParameterSensitivity(const Mesh& mesh,
                                           const ObservationOperator& obs,
                                           const FlowBoundaryConditions& bcs,
                                           double nu,
                                           const SolverSettings& settings,
                                           const Vec3& Uinit,
                                           double pInit, double kInit, double omegaInit)
    : mesh_(mesh), obs_(obs), bcs_(bcs), nu_(nu), settings_(settings),
      Uinit_(Uinit), pInit_(pInit), kInit_(kInit), omegaInit_(omegaInit),
      state_(mesh) {}

SSTModel ParameterSensitivity::makeModel(const std::vector<double>& theta11) const {
    SSTModel sst(InferenceParameterSet::fromVector(theta11));
    sst.variant = static_cast<SSTVariant>(settings_.sstVariant);
    return sst;
}

EvaluationStatus ParameterSensitivity::solveState(const std::vector<double>& theta11) {
    SSTModel sst = makeModel(theta11);
    SIMPLESolver solver(mesh_, sst, bcs_, nu_, settings_);
    FlowFields f(mesh_);
    solver.initUniform(f, Uinit_, pInit_, kInit_, omegaInit_);
    ConvergenceHistory hist = solver.solve(f);
    state_ = f;
    hasState_ = true;
    if (hist.diverged)  return EvaluationStatus::Diverged;
    if (hist.converged) return EvaluationStatus::Converged;
    return EvaluationStatus::Unconverged;
}

std::vector<double> ParameterSensitivity::residual(const std::vector<double>& theta11) {
    SSTModel sst = makeModel(theta11);
    SIMPLESolver solver(mesh_, sst, bcs_, nu_, settings_);
    return solver.assembleResidual(state_, sst.coeffs);
}

std::vector<std::vector<double>>
ParameterSensitivity::dResidualDTheta(const std::vector<double>& theta11) {
    SSTModel sst = makeModel(theta11);
    SIMPLESolver solver(mesh_, sst, bcs_, nu_, settings_);
    return solver.assembleResidualSensitivity(state_, sst.coeffs);
}

void ParameterSensitivity::recompute(const std::vector<double>& theta11, FlowFields& work,
                                     std::vector<SSTClosureSensitivity>* cs) const {
    SSTModel sst = makeModel(theta11);
    work = state_;
    sst.computeFields(mesh_, work.k, work.omega, work.U, nu_,
                      work.nuT, work.F1, work.F2, work.Pk, work.CDkw);
    const double floor = 0.1 * nu_;
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        work.nuT[ci] = std::max(work.nuT[ci], floor);
    if (cs) {
        ScalarField Smag = strainRateMagnitude(computeVelocityGradients(work.U));
        const auto& wd = mesh_.wallDistance();
        cs->resize(mesh_.nCells());
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            (*cs)[ci] = sst.closureSensitivity(work.k[ci], work.omega[ci], Smag[ci],
                                               wd[ci], nu_, work.CDkw[ci], floor);
    }
}

std::vector<double> ParameterSensitivity::observe(const std::vector<double>& theta11) {
    FlowFields work(mesh_);
    recompute(theta11, work, nullptr);
    return obs_.evaluate(mesh_, work, nu_);
}

double ParameterSensitivity::logLik(const std::vector<double>& theta11) {
    FlowFields work(mesh_);
    recompute(theta11, work, nullptr);
    return obs_.logLikelihood(mesh_, work, nu_);
}

int ParameterSensitivity::nearestWallFace(const std::string& patch, const Vec3& loc) const {
    PatchID pid = mesh_.patchByName(patch);
    const Patch& pat = mesh_.patch(pid);
    int best = pat.faces.empty() ? -1 : pat.faces[0];
    double bestDist = 1e30;
    for (FaceID fi : pat.faces) {
        double d = (mesh_.face(fi).center - loc).norm2();
        if (d < bestDist) { bestDist = d; best = fi; }
    }
    return best;
}

// ∂g/∂θ — the Observation operator reads only the STORED state (U, p) and the eddy
// viscosity nuT.  Holding the primary state fixed, the only θ-dependence is through nuT,
// so ∂g/∂θ = (∂g/∂nuT)·(∂nuT/∂θ).  Velocity/pressure QoIs do not read nuT ⇒ exactly 0.
// nuT depends on θ only via a1 and β* (and only where the Bradshaw limiter is active);
// near walls ω is large so a1·ω ≥ S·F2 (limiter inactive, nuT = k/ω), making even those
// columns vanish — hence the documented (near-)zero structure.  No field derivative is
// taken: ∂g/∂nuT is the explicit closure-coefficient and ∂nuT/∂θ is the pointwise block.
std::vector<std::vector<double>>
ParameterSensitivity::dObsDTheta(const std::vector<double>& theta11) {
    FlowFields work(mesh_);
    std::vector<SSTClosureSensitivity> cs;
    recompute(theta11, work, &cs);

    const int nObs = obs_.nObs();
    std::vector<std::vector<double>> dG(nObs, std::vector<double>(11, 0.0));
    const auto& observ = obs_.observables();

    for (int i = 0; i < nObs; ++i) {
        const Observable& ob = observ[i];
        std::vector<double>& row = dG[i];

        switch (ob.type) {
        case ObsType::SkinFriction: {
            int fi = nearestWallFace(ob.patchName, ob.location);
            if (fi < 0) break;
            const Face& face = mesh_.face(fi);
            int ow = face.owner;
            double delta = std::max(face.delta, 1e-20);
            Vec3 Uc = work.U[ow];
            Vec3 Ut = Uc - face.normal * Uc.dot(face.normal);
            double dynP = std::max(0.5 * ob.refVelocity * ob.refVelocity, 1e-20);
            double dCf_dnuT = Ut.norm() / delta / dynP;            // Cf = (ν+nuT)·|Ut|/δ/dynP
            for (int j = 0; j < 11; ++j) row[j] = dCf_dnuT * cs[ow].dnuT[j];
            break;
        }
        case ObsType::Drag: {
            PatchID pid = mesh_.patchByName(ob.patchName);
            const Patch& pat = mesh_.patch(pid);
            double dynP = std::max(0.5 * ob.refVelocity * ob.refVelocity, 1e-20);
            for (FaceID fi : pat.faces) {
                const Face& face = mesh_.face(fi);
                int ow = face.owner;
                double Sf = face.area, delta = std::max(face.delta, 1e-20);
                double Uw_x = work.U.bface(fi).x;
                double dCd_dnuT = (work.U[ow].x - Uw_x) / delta * Sf / (dynP * ob.referenceArea);
                for (int j = 0; j < 11; ++j) row[j] += dCd_dnuT * cs[ow].dnuT[j];
            }
            break;
        }
        case ObsType::ReattachmentLength:
        case ObsType::SeparationPoint: {
            // Sub-cell interpolated zero-crossing of tau_x = (ν+nuT)·Ut.x/δ.  The crossing
            // face PAIR is fixed by sign(Ut.x) (θ-independent, U fixed); only the
            // interpolation weight moves with nuT.  Both QoIs share xc = prevX + Δx·A/(A∓B).
            PatchID pid = mesh_.patchByName(ob.patchName);
            const Patch& pat = mesh_.patch(pid);
            bool   sep = (ob.type == ObsType::SeparationPoint);
            bool   havePrev = false;
            double prevTau = sep ? 1.0 : -1.0, prevX = 0.0, prevP = 0.0;
            int    prevOw = -1;
            for (FaceID fi : pat.faces) {
                const Face& face = mesh_.face(fi);
                int ow = face.owner;
                double nuEff = nu_ + work.nuT[ow];
                double delta = std::max(face.delta, 1e-20);
                Vec3 Uc = work.U[ow];
                Vec3 Ut = Uc - face.normal * Uc.dot(face.normal);
                double Pcur = Ut.x / delta;            // ∂tau/∂nuT at this face
                double tau  = nuEff * Pcur;
                double x    = face.center.x;
                bool cross = sep ? (havePrev && prevTau > 0 && tau <= 0)
                                 : (havePrev && prevTau < 0 && tau >= 0);
                if (cross) {
                    double A = prevTau, B = tau, dx = x - prevX;
                    double den = sep ? (A - B) : (B - A);   // > 0 at a valid crossing
                    if (std::abs(den) > 1e-30) {
                        double den2 = den * den;
                        // sep: xc = prevX + dx·A/(A-B);  reatt: xc = prevX + dx·(−A)/(B−A)
                        // ∂xc/∂A and ∂xc/∂B (see DECISION_RECORD): both reduce to ∓B,±A / den²
                        double dxc_dA = dx * (-B) / den2;
                        double dxc_dB = dx * ( A) / den2;
                        for (int j = 0; j < 11; ++j)
                            row[j] = dxc_dA * prevP * cs[prevOw].dnuT[j]
                                   + dxc_dB * Pcur  * cs[ow].dnuT[j];
                    }
                    break;
                }
                prevTau = tau; prevX = x; prevP = Pcur; prevOw = ow; havePrev = true;
            }
            break;
        }
        case ObsType::VelocityProfile:
        case ObsType::PressureTap:
            // read only U / p (no closure) ⇒ ∂g/∂θ ≡ 0
            break;
        }
    }
    return dG;
}

// ---- RUNG 1 — semi-analytic true-model gradient (NON-HELD) -------------------------------

std::vector<int> ParameterSensitivity::cellColoring(int& nColors) const {
    const int nc  = mesh_.nCells();
    const int nIF = mesh_.nInternalFaces();
    std::vector<std::vector<int>> adj(nc);
    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        adj[face.owner].push_back(face.neighbor);
        adj[face.neighbor].push_back(face.owner);
    }
    std::vector<int> color(nc, -1);
    nColors = 0;
    std::vector<char> used;
    for (int i = 0; i < nc; ++i) {
        used.assign(nColors, 0);
        for (int j : adj[i])
            if (color[j] >= 0) used[color[j]] = 1;
        int c = 0;
        while (c < (int)used.size() && used[c]) ++c;
        color[i] = c;
        if (c + 1 > nColors) nColors = c + 1;
    }
    return color;
}

// Full true-model dη/dθ at the FIXED converged state via the matrix-free frozen-pressure
// tangent solve.  For each coefficient j we solve (∂R/∂U) w_j = −∂R/∂θ_j with a matrix-free,
// column-scaled, Jacobi-(left)-preconditioned BiCGSTAB — the operator is touched ONLY through
// the directional residual difference  Jv ≈ [R(U*+εv) − R(U*−εv)]/(2ε)  (full turbulence
// coupling; the held analytic (∂R/∂U)ᵀ core is never formed).  ∂η_i/∂θ_j is then the
// directional derivative of observable i along w_j, which folds in ∂g/∂U exactly (velocity
// AND through-nuT) — ∂g/∂θ being ≡0.  Pressure is frozen (no continuity row), matching the
// program's 4-block residual definition; any frozen-pressure bias is characterised vs full FD.
TangentGradientResult ParameterSensitivity::etaJacobianTangent(
        const std::vector<double>& theta11, double krylovTol, int maxIter, double fdStep) {
    const int nc = mesh_.nCells();
    const int N  = 4 * nc;
    const int nObsv = obs_.nObs();
    const int BUX = 0, BUY = nc, BK = 2 * nc, BOM = 3 * nc;

    TangentGradientResult out;
    out.dObsDTheta.assign(nObsv, std::vector<double>(11, 0.0));
    out.krylovIters.assign(11, 0);
    out.krylovRelRes.assign(11, 0.0);
    out.krylovConverged.assign(11, 0);
    if (!hasState_ || nc == 0) return out;

    SSTModel sst = makeModel(theta11);
    SIMPLESolver solver(mesh_, sst, bcs_, nu_, settings_);

    // RHS_j = −∂R/∂θ_j of the PICARD-REDUCED model: the transpose-stress
    // deferred correction is lagged in this rung in BOTH U (operator switch
    // below) and theta (includeTransposeTheta = false), exactly like the
    // pressure this rung freezes, so the reduced tangent differentiates a
    // self-consistent model. The full transpose coupling belongs to the
    // coupled tangent and warm-FD, where pressure removes the rigid-rotation
    // near-null mode the completed stress operator otherwise carries; the
    // residual bias of this reduction is characterised against warm-FD by the
    // direction tests, as the frozen-pressure bias always was.
    std::vector<std::vector<double>> dRdTheta =
        solver.assembleResidualSensitivity(state_, sst.coeffs,
                                           /*includeTransposeTheta=*/false);
    solver.setFreezeTransposeStress(true);

    // ---- per-DOF column scaling σ_i = max(|x_i|, floor_block) ---------------------------
    // The tangent unknown [Ux|Uy|k|ω] spans orders of magnitude (ω∼1e6 near walls, ∼1 in
    // the freestream).  Per-DOF |x_i| scaling makes every FD probe a uniform RELATIVE
    // perturbation, so the column-scaled operator Ã = J·diag(σ) is well posed.
    double rmsUx = 0.0;
    for (int ci = 0; ci < nc; ++ci) rmsUx += state_.U[ci].x * state_.U[ci].x;
    rmsUx = std::sqrt(rmsUx / std::max(nc, 1));
    const double floorU = std::max(1e-4 * rmsUx, 1e-12);
    const double floorK = std::max(settings_.kMin, 1e-12);
    const double floorW = std::max(settings_.omegaMin, 1e-12);
    std::vector<double> sigma(N);
    for (int ci = 0; ci < nc; ++ci) {
        sigma[BUX + ci] = std::max(std::abs(state_.U[ci].x),   floorU);
        sigma[BUY + ci] = std::max(std::abs(state_.U[ci].y),   floorU);
        sigma[BK  + ci] = std::max(std::abs(state_.k[ci]),     floorK);
        sigma[BOM + ci] = std::max(std::abs(state_.omega[ci]), floorW);
    }

    // residual at a state delta (assembleResidual copies the state and recomputes closure,
    // so the perturbation propagates through nuT/F1/F2/Pk/CDkw and Smag — full coupling).
    auto residualAt = [&](const std::vector<double>& d) -> std::vector<double> {
        FlowFields pert = state_;
        for (int ci = 0; ci < nc; ++ci) {
            pert.U[ci].x   += d[BUX + ci];
            pert.U[ci].y   += d[BUY + ci];
            pert.k[ci]     += d[BK  + ci];
            pert.omega[ci] += d[BOM + ci];
        }
        return solver.assembleResidual(pert, sst.coeffs);
    };

    // observable vector at a state delta (recompute closure exactly as observe()).
    auto observeAtDelta = [&](const std::vector<double>& d) -> std::vector<double> {
        FlowFields work = state_;
        for (int ci = 0; ci < nc; ++ci) {
            work.U[ci].x   += d[BUX + ci];
            work.U[ci].y   += d[BUY + ci];
            work.k[ci]     += d[BK  + ci];
            work.omega[ci] += d[BOM + ci];
        }
        sst.computeFields(mesh_, work.k, work.omega, work.U, nu_,
                          work.nuT, work.F1, work.F2, work.Pk, work.CDkw);
        const double floor = 0.1 * nu_;
        for (int ci = 0; ci < nc; ++ci) work.nuT[ci] = std::max(work.nuT[ci], floor);
        return obs_.evaluate(mesh_, work, nu_);
    };

    // ---- colored-FD diagonal of Ã = J·diag(σ)  →  Jacobi (left) preconditioner ----------
    int nColors = 0;
    std::vector<int> color = cellColoring(nColors);
    out.nColors = nColors;
    std::vector<double> diagA(N, 0.0);
    std::vector<double> probe(N, 0.0);
    const double inv2s = 1.0 / (2.0 * fdStep);
    for (int f = 0; f < 4; ++f) {
        const int B = f * nc;
        for (int c = 0; c < nColors; ++c) {
            std::fill(probe.begin(), probe.end(), 0.0);
            for (int ci = 0; ci < nc; ++ci)
                if (color[ci] == c) probe[B + ci] = fdStep * sigma[B + ci];
            std::vector<double> Rp = residualAt(probe);
            for (int ci = 0; ci < nc; ++ci)
                if (color[ci] == c) probe[B + ci] = -fdStep * sigma[B + ci];
            std::vector<double> Rm = residualAt(probe);
            out.nResidualEvals += 2;
            for (int ci = 0; ci < nc; ++ci)
                if (color[ci] == c) diagA[B + ci] = (Rp[B + ci] - Rm[B + ci]) * inv2s;
        }
    }
    std::vector<double> invDiag(N);
    for (int i = 0; i < N; ++i)
        invDiag[i] = (std::abs(diagA[i]) > 1e-300) ? 1.0 / diagA[i] : 1.0;

    // ---- left-preconditioned scaled operator  B ṽ = invDiag ∘ (J·(σ∘ṽ)) ----------------
    // Baking invDiag into the operator (not as a separate right-preconditioner) makes the
    // BiCGSTAB stopping norm the *balanced* residual: diag(B)=1, so the ω-block (residual
    // O(1e6)) cannot dominate ‖r‖ and starve the velocity tangent the QoIs depend on.
    std::vector<double> dplus(N), dminus(N);
    auto matvecB = [&](const std::vector<double>& vt, std::vector<double>& yv) {
        double vn = linalg::norm(vt);
        if (vn < 1e-300) { std::fill(yv.begin(), yv.end(), 0.0); return; }
        double eps = fdStep / vn;
        for (int i = 0; i < N; ++i) { double s = eps * sigma[i] * vt[i]; dplus[i] = s; dminus[i] = -s; }
        std::vector<double> Rp = residualAt(dplus);
        std::vector<double> Rm = residualAt(dminus);
        out.nResidualEvals += 2;
        double inv2e = 1.0 / (2.0 * eps);
        for (int i = 0; i < N; ++i) yv[i] = invDiag[i] * (Rp[i] - Rm[i]) * inv2e;
    };

    std::vector<double> ones(N, 1.0), wt(N), w(N);
    for (int j = 0; j < 11; ++j) {
        double bn = 0.0;
        for (int i = 0; i < N; ++i) bn += dRdTheta[j][i] * dRdTheta[j][i];
        if (std::sqrt(bn) < 1e-300) { out.krylovConverged[j] = 1; continue; }  // e.g. κ (∂R/∂κ≡0)

        std::vector<double> b(N);
        for (int i = 0; i < N; ++i) b[i] = -invDiag[i] * dRdTheta[j][i];   // invDiag ∘ (−∂R/∂θ_j)

        MatrixFreeResult kr = bicgstabMatrixFree(matvecB, b, wt, ones, maxIter, krylovTol);
        out.krylovIters[j]     = kr.iterations;
        out.krylovRelRes[j]    = kr.finalRes;
        out.krylovConverged[j] = kr.converged ? 1 : 0;
        for (int i = 0; i < N; ++i) w[i] = sigma[i] * wt[i];   // unscale: w = dU/dθ_j

        // directional observable derivative ∂η_i/∂θ_j = [g_i(x+h w) − g_i(x−h w)]/(2h),
        // h chosen so the largest RELATIVE state perturbation equals fdStep.
        double scale = 0.0;
        for (int i = 0; i < N; ++i) scale = std::max(scale, std::abs(w[i]) / sigma[i]);
        if (scale < 1e-300) continue;                       // w ≈ 0 → zero gradient column
        double hg = fdStep / scale;
        for (int i = 0; i < N; ++i) { dplus[i] = hg * w[i]; dminus[i] = -hg * w[i]; }
        std::vector<double> gP = observeAtDelta(dplus);
        std::vector<double> gM = observeAtDelta(dminus);
        double inv2h = 1.0 / (2.0 * hg);
        for (int i = 0; i < nObsv; ++i) out.dObsDTheta[i][j] = (gP[i] - gM[i]) * inv2h;
    }
    return out;
}

// ---- RUNG 1 (PRESSURE-COUPLED) — continuity, pressure-Poisson, and the full tangent -----

std::vector<double> ParameterSensitivity::massFluxDivergence(
        const std::vector<double>& ux, const std::vector<double>& uy, bool homogeneous) const {
    const int nc  = mesh_.nCells();
    const int nIF = mesh_.nInternalFaces();
    std::vector<double> div(nc, 0.0);
    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        const int o = face.owner, n = face.neighbor;
        const double w = face.weight;
        double Ufx = w * ux[o] + (1.0 - w) * ux[n];
        double Ufy = w * uy[o] + (1.0 - w) * uy[n];
        double mf = (Ufx * face.normal.x + Ufy * face.normal.y) * face.area;
        div[o] += mf; div[n] -= mf;
    }
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch& pat = mesh_.patch(pi);
        for (FaceID fi : pat.faces) {
            const Face& face = mesh_.face(fi);
            const int o = face.owner;
            double Ubx, Uby;
            if (pat.type == "outlet") {            // zero-gradient: responds to interior (both modes)
                Ubx = ux[o]; Uby = uy[o];
            } else if (pat.type == "wall") {       // no-slip: zero flux (both modes)
                Ubx = 0.0; Uby = 0.0;
            } else {                               // inlet/other: fixed value (residual) or 0 (increment)
                if (homogeneous) { Ubx = 0.0; Uby = 0.0; }
                else { Vec3 ub = state_.U.bface(fi); Ubx = ub.x; Uby = ub.y; }
            }
            div[o] += (Ubx * face.normal.x + Uby * face.normal.y) * face.area;
        }
    }
    return div;
}

LinearSystem ParameterSensitivity::assemblePressurePoisson(const std::vector<double>& aP) const {
    LinearSystem sys = makeSystem(mesh_);
    const int nIF = mesh_.nInternalFaces();
    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        const int o = face.owner, n = face.neighbor;
        double Sf = face.area, delta = std::max(face.delta, 1e-20);
        double Vo = mesh_.cell(o).volume, Vn = mesh_.cell(n).volume;
        double dPo = (std::abs(aP[o]) > 1e-30) ? Vo / aP[o] : 0.0;
        double dPn = (std::abs(aP[n]) > 1e-30) ? Vn / aP[n] : 0.0;
        double dPf = face.weight * dPo + (1.0 - face.weight) * dPn;
        double coeff = dPf * Sf / delta;
        sys.diag[o] += coeff; sys.diag[n] += coeff;
        sys.upper[fi] = -coeff; sys.lower[fi] = -coeff;
    }
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {            // outlet Dirichlet p'=0
        const Patch& pat = mesh_.patch(pi);
        if (pat.type != "outlet") continue;
        for (FaceID fi : pat.faces) {
            const Face& face = mesh_.face(fi);
            const int o = face.owner;
            double Sf = face.area, delta = std::max(face.delta, 1e-20);
            double dPo = (std::abs(aP[o]) > 1e-30) ? mesh_.cell(o).volume / aP[o] : 0.0;
            sys.diag[o] += dPo * Sf / delta;
        }
    }
    return sys;
}

// Pressure-coupled (full) semi-analytic tangent: the exact matrix-free augmented 5-block
// [Ux|Uy|k|ω|p] saddle, ROBUSTLY converged.  dη/dθ matches full FD (warm-FD) to ~1-2% on every
// significant coefficient.  Five ingredients made it converge where the naive saddle solves
// (BiCGSTAB diverged, GMRES stalled, segregated tangent-SIMPLE limit-cycled) all failed:
//   1. block-preconditioned FGMRES — a PHYSICS-BASED SIMPLE preconditioner (full solves of the
//      assembled momentum/k/ω operators + pressure-Poisson Schur) captures the stiff ω/k coupling
//      a Jacobi diagonal misses; right-preconditioned FGMRES is monotone on the indefinite saddle.
//   2. wall-ω DIRICHLET rows — the solver PINS near-wall ω = 60ν/(β1·Δy²) post-solve, so q* is a
//      root of the PINNED residual; those rows are overridden to R_ω = ω − ω_BC (identity row,
//      ∂R/∂β1 = ω_BC/β1) in the operator, RHS, and preconditioner block.
//   3. BC re-application — the perturbed state re-applies velocity/pressure BCs so the outlet
//      velocity (continuity) and wall-Neumann pressure gradient (momentum↔pressure coupling the
//      QoIs feel) RESPOND; assembleResidual otherwise freezes them, suppressing the gradient.
//   4. true-residual FGMRES reporting (the flexible-preconditioner |g| estimate is unreliable).
//   5. wall-ω row scaling (ρ=1/σ_ω) — keeps the O(1e7) ∂ω_BC/∂β1 RHS from dominating the residual
//      norm and starving the velocity tangent (this both fixes β1 and makes all columns converge).
// The preconditioner blocks are −∂R_block/∂block approximations (no cross-block coupling, no
// transpose) — the held analytic (∂R/∂U)ᵀ core stays UNTOUCHED.  Cost: O(d_θ) FGMRES solves, each
// with inner block solves — slower than warm-FD, so warm-FD remains the production default; this
// is the clean (no FD-convergence-noise) semi-analytic alternative.  See DECISION_RECORD §4b.
TangentGradientResult ParameterSensitivity::etaJacobianTangentCoupled(
        const std::vector<double>& theta11, double krylovTol, int maxIter, double fdStep) {
    const int nc = mesh_.nCells();
    const int N4 = 4 * nc, N5 = 5 * nc;
    const int nObsv = obs_.nObs();
    const int BUX = 0, BUY = nc, BK = 2 * nc, BOM = 3 * nc, BP = 4 * nc;

    TangentGradientResult out;
    out.dObsDTheta.assign(nObsv, std::vector<double>(11, 0.0));
    out.krylovIters.assign(11, 0);
    out.krylovRelRes.assign(11, 0.0);
    out.krylovConverged.assign(11, 0);
    if (!hasState_ || nc == 0) return out;

    SSTModel sst = makeModel(theta11);
    SIMPLESolver solver(mesh_, sst, bcs_, nu_, settings_);
    std::vector<std::vector<double>> dRdTheta =
        solver.assembleResidualSensitivity(state_, sst.coeffs);   // 11 × N4

    // ---- wall-ω DIRICHLET rows ----------------------------------------------------------
    // The solver PINS near-wall ω = 60ν/(β1·Δy²) by override AFTER each solve, so the converged
    // state is a root of the PINNED ω equation, NOT the PDE ω residual assembleResidual returns.
    // Solving the tangent on the PDE rows there is solving at a non-root ⇒ the stiff columns
    // break down.  Override those rows to R_ω = ω − ω_BC: now q* IS a root, ∂R/∂ω,wall = 1
    // (identity row), and ∂R/∂θ = −∂ω_BC/∂θ (nonzero only for β1, index 2).  Matches the warm-FD
    // truth (which re-solves WITH the pinning).  (useWallFunctions=true uses a blended BC — not
    // handled here; those cases fall back to the PDE rows.)
    std::vector<int> pinCell;
    std::vector<double> pinBC;
    const double beta1 = sst.coeffs.beta1;
    if (!settings_.useWallFunctions) {
        for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
            const Patch& pat = mesh_.patch(pi);
            if (pat.type != "wall") continue;
            for (FaceID fi : pat.faces) {
                int o = mesh_.face(fi).owner;
                double y1 = std::max(mesh_.face(fi).delta, 1e-20);
                double omBC = 60.0 * nu_ / (beta1 * y1 * y1);
                pinCell.push_back(o); pinBC.push_back(omBC);
                for (int j = 0; j < 11; ++j) dRdTheta[j][BOM + o] = 0.0;
                dRdTheta[2][BOM + o] = omBC / beta1;    // ∂R_ω,wall/∂β1 = −∂ω_BC/∂β1 = ω_BC/β1
            }
        }
    }

    // ---- per-DOF column scaling σ (5 blocks; pressure scaled by its RMS) ---------------
    double rmsUx = 0.0, rmsP = 0.0;
    for (int ci = 0; ci < nc; ++ci) { rmsUx += state_.U[ci].x * state_.U[ci].x;
                                      rmsP  += state_.p[ci] * state_.p[ci]; }
    rmsUx = std::sqrt(rmsUx / std::max(nc, 1));
    rmsP  = std::sqrt(rmsP  / std::max(nc, 1));
    const double floorU = std::max(1e-4 * rmsUx, 1e-12);
    const double floorK = std::max(settings_.kMin, 1e-12);
    const double floorW = std::max(settings_.omegaMin, 1e-12);
    const double floorP = std::max(1e-4 * std::max(rmsP, 0.5 * rmsUx * rmsUx), 1e-12);
    std::vector<double> sigma(N5);
    for (int ci = 0; ci < nc; ++ci) {
        sigma[BUX + ci] = std::max(std::abs(state_.U[ci].x),   floorU);
        sigma[BUY + ci] = std::max(std::abs(state_.U[ci].y),   floorU);
        sigma[BK  + ci] = std::max(std::abs(state_.k[ci]),     floorK);
        sigma[BOM + ci] = std::max(std::abs(state_.omega[ci]), floorW);
        sigma[BP  + ci] = std::max(std::abs(state_.p[ci]),     floorP);
    }
    // Row scaling ρ: the wall-ω Dirichlet rows carry an O(1e7) RHS (∂ω_BC/∂β1) that would
    // dominate the FGMRES residual norm and starve the velocity tangent the QoIs feel.  Scale
    // those rows by 1/σ_ω so they sit at O(1) — purely a residual-norm rebalance (the solution
    // ṽ is unchanged), and consistent with the identity-pinned ω preconditioner block.
    std::vector<double> rho(N5, 1.0);
    for (int o : pinCell) rho[BOM + o] = 1.0 / sigma[BOM + o];

    auto augResidualAt = [&](const std::vector<double>& d) -> std::vector<double> {
        FlowFields pert = state_;
        for (int ci = 0; ci < nc; ++ci) {
            pert.U[ci].x += d[BUX + ci]; pert.U[ci].y += d[BUY + ci];
            pert.k[ci] += d[BK + ci];   pert.omega[ci] += d[BOM + ci];
            pert.p[ci] += d[BP + ci];
        }
        // Re-apply the velocity/pressure BCs to the perturbed interior so the boundary faces
        // RESPOND (outlet velocity extrapolates ⇒ continuity can close; wall/outlet pressure
        // Neumann/Dirichlet ⇒ greenGaussGrad(p) is right near walls — the momentum↔pressure
        // coupling the QoIs feel).  assembleResidual otherwise freezes the boundary values.
        applyVelocityBC(pert.U, mesh_, bcs_);
        applyPressureBC(pert.p, mesh_, bcs_);
        std::vector<double> R = solver.assembleResidual(pert, sst.coeffs);
        R.resize(N5);
        std::vector<double> ux(nc), uy(nc);
        for (int ci = 0; ci < nc; ++ci) { ux[ci] = pert.U[ci].x; uy[ci] = pert.U[ci].y; }
        std::vector<double> div = massFluxDivergence(ux, uy, /*homogeneous=*/false);
        for (int ci = 0; ci < nc; ++ci) R[BP + ci] = div[ci];
        for (size_t p = 0; p < pinCell.size(); ++p)              // wall-ω Dirichlet residual
            R[BOM + pinCell[p]] = pert.omega[pinCell[p]] - pinBC[p];
        return R;
    };
    auto observeAtDelta = [&](const std::vector<double>& d) -> std::vector<double> {
        FlowFields work = state_;
        for (int ci = 0; ci < nc; ++ci) {
            work.U[ci].x += d[BUX + ci]; work.U[ci].y += d[BUY + ci];
            work.k[ci] += d[BK + ci];   work.omega[ci] += d[BOM + ci];
            work.p[ci] += d[BP + ci];
        }
        sst.computeFields(mesh_, work.k, work.omega, work.U, nu_,
                          work.nuT, work.F1, work.F2, work.Pk, work.CDkw);
        const double floor = 0.1 * nu_;
        for (int ci = 0; ci < nc; ++ci) work.nuT[ci] = std::max(work.nuT[ci], floor);
        return obs_.evaluate(mesh_, work, nu_);
    };

    // ---- physics-based block-SIMPLE preconditioner blocks (assembled ONCE) --------------
    LinearSystem Amom, Ak, Aom; std::vector<double> aPp;
    solver.assemblePreconBlocks(state_, sst.coeffs, Amom, Ak, Aom, aPp);
    // pin the wall-ω rows of the ω preconditioner block to identity (consistent with the
    // Dirichlet residual): diag = −1, row off-diagonals zeroed ⇒ −A_om⁻¹ acts as identity there.
    if (!pinCell.empty()) {
        std::vector<char> pinned(nc, 0);
        for (int o : pinCell) { pinned[o] = 1; Aom.diag[o] = -1.0; }
        for (int f = 0; f < Aom.nIF; ++f) {
            if (pinned[Aom.own[f]]) Aom.upper[f] = 0.0;
            if (pinned[Aom.nbr[f]]) Aom.lower[f] = 0.0;
        }
    }
    LinearSystem Lp = assemblePressurePoisson(aPp);
    BiCGSTABSolver blockSolver;
    AMGSolver pSolver;
    auto setPrimeBC = [&](ScalarField& pf) {
        for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
            const Patch& pat = mesh_.patch(pi);
            if (pat.type == "outlet") continue;
            for (FaceID fi : pat.faces) pf.bface(fi) = pf[mesh_.face(fi).owner];
        }
    };

    // M_phys⁻¹ r → z (physical 5-block).  z_block = −A_block⁻¹ r_block (since A_block ≈ −∂R/∂block),
    // then the SIMPLE pressure Schur on the velocity predictor.  Loose block solves (tol 0.1).
    auto Mphys = [&](const std::vector<double>& r) -> std::vector<double> {
        std::vector<double> z(N5, 0.0);
        std::vector<double> rb(nc), zb(nc, 0.0);
        for (int ci = 0; ci < nc; ++ci) rb[ci] = r[BOM + ci];
        Aom.source = rb; std::fill(zb.begin(), zb.end(), 0.0);
        blockSolver.solve(Aom, zb, 400, 1e-4);
        for (int ci = 0; ci < nc; ++ci) z[BOM + ci] = -zb[ci];
        for (int ci = 0; ci < nc; ++ci) rb[ci] = r[BK + ci];
        Ak.source = rb; std::fill(zb.begin(), zb.end(), 0.0);
        blockSolver.solve(Ak, zb, 400, 1e-4);
        for (int ci = 0; ci < nc; ++ci) z[BK + ci] = -zb[ci];

        std::vector<double> zuhx(nc, 0.0), zuhy(nc, 0.0), rux(nc), ruy(nc);
        for (int ci = 0; ci < nc; ++ci) { rux[ci] = r[BUX + ci]; ruy[ci] = r[BUY + ci]; }
        Amom.source = rux; blockSolver.solve(Amom, zuhx, 400, 1e-4);
        Amom.source = ruy; blockSolver.solve(Amom, zuhy, 400, 1e-4);
        for (int ci = 0; ci < nc; ++ci) { zuhx[ci] = -zuhx[ci]; zuhy[ci] = -zuhy[ci]; }

        std::vector<double> div = massFluxDivergence(zuhx, zuhy, /*homogeneous=*/true);
        std::vector<double> prhs(nc);
        for (int ci = 0; ci < nc; ++ci) prhs[ci] = -div[ci] + r[BP + ci];
        Lp.source = prhs;
        std::vector<double> zp(nc, 0.0);
        pSolver.solve(Lp, zp, 500, 1e-8);
        ScalarField zpF(mesh_, "zp");
        for (int ci = 0; ci < nc; ++ci) zpF[ci] = zp[ci];
        setPrimeBC(zpF);
        VectorField gradZp = greenGaussGrad(zpF);
        for (int ci = 0; ci < nc; ++ci) {
            double rap = mesh_.cell(ci).volume / (std::abs(aPp[ci]) > 1e-30 ? aPp[ci] : 1.0);
            z[BUX + ci] = zuhx[ci] - rap * gradZp[ci].x;
            z[BUY + ci] = zuhy[ci] - rap * gradZp[ci].y;
            z[BP  + ci] = zp[ci];
        }
        return z;
    };
    // preconditioner for the row-scaled, column-scaled operator B = ρ∘Ã:  M⁻¹ ≈ B⁻¹ = Ã⁻¹∘ρ⁻¹,
    // so un-row-scale (÷ρ), apply the physical block-SIMPLE M_phys⁻¹, then un-column-scale (÷σ).
    auto precond = [&](const std::vector<double>& r, std::vector<double>& z) {
        std::vector<double> ru(N5);
        for (int i = 0; i < N5; ++i) ru[i] = r[i] / rho[i];
        std::vector<double> zp = Mphys(ru);
        for (int i = 0; i < N5; ++i) z[i] = zp[i] / sigma[i];
    };

    // exact σ-scaled augmented Jv:  Ã ṽ = J·(σ∘ṽ)  via central differences
    std::vector<double> dplus(N5), dminus(N5);
    auto matvec = [&](const std::vector<double>& vt, std::vector<double>& yv) {
        double vn = linalg::norm(vt);
        if (vn < 1e-300) { std::fill(yv.begin(), yv.end(), 0.0); return; }
        double eps = fdStep / vn;
        for (int i = 0; i < N5; ++i) { double s = eps * sigma[i] * vt[i]; dplus[i] = s; dminus[i] = -s; }
        std::vector<double> Rp = augResidualAt(dplus);
        std::vector<double> Rm = augResidualAt(dminus);
        out.nResidualEvals += 2;
        double inv2e = 1.0 / (2.0 * eps);
        for (int i = 0; i < N5; ++i) yv[i] = rho[i] * (Rp[i] - Rm[i]) * inv2e;
    };

    std::vector<double> vt(N5), w(N5);
    for (int j = 0; j < 11; ++j) {
        double bn = 0.0;
        for (int i = 0; i < N4; ++i) bn += dRdTheta[j][i] * dRdTheta[j][i];
        if (std::sqrt(bn) < 1e-300) { out.krylovConverged[j] = 1; continue; }   // κ
        std::vector<double> b(N5, 0.0);
        for (int i = 0; i < N4; ++i) b[i] = -dRdTheta[j][i];                    // continuity RHS = 0
        for (int i = 0; i < N5; ++i) b[i] *= rho[i];                            // row-balance wall-ω

        std::fill(vt.begin(), vt.end(), 0.0);
        MatrixFreeResult kr = fgmresMatrixFree(matvec, precond, b, vt, /*restart=*/150, maxIter, krylovTol);
        out.krylovIters[j]     = kr.iterations;
        out.krylovRelRes[j]    = kr.finalRes;
        out.krylovConverged[j] = kr.converged ? 1 : 0;
        for (int i = 0; i < N5; ++i) w[i] = sigma[i] * vt[i];

        // directional-derivative step from the U/k/ω blocks ONLY: the pressure tangent dp can be
        // large in magnitude but the QoIs do not read p, so including it would shrink hg until the
        // velocity perturbation fell below FD roundoff (collapsing the gradient to ~0).
        double scale = 0.0;
        for (int i = 0; i < N4; ++i) scale = std::max(scale, std::abs(w[i]) / sigma[i]);
        if (scale < 1e-300) continue;
        double hg = fdStep / scale;
        for (int i = 0; i < N5; ++i) { dplus[i] = hg * w[i]; dminus[i] = -hg * w[i]; }
        std::vector<double> gP = observeAtDelta(dplus);
        std::vector<double> gM = observeAtDelta(dminus);
        double inv2h = 1.0 / (2.0 * hg);
        for (int i = 0; i < nObsv; ++i) out.dObsDTheta[i][j] = (gP[i] - gM[i]) * inv2h;
    }
    return out;
}

// ---- RUNG 1 (WARM-FD) — robust full true-model gradient by warm-started central FD --------
TangentGradientResult ParameterSensitivity::etaJacobianWarmFD(
        const std::vector<double>& theta11, double hRel, double hFloor,
        int warmMaxIter, double warmTol) {
    const int nObsv = obs_.nObs();
    TangentGradientResult out;
    out.dObsDTheta.assign(nObsv, std::vector<double>(11, 0.0));
    out.logLikGradient.assign(11, 0.0);
    out.krylovIters.assign(11, 0);
    out.krylovRelRes.assign(11, 0.0);
    out.krylovConverged.assign(11, 0);
    if (!hasState_) return out;

    // looser cap/tol for the WARM re-solves (they start ~converged) — the speedup source.
    SolverSettings warm = settings_;
    if (warmMaxIter > 0) warm.maxIterations  = warmMaxIter;
    if (warmTol > 0.0)   warm.convergenceTol = warmTol;

    // β* (8) and a1 (9) are the only coefficients nonlinear in the closure — smaller step.
    auto stepFor = [&](int j, double tj) {
        double hr = (j == 8 || j == 9) ? std::min(hRel, 1e-5) : hRel;
        return std::max(hr * std::abs(tj), hFloor);
    };
    // Re-solve SIMPLE at theta_pert WARM-STARTED from the fixed converged state, then observe
    // (closure recomputed exactly as observe()) and also score the Gaussian log-likelihood —
    // both η and logL from the SAME warm solve, so ∂η/∂θ and ∂logL/∂θ cost one warm-FD pass.
    auto etaWarm = [&](const std::vector<double>& tp, bool& ok, double& logL) -> std::vector<double> {
        SSTModel sstp = makeModel(tp);
        SIMPLESolver solverp(mesh_, sstp, bcs_, nu_, warm);
        FlowFields f = state_;                            // WARM start
        ConvergenceHistory hist = solverp.solve(f);
        out.nResidualEvals += hist.finalIter;             // total SIMPLE iterations (cost proxy)
        ok = !hist.diverged;
        sstp.computeFields(mesh_, f.k, f.omega, f.U, nu_, f.nuT, f.F1, f.F2, f.Pk, f.CDkw);
        const double floor = 0.1 * nu_;
        for (int ci = 0; ci < mesh_.nCells(); ++ci) f.nuT[ci] = std::max(f.nuT[ci], floor);
        logL = obs_.logLikelihood(mesh_, f, nu_);
        return obs_.evaluate(mesh_, f, nu_);
    };

    for (int j = 0; j < 11; ++j) {
        double hj = stepFor(j, theta11[j]);
        std::vector<double> tp = theta11, tm = theta11;
        tp[j] += hj; tm[j] -= hj;
        bool okp = true, okm = true;
        double lp = 0.0, lm = 0.0;
        std::vector<double> ep = etaWarm(tp, okp, lp);
        std::vector<double> em = etaWarm(tm, okm, lm);
        out.krylovConverged[j] = (okp && okm) ? 1 : 0;
        if (!(okp && okm)) continue;                      // leave the column 0 on divergence
        for (int i = 0; i < nObsv; ++i) out.dObsDTheta[i][j] = (ep[i] - em[i]) / (2.0 * hj);
        out.logLikGradient[j] = (lp - lm) / (2.0 * hj);   // ∂logL/∂θ_j (Gaussian)
    }
    return out;
}
