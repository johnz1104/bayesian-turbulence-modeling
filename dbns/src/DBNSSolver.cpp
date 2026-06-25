#include "DBNSSolver.hpp"
#include "HLLCFlux.hpp"
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace dbns {

// primitive-variable indices into grad_ / limiter_ arrays
enum GradIdx { G_RHO = 0, G_U = 1, G_V = 2, G_P = 3, G_K = 4, G_W = 5 };

DBNSSolver::DBNSSolver(const Mesh& mesh, const IdealGasEOS& eos,
                       const SSTCoefficients& sst,
                       const DBNSBoundaryConditions& bcs,
                       const DBNSSettings& settings)
    : mesh_(mesh), eos_(eos), sstCoeffs_(sst), sst_(sst), bcs_(bcs),
      settings_(settings) {
    int nc = mesh_.nCells();
    W_.assign(nc, StateVec{});
    // Consistent per-unit-depth cell volume via the 2D divergence theorem,
    //   V = 1/2 sum_f (x_f n_x + y_f n_y) |S_f|,  outward normal per cell.
    // The core Mesh stores a 3D divergence-theorem volume that, for these
    // z-open 2D cells, is 2/3 of the per-unit-depth area; that is consistent for
    // the steady pressure solvers but would mis-time an explicit unsteady march,
    // so the density-based solver carries its own volume.  (Mesh is not changed.)
    vol_.assign(nc, 0.0);
    for (int ci = 0; ci < nc; ++ci) {
        double s = 0.0;
        for (FaceID fi : mesh_.cell(ci).faces) {
            const Face& f = mesh_.face(fi);
            double sgn = (f.owner == ci) ? 1.0 : -1.0;
            double nx = f.normal.x * sgn, ny = f.normal.y * sgn;
            s += (f.center.x * nx + f.center.y * ny) * f.area;
        }
        vol_[ci] = 0.5 * s;
    }
    muLam_.assign(nc, 0.0);
    muT_.assign(nc, 0.0);
    grad_.assign(nc, std::array<Vec3, 6>{});
    limiter_.assign(nc, std::array<double, 6>{});
    res_.assign(nc, StateVec{});
    dtCell_.assign(nc, 0.0);
}

void DBNSSolver::initUniform(const Primitive& V) {
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        W_[ci] = GasState::toConserved(V, eos_);
}

void DBNSSolver::initField(const std::vector<Primitive>& V) {
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        W_[ci] = GasState::toConserved(V[ci], eos_);
}

// --- properties: laminar viscosity (Sutherland) and SST eddy viscosity ------
void DBNSSolver::updateProperties() {
    double a1 = sstCoeffs_.a1, bStar = sstCoeffs_.betaStar;
    const auto& wallDist = mesh_.wallDistance();
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        Primitive V = GasState::toPrimitive(W_[ci], eos_);
        double T = GasState::temperature(V, eos_);
        muLam_[ci] = eos_.viscosity(T);
        if (!settings_.turbulent) { muT_[ci] = 0.0; continue; }

        double nu = muLam_[ci] / V.rho;
        double k = std::max(V.k, settings_.kFloor);
        double w = std::max(V.omega, settings_.omegaFloor);
        double y = (ci < (int)wallDist.size()) ? std::max(wallDist[ci], 1e-9) : 1e-9;

        // strain-rate magnitude |S| = sqrt(2 S_ij S_ij)
        double dudx = grad_[ci][G_U].x, dudy = grad_[ci][G_U].y;
        double dvdx = grad_[ci][G_V].x, dvdy = grad_[ci][G_V].y;
        double S12 = 0.5 * (dudy + dvdx);
        double Smag = std::sqrt(2.0 * (dudx * dudx + dvdy * dvdy) + 4.0 * S12 * S12);

        // F2 blending (Menter 2003): arg2 = max(2 sqrt(k)/(beta* w y), 500 nu/(y^2 w))
        double arg2 = std::max(2.0 * std::sqrt(k) / (bStar * w * y),
                               500.0 * nu / (y * y * w));
        double F2 = std::tanh(arg2 * arg2);

        // Bradshaw shear-stress limited eddy viscosity: mu_t = rho a1 k/max(a1 w, S F2)
        double denom = std::max(a1 * w, Smag * F2);
        muT_[ci] = V.rho * a1 * k / std::max(denom, 1e-30);
        if (!std::isfinite(muT_[ci]) || muT_[ci] < 0.0) muT_[ci] = 0.0;
    }
}

// --- Green-Gauss cell gradients of the six primitive variables --------------
void DBNSSolver::computeGradients() {
    int nc = mesh_.nCells();
    for (int ci = 0; ci < nc; ++ci)
        for (int v = 0; v < 6; ++v) grad_[ci][v] = Vec3{};

    auto primVar = [&](const Primitive& V, int v) -> double {
        switch (v) { case G_RHO: return V.rho; case G_U: return V.u;
                     case G_V: return V.v; case G_P: return V.p;
                     case G_K: return V.k; default: return V.omega; }
    };

    int nIF = mesh_.nInternalFaces();
    // internal faces: contribute interpolated face value to owner (+) and neighbor (-)
    for (int fi = 0; fi < nIF; ++fi) {
        const Face& f = mesh_.face(fi);
        Primitive VP = GasState::toPrimitive(W_[f.owner], eos_);
        Primitive VN = GasState::toPrimitive(W_[f.neighbor], eos_);
        Vec3 Sf = f.normal * f.area;
        for (int v = 0; v < 6; ++v) {
            double phiF = primVar(VP, v) * f.weight + primVar(VN, v) * (1.0 - f.weight);
            grad_[f.owner][v]    = grad_[f.owner][v]    + Sf * phiF;
            grad_[f.neighbor][v] = grad_[f.neighbor][v] - Sf * phiF;
        }
    }
    // boundary faces: face value = 0.5 (interior + ghost)
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face& f = mesh_.face(fi);
        int bidx = fi - nIF;
        Primitive VP = GasState::toPrimitive(W_[f.owner], eos_);
        const Patch& pat = mesh_.patch(f.patchID);
        Primitive Vg = bcs_.has(pat.name)
                     ? ghostState(VP, fi, bcs_.get(pat.name), bidx) : VP;
        Vec3 Sf = f.normal * f.area;
        for (int v = 0; v < 6; ++v) {
            double phiF = 0.5 * (primVar(VP, v) + primVar(Vg, v));
            grad_[f.owner][v] = grad_[f.owner][v] + Sf * phiF;
        }
    }
    for (int ci = 0; ci < nc; ++ci) {
        double V = vol_[ci];
        if (V > 1e-30)
            for (int v = 0; v < 6; ++v) grad_[ci][v] = grad_[ci][v] / V;
    }
}

// --- Venkatakrishnan / Barth-Jespersen multidimensional limiter -------------
void DBNSSolver::computeLimiters() {
    int nc = mesh_.nCells();
    if (settings_.reconstructOrder < 2) {
        for (int ci = 0; ci < nc; ++ci) limiter_[ci].fill(0.0);  // 1st order
        return;
    }
    if (!settings_.limitReconstruction) {
        for (int ci = 0; ci < nc; ++ci) limiter_[ci].fill(1.0);  // unlimited (MMS)
        return;
    }
    auto primVar = [&](const Primitive& V, int v) -> double {
        switch (v) { case G_RHO: return V.rho; case G_U: return V.u;
                     case G_V: return V.v; case G_P: return V.p;
                     case G_K: return V.k; default: return V.omega; }
    };
    for (int ci = 0; ci < nc; ++ci) {
        Primitive VC = GasState::toPrimitive(W_[ci], eos_);
        double qC[6]; for (int v = 0; v < 6; ++v) qC[v] = primVar(VC, v);
        double qMax[6], qMin[6];
        for (int v = 0; v < 6; ++v) { qMax[v] = qC[v]; qMin[v] = qC[v]; }
        // neighbour extrema
        for (FaceID fi : mesh_.cell(ci).faces) {
            const Face& f = mesh_.face(fi);
            int other = (f.owner == ci) ? f.neighbor : f.owner;
            if (other < 0) continue;
            Primitive VO = GasState::toPrimitive(W_[other], eos_);
            for (int v = 0; v < 6; ++v) {
                double q = primVar(VO, v);
                qMax[v] = std::max(qMax[v], q); qMin[v] = std::min(qMin[v], q);
            }
        }
        double phi[6]; for (int v = 0; v < 6; ++v) phi[v] = 1.0;
        for (FaceID fi : mesh_.cell(ci).faces) {
            const Face& f = mesh_.face(fi);
            Vec3 dr = f.center - mesh_.cell(ci).center;
            for (int v = 0; v < 6; ++v) {
                double d = grad_[ci][v].dot(dr);   // unlimited increment to face
                double p = Limiters::venkat(d, qC[v], qMax[v], qMin[v], 0.0);
                phi[v] = std::min(phi[v], p);
            }
        }
        for (int v = 0; v < 6; ++v) limiter_[ci][v] = phi[v];
    }
}

// reconstruct primitive at face center xf from cell ci, with positivity guard
Primitive DBNSSolver::reconstruct(int ci, const Vec3& xf) const {
    Primitive V = GasState::toPrimitive(W_[ci], eos_);
    if (settings_.reconstructOrder < 2) return V;
    Vec3 dr = xf - mesh_.cell(ci).center;
    Primitive R = V;
    R.rho   = V.rho   + limiter_[ci][G_RHO] * grad_[ci][G_RHO].dot(dr);
    R.u     = V.u     + limiter_[ci][G_U]   * grad_[ci][G_U].dot(dr);
    R.v     = V.v     + limiter_[ci][G_V]   * grad_[ci][G_V].dot(dr);
    R.p     = V.p     + limiter_[ci][G_P]   * grad_[ci][G_P].dot(dr);
    R.k     = V.k     + limiter_[ci][G_K]   * grad_[ci][G_K].dot(dr);
    R.omega = V.omega + limiter_[ci][G_W]   * grad_[ci][G_W].dot(dr);
    // drop to first order if reconstruction left the physical set
    if (!(R.rho > 0.0) || !(R.p > 0.0)) return V;
    if (R.k < 0.0) R.k = V.k;
    if (R.omega < settings_.omegaFloor) R.omega = V.omega;
    return R;
}

// boundary ghost state ------------------------------------------------------
Primitive DBNSSolver::ghostState(const Primitive& in, int faceId,
                                 const BoundarySpec& spec, int boundaryIdx) const {
    if (!bndOverride_.empty()) return bndOverride_[boundaryIdx];
    const Face& f = mesh_.face(faceId);
    double nx = f.normal.x, ny = f.normal.y;
    Primitive g = in;
    switch (spec.kind) {
        case BoundaryKind::SupersonicInflow:
        case BoundaryKind::FixedState:
            return spec.freestream;
        case BoundaryKind::Extrapolate:
            return in;
        case BoundaryKind::SubsonicInflow: {
            double Tfs = spec.freestream.p / (spec.freestream.rho * eos_.R);
            g.u = spec.freestream.u; g.v = spec.freestream.v;
            g.k = spec.freestream.k; g.omega = spec.freestream.omega;
            g.p = in.p;                          // extrapolate pressure
            g.rho = g.p / (eos_.R * Tfs);
            return g;
        }
        case BoundaryKind::SubsonicOutflow: {
            g = in; g.p = spec.backPressure;     // impose back pressure
            return g;
        }
        case BoundaryKind::SlipWall: {
            double un = in.u * nx + in.v * ny;   // reflect normal velocity
            g.u = in.u - 2.0 * un * nx;
            g.v = in.v - 2.0 * un * ny;
            return g;
        }
        case BoundaryKind::NoSlipAdiabatic: {
            g.u = -in.u; g.v = -in.v;            // face velocity 0
            g.p = in.p; g.rho = in.rho;          // dT/dn = 0
            g.k = -in.k;                         // face k = 0
            return g;
        }
        case BoundaryKind::NoSlipIsothermal: {
            g.u = -in.u; g.v = -in.v;
            double Tw = spec.wallTemp;
            double Tin = GasState::temperature(in, eos_);
            double Tg = 2.0 * Tw - Tin;          // face T = Tw
            g.p = in.p; g.rho = g.p / (eos_.R * std::max(Tg, 1.0));
            g.k = -in.k;
            return g;
        }
    }
    return in;
}

// --- viscous flux on an internal face (corrected face gradient) -------------
void DBNSSolver::addViscousFace(int faceId) {
    const Face& f = mesh_.face(faceId);
    int P = f.owner, N = f.neighbor;
    Primitive VP = GasState::toPrimitive(W_[P], eos_);
    Primitive VN = GasState::toPrimitive(W_[N], eos_);
    double nx = f.normal.x, ny = f.normal.y;

    Vec3 d = mesh_.cell(N).center - mesh_.cell(P).center;
    double dmag = std::max(d.norm(), 1e-30);
    Vec3 dhat = d / dmag;

    // corrected face gradient: averaged cell gradient + normal-difference fix
    auto faceGrad = [&](int v, double phiP, double phiN) -> Vec3 {
        Vec3 gAvg = (grad_[P][v] + grad_[N][v]) * 0.5;
        double corr = (phiN - phiP) / dmag - gAvg.dot(dhat);
        return gAvg + dhat * corr;
    };
    Vec3 gU = faceGrad(G_U, VP.u, VN.u);
    Vec3 gV = faceGrad(G_V, VP.v, VN.v);
    double TP = GasState::temperature(VP, eos_), TN = GasState::temperature(VN, eos_);
    // Cell-centred T gradients are not stored, so use the wall-normal-consistent
    // difference estimate (dominant term for boundary-layer heat conduction).
    Vec3 gT = dhat * ((TN - TP) / dmag);

    double muP = muLam_[P] + muT_[P], muN = muLam_[N] + muT_[N];
    double muEff = 0.5 * (muP + muN);
    double muLamF = 0.5 * (muLam_[P] + muLam_[N]);
    double muTF   = 0.5 * (muT_[P] + muT_[N]);

    double dudx = gU.x, dudy = gU.y, dvdx = gV.x, dvdy = gV.y;
    double div = dudx + dvdy;
    double tau_xx = muEff * (2.0 * dudx - 2.0 / 3.0 * div);
    double tau_yy = muEff * (2.0 * dvdy - 2.0 / 3.0 * div);
    double tau_xy = muEff * (dudy + dvdx);
    if (settings_.turbulent) {
        double rhok = 0.5 * (VP.rho * VP.k + VN.rho * VN.k);
        tau_xx -= 2.0 / 3.0 * rhok;     // Boussinesq turbulent normal stress
        tau_yy -= 2.0 / 3.0 * rhok;
    }
    double uF = 0.5 * (VP.u + VN.u), vF = 0.5 * (VP.v + VN.v);
    double lamEff = eos_.Cp() * (muLamF / eos_.Pr + muTF / eos_.Pr_T);

    StateVec Fv{};
    Fv[I_RHOU] = tau_xx * nx + tau_xy * ny;
    Fv[I_RHOV] = tau_xy * nx + tau_yy * ny;
    double qn = -lamEff * gT.dot(f.normal);                 // q.n = -lambda dT/dn
    Fv[I_RHOE] = (tau_xx * uF + tau_xy * vF) * nx
               + (tau_xy * uF + tau_yy * vF) * ny - qn;
    if (settings_.turbulent) {
        Vec3 gK = faceGrad(G_K, VP.k, VN.k);
        Vec3 gW = faceGrad(G_W, VP.omega, VN.omega);
        double sk = sstCoeffs_.sigma_k1, sw = sstCoeffs_.sigma_w1;  // near-wall blend dominant
        Fv[I_RHOK] = (muLamF + sk * muTF) * gK.dot(f.normal);
        Fv[I_RHOW] = (muLamF + sw * muTF) * gW.dot(f.normal);
    }
    double A = f.area;
    for (int i = 0; i < NVAR; ++i) {
        res_[P][i] -= Fv[i] * A;   // viscous (diffusive) flux subtracts
        res_[N][i] += Fv[i] * A;
    }
}

// --- boundary face flux (convective + viscous wall treatment) ---------------
void DBNSSolver::addBoundaryFlux(int faceId, int patchIdx) {
    const Face& f = mesh_.face(faceId);
    int P = f.owner;
    int bidx = faceId - mesh_.nInternalFaces();
    double nx = f.normal.x, ny = f.normal.y, A = f.area;
    const Patch& pat = mesh_.patch(patchIdx);
    Primitive VP = GasState::toPrimitive(W_[P], eos_);

    if (!bcs_.has(pat.name)) {                  // default: extrapolate
        StateVec F = GasState::normalFlux(VP, nx, ny, eos_);
        for (int i = 0; i < NVAR; ++i) res_[P][i] += F[i] * A;
        return;
    }
    const BoundarySpec& spec = bcs_.get(pat.name);
    bool isWall = (spec.kind == BoundaryKind::SlipWall ||
                   spec.kind == BoundaryKind::NoSlipAdiabatic ||
                   spec.kind == BoundaryKind::NoSlipIsothermal);

    if (isWall && bndOverride_.empty()) {
        // Inviscid wall flux: pressure only (zero mass flux through the wall).
        double pw = VP.p;
        res_[P][I_RHOU] += pw * nx * A;
        res_[P][I_RHOV] += pw * ny * A;
        if (spec.kind == BoundaryKind::SlipWall) return;

        // No-slip viscous wall flux from one-sided near-wall gradients.
        double delta = std::max(f.delta, 1e-12);
        double muEff = muLam_[P] + muT_[P];
        // tangential velocity gradient (wall velocity = 0)
        Vec3 Uc{VP.u, VP.v, 0.0};
        double un = Uc.dot(f.normal);
        Vec3 Ut = Uc - f.normal * un;            // tangential part
        Vec3 dUdn = Uc / delta;                  // (U_cell - 0)/delta
        double dUdn_n = dUdn.dot(f.normal);
        // tau.n = mu_eff [ dU/dn + 1/3 (dU/dn . n) n ]
        Vec3 taun = (dUdn + f.normal * (dUdn_n / 3.0)) * muEff;
        res_[P][I_RHOU] -= taun.x * A;
        res_[P][I_RHOV] -= taun.y * A;
        // wall heat flux: q.n = -lambda dT/dn ; adiabatic -> 0
        if (spec.kind == BoundaryKind::NoSlipIsothermal) {
            double lamEff = eos_.Cp() * (muLam_[P] / eos_.Pr + muT_[P] / eos_.Pr_T);
            double Tin = GasState::temperature(VP, eos_);
            double dTdn = (spec.wallTemp - Tin) / delta;   // (T_wall - T_cell)/delta
            double qn = -lamEff * dTdn;
            res_[P][I_RHOE] -= -qn * A;   // energy viscous flux contribution
        }
        // turbulent transport at wall: k=0 imposed via source clamp; omega via BC
        return;
    }

    // open boundary (inflow / outflow / far-field / fixed): reconstruct interior,
    // build ghost, take HLLC; add viscous using ghost as the outer state.
    Primitive Vin = reconstruct(P, f.center);
    Primitive Vg = ghostState(Vin, faceId, spec, bidx);
    StateVec F = HLLCFlux::flux(Vin, Vg, nx, ny, eos_);
    for (int i = 0; i < NVAR; ++i) res_[P][i] += F[i] * A;

    if (settings_.viscous && spec.kind != BoundaryKind::Extrapolate) {
        // light viscous boundary contribution from one-sided gradient to ghost
        double delta = std::max(f.delta, 1e-12);
        double muEff = muLam_[P] + muT_[P];
        Vec3 dUdn{(Vg.u - VP.u) / delta, (Vg.v - VP.v) / delta, 0.0};
        double div = dUdn.x * nx + dUdn.y * ny;
        Vec3 taun{muEff * (dUdn.x), muEff * (dUdn.y), 0.0};
        taun = taun + f.normal * (muEff * div / 3.0);
        res_[P][I_RHOU] -= taun.x * A;
        res_[P][I_RHOV] -= taun.y * A;
    }
}

// --- turbulence sources (compressible SST) ----------------------------------
void DBNSSolver::addTurbulenceSources() {
    if (!settings_.turbulent) return;
    double bStar = sstCoeffs_.betaStar, a1 = sstCoeffs_.a1;
    const auto& wallDist = mesh_.wallDistance();
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        Primitive V = GasState::toPrimitive(W_[ci], eos_);
        double rho = V.rho;
        double k = std::max(V.k, settings_.kFloor);
        double w = std::max(V.omega, settings_.omegaFloor);
        double nu = muLam_[ci] / rho;
        double y = (ci < (int)wallDist.size()) ? std::max(wallDist[ci], 1e-9) : 1e-9;

        double dudx = grad_[ci][G_U].x, dudy = grad_[ci][G_U].y;
        double dvdx = grad_[ci][G_V].x, dvdy = grad_[ci][G_V].y;
        double S12 = 0.5 * (dudy + dvdx);
        double S2 = 2.0 * (dudx * dudx + dvdy * dvdy) + 4.0 * S12 * S12;  // |S|^2

        // cross-diffusion CD = 2 rho sigma_w2 / w * gradK.gradOmega
        double gkgw = grad_[ci][G_K].dot(grad_[ci][G_W]);
        double CDkw = 2.0 * rho * sstCoeffs_.sigma_w2 / w * gkgw;
        double CDpos = std::max(CDkw, 1e-10);

        // F1 blending (Menter 2003)
        double a1arg = std::max(std::sqrt(k) / (bStar * w * y), 500.0 * nu / (y * y * w));
        double arg1 = std::min(a1arg, 4.0 * rho * sstCoeffs_.sigma_w2 * k / (CDpos * y * y));
        double F1 = std::tanh(arg1 * arg1 * arg1 * arg1);

        double alpha = sstCoeffs_.blend(sstCoeffs_.alpha1, sstCoeffs_.alpha2, F1);
        double beta  = sstCoeffs_.blend(sstCoeffs_.beta1,  sstCoeffs_.beta2,  F1);

        double muT = muT_[ci];
        double Pk = muT * S2;                                   // production
        Pk = std::min(Pk, 20.0 * bStar * rho * k * w);          // Menter limiter

        // dilatational-dissipation (compressibility) correction
        double Mt2 = 2.0 * k / (eos_.gamma * V.p / rho);        // M_t^2 = 2k/a^2
        double comprFactor = 1.0;
        if (settings_.compressibility == CompressibilityModel::Sarkar) {
            comprFactor = 1.0 + settings_.comprXiStar * Mt2;
        } else if (settings_.compressibility == CompressibilityModel::Zeman) {
            double Mt0 = settings_.turbMachCutoff;
            double F = std::max(0.0, Mt2 - Mt0 * Mt0);
            comprFactor = 1.0 + settings_.comprXiStar * F;
        }

        double Dk = bStar * rho * k * w * comprFactor;          // k dissipation
        double sourceK = Pk - Dk;
        double Pw = alpha * rho * S2;                           // omega production
        double Dw = beta * rho * w * w;
        double crossW = 2.0 * (1.0 - F1) * rho * sstCoeffs_.sigma_w2 / w * gkgw;
        double sourceW = Pw - Dw + crossW;

        double Vol = vol_[ci];
        res_[ci][I_RHOK] -= sourceK * Vol;    // sources move to RHS of dW/dt=-res/V
        res_[ci][I_RHOW] -= sourceW * Vol;
    }
}

void DBNSSolver::computeResidual() {
    int nc = mesh_.nCells();
    for (int ci = 0; ci < nc; ++ci) res_[ci].fill(0.0);

    int nIF = mesh_.nInternalFaces();
    for (int fi = 0; fi < nIF; ++fi) {
        const Face& f = mesh_.face(fi);
        Primitive VL = reconstruct(f.owner, f.center);
        Primitive VR = reconstruct(f.neighbor, f.center);
        StateVec F = HLLCFlux::flux(VL, VR, f.normal.x, f.normal.y, eos_);
        double A = f.area;
        for (int i = 0; i < NVAR; ++i) {
            res_[f.owner][i]    += F[i] * A;
            res_[f.neighbor][i] -= F[i] * A;
        }
        if (settings_.viscous) addViscousFace(fi);
    }
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi)
        addBoundaryFlux(fi, mesh_.face(fi).patchID);

    addTurbulenceSources();

    // manufactured-solution source: dW/dt = -(res - S V)/V
    if (!mmsSource_.empty())
        for (int ci = 0; ci < nc; ++ci) {
            double Vol = vol_[ci];
            for (int i = 0; i < NVAR; ++i) res_[ci][i] -= mmsSource_[ci][i] * Vol;
        }
}

void DBNSSolver::computeTimeStep() {
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        Primitive V = GasState::toPrimitive(W_[ci], eos_);
        double a = GasState::soundSpeed(V, eos_);
        double lamC = 0.0, lamV = 0.0;
        double Vol = vol_[ci];
        for (FaceID fi : mesh_.cell(ci).faces) {
            const Face& f = mesh_.face(fi);
            double sgn = (f.owner == ci) ? 1.0 : -1.0;
            double un = (V.u * f.normal.x + V.v * f.normal.y) * sgn;
            lamC += (std::abs(un) + a) * f.area;
            if (settings_.viscous) {
                double muEff = muLam_[ci] + muT_[ci];
                double diff = std::max(4.0 / 3.0, eos_.gamma / eos_.Pr) * muEff / V.rho;
                lamV += diff * f.area * f.area / std::max(Vol, 1e-30);
            }
        }
        dtCell_[ci] = settings_.cfl * Vol / std::max(lamC + 2.0 * lamV, 1e-30);
    }
}

double DBNSSolver::rhoResidualNorm() const {
    double s = 0.0;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double r = res_[ci][I_RHO] / std::max(vol_[ci], 1e-30);
        s += r * r;
    }
    return std::sqrt(s / std::max(1, mesh_.nCells()));
}

SolveReport DBNSSolver::solve() {
    SolveReport rep;
    int nc = mesh_.nCells();
    double t = 0.0;
    double res0 = -1.0;

    // SSP-RK3 coefficients (Shu and Osher 1988); rkStages==1 -> forward Euler.
    auto evalSpatial = [&]() { computeGradients(); computeLimiters(); computeResidual(); };

    std::vector<StateVec> W0(nc);
    int iter = 0;
    for (; iter < settings_.maxIterations; ++iter) {
        updateProperties();
        computeTimeStep();
        double dtGlobal = 1e30;
        if (settings_.timeMode == TimeMode::Unsteady) {
            for (int ci = 0; ci < nc; ++ci) dtGlobal = std::min(dtGlobal, dtCell_[ci]);
            if (t + dtGlobal > settings_.tEnd) dtGlobal = settings_.tEnd - t;
        }
        auto dtOf = [&](int ci) {
            return settings_.timeMode == TimeMode::Unsteady ? dtGlobal : dtCell_[ci];
        };

        W0 = W_;
        if (settings_.rkStages >= 3) {
            // stage 1
            evalSpatial();
            for (int ci = 0; ci < nc; ++ci) {
                double f = dtOf(ci) / std::max(vol_[ci], 1e-30);
                for (int i = 0; i < NVAR; ++i) W_[ci][i] = W0[ci][i] - f * res_[ci][i];
            }
            clampPositivity();
            // stage 2
            evalSpatial();
            for (int ci = 0; ci < nc; ++ci) {
                double f = dtOf(ci) / std::max(vol_[ci], 1e-30);
                for (int i = 0; i < NVAR; ++i)
                    W_[ci][i] = 0.75 * W0[ci][i] + 0.25 * (W_[ci][i] - f * res_[ci][i]);
            }
            clampPositivity();
            // stage 3
            evalSpatial();
            for (int ci = 0; ci < nc; ++ci) {
                double f = dtOf(ci) / std::max(vol_[ci], 1e-30);
                for (int i = 0; i < NVAR; ++i)
                    W_[ci][i] = (1.0 / 3.0) * W0[ci][i]
                              + (2.0 / 3.0) * (W_[ci][i] - f * res_[ci][i]);
            }
            clampPositivity();
        } else {
            evalSpatial();
            for (int ci = 0; ci < nc; ++ci) {
                double f = dtOf(ci) / std::max(vol_[ci], 1e-30);
                for (int i = 0; i < NVAR; ++i) W_[ci][i] = W0[ci][i] - f * res_[ci][i];
            }
            clampPositivity();
        }

        // divergence check
        for (int ci = 0; ci < nc; ++ci)
            if (!GasState::admissible(W_[ci], eos_)) {
                rep.status = EvaluationStatus::Diverged;
                rep.iterations = iter; rep.tFinal = t;
                return rep;
            }

        double rn = rhoResidualNorm();
        if (res0 < 0.0) res0 = std::max(rn, 1e-30);
        if (settings_.verbose && (iter % settings_.reportInterval == 0))
            std::printf("  iter %6d  res_rho %.3e  (rel %.3e)\n", iter, rn, rn / res0);
        if (iter % settings_.reportInterval == 0) rep.residualHistory.push_back(rn);

        if (settings_.timeMode == TimeMode::Unsteady) {
            t += dtGlobal;
            if (t >= settings_.tEnd - 1e-14) {
                rep.status = EvaluationStatus::Converged; break;
            }
        } else {
            if (rn / res0 < settings_.convergenceTol) {
                rep.status = EvaluationStatus::Converged; break;
            }
        }
    }
    if (rep.status == EvaluationStatus::Unknown)
        rep.status = (settings_.timeMode == TimeMode::Unsteady)
                     ? EvaluationStatus::Converged : EvaluationStatus::Unconverged;
    rep.iterations = iter; rep.tFinal = t;
    rep.finalResidual = (res0 > 0.0) ? rhoResidualNorm() / res0 : 0.0;
    return rep;
}

// floor density / pressure / turbulence to keep states physical
void DBNSSolver::clampPositivity() {
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        StateVec& W = W_[ci];
        if (W[I_RHO] < 1e-10) W[I_RHO] = 1e-10;
        Primitive V = GasState::toPrimitive(W, eos_);
        bool fix = false;
        if (!(V.p > 0.0)) { V.p = 1.0; fix = true; }
        if (settings_.turbulent) {
            if (V.k < settings_.kFloor) { V.k = settings_.kFloor; fix = true; }
            if (V.omega < settings_.omegaFloor) { V.omega = settings_.omegaFloor; fix = true; }
        } else { V.k = 0.0; V.omega = 0.0; }
        if (fix) W_[ci] = GasState::toConserved(V, eos_);
        else { W_[ci][I_RHOK] = V.rho * V.k; W_[ci][I_RHOW] = V.rho * V.omega; }
    }
}

}  // namespace dbns
