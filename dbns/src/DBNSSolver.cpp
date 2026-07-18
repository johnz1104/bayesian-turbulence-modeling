#include "DBNSSolver.hpp"
#include "HLLCFlux.hpp"
#include "AnisotropyTools.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <stdexcept>

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
    // patch-local ordinal of each boundary face (for per-face boundary
    // profiles): patchLocalIdx_[globalFace - nInternalFaces] = position of
    // that face within its patch's own face list
    patchLocalIdx_.assign(mesh_.nFaces() - mesh_.nInternalFaces(), 0);
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch& pat = mesh_.patch(pi);
        for (size_t j = 0; j < pat.faces.size(); ++j) {
            int bidx = pat.faces[j] - mesh_.nInternalFaces();
            if (bidx >= 0 && bidx < (int)patchLocalIdx_.size())
                patchLocalIdx_[bidx] = (int)j;
        }
    }
    grad_.assign(nc, std::array<Vec3, 6>{});
    limiter_.assign(nc, std::array<double, 6>{});
    res_.assign(nc, StateVec{});
    limiterActive_.assign(nc, 0);
    dtCell_.assign(nc, 0.0);

    // wall distance to the no-slip patches of THESE boundary conditions
    // (the mesh's own wallDistance() is lazily populated and was empty for
    // every density-based run, silently breaking the SST blending; and the
    // generator's patch types need not match the imposed conditions, e.g.
    // an imposed-shock top on a channel mesh)
    std::vector<Vec3> wallPts;
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch& pat = mesh_.patch(pi);
        if (!bcs_.has(pat.name)) continue;
        BoundaryKind kind = bcs_.get(pat.name).kind;
        if (kind != BoundaryKind::NoSlipAdiabatic
            && kind != BoundaryKind::NoSlipIsothermal) continue;
        for (FaceID fi : pat.faces)
            wallPts.push_back(mesh_.face(fi).center);
    }
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch& pat = mesh_.patch(pi);
        if (!bcs_.has(pat.name)) continue;
        BoundaryKind kind = bcs_.get(pat.name).kind;
        if (kind != BoundaryKind::NoSlipAdiabatic
            && kind != BoundaryKind::NoSlipIsothermal) continue;
        for (FaceID fi : pat.faces) {
            wallAdjCell_.push_back(mesh_.face(fi).owner);
            wallAdjDelta_.push_back(std::max(mesh_.face(fi).delta, 1e-12));
        }
    }
    wallDist_.assign(nc, 1e10);
    if (!wallPts.empty()) {
        for (int ci = 0; ci < nc; ++ci) {
            const Vec3& c = mesh_.cell(ci).center;
            double best = 1e30;
            for (const Vec3& wpt : wallPts) {
                double dx = c.x - wpt.x, dy = c.y - wpt.y;
                double d2 = dx * dx + dy * dy;
                if (d2 < best) best = d2;
            }
            wallDist_[ci] = std::sqrt(best);
        }
    }
}

void DBNSSolver::initUniform(const Primitive& V) {
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        W_[ci] = GasState::toConserved(V, eos_);
    // scale-aware positivity floors (units differ by orders across the test
    // ladder: nondimensional shock tubes against dimensional wall flows)
    rhoFloor_ = 1e-6 * V.rho;
    pFloor_ = 1e-6 * V.p;
}

void DBNSSolver::initField(const std::vector<Primitive>& V) {
    double rhoMax = 0.0, pMax = 0.0;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        W_[ci] = GasState::toConserved(V[ci], eos_);
        rhoMax = std::max(rhoMax, V[ci].rho);
        pMax = std::max(pMax, V[ci].p);
    }
    rhoFloor_ = 1e-6 * rhoMax;
    pFloor_ = 1e-6 * pMax;
}

// --- properties: laminar viscosity (Sutherland) and SST eddy viscosity ------
void DBNSSolver::updateProperties() {
    double a1 = sstCoeffs_.a1, bStar = sstCoeffs_.betaStar;
    const auto& wallDist = wallDist_;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        Primitive V = GasState::toPrimitive(W_[ci], eos_);
        double T = GasState::temperature(V, eos_);
        muLam_[ci] = (settings_.constMu >= 0.0) ? settings_.constMu : eos_.viscosity(T);
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

// the Menter omega wall value seen through the ghost: face value
// omega_w = 60 nu_w / (beta1 dy1^2) with dy1 twice the owner-cell wall
// distance, so the near-wall gradients feel the singular growth the omega
// equation cannot generate on its own
double DBNSSolver::wallOmegaGhost(const Primitive& in, int faceId) const {
    if (!settings_.turbulent) return in.omega;
    const Face& f = mesh_.face(faceId);
    double delta = std::max(f.delta, 1e-12);
    double nuW = muLam_[f.owner] / std::max(in.rho, 1e-30);
    double dy1 = 2.0 * delta;
    double omegaWall = 60.0 * nuW / (sstCoeffs_.beta1 * dy1 * dy1);
    return std::max(2.0 * omegaWall - in.omega, omegaWall);
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
            // spatially varying prescribed state (incoming-layer profile or
            // an imposed-shock top boundary) when the patch carries one
            if (!spec.profile.empty()) {
                int j = patchLocalIdx_[boundaryIdx];
                if (j < (int)spec.profile.size()) return spec.profile[j];
            }
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
            g.u = 2.0 * spec.wallVelocity.x - in.u;   // face velocity = wallVelocity
            g.v = 2.0 * spec.wallVelocity.y - in.v;
            g.p = in.p; g.rho = in.rho;          // dT/dn = 0
            g.k = -in.k;                         // face k = 0
            g.omega = wallOmegaGhost(in, faceId);
            return g;
        }
        case BoundaryKind::NoSlipIsothermal: {
            g.u = 2.0 * spec.wallVelocity.x - in.u;
            g.v = 2.0 * spec.wallVelocity.y - in.v;
            double Tw = spec.wallTemp;
            if (!spec.wallTempProfile.empty()) {
                int j = patchLocalIdx_[boundaryIdx];
                if (j < (int)spec.wallTempProfile.size())
                    Tw = spec.wallTempProfile[j];
            }
            double Tin = GasState::temperature(in, eos_);
            double Tg = 2.0 * Tw - Tin;          // face T = Tw
            g.p = in.p; g.rho = g.p / (eos_.R * std::max(Tg, 1.0));
            g.k = -in.k;
            g.omega = wallOmegaGhost(in, faceId);
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
        // velocity gradient along the OUTWARD normal (cell value to wall
        // value over delta), matching the energy line's dTdn convention;
        // tau.n_out is then the force per area the wall exerts on the cell:
        // a stationary wall decelerates, a moving wall drags toward its
        // speed. (The inward-gradient form had the opposite sign, a latent
        // defect no committed case exercised: every prior viscous-wall rung
        // was gated on the implicit driver.)
        Vec3 Uw = spec.wallVelocity;
        Vec3 dUdn{(Uw.x - VP.u) / delta, (Uw.y - VP.v) / delta, 0.0};
        double dUdn_n = dUdn.dot(f.normal);
        // tau.n = mu_eff [ dU/dn + 1/3 (dU/dn . n) n ]
        Vec3 taun = (dUdn + f.normal * (dUdn_n / 3.0)) * muEff;
        res_[P][I_RHOU] -= taun.x * A;
        res_[P][I_RHOV] -= taun.y * A;
        // wall heat flux: q.n = -lambda dT/dn ; adiabatic -> 0
        if (spec.kind == BoundaryKind::NoSlipIsothermal) {
            double lamEff = eos_.Cp() * (muLam_[P] / eos_.Pr + muT_[P] / eos_.Pr_T);
            double Tin = GasState::temperature(VP, eos_);
            double Tw = spec.wallTemp;
            if (!spec.wallTempProfile.empty()) {
                int j = patchLocalIdx_[bidx];
                if (j < (int)spec.wallTempProfile.size())
                    Tw = spec.wallTempProfile[j];
            }
            double dTdn = (Tw - Tin) / delta;   // (T_wall - T_cell)/delta
            double qn = -lamEff * dTdn;
            res_[P][I_RHOE] -= -qn * A;   // energy viscous flux contribution
        }
        // turbulent transport wall fluxes (previously absent entirely, which
        // starved the buffer layer: no k sink to the wall and no omega feed
        // from its wall value, so k and omega rode to an over-mixed state
        // there). One-sided diffusion with k_wall = 0 and the Menter omega
        // wall value omega_w = 60 nu_w / (beta1 dy1^2), dy1 = 2 delta (the
        // first-cell height); together with the sublayer cell floor this is
        // the standard low-Reynolds omega wall treatment.
        if (settings_.turbulent) {
            double sk = sstCoeffs_.sigma_k1, sw = sstCoeffs_.sigma_w1;
            double nuW = muLam_[P] / VP.rho;
            double dy1 = 2.0 * delta;
            double omegaWall = 60.0 * nuW / (sstCoeffs_.beta1 * dy1 * dy1);
            double fluxK = (muLam_[P] + sk * muT_[P]) * (0.0 - VP.k) / delta;
            double fluxW = (muLam_[P] + sw * muT_[P])
                           * (omegaWall - VP.omega) / delta;
            res_[P][I_RHOK] -= fluxK * A;
            res_[P][I_RHOW] -= fluxW * A;
        }
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
    const auto& wallDist = wallDist_;
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
        Pk = std::min(Pk, 10.0 * bStar * rho * k * w);          // Menter 2003 limiter

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
        // omega production: alpha*rho*Pk_limited/muT = alpha*rho*min(S2, 10 bStar rho k w / muT)
        // (SST-2003 corrected form; alpha*rho*S2 is the paper's documented misprint, see the
        // NASA TMR SST page). Singular-safe: as muT -> 0 the min selects S2, the correct
        // limit; min(S2, lim) <= S2 so the corrected term is bounded by the old one.
        double lim = 10.0 * bStar * rho * k * w / std::max(muT, 1e-30);
        double PwSpec = std::min(S2, lim);
        if (!limiterActive_.empty())
            limiterActive_[ci] = (lim < S2) ? 1 : 0;
        double Pw = alpha * rho * PwSpec;                       // omega production
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

    if (!bTarget6_.empty()) addInjectionFluxes();

    addTurbulenceSources();

    // manufactured-solution source: dW/dt = -(res - S V)/V
    if (!mmsSource_.empty())
        for (int ci = 0; ci < nc; ++ci) {
            double Vol = vol_[ci];
            for (int i = 0; i < NVAR; ++i) res_[ci][i] -= mmsSource_[ci][i] * Vol;
        }
}

// --- model-form injection (the deferred-correction coupling) -----------------
void DBNSSolver::setTargetCorrection(const std::vector<double>& b6,
                                     const std::vector<double>& dq2,
                                     bool energyReach,
                                     const std::vector<std::uint8_t>& mask) {
    injMask_ = mask;
    int nc = mesh_.nCells();
    if ((int)b6.size() != 6 * nc)
        throw std::runtime_error("setTargetCorrection: b6 must be nCells*6");
    if (!dq2.empty() && (int)dq2.size() != 2 * nc)
        throw std::runtime_error("setTargetCorrection: dq2 must be nCells*2 "
                                 "or empty");
    bTarget6_ = b6;
    dqTarget2_ = dq2;
    injectEnergyReach_ = energyReach;
    injDiag_ = InjectionDiagnostics{};
    injDiag_.active = true;
    for (double v : bTarget6_)
        injDiag_.maxDb = std::max(injDiag_.maxDb, std::abs(v));
    for (double v : dqTarget2_)
        injDiag_.maxDq = std::max(injDiag_.maxDq, std::abs(v));
}

void DBNSSolver::clearTargetCorrection() {
    bTarget6_.clear();
    dqTarget2_.clear();
    injMask_.clear();
    injDiag_ = InjectionDiagnostics{};
}

// Deferred-correction fluxes of the sampled closure, mirroring the
// incompressible injection and extending it to the energy equation (the
// heat-flux reach the pre-registered scheme names as the compressible
// novelty). Per cell the injected turbulent stress difference is
//   tau_inj = -(2 <rho> k b_target + 2 mu_t dev(S)),
// the difference between the target deviatoric Favre stress and the
// Boussinesq deviatoric stress the mu_t diffusion already carries (running
// rho, k, mu_t and strain, so a target equal to the solver's own Boussinesq
// anisotropy reproduces the baseline solve identically). It is assembled as
// a conservative internal-face flux (distance-weighted face values); wall
// and open boundary faces carry no injected flux (the near-wall rows have
// k ~ 0, the same negligible-contribution note as the incompressible
// scheme). With the energy reach on, the consistent stress work
// (tau_inj . u) . n and the injected turbulent heat flux
//   q_inj = <rho> cp dq_target
// (the deferred correction against the implicit gradient-diffusion term)
// enter the energy flux. The barycentric margin of the target is re-checked
// once per residual evaluation and violations are recorded, never silently
// projected (the caller projects before setting).
void DBNSSolver::addInjectionFluxes() {
    int nc = mesh_.nCells();
    injDiag_.checkedIters += 1;
    const bool masked = (int)injMask_.size() == nc;
    for (int ci = 0; ci < nc; ++ci) {
        // realizability is a property of ACTIVE targets; outside the
        // injection mask the stored rows are the raw baseline placeholder
        // and carry no injected flux (skipped below), so they are not
        // checked (the review's convention item)
        if (masked && !injMask_[ci]) continue;
        double margin = aniso::barycentricMargin(bTarget6_.data() + 6 * ci);
        if (margin < -1e-9) {
            injDiag_.allRealizable = false;
            injDiag_.maxViolation = std::max(injDiag_.maxViolation, -margin);
        }
    }

    // per-cell injected stress difference tau_inj (xx, yy, xy needed for the
    // 2-D fluxes; zz never crosses a face) and heat-flux vector rho cp dq
    std::vector<double> txx(nc), tyy(nc), txy(nc), qx(nc, 0.0), qy(nc, 0.0);
    bool withHeat = injectEnergyReach_ && !dqTarget2_.empty();
    double cp = eos_.Cp();
    for (int ci = 0; ci < nc; ++ci) {
        Primitive V = GasState::toPrimitive(W_[ci], eos_);
        double rho = V.rho;
        double k = std::max(V.k, 0.0);
        double muT = muT_[ci];
        double dudx = grad_[ci][G_U].x, dudy = grad_[ci][G_U].y;
        double dvdx = grad_[ci][G_V].x, dvdy = grad_[ci][G_V].y;
        double div3 = (dudx + dvdy) / 3.0;      // trace/3 of the 2-D strain
        double devSxx = dudx - div3;
        double devSyy = dvdy - div3;
        double Sxy = 0.5 * (dudy + dvdx);
        const double* bt = bTarget6_.data() + 6 * ci;
        // b ordered xx, yy, zz, xy, xz, yz; only xx, yy, xy enter the fluxes
        txx[ci] = -(2.0 * rho * k * bt[0] + 2.0 * muT * devSxx);
        tyy[ci] = -(2.0 * rho * k * bt[1] + 2.0 * muT * devSyy);
        txy[ci] = -(2.0 * rho * k * bt[3] + 2.0 * muT * Sxy);
        if (withHeat) {
            qx[ci] = rho * cp * dqTarget2_[2 * ci];
            qy[ci] = rho * cp * dqTarget2_[2 * ci + 1];
        }
    }

    int nIF = mesh_.nInternalFaces();
    for (int fi = 0; fi < nIF; ++fi) {
        const Face& f = mesh_.face(fi);
        int P = f.owner, N = f.neighbor;
        // the solver-side injection mask: correction fluxes exist only on
        // faces INTERIOR to the active region (both cells active). A flux
        // either crosses a face fully or not at all, so the correction ends
        // conservatively at the mask boundary and cells outside the training
        // support receive no injection as the member state evolves (the
        // python-side baseline placeholder only cancelled at the initial
        // state; this gate makes the inactivity exact for all iterations).
        if (masked && (!injMask_[P] || !injMask_[N])) continue;
        double w = f.weight;
        double fxx = w * txx[P] + (1.0 - w) * txx[N];
        double fyy = w * tyy[P] + (1.0 - w) * tyy[N];
        double fxy = w * txy[P] + (1.0 - w) * txy[N];
        double nx = f.normal.x, ny = f.normal.y, A = f.area;
        StateVec Fv{};
        Fv[I_RHOU] = fxx * nx + fxy * ny;
        Fv[I_RHOV] = fxy * nx + fyy * ny;
        if (injectEnergyReach_) {
            Primitive VP = GasState::toPrimitive(W_[P], eos_);
            Primitive VN = GasState::toPrimitive(W_[N], eos_);
            double uF = 0.5 * (VP.u + VN.u), vF = 0.5 * (VP.v + VN.v);
            Fv[I_RHOE] = (fxx * uF + fxy * vF) * nx
                       + (fxy * uF + fyy * vF) * ny;
            if (withHeat) {
                double qnx = w * qx[P] + (1.0 - w) * qx[N];
                double qny = w * qy[P] + (1.0 - w) * qy[N];
                Fv[I_RHOE] -= qnx * nx + qny * ny;
            }
        }
        for (int i = 0; i < NVAR; ++i) {
            res_[P][i] -= Fv[i] * A;    // same sign convention as the viscous flux
            res_[N][i] += Fv[i] * A;
        }
    }
}

void DBNSSolver::computeTimeStep(double cfl, bool includeViscous) {
    // includeViscous: the explicit march must respect the viscous stability
    // limit (spectral radius ~ 1/dy^2, the documented wall-clustered-mesh
    // blocker); the implicit march drops it from the STEP SIZE and absorbs
    // the stiffness in its diagonal instead, which is the whole point of the
    // implicit driver (a viscous-limited pseudo-time step would freeze the
    // wall cells at any Courant number).
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
            if (settings_.viscous && includeViscous) {
                double muEff = muLam_[ci] + muT_[ci];
                double diff = std::max(4.0 / 3.0, eos_.gamma / eos_.Pr) * muEff / V.rho;
                lamV += diff * f.area * f.area / std::max(Vol, 1e-30);
            }
        }
        dtCell_[ci] = cfl * Vol / std::max(lamC + 2.0 * lamV, 1e-30);
    }
}

// face spectral radii for the LU-SGS diagonal and off-diagonal relaxation:
// convective |u.n| + a from the larger side, viscous mirroring the explicit
// stability estimate (diff A / V), both entering as lam with velocity units
// so that lam * A matches the V/dt aggregate.
void DBNSSolver::computeFaceSpectralRadii() {
    int nF = mesh_.nFaces();
    if ((int)lamFace_.size() != nF) lamFace_.assign(nF, 0.0);
    double diffFac = std::max(4.0 / 3.0, eos_.gamma / eos_.Pr);
    for (int fi = 0; fi < nF; ++fi) {
        const Face& f = mesh_.face(fi);
        Primitive VP = GasState::toPrimitive(W_[f.owner], eos_);
        double unP = VP.u * f.normal.x + VP.v * f.normal.y;
        double lam = std::abs(unP) + GasState::soundSpeed(VP, eos_);
        double nuP = (muLam_[f.owner] + muT_[f.owner]) / VP.rho;
        double volP = vol_[f.owner];
        double nu = nuP, vol = volP;
        if (f.neighbor >= 0) {
            Primitive VN = GasState::toPrimitive(W_[f.neighbor], eos_);
            double unN = VN.u * f.normal.x + VN.v * f.normal.y;
            lam = std::max(lam, std::abs(unN) + GasState::soundSpeed(VN, eos_));
            double nuN = (muLam_[f.neighbor] + muT_[f.neighbor]) / VN.rho;
            nu = 0.5 * (nuP + nuN);
            vol = 0.5 * (volP + vol_[f.neighbor]);
        }
        if (settings_.viscous)
            lam += 2.0 * diffFac * nu * f.area / std::max(vol, 1e-30);
        lamFace_[fi] = lam;
    }
}

std::array<double, NVAR> DBNSSolver::equationResidualNorms() const {
    std::array<double, NVAR> s{};
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        for (int v = 0; v < NVAR; ++v) {
            double r = res_[ci][v] / std::max(vol_[ci], 1e-30);
            s[v] += r * r;
        }
    for (int v = 0; v < NVAR; ++v)
        s[v] = std::sqrt(s[v] / std::max(1, mesh_.nCells()));
    return s;
}

bool DBNSSolver::stateIsValid() const {
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        for (int v = 0; v < NVAR; ++v)
            if (!std::isfinite(W_[ci][v])) return false;
        if (!GasState::admissible(W_[ci], eos_)) return false;
    }
    return true;
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
    if (settings_.timeMode == TimeMode::Steady && settings_.implicitSteady)
        return solveImplicitSteady();

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
        computeTimeStep(settings_.cfl, true);
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

// --- implicit (LU-SGS) steady march ------------------------------------------
// Backward-Euler pseudo-time on the second-order residual:
//   (V/dt) dW + dRes(dW) = -Res(W^n),
// with the face-flux Jacobian replaced by its spectral-radius-split
// first-order (Rusanov) form (Yoon and Jameson 1988). Per face between i and
// its neighbor j (outward normal n from i):
//   dRes_i ~= 0.5 A [ dFn(W_i) + lam I ] dW_i + 0.5 A [ dFn(W_j) - lam I ] dW_j,
// the diagonal bounded by its spectral radius (drop dFn(W_i), keep lam), the
// off-diagonal applied matrix-free as Fn(W_j + dW_j) - Fn(W_j). Two sweeps:
//   forward   dW*_i = D^-1 ( -Res_i - sum_{j<i} c_ij(dW*_j) )
//   backward  dW_i  = dW*_i - D^-1 sum_{j>i} c_ij(dW_j)
// with c_ij(dW_j) = 0.5 A ( Fn(W_j+dW_j) - Fn(W_j) - lam dW_j ) and
//   D_i = V/dt + 0.5 sum_f lam_f A_f  (+ the point-implicit SST destruction
// on the k and omega rows: d(beta* rho k w)/d(rho k) = beta* w and
// d(beta rho w^2)/d(rho w) = 2 beta w, with beta bounded by max(beta1, beta2)
// so the diagonal over-estimates the blended value, which only stabilizes).
// Boundary faces couple to no unknown neighbor; their lam enters the diagonal.
SolveReport DBNSSolver::solveImplicitSteady() {
    SolveReport rep;
    int nc = mesh_.nCells();
    double res0 = -1.0;
    std::array<double, NVAR> eqRes0{};
    if ((int)dW_.size() != nc) dW_.assign(nc, StateVec{});

    double betaMax = std::max(sstCoeffs_.beta1, sstCoeffs_.beta2);
    double bStar = sstCoeffs_.betaStar;

    // matrix-free off-diagonal action of neighbor `other` on cell ci
    auto neighborTerm = [&](int ci, int other, FaceID fi) -> StateVec {
        const Face& f = mesh_.face(fi);
        double sgn = (f.owner == ci) ? 1.0 : -1.0;
        double nx = f.normal.x * sgn, ny = f.normal.y * sgn;
        double A = f.area, lam = lamFace_[fi];
        StateVec Wp;
        for (int i = 0; i < NVAR; ++i) Wp[i] = W_[other][i] + dW_[other][i];
        StateVec out{};
        if (GasState::admissible(Wp, eos_)) {
            Primitive V0 = GasState::toPrimitive(W_[other], eos_);
            Primitive V1 = GasState::toPrimitive(Wp, eos_);
            StateVec F0 = GasState::normalFlux(V0, nx, ny, eos_);
            StateVec F1 = GasState::normalFlux(V1, nx, ny, eos_);
            for (int i = 0; i < NVAR; ++i)
                out[i] = 0.5 * A * (F1[i] - F0[i] - lam * dW_[other][i]);
        } else {
            // inadmissible perturbed state: keep only the relaxation part
            for (int i = 0; i < NVAR; ++i)
                out[i] = -0.5 * A * lam * dW_[other][i];
        }
        return out;
    };

    std::vector<StateVec> D(nc);
    std::vector<StateVec> Wsave(nc);
    // step-rejection backoff: an inadmissible update is rejected (state
    // restored) and the CFL scaled down; it recovers multiplicatively on
    // success. Persistent rejection classifies as Diverged.
    double cflScale = 1.0;
    int consecutiveRejects = 0;
    // the uniform initial state has a machine-zero density residual, so the
    // relative-convergence normalizer is the residual maximum over the ramp
    // (frozen afterwards), not the first iterate
    int freezeIter = std::max(settings_.cflRampIters, 50);
    // the hard (eps = 0) Venkatakrishnan limiter chatters at steady state and
    // stalls deep convergence; freezing it once the residual has dropped two
    // orders is the standard remedy and changes no converged answer
    bool limiterFrozen = false;
    int iter = 0;
    for (; iter < settings_.maxIterations; ++iter) {
        // linear CFL ramp: low while the transient is strong, then the target
        double frac = (settings_.cflRampIters > 0)
            ? std::min(1.0, double(iter) / settings_.cflRampIters) : 1.0;
        double cfl = cflScale * (settings_.cflRampStart
                   + frac * (settings_.cflImplicit - settings_.cflRampStart));

        updateProperties();
        computeTimeStep(cfl, false);   // convective step; viscosity in the diagonal
        computeGradients();
        if (!limiterFrozen) computeLimiters();
        computeResidual();
        computeFaceSpectralRadii();

        for (int ci = 0; ci < nc; ++ci) {
            double sumLam = 0.0;
            for (FaceID fi : mesh_.cell(ci).faces) {
                const Face& f = mesh_.face(fi);
                // boundary ghosts move with the interior state, so their
                // Jacobian falls fully on this cell: full weight there,
                // half on internal faces (the split Rusanov diagonal)
                double weight = (f.neighbor < 0) ? 1.0 : 0.5;
                sumLam += weight * lamFace_[fi] * f.area;
            }
            double base = vol_[ci] / std::max(dtCell_[ci], 1e-30)
                        + sumLam;
            for (int i = 0; i < NVAR; ++i) D[ci][i] = base;
            if (settings_.turbulent) {
                Primitive V = GasState::toPrimitive(W_[ci], eos_);
                double w = std::max(V.omega, settings_.omegaFloor);
                D[ci][I_RHOK] += vol_[ci] * bStar * w;
                D[ci][I_RHOW] += vol_[ci] * 2.0 * betaMax * w;
            }
        }

        // forward sweep (lower neighbors already updated this sweep)
        for (int ci = 0; ci < nc; ++ci) {
            StateVec rhs;
            for (int i = 0; i < NVAR; ++i) rhs[i] = -res_[ci][i];
            for (FaceID fi : mesh_.cell(ci).faces) {
                const Face& f = mesh_.face(fi);
                if (f.neighbor < 0) continue;
                int other = (f.owner == ci) ? f.neighbor : f.owner;
                if (other >= ci) continue;
                StateVec c = neighborTerm(ci, other, fi);
                for (int i = 0; i < NVAR; ++i) rhs[i] -= c[i];
            }
            for (int i = 0; i < NVAR; ++i) dW_[ci][i] = rhs[i] / D[ci][i];
        }
        // backward sweep (upper neighbors carry their final increments)
        for (int ci = nc - 1; ci >= 0; --ci) {
            StateVec corr{};
            for (FaceID fi : mesh_.cell(ci).faces) {
                const Face& f = mesh_.face(fi);
                if (f.neighbor < 0) continue;
                int other = (f.owner == ci) ? f.neighbor : f.owner;
                if (other <= ci) continue;
                StateVec c = neighborTerm(ci, other, fi);
                for (int i = 0; i < NVAR; ++i) corr[i] += c[i];
            }
            for (int i = 0; i < NVAR; ++i) dW_[ci][i] -= corr[i] / D[ci][i];
        }

        Wsave = W_;
        for (int ci = 0; ci < nc; ++ci)
            for (int i = 0; i < NVAR; ++i) W_[ci][i] += dW_[ci][i];

        bool admissible = true;
        for (int ci = 0; ci < nc; ++ci)
            if (!GasState::admissible(W_[ci], eos_)) { admissible = false; break; }
        if (!admissible) {
            W_ = Wsave;                     // reject the step, back the CFL off
            cflScale *= 0.5;
            if (++consecutiveRejects > 20) {
                rep.status = EvaluationStatus::Diverged;
                rep.iterations = iter;
                return rep;
            }
            continue;
        }
        consecutiveRejects = 0;
        cflScale = std::min(1.0, cflScale * 1.5);
        clampPositivity();

        const std::array<double, NVAR> rns = equationResidualNorms();
        double rn = rns[I_RHO];
        const int nEq = settings_.turbulent ? NVAR : 4;
        if (iter <= freezeIter || res0 < 0.0) {
            res0 = std::max(res0, std::max(rn, 1e-30));
            for (int v = 0; v < nEq; ++v)
                eqRes0[v] = std::max(eqRes0[v], std::max(rns[v], 1e-30));
        }
        // an equation whose reference residual is negligible against the
        // system scale (a quiescent direction) must not gate convergence on
        // round-off; its reference is floored at 1e-6 of the largest
        double eqRefMax = 1e-30;
        for (int v = 0; v < nEq; ++v) eqRefMax = std::max(eqRefMax, eqRes0[v]);
        double relMax = 0.0;
        for (int v = 0; v < nEq; ++v)
            relMax = std::max(relMax,
                              rns[v] / std::max(eqRes0[v], 1e-6 * eqRefMax));
        if (settings_.verbose && (iter % settings_.reportInterval == 0))
            std::printf("  [lusgs] iter %6d  cfl %.1f  res_rho %.3e (rel %.3e)"
                        "  rel_max %.3e\n",
                        iter, cfl, rn, rn / res0, relMax);
        if (iter % settings_.reportInterval == 0)
            rep.residualHistory.push_back(rn);
        if (iter > freezeIter && !limiterFrozen && rn / res0 < 1e-2)
            limiterFrozen = true;
        // convergence requires EVERY live equation's relative decay below the
        // tolerance AND a directly validated state (the per-cell sweep; the
        // acceptance logic alone cannot prove finiteness)
        if (iter > freezeIter && relMax < settings_.convergenceTol) {
            if (stateIsValid()) {
                rep.status = EvaluationStatus::Converged;
            } else {
                rep.status = EvaluationStatus::Diverged;
                rep.iterations = iter;
                return rep;
            }
            break;
        }
    }
    if (rep.status == EvaluationStatus::Unknown)
        rep.status = EvaluationStatus::Unconverged;
    rep.iterations = iter;
    {
        const std::array<double, NVAR> rnsF = equationResidualNorms();
        const int nEq = settings_.turbulent ? NVAR : 4;
        double eqRefMax = 1e-30, relMax = 0.0;
        for (int v = 0; v < nEq; ++v) eqRefMax = std::max(eqRefMax, eqRes0[v]);
        for (int v = 0; v < nEq; ++v)
            relMax = std::max(relMax,
                              rnsF[v] / std::max(eqRes0[v], 1e-6 * eqRefMax));
        rep.finalResidual = relMax;
    }
    return rep;
}

// floor density / pressure / turbulence to keep states physical; then re-pin
// the wall-adjacent omega. The validated incompressible treatment OVERRIDES
// the first cell's omega every iteration with the Menter value
// 60 nu / (beta1 y1^2) (ten times the analytic sublayer solution at the
// first-cell scale): that deliberately strong anchor feeds the diffusive
// omega tail that selects the log-law branch. A floor at the analytic value
// alone leaves a second, spurious low-shear/high-mixing near-wall
// equilibrium available, which the interaction baseline bring-up measured
// as a mesh-converged 27 percent skin-friction excess.
void DBNSSolver::clampPositivity() {
    if (settings_.turbulent) {
        for (size_t idx = 0; idx < wallAdjCell_.size(); ++idx) {
            int ci = wallAdjCell_[idx];
            double y1 = wallAdjDelta_[idx];
            Primitive V = GasState::toPrimitive(W_[ci], eos_);
            double nu = muLam_[ci] / std::max(V.rho, rhoFloor_);
            V.omega = 60.0 * nu / (sstCoeffs_.beta1 * y1 * y1);
            W_[ci] = GasState::toConserved(V, eos_);
        }
    }
    clampPositivityCore();
}

void DBNSSolver::clampPositivityCore() {
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        StateVec& W = W_[ci];
        if (W[I_RHO] < rhoFloor_) W[I_RHO] = rhoFloor_;
        Primitive V = GasState::toPrimitive(W, eos_);
        bool fix = false;
        if (!(V.p > pFloor_)) { V.p = pFloor_; fix = true; }
        if (settings_.turbulent) {
            if (V.k < settings_.kFloor) { V.k = settings_.kFloor; fix = true; }
            if (V.omega < settings_.omegaFloor) { V.omega = settings_.omegaFloor; fix = true; }
            // omega wall anchoring: the SST omega equation needs its singular
            // near-wall value, and the wall ghost is zero-gradient (the
            // boundary comment promised a BC no code implemented), so the
            // viscous-sublayer analytic solution omega = 6 nu / (beta1 y^2)
            // is imposed as a floor. It decays as 1/y^2 and binds only inside
            // the sublayer (the automatic-wall-treatment limit); without it
            // an imposed turbulent layer LAMINARIZES, which the interaction
            // baseline bring-up measured as a laminar-like Cf decay.
            const auto& wd = wallDist_;
            if (ci < (int)wd.size()) {
                double y = std::max(wd[ci], 1e-12);
                double nu = muLam_[ci] / std::max(V.rho, rhoFloor_);
                double omegaVis = 6.0 * nu / (sstCoeffs_.beta1 * y * y);
                if (V.omega < omegaVis) { V.omega = omegaVis; fix = true; }
            }
        } else { V.k = 0.0; V.omega = 0.0; }
        if (fix) W_[ci] = GasState::toConserved(V, eos_);
        else { W_[ci][I_RHOK] = V.rho * V.k; W_[ci][I_RHOW] = V.rho * V.omega; }
    }
}

}  // namespace dbns
