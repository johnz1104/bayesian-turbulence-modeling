#include "CompressibleSIMPLESolver.hpp"
#include "LinearSolver.hpp"
#include "BoundaryCondition.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

// ─── Constructor ──────────────────────────────────────────────────────────────

CompressibleSIMPLESolver::CompressibleSIMPLESolver(
    const Mesh& mesh, const SSTModel& sst,
    const CompressibleBoundaryConditions& bcs,
    const IdealGasEOS& eos, const SolverSettings& settings)
    : mesh_(mesh), sst_(sst), bcs_(bcs), eos_(eos), settings_(settings),
      aP_(mesh.nCells(), 0.0), aPunrelaxed_(mesh.nCells(), 0.0),
      mu_(mesh.nCells(), eos.mu_ref)
{
    pSolver_ = makeSolver(settings_.pressureSolver);
    mSolver_ = makeSolver(settings_.momentumSolver);
    tSolver_ = makeSolver(settings_.turbulenceSolver);
    eSolver_ = makeSolver("BiCGSTAB");  // energy equation
}

// ─── Initialisation ───────────────────────────────────────────────────────────

void CompressibleSIMPLESolver::initUniform(CompressibleFlowFields& f,
                                            const Vec3& Uinit, double p_init,
                                            double T_init,
                                            double kInit, double omegaInit)
{
    f.U.setUniform(Uinit);
    f.p.setUniform(p_init);
    f.T.setUniform(T_init);
    f.k.setUniform(kInit);
    f.omega.setUniform(omegaInit);
    f.F1.setUniform(1.0);
    f.F2.setUniform(1.0);
    f.Pk.setUniform(0.0);
    f.CDkw.setUniform(0.0);

    // Density from EOS
    const double rho0 = eos_.density(p_init, T_init);
    f.rho.setUniform(rho0);

    // Sutherland viscosity at T_init
    const double mu0 = eos_.viscosity(T_init);
    for (int ci = 0; ci < mesh_.nCells(); ++ci) mu_[ci] = mu0;

    // Wall-distance omega profile (same logic as incompressible)
    const auto& wd = mesh_.wallDistance();
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double y = std::max(wd[ci], 1e-20);
        double nu0 = mu0 / rho0;
        double omWall = 60.0 * nu0 / (sst_.coeffs.beta1 * y * y);
        f.omega[ci] = std::max(omWall, omegaInit);
    }

    // Initial nuT
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        f.nuT[ci] = sst_.coeffs.a1 * kInit / std::max(sst_.coeffs.a1 * f.omega[ci], 1e-20);

    // Apply BCs
    double nu0 = mu0 / rho0;
    applyAllCompressibleBCs(f.U, f.p, f.T, f.k, f.omega, mesh_, bcs_, nu0);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

void CompressibleSIMPLESolver::updateViscosity(const CompressibleFlowFields& f) {
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        mu_[ci] = eos_.viscosity(f.T[ci]);
}

void CompressibleSIMPLESolver::updateDensity(CompressibleFlowFields& f) {
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        f.rho[ci] = eos_.density(f.p[ci], f.T[ci]);
}

// ─── Momentum equation ────────────────────────────────────────────────────────
// ∇·(ρ U U) = -∇p + ∇·(μ_eff ∇U)
// Convection uses density-weighted face flux: ṁ = ρ_f (U_f·S_f)

void CompressibleSIMPLESolver::assembleMomentum(LinearSystem& sys,
                                                 const CompressibleFlowFields& f,
                                                 int component,
                                                 std::vector<double>& aP)
{
    sys.zero();
    int nIF = mesh_.nInternalFaces();

    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner, n = face.neighbor;
        double Sf    = face.area;
        double delta = std::max(face.delta, 1e-20);

        // Density and dynamic viscosity at face (linear interpolation)
        double rho_f = face.weight * f.rho[o] + (1.0 - face.weight) * f.rho[n];
        double mu_f  = face.weight * mu_[o]  + (1.0 - face.weight) * mu_[n];
        double nuT_f = face.weight * f.nuT[o] + (1.0 - face.weight) * f.nuT[n];
        double muEff_f = mu_f + rho_f * nuT_f;   // total effective dynamic viscosity

        // Diffusion coefficient (units: Pa·s · m / m = Pa = N/m²)
        double Df = muEff_f * Sf / delta;

        // Density-weighted convective mass flux ṁ = ρ_f (U_f·S_f)
        Vec3   Uf    = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double mFlux = rho_f * (Uf.x * face.normal.x
                               + Uf.y * face.normal.y
                               + Uf.z * face.normal.z) * Sf;

        double cPos = std::max( mFlux, 0.0);
        double cNeg = std::max(-mFlux, 0.0);

        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);
        sys.lower[fi] = -(Df + cPos);
    }

    // Boundary faces
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face& face = mesh_.face(fi);
        int o     = face.owner;
        double Sf = face.area, delta = std::max(face.delta, 1e-20);
        double mu_b  = mu_[o];
        double nuT_b = f.nuT[o];
        double rho_b = f.rho[o];
        double muEff = mu_b + rho_b * nuT_b;
        double Db    = muEff * Sf / delta;

        double Ub_comp;
        if (component == 0) Ub_comp = f.U.bface(fi).x;
        else if (component == 1) Ub_comp = f.U.bface(fi).y;
        else Ub_comp = f.U.bface(fi).z;

        Vec3   Ubv   = f.U.bface(fi);
        double mFlux = rho_b * (Ubv.x * face.normal.x
                               + Ubv.y * face.normal.y
                               + Ubv.z * face.normal.z) * Sf;

        if (mFlux >= 0) {
            sys.diag[o]   += mFlux + Db;
            sys.source[o] += Db * Ub_comp;
        } else {
            sys.diag[o]   += Db;
            sys.source[o] += (Db - mFlux) * Ub_comp;
        }
    }

    // Pressure gradient source: -∂p/∂x_k · V
    VectorField gradP = greenGaussGrad(f.p);
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol = mesh_.cell(ci).volume;
        double dpdx = (component == 0) ? gradP[ci].x
                    : (component == 1) ? gradP[ci].y
                                       : gradP[ci].z;
        sys.source[ci] -= dpdx * vol;
    }

    // Under-relaxation
    const double alphaU = settings_.alphaU;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double phi_old = (component == 0) ? f.U[ci].x
                       : (component == 1) ? f.U[ci].y
                                          : f.U[ci].z;
        aPunrelaxed_[ci] = sys.diag[ci];
        sys.source[ci] += (1.0 - alphaU) / alphaU * sys.diag[ci] * phi_old;
        sys.diag[ci]   /= alphaU;
        aP[ci] = sys.diag[ci];
    }
}

// ─── Pressure-correction equation ─────────────────────────────────────────────
// Continuity: ∑_f (ρ_f U_f · S_f) = 0
// Correction: ρ = p/(RT) → ρ' ≈ p'/(RT) at constant T
// Velocity:   U' = -(V/aP) ∇p'
// → Poisson:  ∑_f (ρ_f V_f / aP_f) / delta * (p'_P - p'_F) = -∑_f ṁ*_f

void CompressibleSIMPLESolver::assemblePressureCorrection(
    LinearSystem& sys, const CompressibleFlowFields& f,
    const std::vector<double>& aP, ScalarField& pPrime)
{
    sys.zero();
    pPrime.setUniform(0.0);
    int nIF = mesh_.nInternalFaces();

    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner, n = face.neighbor;
        double Sf    = face.area;
        double delta = std::max(face.delta, 1e-20);

        double Vo = mesh_.cell(o).volume;
        double Vn = mesh_.cell(n).volume;
        double rho_f = face.weight * f.rho[o] + (1.0 - face.weight) * f.rho[n];

        double dP_o = (std::abs(aP[o]) > 1e-30) ? f.rho[o] * Vo / aP[o] : 0.0;
        double dP_n = (std::abs(aP[n]) > 1e-30) ? f.rho[n] * Vn / aP[n] : 0.0;
        double dP_f = face.weight * dP_o + (1.0 - face.weight) * dP_n;

        double coeff = dP_f * Sf / delta;

        sys.diag[o] += coeff;
        sys.diag[n] += coeff;
        sys.upper[fi] = -coeff;
        sys.lower[fi] = -coeff;

        // Density-weighted mass flux source -∑ ρ_f U*·S_f
        Vec3   Uf    = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double mFlux = rho_f * (Uf.x * face.normal.x
                               + Uf.y * face.normal.y
                               + Uf.z * face.normal.z) * Sf;
        sys.source[o] -= mFlux;
        sys.source[n] += mFlux;
    }

    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch& pat   = mesh_.patch(pi);
        bool isOutlet = (pat.type == "outlet");
        for (FaceID fi : pat.faces) {
            const Face& face = mesh_.face(fi);
            int o = face.owner;
            double Sf    = face.area;
            double delta = std::max(face.delta, 1e-20);

            Vec3   Ub    = f.U.bface(fi);
            double mFlux = f.rho[o] * (Ub.x * face.normal.x
                                      + Ub.y * face.normal.y
                                      + Ub.z * face.normal.z) * Sf;
            sys.source[o] -= mFlux;

            if (isOutlet) {
                double Vo    = mesh_.cell(o).volume;
                double dP_o  = (std::abs(aP[o]) > 1e-30) ? f.rho[o] * Vo / aP[o] : 0.0;
                double coeff = dP_o * Sf / delta;
                sys.diag[o] += coeff;
            }
        }
    }
}

// ─── Energy equation ──────────────────────────────────────────────────────────
// Steady: ∇·(ρ U Cp T) = ∇·(λ_eff ∇T)
// λ_eff = μ Cp/Pr + μ_T Cp/Pr_T

void CompressibleSIMPLESolver::assembleEnergy(LinearSystem& sys,
                                               const CompressibleFlowFields& f)
{
    sys.zero();
    int nIF = mesh_.nInternalFaces();

    const double Cp   = eos_.Cp();

    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner, n = face.neighbor;
        double Sf    = face.area;
        double delta = std::max(face.delta, 1e-20);

        double rho_f = face.weight * f.rho[o]   + (1.0 - face.weight) * f.rho[n];
        double mu_f  = face.weight * mu_[o]     + (1.0 - face.weight) * mu_[n];
        double nuT_f = face.weight * f.nuT[o]   + (1.0 - face.weight) * f.nuT[n];
        double muT_f = rho_f * nuT_f;

        double lambda_f = eos_.conductivity(mu_f, muT_f);

        // Diffusion coefficient: λ_eff * Sf / delta
        double Df = lambda_f * Sf / delta;

        // Convection mass flux (same density-weighted flux as momentum)
        Vec3   Uf    = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double mFlux = rho_f * Cp * (Uf.x * face.normal.x
                                    + Uf.y * face.normal.y
                                    + Uf.z * face.normal.z) * Sf;

        double cPos = std::max( mFlux, 0.0);
        double cNeg = std::max(-mFlux, 0.0);

        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);
        sys.lower[fi] = -(Df + cPos);
    }

    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face& face = mesh_.face(fi);
        int o     = face.owner;
        double Sf = face.area, delta = std::max(face.delta, 1e-20);

        double mu_o  = mu_[o];
        double muT_o = f.rho[o] * f.nuT[o];
        double lam   = eos_.conductivity(mu_o, muT_o);
        double Db    = lam * Sf / delta;

        double Tb    = f.T.bface(fi);
        Vec3   Ubv   = f.U.bface(fi);
        double mFlux = f.rho[o] * Cp * (Ubv.x * face.normal.x
                                        + Ubv.y * face.normal.y
                                        + Ubv.z * face.normal.z) * Sf;

        if (mFlux >= 0) {
            sys.diag[o]   += mFlux + Db;
            sys.source[o] += Db * Tb;
        } else {
            sys.diag[o]   += Db;
            sys.source[o] += (Db - mFlux) * Tb;
        }
    }

    // Under-relaxation for temperature (dedicated alphaT, default 0.7)
    const double alphaT = settings_.alphaT;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        sys.source[ci] += (1.0 - alphaT) / alphaT * sys.diag[ci] * f.T[ci];
        sys.diag[ci]   /= alphaT;
    }
}

// ─── Turbulence: k equation ───────────────────────────────────────────────────
// ∇·(ρ U k) = ∇·((μ + ρ σ_k νT) ∇k) + ρ Pk − ρ β* ω k

void CompressibleSIMPLESolver::assembleKEquation(LinearSystem& sys,
                                                  const CompressibleFlowFields& f)
{
    sys.zero();
    int nIF = mesh_.nInternalFaces();

    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner, n = face.neighbor;
        double Sf    = face.area;
        double delta = std::max(face.delta, 1e-20);

        double F1_f  = face.weight * f.F1[o] + (1.0 - face.weight) * f.F1[n];
        double sk    = sst_.coeffs.sigma_k(F1_f);
        double rho_f = face.weight * f.rho[o] + (1.0 - face.weight) * f.rho[n];
        double mu_f  = face.weight * mu_[o]   + (1.0 - face.weight) * mu_[n];
        double nuT_f = face.weight * f.nuT[o] + (1.0 - face.weight) * f.nuT[n];
        double muEff = mu_f + rho_f * sk * nuT_f;
        double Df    = muEff * Sf / delta;

        Vec3   Uf    = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double mFlux = rho_f * (Uf.x * face.normal.x
                               + Uf.y * face.normal.y
                               + Uf.z * face.normal.z) * Sf;
        double cPos = std::max( mFlux, 0.0);
        double cNeg = std::max(-mFlux, 0.0);

        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);
        sys.lower[fi] = -(Df + cPos);
    }

    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face& face = mesh_.face(fi);
        int o     = face.owner;
        double Sf = face.area, delta = std::max(face.delta, 1e-20);
        double sk    = sst_.coeffs.sigma_k(f.F1[o]);
        double muEff = mu_[o] + f.rho[o] * sk * f.nuT[o];
        double Db    = muEff * Sf / delta;

        double kb    = f.k.bface(fi);
        Vec3   Ubv   = f.U.bface(fi);
        double mFlux = f.rho[o] * (Ubv.x * face.normal.x
                                  + Ubv.y * face.normal.y
                                  + Ubv.z * face.normal.z) * Sf;
        if (mFlux >= 0) {
            sys.diag[o] += mFlux + Db;
            sys.source[o] += Db * kb;
        } else {
            sys.diag[o] += Db;
            sys.source[o] += (Db - mFlux) * kb;
        }
    }

    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol  = mesh_.cell(ci).volume;
        double rhoC = f.rho[ci];
        sys.source[ci] += rhoC * f.Pk[ci] * vol;
        double dest = rhoC * sst_.coeffs.betaStar * std::max(f.omega[ci], 1e-20);
        sys.diag[ci] += dest * vol;
    }

    const double alphaK = settings_.alphaK;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        sys.source[ci] += (1.0 - alphaK) / alphaK * sys.diag[ci] * f.k[ci];
        sys.diag[ci]   /= alphaK;
    }
}

// ─── Turbulence: omega equation ───────────────────────────────────────────────

void CompressibleSIMPLESolver::assembleOmegaEquation(LinearSystem& sys,
                                                      const CompressibleFlowFields& f,
                                                      const ScalarField& Smag)
{
    sys.zero();
    int nIF = mesh_.nInternalFaces();

    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner, n = face.neighbor;
        double Sf    = face.area;
        double delta = std::max(face.delta, 1e-20);

        double F1_f  = face.weight * f.F1[o] + (1.0 - face.weight) * f.F1[n];
        double sw    = sst_.coeffs.sigma_w(F1_f);
        double rho_f = face.weight * f.rho[o] + (1.0 - face.weight) * f.rho[n];
        double mu_f  = face.weight * mu_[o]   + (1.0 - face.weight) * mu_[n];
        double nuT_f = face.weight * f.nuT[o] + (1.0 - face.weight) * f.nuT[n];
        double muEff = mu_f + rho_f * sw * nuT_f;
        double Df    = muEff * Sf / delta;

        Vec3   Uf    = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double mFlux = rho_f * (Uf.x * face.normal.x
                               + Uf.y * face.normal.y
                               + Uf.z * face.normal.z) * Sf;
        double cPos = std::max( mFlux, 0.0);
        double cNeg = std::max(-mFlux, 0.0);

        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);
        sys.lower[fi] = -(Df + cPos);
    }

    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face& face = mesh_.face(fi);
        int o     = face.owner;
        double Sf = face.area, delta = std::max(face.delta, 1e-20);
        double sw    = sst_.coeffs.sigma_w(f.F1[o]);
        double muEff = mu_[o] + f.rho[o] * sw * f.nuT[o];
        double Db    = muEff * Sf / delta;

        double wb    = f.omega.bface(fi);
        Vec3   Ubv   = f.U.bface(fi);
        double mFlux = f.rho[o] * (Ubv.x * face.normal.x
                                  + Ubv.y * face.normal.y
                                  + Ubv.z * face.normal.z) * Sf;
        if (mFlux >= 0) {
            sys.diag[o]   += mFlux + Db;
            sys.source[o] += Db * wb;
        } else {
            sys.diag[o]   += Db;
            sys.source[o] += (Db - mFlux) * wb;
        }
    }

    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol   = mesh_.cell(ci).volume;
        double F1    = f.F1[ci];
        double rhoC  = f.rho[ci];
        double omC   = std::max(f.omega[ci], 1e-20);

        double alphaB = sst_.coeffs.alpha(F1);
        double S      = Smag[ci];
        sys.source[ci] += rhoC * alphaB * S * S * vol;

        double betaB = sst_.coeffs.beta(F1);
        sys.diag[ci] += rhoC * betaB * omC * vol;

        sys.source[ci] += (1.0 - F1) * std::max(f.CDkw[ci], 0.0) * vol;
    }

    const double alphaW = settings_.alphaOmega;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        sys.source[ci] += (1.0 - alphaW) / alphaW * sys.diag[ci] * f.omega[ci];
        sys.diag[ci]   /= alphaW;
    }
}

// ─── Correction steps ─────────────────────────────────────────────────────────

void CompressibleSIMPLESolver::correctVelocity(CompressibleFlowFields& f,
                                                const ScalarField& pPrime,
                                                const std::vector<double>& aP)
{
    VectorField gradPp = greenGaussGrad(pPrime);
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double rAP = (std::abs(aP[ci]) > 1e-30) ? 1.0 / aP[ci] : 0.0;
        double vol = mesh_.cell(ci).volume;
        f.U[ci].x -= rAP * gradPp[ci].x * vol;
        f.U[ci].y -= rAP * gradPp[ci].y * vol;
        f.U[ci].z -= rAP * gradPp[ci].z * vol;
    }
    // Re-apply BCs via the FlowBoundaryConditions form
    FlowBoundaryConditions fbc;
    fbc.velocityBC = bcs_.velocityBC;
    fbc.pressureBC = bcs_.pressureBC;
    fbc.kBC        = bcs_.kBC;
    fbc.omegaBC    = bcs_.omegaBC;
    applyVelocityBC(f.U, mesh_, fbc);
}

void CompressibleSIMPLESolver::correctPressure(CompressibleFlowFields& f,
                                                const ScalarField& pPrime)
{
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        f.p[ci] += settings_.alphaP * pPrime[ci];

    FlowBoundaryConditions fbc;
    fbc.velocityBC = bcs_.velocityBC;
    fbc.pressureBC = bcs_.pressureBC;
    fbc.kBC        = bcs_.kBC;
    fbc.omegaBC    = bcs_.omegaBC;
    applyPressureBC(f.p, mesh_, fbc);
}

double CompressibleSIMPLESolver::computeResidual(const CompressibleFlowFields&, int) {
    return 0.0;  // tracked via solver residuals in the main loop
}

double CompressibleSIMPLESolver::computeScalarResidual(const ScalarField& old_phi,
                                                        const ScalarField& new_phi)
{
    double maxVal  = 1e-30, maxDiff = 0.0;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        maxVal  = std::max(maxVal,  std::abs(new_phi[ci]));
        maxDiff = std::max(maxDiff, std::abs(new_phi[ci] - old_phi[ci]));
    }
    return maxDiff / maxVal;
}

// ─── Main SIMPLE loop ─────────────────────────────────────────────────────────

CompressibleConvergenceHistory CompressibleSIMPLESolver::solve(CompressibleFlowFields& f) {
    CompressibleConvergenceHistory hist;

    LinearSystem momSys = makeSystem(mesh_);
    LinearSystem pSys   = makeSystem(mesh_);
    LinearSystem kSys   = makeSystem(mesh_);
    LinearSystem omSys  = makeSystem(mesh_);
    LinearSystem eSys   = makeSystem(mesh_);
    ScalarField  pPrime(mesh_, "p'");
    ScalarField  SmagFrozen(mesh_, "Smag");

    // Reference nu for BCs (at inlet temperature; updated each iteration)
    double nu_ref = eos_.mu_ref / eos_.density(f.p[0], f.T[0]);

    for (int iter = 0; iter < settings_.maxIterations; ++iter) {

        // 1. Update thermodynamic state: ρ, μ from current p, T
        updateDensity(f);
        updateViscosity(f);
        nu_ref = mu_[0] / f.rho[0];

        // 2. SST turbulence fields
        bool turbActive = (iter >= settings_.turbStartIter);
        bool turbUpdate = turbActive &&
                          ((iter - settings_.turbStartIter) % settings_.turbUpdateInterval == 0);

        if (turbUpdate) {
            // Compute kinematic eddy viscosity nuT = μT/ρ for SST model input
            // (SSTModel expects kinematic quantities)
            ScalarField nuT_k(mesh_, "nuT_k");
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                nuT_k[ci] = f.nuT[ci];   // already kinematic in our storage

            ScalarField k_scaled(mesh_, "k_s");
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                k_scaled[ci] = f.k[ci];

            double nu0 = mu_[0] / std::max(f.rho[0], 1e-30);
            sst_.computeFields(mesh_, k_scaled, f.omega, f.U, nu0,
                               f.nuT, f.F1, f.F2, f.Pk, f.CDkw);

            SmagFrozen = strainRateMagnitude(computeVelocityGradients(f.U));

            const double nuTMin = 0.1 * nu0;
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                f.nuT[ci] = std::max(f.nuT[ci], nuTMin);
        }

        // 3. Momentum equations
        assembleMomentum(momSys, f, 0, aP_);
        std::vector<double> Ux(mesh_.nCells());
        for (int ci = 0; ci < mesh_.nCells(); ++ci) Ux[ci] = f.U[ci].x;
        SolverResult resUx = mSolver_->solve(momSys, Ux,
                                              settings_.innerIterations,
                                              settings_.innerTolerance);
        for (int ci = 0; ci < mesh_.nCells(); ++ci) f.U[ci].x = Ux[ci];

        std::vector<double> aPstore = aP_;

        assembleMomentum(momSys, f, 1, aP_);
        std::vector<double> Uy(mesh_.nCells());
        for (int ci = 0; ci < mesh_.nCells(); ++ci) Uy[ci] = f.U[ci].y;
        SolverResult resUy = mSolver_->solve(momSys, Uy,
                                              settings_.innerIterations,
                                              settings_.innerTolerance);
        for (int ci = 0; ci < mesh_.nCells(); ++ci) f.U[ci].y = Uy[ci];

        assembleMomentum(momSys, f, 2, aP_);
        std::vector<double> Uz(mesh_.nCells());
        for (int ci = 0; ci < mesh_.nCells(); ++ci) Uz[ci] = f.U[ci].z;
        SolverResult resUz = mSolver_->solve(momSys, Uz,
                                              settings_.innerIterations,
                                              settings_.innerTolerance);
        for (int ci = 0; ci < mesh_.nCells(); ++ci) f.U[ci].z = Uz[ci];

        // Re-apply velocity BCs
        FlowBoundaryConditions fbc_tmp;
        fbc_tmp.velocityBC = bcs_.velocityBC;
        fbc_tmp.pressureBC = bcs_.pressureBC;
        fbc_tmp.kBC = bcs_.kBC;
        fbc_tmp.omegaBC = bcs_.omegaBC;
        applyVelocityBC(f.U, mesh_, fbc_tmp);

        // 4. Pressure correction (with density weighting)
        assemblePressureCorrection(pSys, f, aPstore, pPrime);
        std::vector<double> ppVec(mesh_.nCells(), 0.0);
        SolverResult resP = pSolver_->solve(pSys, ppVec,
                                             settings_.innerIterations,
                                             settings_.innerTolerance);
        for (int ci = 0; ci < mesh_.nCells(); ++ci) pPrime[ci] = ppVec[ci];

        // p' boundary conditions
        for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
            const Patch& pat = mesh_.patch(pi);
            if (pat.type == "outlet") continue;
            for (FaceID fi : pat.faces)
                pPrime.bface(fi) = pPrime[mesh_.face(fi).owner];
        }

        correctVelocity(f, pPrime, aPstore);
        correctPressure(f, pPrime);

        // 5. Energy equation for T
        {
            ScalarField T_old(mesh_, "T_old");
            for (int ci = 0; ci < mesh_.nCells(); ++ci) T_old[ci] = f.T[ci];

            assembleEnergy(eSys, f);
            std::vector<double> Tvec(mesh_.nCells());
            for (int ci = 0; ci < mesh_.nCells(); ++ci) Tvec[ci] = f.T[ci];
            SolverResult resT = eSolver_->solve(eSys, Tvec,
                                                 settings_.innerIterations,
                                                 settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) {
                f.T[ci] = std::max(Tvec[ci], 1.0);  // prevent non-physical T ≤ 0
            }
            applyTemperatureBC(f.T, mesh_, bcs_);

            // Update density from updated p, T
            updateDensity(f);
        }

        // 6. Turbulence equations
        SolverResult resK = {}, resOm = {};
        if (turbUpdate) {
            double nu0 = mu_[0] / std::max(f.rho[0], 1e-30);

            assembleKEquation(kSys, f);
            std::vector<double> kVec = f.k.data();
            resK = tSolver_->solve(kSys, kVec, settings_.innerIterations, settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) f.k[ci] = kVec[ci];
            f.k.clamp(settings_.kMin, settings_.kMax);
            applyKBC(f.k, mesh_, fbc_tmp);

            assembleOmegaEquation(omSys, f, SmagFrozen);
            std::vector<double> omVec = f.omega.data();
            resOm = tSolver_->solve(omSys, omVec, settings_.innerIterations, settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) f.omega[ci] = omVec[ci];
            f.omega.clamp(settings_.omegaMin, 1e15);
            applyOmegaBC(f.omega, mesh_, fbc_tmp, nu0, sst_.coeffs.beta1);

            // Re-pin near-wall omega.  PHASE 7 — when settings_.useWallFunctions
            // is on, blend the resolved-LES form with the log-law form so the
            // solver runs safely for y+ ≥ 30.
            const double betaStar = sst_.coeffs.betaStar;
            const double kappa    = settings_.vonKarman;
            for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
                const Patch& pat = mesh_.patch(pi);
                if (pat.type != "wall") continue;
                for (FaceID fi : pat.faces) {
                    const Face& face = mesh_.face(fi);
                    int       o    = face.owner;
                    double    y1   = std::max(face.delta, 1e-20);
                    double omRes = 60.0 * nu0 / (sst_.coeffs.beta1 * y1 * y1);
                    if (settings_.useWallFunctions) {
                        double k_p   = std::max(f.k[o], 1e-30);
                        double uTau  = std::sqrt(std::sqrt(betaStar) * k_p);
                        double omLog = uTau / (std::sqrt(betaStar) * kappa * y1);
                        f.omega[o] = std::sqrt(omRes * omRes + omLog * omLog);
                    } else {
                        f.omega[o] = omRes;
                    }
                }
            }
        }

        // 7. Convergence and divergence check
        CompressibleResidualEntry entry;
        entry.iteration = iter + 1;
        entry.Ux    = resUx.initialRes;
        entry.Uy    = resUy.initialRes;
        entry.p     = resP.initialRes;
        entry.k     = turbUpdate ? resK.initialRes  : 0.0;
        entry.omega = turbUpdate ? resOm.initialRes : 0.0;
        hist.entries.push_back(entry);

        // Check for divergence
        bool diverged = !std::isfinite(entry.Ux) || !std::isfinite(entry.p)
                     || !std::isfinite(entry.k)  || !std::isfinite(entry.omega);
        if (!diverged) {
            double pMax = 0;
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                pMax = std::max(pMax, std::abs(f.p[ci]));
            diverged = (pMax > settings_.divergenceLimit);
        }
        if (diverged) {
            hist.diverged  = true;
            hist.finalIter = iter + 1;
            if (settings_.verbose)
                std::cout << "  DIVERGED at iteration " << iter + 1 << "\n";
            return hist;
        }

        if (settings_.verbose && ((iter + 1) % settings_.reportInterval == 0)) {
            std::cout << "  iter " << iter + 1
                      << "  Ux=" << entry.Ux << "  p=" << entry.p
                      << "  k="  << entry.k  << "  om=" << entry.omega << "\n";
        }

        // Check convergence
        double tol = settings_.convergenceTol;
        bool converged = (entry.Ux < tol) && (entry.p < tol);
        if (converged) {
            hist.converged = true;
            hist.finalIter = iter + 1;
            if (settings_.verbose)
                std::cout << "  Converged at iteration " << iter + 1 << "\n";
            return hist;
        }
    }

    hist.finalIter = settings_.maxIterations;
    return hist;
}
