#include "SIMPLESolver.hpp"
#include "AnisotropyTools.hpp"
#include "StressOperators.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

// Constructor 
// Returns initialized SIMPLESolver object
SIMPLESolver::SIMPLESolver(const Mesh& mesh, const SSTModel& sst,
                           const FlowBoundaryConditions& bcs, double nu,
                           const SolverSettings& settings)
    : mesh_(mesh), sst_(sst), bcs_(bcs), nu_(nu), settings_(settings),
      aP_(mesh.nCells(), 0.0), aPunrelaxed_(mesh.nCells(), 0.0)
{
    pSolver_ = makeSolver(settings_.pressureSolver);
    mSolver_ = makeSolver(settings_.momentumSolver);
    tSolver_ = makeSolver(settings_.turbulenceSolver);
}

// Sets uniform initial conditions for velocity, pressure, and turbulence variables (k, omega)
// initializes derived SST fields (nuT, F1, F2, Pk, CDkw), and applies boundary conditions before iteration
void SIMPLESolver::initUniform(FlowFields& f, const Vec3& Uinit, double pInit, double kInit, double omegaInit) {
    f.U.setUniform(Uinit);
    f.p.setUniform(pInit);
    f.k.setUniform(kInit);
    f.omega.setUniform(omegaInit);
    f.F1.setUniform(1.0);
    f.F2.setUniform(1.0);
    f.Pk.setUniform(0.0);
    f.CDkw.setUniform(0.0);
    f.turbEstablished = false;   // cold start: the startup nuT floor window applies

    // Initialize omega with the wall-distance profile omega = max(60nu/(beta1*y^2), omegaInit).
    // A uniform omega = omegaInit everywhere creates a cliff-edge gradient when wall cells are
    // later pinned to ~5800: diffusion from the wall cell drives the 2nd-row cell from 2.55 to
    // ~350 in the first turbulence iteration and the spike propagates inward until divergence.
    // Using the 1/y^2 profile from the start gives smooth, physics-consistent gradients.
    const auto& wd = mesh_.wallDistance();
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double y = std::max(wd[ci], 1e-20);
        double omWall = 60.0 * nu_ / (sst_.coeffs.beta1 * y * y);
        f.omega[ci] = std::max(omWall, omegaInit);
    }

    // nuT from the updated omega profile
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        f.nuT[ci] = sst_.coeffs.a1 * kInit / std::max(sst_.coeffs.a1 * f.omega[ci], 1e-20);

    applyAllBCs(f.U, f.p, f.k, f.omega, mesh_, bcs_, nu_);
}

// Momentum equation assembly (Ux, Uy, or Uz)
// constructs a linear system for finite-volume discretization of momentum equation
// builds A_U * U_component = b (where A_U is in LDU and b is source)
void SIMPLESolver::assembleMomentum(LinearSystem& sys, const FlowFields& f, int component, std::vector<double>& aP) {
    sys.zero();         // resets linear system
    int nIF = mesh_.nInternalFaces();

    // internal faces: convection + diffusion
    for (int fi = 0; fi < nIF; ++fi) {

        // face geometry
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        int n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        // effective viscosity at face (molecular + eddy)
        double nuEff_o = nu_ + f.nuT[o];        // molecular viscosity
        double nuEff_n = nu_ + f.nuT[n];        // eddy viscosity
        double nuEff_f = face.weight * nuEff_o + (1.0 - face.weight) * nuEff_n;     // interpolation

        // diffusion coefficient
        double Df = nuEff_f * Sf / delta;

        // mass flux through face (first-order upwind)
        // flux = rho * U_f . S_f   (rho = 1 for incompressible)
        Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double mFlux = (Uf.x * face.normal.x + Uf.y * face.normal.y + Uf.z * face.normal.z) * Sf;

        // upwind convection coefficients
        double cPos = std::max( mFlux, 0.0);   // flux from owner to neighbor
        double cNeg = std::max(-mFlux, 0.0);   // flux from neighbor to owner

        // owner equation: aP += Df + cPos,  aN = -(Df + cNeg)
        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);   // off-diag contribution to owner from neighbor
        sys.lower[fi] = -(Df + cPos);   // off-diag contribution to neighbor from owner
    }

    // boundary faces. At RESOLVED walls the eddy viscosity vanishes on the
    // face (SST asymptotics), so the implicit wall diffusion uses the
    // MOLECULAR viscosity only: extrapolating the owner-cell nuT (nonzero at
    // y+ ~ 1) overestimates the discrete wall shear the momentum equation
    // carries, inconsistently with the molecular wall-stress observation and
    // the wall-zero transpose treatment. Under wall functions the owner
    // value stays (the modeled stress carrier). Non-wall boundaries keep
    // owner extrapolation.
    std::vector<char> isWall(mesh_.nFaces(), 0);
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch& pat = mesh_.patch(pi);
        if (pat.type != "wall") continue;
        for (FaceID wfi : pat.faces) isWall[wfi] = 1;
    }
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {

        // face geometry
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);
        double nuEff = (isWall[fi] && !settings_.useWallFunctions)
                       ? nu_ : nu_ + f.nuT[o];

        // diffusion term
        double Db = nuEff * Sf / delta;

        // computes boundary velocity component
        double Ub;
        if (component == 0) Ub = f.U.bface(fi).x;
        else if (component == 1) Ub = f.U.bface(fi).y;
        else Ub = f.U.bface(fi).z;

        // boundary mass flux
        Vec3 Ubv = f.U.bface(fi);
        double mFlux = (Ubv.x * face.normal.x + Ubv.y * face.normal.y
                      + Ubv.z * face.normal.z) * Sf;

        if (mFlux >= 0) {
            // outflow: convection from owner
            sys.diag[o] += mFlux + Db;
            sys.source[o] += Db * Ub;
        } else {
            // inflow: fixed boundary value
            sys.diag[o] += Db;
            sys.source[o] += (Db - mFlux) * Ub;
        }
    }

    // pressure gradient source (-dP/dx_component * V)
    VectorField gradP = greenGaussGrad(f.p);
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol = mesh_.cell(ci).volume;
        double dpdx;
        if (component == 0) dpdx = gradP[ci].x;
        else if (component == 1) dpdx = gradP[ci].y;
        else dpdx = gradP[ci].z;
        sys.source[ci] -= dpdx * vol;
    }

    // constant body force (drives streamwise-periodic domains, which have no
    // inlet; represents the mean pressure gradient)
    const double fb = (component == 0) ? settings_.bodyForce.x
                    : (component == 1) ? settings_.bodyForce.y
                                       : settings_.bodyForce.z;
    if (fb != 0.0)
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            sys.source[ci] += fb * mesh_.cell(ci).volume;

    // variable-viscosity transpose stress div(nuT (grad U)^T), the explicit
    // half completing the Boussinesq deviatoric stress divergence (the
    // implicit diffusion above is only the componentwise Laplacian); the
    // constant-nu molecular transpose is analytically zero for solenoidal U,
    // so only nuT enters, and the -2/3 k I normal stress stays absorbed in
    // the working pressure. See core/include/StressOperators.hpp. The
    // freeze switch is the frozen-pressure tangent's reduced operator (the
    // source is lagged there, matching the solver's own Picard operator).
    if (!freezeTransposeStress_) {
        // wall faces carry ZERO transpose coefficient: the eddy viscosity
        // vanishes AT a no-slip wall even though the owner-cell value does
        // not, so owner extrapolation would overweight the wall flux
        // (review fix, pinned by the wall-adjacent manufactured test)
        std::vector<double> zeroWall(mesh_.nCells(), 0.0);
        std::vector<double> tsrc =
            transposeStressSource(mesh_, f.nuT, f.U, component, &zeroWall);
        for (int ci = 0; ci < mesh_.nCells(); ++ci)
            sys.source[ci] += tsrc[ci];
    }

    // a-posteriori Reynolds-stress injection (explicit deferred-correction)
    if (bTarget6_) addInjectionSource(sys, f, component);

    // In SIMPLE, the momentum equation is usually under-relaxed to improve stability.
    // under-relaxation: numerical stabilization technique - prevents solution from changing too drastically between iterations
    // makes small adjustments to allow for smooth convergence

    // under-relaxation:  aP_new = aP / alphaU,  source += (1-alphaU)/alphaU * aP * phi_old
    double alphaU = settings_.alphaU;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double phi_old;
        if (component == 0) phi_old = f.U[ci].x;
        else if (component == 1) phi_old = f.U[ci].y;
        else phi_old = f.U[ci].z;

        aPunrelaxed_[ci] = sys.diag[ci];  // store UNRELAXED diagonal for Rhie-Chow pressure Laplacian
        sys.source[ci] += (1.0 - alphaU) / alphaU * sys.diag[ci] * phi_old;
        sys.diag[ci] /= alphaU;
        aP[ci] = sys.diag[ci];   // store RELAXED diagonal (after /alphaU) for velocity correction
    }
}

// A-posteriori Reynolds-stress injection source (explicit deferred-correction).
//
// Adds to the momentum source of `component` the body force
//   f_inj V = -[ sum_f (D_f . n_f)_i A_f ],   D = 2 k b_target + 2 nuT dev(S),
// i.e. the Green-Gauss divergence of the difference between the injected
// deviatoric Reynolds stress 2 k b_target and the Boussinesq-modeled deviatoric
// stress -2 nuT dev(S) the implicit nuEff diffusion stands for. k and S are the
// RUNNING fields, so the force vanishes identically when b_target equals the
// solver's own Boussinesq anisotropy b_B = -(nuT/k) dev(S), and the injected
// difference from the baseline solve is exactly -div(2 k (b_target - b_B)).
// Face values of D are distance-weight interpolated; boundary faces take the
// owner value (the wall rows carry k ~ 0, so their stress contribution is
// negligible there). nuT diffusion stays implicit as the stabilizer.
//
// On the x-component pass the target anisotropy is re-checked for realizability
// (barycentric margin of its eigenvalues), fulfilling the every-outer-iteration
// realizability assertion of the pre-registered injection scheme; violations are
// recorded in injDiag_ rather than silently projected, so a study can report
// them (the caller projects before setting the target).
void SIMPLESolver::addInjectionSource(LinearSystem& sys, const FlowFields& f,
                                      int component) {
    const int nc = mesh_.nCells();

    // The x-component pass recomputes the full Vec3 source from the current
    // fields, blends it into injSrcBlend_ at injRelax_, and re-asserts
    // realizability; the y and z passes reuse the stored blend. This relies on
    // the momentum components being assembled in order 0, 1, 2, which both
    // solve() and assembleResidual() do.
    if (component == 0) {
        // per-cell deferred-correction tensor D = 2 k b_target + 2 nuT dev(S),
        // stored as xx, yy, zz, xy, xz, yz
        VelocityGradients vg = computeVelocityGradients(f.U);
        std::vector<double> D(6 * nc);
        for (int ci = 0; ci < nc; ++ci) {
            const double S11 = vg.dudx[ci].x;
            const double S22 = vg.dvdx[ci].y;
            const double S33 = vg.dwdx[ci].z;
            const double S12 = 0.5 * (vg.dudx[ci].y + vg.dvdx[ci].x);
            const double S13 = 0.5 * (vg.dudx[ci].z + vg.dwdx[ci].x);
            const double S23 = 0.5 * (vg.dvdx[ci].z + vg.dwdx[ci].y);
            const double trS3 = (S11 + S22 + S33) / 3.0;   // discrete div(U)/3 residue

            const double twoK  = 2.0 * f.k[ci];
            const double twoNu = 2.0 * f.nuT[ci];
            const double* bt = bTarget6_->data() + 6 * ci;
            D[6 * ci + 0] = twoK * bt[0] + twoNu * (S11 - trS3);
            D[6 * ci + 1] = twoK * bt[1] + twoNu * (S22 - trS3);
            D[6 * ci + 2] = twoK * bt[2] + twoNu * (S33 - trS3);
            D[6 * ci + 3] = twoK * bt[3] + twoNu * S12;
            D[6 * ci + 4] = twoK * bt[4] + twoNu * S13;
            D[6 * ci + 5] = twoK * bt[5] + twoNu * S23;
        }

        // fresh force per cell: (f_inj V)_i = -sum_f (D_f . n_f)_i A_f
        std::vector<Vec3> q(nc, Vec3(0.0, 0.0, 0.0));
        auto dDotN = [](const double* d, const Vec3& n) {
            return Vec3(d[0] * n.x + d[3] * n.y + d[4] * n.z,
                        d[3] * n.x + d[1] * n.y + d[5] * n.z,
                        d[4] * n.x + d[5] * n.y + d[2] * n.z);
        };
        const int nIF = mesh_.nInternalFaces();
        for (int fi = 0; fi < nIF; ++fi) {
            const Face& face = mesh_.face(fi);
            const int o = face.owner;
            const int n = face.neighbor;
            double Df[6];
            for (int c = 0; c < 6; ++c)
                Df[c] = face.weight * D[6 * o + c]
                      + (1.0 - face.weight) * D[6 * n + c];
            const Vec3 flux = dDotN(Df, face.normal) * face.area;
            q[o] = q[o] - flux;     // f_inj = -div(D): outflux lowers the owner
            q[n] = q[n] + flux;     // normal points owner -> neighbor
        }
        // boundary faces: owner-value extrapolation
        for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
            const Face& face = mesh_.face(fi);
            q[face.owner] = q[face.owner]
                - dDotN(D.data() + 6 * face.owner, face.normal) * face.area;
        }

        // blend (the first pass of a solve, or a fresh evaluation, has
        // injSrcBlend_ empty or injRelax_ = 1 and takes q as-is)
        if (injSrcBlend_.size() != static_cast<size_t>(nc) || injRelax_ >= 1.0) {
            injSrcBlend_ = q;
        } else {
            for (int ci = 0; ci < nc; ++ci)
                injSrcBlend_[ci] = injSrcBlend_[ci] * (1.0 - injRelax_)
                                 + q[ci] * injRelax_;
        }

        // realizability re-assertion, once per outer iteration
        injDiag_.active = true;
        injDiag_.checkedIters += 1;
        for (int ci = 0; ci < nc; ++ci) {
            const double margin = aniso::barycentricMargin(bTarget6_->data() + 6 * ci);
            // tolerance above the trig eigensolver's conditioning floor at
            // degenerate eigenvalues (see AnisotropyTools::isRealizable): corner
            // targets from the eigenspace family sit exactly on the boundary
            if (margin < -1e-6) {
                injDiag_.allRealizable = false;
                injDiag_.maxViolation = std::max(injDiag_.maxViolation, -margin);
            }
        }
    }

    for (int ci = 0; ci < nc; ++ci) {
        if (component == 0)      sys.source[ci] += injSrcBlend_[ci].x;
        else if (component == 1) sys.source[ci] += injSrcBlend_[ci].y;
        else                     sys.source[ci] += injSrcBlend_[ci].z;
    }
}

// Pressure correction equation assembly
// constructs a finite-volume linear system for the pressure correction equation in SIMPLE algorithm
// constructs Laplacian operator using inverse momentum diagonal and adding source terms from divergence of predicted velocity field
// Laplacian(p') = div(U*)  where coefficients use rAP = 1/aP per cell
// ensures that div(U*) = 0
void SIMPLESolver::assemblePressureCorrection(LinearSystem& sys,
                                              const FlowFields& f,
                                              const std::vector<double>& aP,
                                              ScalarField& pPrime) {
    sys.zero();
    pPrime.setUniform(0.0);
    int nIF = mesh_.nInternalFaces();

    // Outlet-free (streamwise-periodic) domains need two guards a bounded
    // domain does not: the Rhie-Chow face-flux dissipation (the periodic
    // direction carries an exact odd-even null mode that boundaries otherwise
    // break) and a pressure reference pin (the all-Neumann Poisson system is
    // singular). Both are gated on the domain type so every bounded case keeps
    // the legacy discretization bit-for-bit (the semi-analytic coupled tangent
    // linearizes exactly that operator).
    bool hasOutlet = false;
    for (int pi = 0; pi < mesh_.nPatches(); ++pi)
        if (mesh_.patch(pi).type == "outlet" && !mesh_.patch(pi).faces.empty())
            hasOutlet = true;

    // Audit adjudication of this gate: the outlet test identifies where the
    // p' system is all-Neumann and the odd-even mode is an EXACT null mode.
    // An outlet's Dirichlet row removes the exact null mode and the
    // singularity, but interior odd-even susceptibility on a collocated
    // grid is a local stencil property that boundary rows only damp, so the
    // question on bounded meshes is EMPIRICAL, not settled by the gate:
    // the bounded production cases are DNS-validated without the
    // dissipation and their solved-pressure checkerboard energy measures
    // low (oddEvenEnergyRatio, pinned by test on the bounded channel), and
    // the coupled tangent linearises the legacy bounded operator
    // bit-for-bit. The term therefore stays gated by default with
    // settings_.rhieChowAllMeshes as the standing probe; enabling it
    // globally is a reviewed physics change to make if the diagnostic ever
    // measures otherwise on a production case. The pressure PIN below
    // remains outlet-free-only, where the singular system needs it.
    const bool rcActive = !hasOutlet || settings_.rhieChowAllMeshes;

    // cell pressure gradient for the Rhie-Chow face-flux dissipation below
    VectorField gradP(mesh_, "gradP");
    if (rcActive) gradP = greenGaussGrad(f.p);

    // internal faces
    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        int n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        // Pressure correction Laplacian: ∇·((V/aP) ∇p') = div(U*)
        // Face coefficient D_f = (V/aP)_f * A_f / delta
        // Interpolate (V/aP) from owner and neighbor cells
        double Vo = mesh_.cell(o).volume;
        double Vn = mesh_.cell(n).volume;
        double dP_o = (std::abs(aP[o]) > 1e-30) ? Vo / aP[o] : 0.0;
        double dP_n = (std::abs(aP[n]) > 1e-30) ? Vn / aP[n] : 0.0;
        double dP_f = face.weight * dP_o + (1.0 - face.weight) * dP_n;

        // pressure correction Laplacian coefficient
        double coeff = dP_f * Sf / delta;

        sys.diag[o] += coeff;
        sys.diag[n] += coeff;
        sys.upper[fi] = -coeff;
        sys.lower[fi] = -coeff;

        // mass flux source: U*.Sf with the Rhie-Chow dissipation
        //   m_f = Ubar_f.Sf - dP_f A_f [ (p_N - p_O)/delta - grad(p)bar_f . e ]
        // The added term vanishes on smooth pressure ((p_N-p_O)/delta equals the
        // interpolated gradient along e to truncation order) and penalises only
        // the odd-even component, which the compact Laplacian cannot see through
        // centrally interpolated fluxes. Without it a fully periodic direction
        // has an exact checkerboard null mode that grows from round-off (bounded
        // inlet/outlet domains break the mode at their boundaries, which is why
        // this solver never needed it before).
        Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double massFlux = (Uf.x * face.normal.x + Uf.y * face.normal.y
                         + Uf.z * face.normal.z) * Sf;
        if (rcActive) {
            Vec3 ehat = face.d / std::max(face.delta, 1e-30);
            Vec3 gbar = gradP[o] * face.weight + gradP[n] * (1.0 - face.weight);
            massFlux += -dP_f * Sf * ((f.p[n] - f.p[o]) / delta - gbar.dot(ehat));
        }

        // Source = -div(U*)*V: FV Laplacian sum(coeff*(p'_P - p'_N)) = -∇²p'*V,
        // so the Poisson equation rAP*∇²p' = div(U*) maps to source = -div(U*)*V.
        sys.source[o] -= massFlux;
        sys.source[n] += massFlux;
    }

    // boundary faces: mass flux source + Dirichlet diffusion for outlet (p'=0 there)
    // Wall and inlet use zero-gradient p' (Neumann) — no diffusion term needed.
    // Outlet uses Dirichlet p'=0 — the Laplacian coefficient must be added to the diagonal
    // so the pressure correction at the outlet-adjacent cell is actually constrained.
    // Without this, the p' Poisson system has no Dirichlet row and is singular.
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch& pat = mesh_.patch(pi);
        bool isOutlet = (pat.type == "outlet");
        for (FaceID fi : pat.faces) {
            const Face& face = mesh_.face(fi);
            int o = face.owner;
            double Sf = face.area;
            double delta = std::max(face.delta, 1e-20);

            // boundary mass flux source (same sign convention as internal)
            Vec3 Ub = f.U.bface(fi);
            double massFlux = (Ub.x * face.normal.x + Ub.y * face.normal.y
                             + Ub.z * face.normal.z) * Sf;
            sys.source[o] -= massFlux;

            // outlet Dirichlet: p'=0 → add diffusion coeff, source += coeff*0 = 0
            if (isOutlet) {
                double Vo = mesh_.cell(o).volume;
                double dP_o = (std::abs(aP[o]) > 1e-30) ? Vo / aP[o] : 0.0;
                double coeff = dP_o * Sf / delta;
                sys.diag[o] += coeff;
                // sys.source[o] += coeff * 0.0 — omitted (p'_outlet = 0)
            }
        }
    }

    // Pressure reference for domains with NO outlet (streamwise-periodic
    // channels bounded by walls and wrap faces): the all-Neumann p' Poisson
    // system is singular up to a constant, so pin cell 0 by doubling its
    // diagonal, a Dirichlet-strength link to p' = 0 at the natural scale of
    // its own row. The physical pressure level is arbitrary in such domains.
    if (!hasOutlet && mesh_.nCells() > 0)
        sys.diag[0] += std::abs(sys.diag[0]) > 1e-30 ? sys.diag[0] : 1.0;
}

// Correction step of the SIMPLE Algorithm

// Velocity correction:  U = U* - gradP' * V / aP
// adjusts predicted velocity using pressure-correction gradient and momentum diagonal coefficients
void SIMPLESolver::correctVelocity(FlowFields& f, const ScalarField& pPrime, const std::vector<double>& aP) {
    VectorField gradPp = greenGaussGrad(pPrime);
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double rAP = (std::abs(aP[ci]) > 1e-30) ? 1.0 / aP[ci] : 0.0;
        double vol = mesh_.cell(ci).volume;
        f.U[ci].x -= rAP * gradPp[ci].x * vol;
        f.U[ci].y -= rAP * gradPp[ci].y * vol;
        f.U[ci].z -= rAP * gradPp[ci].z * vol;
    }
    applyVelocityBC(f.U, mesh_, bcs_);      // boundary values must be re-enforced after modifying interior cells
}

// Pressure correction:  p = p + alphaP * p'
// updates pressure field with an under-relaxed pressure correction so that corrected U satisfies incompressible continuity equation (div U = 0)
void SIMPLESolver::correctPressure(FlowFields& f, const ScalarField& pPrime) {
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        f.p[ci] += settings_.alphaP * pPrime[ci];   // SIMPLE pressure update with under-relaxation to prevent instability
    }
    applyPressureBC(f.p, mesh_, bcs_);
}


// k-equation assembly
// converts continuous k-transport equation into a finite-volume matrix form (A_k * k = b_k)
//      discretizing convection and diffusion terms across faces, applying boundary conditions, 
//      adding production and destruction source terms, and applying under-relaxation for stability 
//      before the equation is solved
void SIMPLESolver::assembleKEquation(LinearSystem& sys, const FlowFields& f) {
    sys.zero();
    int nIF = mesh_.nInternalFaces();

    // loop over internal faces
    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        int n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        // diffusion term
        double F1_f = face.weight * f.F1[o] + (1.0 - face.weight) * f.F1[n];
        double sk = sst_.coeffs.sigma_k(F1_f);
        double nuEff_o = nu_ + sk * f.nuT[o];                                       // effective diffusivity: nu + sigma_k * nuT
        double nuEff_n = nu_ + sk * f.nuT[n];
        double nuEff_f = face.weight * nuEff_o + (1.0 - face.weight) * nuEff_n;     // interpolate to face
        double Df = nuEff_f * Sf / delta;

        // convection term
        Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);                                  // velocity at the face
        double mFlux = (Uf.x * face.normal.x + Uf.y * face.normal.y + Uf.z * face.normal.z) * Sf;       // mass flux
        double cPos = std::max( mFlux, 0.0);                                                            // upwind discretization
        double cNeg = std::max(-mFlux, 0.0);
        
        // add matrix coefficients
        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);
        sys.lower[fi] = -(Df + cPos);
    }

    // boundary faces
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);
        double sk = sst_.coeffs.sigma_k(f.F1[o]);
        double nuEff = nu_ + sk * f.nuT[o];
        double Db = nuEff * Sf / delta;         // diffusion coefficient

        double kb = f.k.bface(fi);              // boundary k value
        Vec3 Ub = f.U.bface(fi);                // boundary velocity
        double mFlux = (Ub.x * face.normal.x + 
                        Ub.y * face.normal.y + 
                        Ub.z * face.normal.z) * Sf; // mass flux

        // outflow boundary
        if (mFlux >= 0) {
            sys.diag[o] += mFlux + Db;
            sys.source[o] += Db * kb;
        } 
        // inflow boundary
        else {
            sys.diag[o] += Db;
            sys.source[o] += (Db - mFlux) * kb;
        }
    }

    // source terms: Pk (explicit), betaStar*omega*k (linearised destruction)
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol = mesh_.cell(ci).volume;
        // production term (explicit)
        sys.source[ci] += f.Pk[ci] * vol;
        // destruction term (linearised): betaStar * omega * k → diag += betaStar*omega*V
        double destruction = sst_.coeffs.betaStar * std::max(f.omega[ci], 1e-20);
        sys.diag[ci] += destruction * vol;
    }

    // under-relaxation
    double alphaK = settings_.alphaK;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        sys.source[ci] += (1.0 - alphaK) / alphaK * sys.diag[ci] * f.k[ci];
        sys.diag[ci] /= alphaK;
    }
}

// omega equation assembly
// converts continous SST omega transport equation into finite-volume matrix form (A_ω * ω = b_ω)
//      discretize convection and diffusion fluxes across faces, applying boundary conditions, 
//      adding turbulence production, destruction, and cross-diffusion source terms, and 
//      applying under-relaxation before solving for the updated ω field
void SIMPLESolver::assembleOmegaEquation(LinearSystem& sys, const FlowFields& f, const ScalarField& Smag) {
    sys.zero();
    int nIF = mesh_.nInternalFaces();
    // Smag is passed in (frozen at the pre-correction U from computeFields).
    // Do NOT recompute from f.U here: the post-correction velocity may carry
    // pressure-correction oscillations that would inflate S^2 and blow up omega.

    // internal face loop
    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        int n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        // diffusion term div((nu + sw * nuT) * grad(omega))
        double F1_f = face.weight * f.F1[o] + (1.0 - face.weight) * f.F1[n];
        double sw = sst_.coeffs.sigma_w(F1_f);                  
        double nuEff_o = nu_ + sw * f.nuT[o];                   // effective diffusivity: nu + sigma_w * nuT
        double nuEff_n = nu_ + sw * f.nuT[n];
        double nuEff_f = face.weight * nuEff_o + (1.0 - face.weight) * nuEff_n;
        double Df = nuEff_f * Sf / delta;                       // face diffusion coefficient

        // convection term (U dot grad(omega))
        Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double mFlux = (Uf.x * face.normal.x + 
                        Uf.y * face.normal.y + 
                        Uf.z * face.normal.z) * Sf;             // mass flux

        // upwind discretization (determines flow direction)
        double cPos = std::max( mFlux, 0.0);
        double cNeg = std::max(-mFlux, 0.0);
        
        // matrix coefficients: 
        // builds convection-diffusion operator
        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);
        sys.lower[fi] = -(Df + cPos);
    }

    // boundary faces
    // computes diffusion coefficient, boundary omega value, and mass flux
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);
        double sw = sst_.coeffs.sigma_w(f.F1[o]);
        double nuEff = nu_ + sw * f.nuT[o];
        double Db = nuEff * Sf / delta;

        double wb = f.omega.bface(fi);
        Vec3 Ub = f.U.bface(fi);
        double mFlux = (Ub.x * face.normal.x + 
                        Ub.y * face.normal.y +
                        Ub.z * face.normal.z) * Sf;
        
        // outflow boundary
        if (mFlux >= 0) {
            sys.diag[o] += mFlux + Db;
            sys.source[o] += Db * wb;
        } 
        // inflow boundary
        else {
            sys.diag[o] += Db;
            sys.source[o] += (Db - mFlux) * wb;
        }
    }

    // source terms
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol = mesh_.cell(ci).volume;
        double F1  = f.F1[ci];
        double omC = std::max(f.omega[ci], 1e-20);

        // production: alpha * Pk_limited/nuT = alpha * min(S^2, 10*betaStar*k*omega/nuT)
        // (SST-2003 corrected form; the 2003 paper's alpha*S^2 is a documented misprint,
        // see the NASA TMR SST page). min(S^2, lim) <= S^2 pointwise, so this term is
        // bounded above by the S^2 form: reducing nuT cannot amplify it, and destruction
        // (beta*omega^2, implicit below) still grows quadratically, so the equation stays
        // self-limiting. The two forms coincide wherever the k-production limiter is
        // inactive (equilibrium attached flows).
        double alphaB = sst_.coeffs.alpha(F1);
        double S = Smag[ci];
        sys.source[ci] += alphaB * sst_.productionOmega(f.nuT[ci], S, f.k[ci], f.omega[ci]) * vol;

        // destruction term (linearised) beta*omega^2 → diag += beta*omega*V
        double betaB = sst_.coeffs.beta(F1);
        sys.diag[ci] += betaB * omC * vol;

        // cross-diffusion (explicit): (1-F1)*CDkw, UNCLIPPED per SST-2003 (only the CDkw
        // inside the F1 argument is clipped; the source term itself may be negative)
        sys.source[ci] += (1.0 - F1) * f.CDkw[ci] * vol;
    }

    // under-relaxation
    double alphaW = settings_.alphaOmega;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        sys.source[ci] += (1.0 - alphaW) / alphaW * sys.diag[ci] * f.omega[ci];
        sys.diag[ci] /= alphaW;
    }
}

// placeholder for tracking field residuals
// currently we rely on linear solver residuals 
double SIMPLESolver::computeResidual(const FlowFields& f, int component) {
    (void)f; (void)component;
    return 0.0; // actual convergence tracked through linear solver results
}

// Main SIMPLE loop     
// SIMPLE algorithm runs until flow solution converges or diverges
ConvergenceHistory SIMPLESolver::solve(FlowFields& f) { // input: FlowField object / flow variables
    ConvergenceHistory hist;
    // Reynolds-stress injection: fresh blend state and diagnostics per solve;
    // the explicit source is under-relaxed inside the outer loop (see
    // addInjectionSource)
    injSrcBlend_.clear();
    injDiag_ = InjectionDiagnostics{};
    injRelax_ = settings_.alphaInjection;
    // Create working linear systems
    LinearSystem momSys  = makeSystem(mesh_);
    LinearSystem pSys    = makeSystem(mesh_);
    LinearSystem kSys    = makeSystem(mesh_);
    LinearSystem omSys   = makeSystem(mesh_);
    ScalarField  pPrime(mesh_, "p'");

    // Smag = Strain-rate magnitude
    // Frozen strain-rate field: computed from the pre-correction U (same U computeFields uses)
    // Passed to assembleOmegaEquation so both k and omega production see the same velocity gradients, preventing omega blow-up.
    ScalarField SmagFrozen(mesh_, "Smag");

    // Last-computed turbulence field-change norms, CARRIED ACROSS iterations:
    // on non-update iterations (turb_update_interval > 1) the convergence
    // check must see the most recent turbulence state, never a fresh zero
    // (a zero would let the solve declare convergence between turbulence
    // updates while k/omega are still moving). Initialised to 1 so nothing
    // can claim turbulence convergence before the first update.
    double lastKChange = 1.0, lastOmChange = 1.0;

    // SIMPLE Iteration loop
    for (int iter = 0; iter < settings_.maxIterations; ++iter) {
        // 1. Update SST fields (at turbUpdateInterval cadence after turbStartIter)
        bool turbActive = (iter >= settings_.turbStartIter);
        bool turbUpdate = turbActive &&
                          ((iter - settings_.turbStartIter) % settings_.turbUpdateInterval == 0);
        if (turbUpdate) {
            // Save old nuT for under-relaxation
            std::vector<double> nuT_old(mesh_.nCells());
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                nuT_old[ci] = f.nuT[ci];

            sst_.computeFields(mesh_, f.k, f.omega,
                               f.U, nu_, f.nuT, f.F1,
                               f.F2, f.Pk, f.CDkw); // Computes SST quantities (nuT (eddy viscosity), F1, F2, Pk, CDkw)

            // Freeze Smag from the same U that computeFields used (pre-correction)
            // assembleOmegaEquation will use this instead of recomputing from the post-correction U
            // Post-corrected velocity can contain numerical oscillations because it is updated by discrete pressure gradients
            // Since turbulence production depends on velocity derivatives
            //      - those oscillations can artificially inflate strain rate and destabilize the ω equation
            // Using the frozen Smag decouples the turbulence model from these numerical oscillations
            SmagFrozen = strainRateMagnitude(computeVelocityGradients(f.U));

            // nuT floor, STARTUP-ONLY: guards against early k-omega collapse
            // (cells where the Bradshaw limiter drives nuT toward zero before
            // the turbulence fields establish). SST requires nuT -> 0 at a
            // resolved no-slip wall, so the floor releases after the startup
            // window and the converged state is floor-free; a permanent floor
            // biases the near-wall diffusion and the wall stress (~+10 percent
            // where it binds at y+ ~ 1). Warm restarts carry
            // f.turbEstablished = true and never re-engage the floor.
            if (!f.turbEstablished
                && iter >= settings_.turbStartIter + settings_.nuTFloorIters)
                f.turbEstablished = true;
            const double nuTMin = f.turbEstablished ? 0.0 : 0.1 * nu_;
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                f.nuT[ci] = std::max(f.nuT[ci], nuTMin);
        }

        // 2. Assemble + solve momentum x, y, z 
        SolverResult resUx, resUy, resUz;
        {
            assembleMomentum(momSys, f, 0, aP_);
            std::vector<double> Ux(mesh_.nCells());
            for (int ci = 0; ci < mesh_.nCells(); ++ci) Ux[ci] = f.U[ci].x;
            resUx = mSolver_->solve(momSys, Ux, settings_.innerIterations,
                                    settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) f.U[ci].x = Ux[ci];
        }
        // store aP from Ux for corrections (diagonal dominance is similar for all components)
        std::vector<double> aPstore = aP_;              // relaxed — for velocity correction
        std::vector<double> aPrhie = aPunrelaxed_;      // unrelaxed — for pressure Laplacian
        {
            assembleMomentum(momSys, f, 1, aP_);
            std::vector<double> Uy(mesh_.nCells());
            for (int ci = 0; ci < mesh_.nCells(); ++ci) Uy[ci] = f.U[ci].y;
            resUy = mSolver_->solve(momSys, Uy, settings_.innerIterations, settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) f.U[ci].y = Uy[ci];
        }
        {
            assembleMomentum(momSys, f, 2, aP_);
            std::vector<double> Uz(mesh_.nCells());
            for (int ci = 0; ci < mesh_.nCells(); ++ci) Uz[ci] = f.U[ci].z;
            resUz = mSolver_->solve(momSys, Uz, settings_.innerIterations, settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) f.U[ci].z = Uz[ci];
        }
        applyVelocityBC(f.U, mesh_, bcs_);

        // 3. Assemble + solve pressure correction
        // Use RELAXED aP (= aP_raw / alphaU) for both pressure Laplacian and velocity correction 
        // This is necessary for consistency in SIMPLE algorithm: the momentum equation is solved with relaxed diagonal, 
        // and the correction step must use the same coefficients (so div(U_corrected) = 0). 
        // Note: U_corrected = U* - U' = U* - r_AP * ∇p' (r_AP = V/a_p)
        //  - when you solve the momentum eq, you get provisional velocity U* which does not satisfy continuity
        //  - SIMPLE introduces a correction U = U* + U' where U' is velocity correction derived from pressure correction or U_corrected
        assemblePressureCorrection(pSys, f, aPstore, pPrime);
        std::vector<double> ppVec(mesh_.nCells(), 0.0);
        SolverResult resP = pSolver_->solve(pSys,
                                            ppVec, 
                                            settings_.innerIterations,
                                            settings_.innerTolerance);
        for (int ci = 0; ci < mesh_.nCells(); ++ci) pPrime[ci] = ppVec[ci];
        
        // Gradient of pressure correction p' was being computed using incorrect boundary face values
        // Apply p' boundary conditions before computing grad(p') in correctVelocity
        //  - At wall/inlet, grad(p') = 0, but not necessarily p' itself
        //      - to enforce grad(p') = 0, we set boundary face equal to value of adjacent interior cell
        //  - At outlet, BC is p' = 0 (Dirichlet) - bface stays at 0 (default)
        // Without this, greenGaussGrad sees bface=0 on all faces (initialization default)
        //  - this would give wrong gradients at boundary-adjacent cells and causing velocity to grow without bound
        for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
            const Patch& pat = mesh_.patch(pi);
            if (pat.type == "outlet") continue; // keep Dirichlet p'=0
            for (FaceID fi : pat.faces)
                pPrime.bface(fi) = pPrime[mesh_.face(fi).owner];
        }

        // 4. Correct velocity and pressure (same relaxed aP)
        correctVelocity(f, pPrime, aPstore);
        correctPressure(f, pPrime);

        // 5. Assembly + solve Turbulence equations (k and omega) (on turbulence update iterations)
        SolverResult resK = {}, resOm = {};
        // Field-change norms for turbulence convergence tracking.
        // Linear solver initialRes is unreliable for omega because wall re-pinning
        // overrides the PDE solution at wall cells, creating a persistent equation
        // imbalance that never goes to zero.  Instead, track whether the fields
        // have stopped changing between iterations.
        double omChangeNorm = 0.0, kChangeNorm = 0.0;
        if (turbUpdate) {
            // Save pre-solve fields for convergence metric
            std::vector<double> kOld = f.k.data();
            std::vector<double> omOld = f.omega.data();

            // k equation
            assembleKEquation(kSys, f);
            std::vector<double> kVec = f.k.data();
            resK = tSolver_->solve(kSys, kVec, settings_.innerIterations,
                                   settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) f.k[ci] = kVec[ci];
            f.k.clamp(settings_.kMin, settings_.kMax);
            applyKBC(f.k, mesh_, bcs_);

            // omega equation (uses frozen Smag from pre-correction U)
            assembleOmegaEquation(omSys, f, SmagFrozen);
            std::vector<double> omVec = f.omega.data();
            resOm = tSolver_->solve(omSys, omVec, settings_.innerIterations,
                                    settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) f.omega[ci] = omVec[ci];
            f.omega.clamp(settings_.omegaMin, 1e15);
            applyOmegaBC(f.omega, mesh_, bcs_, nu_, sst_.coeffs.beta1);

            // Wall omega: re-pin near-wall cell centers each iteration.  The ω
            // equation strongly overproduces near walls (α·S² → ω ~ 10⁵) and
            // diffusion alone cannot enforce the Dirichlet condition; pinning
            // is standard practice in SST implementations.
            //
            // PHASE 7 — wall-function blend.  When useWallFunctions=true we
            // combine the resolved-LES form (Menter low-Re) with the log-law
            // form ω_log = u_τ / (√β* κ y), giving a single expression that
            // is correct for both y⁺ ≈ 1 and y⁺ ≥ 30.  When the flag is off
            // the legacy resolved-only behaviour is preserved bit-for-bit.
            const double betaStar = sst_.coeffs.betaStar;
            const double kappa    = settings_.vonKarman;
            for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
                const Patch& pat = mesh_.patch(pi);
                if (pat.type != "wall") continue;
                for (FaceID fi : pat.faces) {
                    const Face& face = mesh_.face(fi);
                    int       o    = face.owner;
                    double    y1   = std::max(face.delta, 1e-20);
                    double omRes = 60.0 * nu_ / (sst_.coeffs.beta1 * y1 * y1);
                    if (settings_.useWallFunctions) {
                        // Estimate u_τ from the cell-centre velocity tangential
                        // to the wall: |u_p| ≈ u_τ * (1/κ) ln(E y⁺) implies
                        // for y⁺ ≥ 30 that u_τ ≈ |u_p| κ / ln(E y⁺).  We use a
                        // conservative low-bound of u_τ from k via u_τ²= √β* k_p
                        // (Menter), which is robust early in the solve.
                        double k_p = std::max(f.k[o], 1e-30);
                        double uTau = std::sqrt(std::sqrt(betaStar) * k_p);
                        double omLog = uTau / (std::sqrt(betaStar) * kappa * y1);
                        f.omega[o] = std::sqrt(omRes * omRes + omLog * omLog);
                    } else {
                        f.omega[o] = omRes;
                    }
                }
            }

            // Compute field-change norms: ||phi_new - phi_old||_inf / ||phi_new||_inf
            double kMaxVal = 1e-30, omMaxVal = 1e-30;
            double kMaxDiff = 0.0, omMaxDiff = 0.0;
            for (int ci = 0; ci < mesh_.nCells(); ++ci) {
                kMaxVal  = std::max(kMaxVal,  std::abs(f.k[ci]));
                omMaxVal = std::max(omMaxVal, std::abs(f.omega[ci]));
                kMaxDiff  = std::max(kMaxDiff,  std::abs(f.k[ci] - kOld[ci]));
                omMaxDiff = std::max(omMaxDiff, std::abs(f.omega[ci] - omOld[ci]));
            }
            kChangeNorm  = kMaxDiff / kMaxVal;
            omChangeNorm = omMaxDiff / omMaxVal;
            lastKChange  = kChangeNorm;
            lastOmChange = omChangeNorm;
        }

        // 6. Track residuals
        //    Momentum and pressure: absolute initial residual normalised by iter-0 values
        //    (convergence = orders-of-magnitude reduction in equation imbalance).
        //    k and omega: field-change norms (||new - old||_inf / ||new||_inf) because
        //    wall re-pinning creates a persistent equation imbalance that never goes to zero.
        ResidualEntry entry;
        entry.iteration = iter;
        entry.Ux    = resUx.initialRes;
        entry.Uy    = resUy.initialRes;
        entry.Uz    = resUz.initialRes;
        entry.p     = resP.initialRes;
        // the CARRIED norms: on non-update iterations these hold the most
        // recent turbulence change, so the recorded history and the
        // convergence check below never see a false zero
        entry.k     = turbActive ? lastKChange  : 0.0;
        entry.omega = turbActive ? lastOmChange : 0.0;

        // store iter-0 norms for normalisation. The pressure norm keeps a
        // running max over a short warmup: on a fully periodic domain a
        // uniform initial field is EXACTLY divergence-free, so the iter-0 mass
        // imbalance is machine round-off and normalising by it turns every
        // later tiny residual into a false divergence alarm; the true residual
        // scale only appears once the flow develops. Bounded inlet/outlet
        // cases peak at iter 0 anyway, so their behaviour is unchanged.
        if (iter == 0) {
            normUx0_ = std::max(entry.Ux, 1e-30);
            normUy0_ = std::max(entry.Uy, 1e-30);
            normP0_  = std::max(entry.p,  1e-30);
        } else if (iter < 50) {
            normP0_  = std::max(normP0_, entry.p);
        }

        // normalised residuals (skip equations with negligible iter-0 residual)
        // in 2D, Uy/Uz have zero forcing so normUy0_ ~ 1e-30.  dividing any tiny
        // Uy residual by 1e-30 produces O(1e10), triggering false divergence.
        // safeNorm returns 0 for these equations so they don't affect convergence.
        auto safeNorm = [](double val, double ref) -> double {
            if (ref < 1e-20) return 0.0; // equation has no forcing; always "converged"
            return val / ref;
        };
        // Both momentum components are judged against the common momentum
        // scale max(normUx0_, normUy0_). Normalising Uy by its own iter-0
        // value is meaningless when that value is tiny but nonzero (e.g. the
        // Reynolds-stress injection contributes a small y-force at iter 0):
        // the y-equation would be held to a reference orders of magnitude
        // below the momentum balance of the problem and never "converge".
        const double normMom0 = std::max(normUx0_, normUy0_);
        double nUx = safeNorm(entry.Ux, normMom0);
        double nUy = safeNorm(entry.Uy, normMom0);
        double nP  = safeNorm(entry.p,  normP0_);
        // k and omega: carried field-change norms (already normalised; the
        // most recent update's change, never a fresh zero between updates)
        double nK  = lastKChange;
        double nOm = lastOmChange;
        hist.entries.push_back(entry);

        // computes maximum normalised residual
        double maxRes = std::max({nUx, nUy, nP});
        if (turbActive)
            maxRes = std::max({maxRes, nK, nOm});

        if (settings_.verbose && (iter % settings_.reportInterval == 0 || iter == 0)) {
            std::cout << "  SIMPLE iter " << iter
                      << "  Ux=" << nUx << "  Uy=" << nUy
                      << "  p=" << nP;
            if (turbUpdate)
                std::cout << "  k=" << nK << "  w=" << nOm;
            std::cout << "\n";
        }

        // If any individual residual or carried norm is non-finite, force
        // maxRes to infinity BEFORE the convergence test so a NaN can neither
        // be masked by chained std::max operand ordering nor slip past the
        // (NaN < tol) == false accident into a later iteration
        if (!std::isfinite(entry.Ux) || !std::isfinite(entry.Uy)
            || !std::isfinite(entry.Uz) || !std::isfinite(entry.p)
            || !std::isfinite(entry.k) || !std::isfinite(entry.omega)
            || !std::isfinite(lastKChange) || !std::isfinite(lastOmChange))
            maxRes = std::numeric_limits<double>::infinity();

        // convergence check (all normalised residuals below tolerance).
        // Scheduled turbulence must have RUN and its startup floor released:
        // without the turbScheduled gate a solve could declare convergence
        // before its first turbulence update, on a laminar transient the
        // criterion never sees; with it, no run can freeze a floored or
        // turbulence-free state as its converged solution.
        const bool turbScheduled =
            settings_.turbStartIter < settings_.maxIterations;
        if (maxRes < settings_.convergenceTol && iter > 0
            && (!turbScheduled || (turbActive && f.turbEstablished))) {
            hist.converged = true;
            hist.finalIter = iter;
            if (settings_.verbose)
                std::cout << "  SIMPLE converged at iteration " << iter << "\n";
            return hist;
        }

        // divergence check
        if (std::isnan(maxRes) || std::isinf(maxRes) || maxRes > settings_.divergenceLimit) {
            hist.diverged = true;
            hist.finalIter = iter;
            if (settings_.verbose)
                std::cout << "  SIMPLE diverged at iteration " << iter << "\n";
            return hist;
        }
    }

    hist.finalIter = settings_.maxIterations;
    if (settings_.verbose)
        std::cout << "  SIMPLE reached maxIter (" << settings_.maxIterations << ")\n";
    return hist;
}

// ===================================================================================
// ADJOINT GROUNDWORK — exact parameter sensitivities ∂R/∂θ and ∂g/∂θ
// (the first, NON-HELD increment of the discrete adjoint; see IMPLEMENTATION_SUMMARY.md
//  §8 and the standing adjoint plan).  We differentiate ONLY w.r.t. the 11 SST
//  coefficients θ, holding the converged primary state (U, p, k, ω) FIXED.  No code path
//  below differentiates the residual w.r.t. the FIELDS — the held (∂R/∂U)ᵀ core is not
//  touched.
// ===================================================================================

// Recompute the SST closure of `work` from sst_.coeffs, exactly mirroring the
// turbulence-update branch of solve() at a CONVERGED state: the startup-only
// nuT floor has released by convergence, so the mirror applies only the
// non-negativity clamp (a floored mirror would make residual/derivative
// evaluations disagree with the converged assembly).
void SIMPLESolver::recomputeClosure(FlowFields& work, ScalarField& Smag) {
    sst_.computeFields(mesh_, work.k, work.omega, work.U, nu_,
                       work.nuT, work.F1, work.F2, work.Pk, work.CDkw);
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        work.nuT[ci] = std::max(work.nuT[ci], 0.0);
    Smag = strainRateMagnitude(computeVelocityGradients(work.U));
}

// R(U,θ): assemble the discrete residual of momentum-x, -y, k, ω at the FIXED state with
// closure recomputed from θ.  Reuses the solver's own assemble* so the discretisation is
// bit-identical to solve().  The unrelaxed residual r = b − A·φ is recovered by evaluating
// LinearSystem::residual at φ = φ_old: the under-relaxation source term (1−α)/α·a_P·φ_old
// cancels the diagonal scaling a_P/α exactly at the current field value.
std::vector<double> SIMPLESolver::assembleResidual(const FlowFields& state,
                                                   const SSTCoefficients& theta) {
    const int nc = mesh_.nCells();
    SSTCoefficients saved = sst_.coeffs;
    sst_.coeffs = theta;

    // residuals must be state-consistent: the injection source (if any) is
    // evaluated fresh from `state`, not blended across calls
    injRelax_ = 1.0;
    injSrcBlend_.clear();

    FlowFields work = state;
    ScalarField Smag(mesh_, "Smag");
    recomputeClosure(work, Smag);

    std::vector<double> R(4 * nc, 0.0);
    std::vector<double> aP(nc, 0.0);
    LinearSystem sys = makeSystem(mesh_);
    std::vector<double> phi(nc), r(nc);

    auto block = [&](int b) { return b * nc; };  // [Rux|Ruy|Rk|Rω] block offset

    // momentum x
    assembleMomentum(sys, work, 0, aP);
    for (int ci = 0; ci < nc; ++ci) phi[ci] = work.U[ci].x;
    sys.residual(phi, r);
    for (int ci = 0; ci < nc; ++ci) R[block(0) + ci] = r[ci];
    // momentum y
    assembleMomentum(sys, work, 1, aP);
    for (int ci = 0; ci < nc; ++ci) phi[ci] = work.U[ci].y;
    sys.residual(phi, r);
    for (int ci = 0; ci < nc; ++ci) R[block(1) + ci] = r[ci];
    // k
    assembleKEquation(sys, work);
    for (int ci = 0; ci < nc; ++ci) phi[ci] = work.k[ci];
    sys.residual(phi, r);
    for (int ci = 0; ci < nc; ++ci) R[block(2) + ci] = r[ci];
    // omega (frozen Smag, as in solve())
    assembleOmegaEquation(sys, work, Smag);
    for (int ci = 0; ci < nc; ++ci) phi[ci] = work.omega[ci];
    sys.residual(phi, r);
    for (int ci = 0; ci < nc; ++ci) R[block(3) + ci] = r[ci];

    sst_.coeffs = saved;
    return R;
}

// ∂R/∂θ: the exact derivative of assembleResidual.  Each closure field enters the
// assembly linearly (nuT in the diffusion coefficients, Pk and the cross-diffusion in the
// volumetric sources, F1 in the σ_k/σ_w/α/β blends and the (1−F1) cross-diffusion
// factor), so ∂R/∂θ is the same face/cell walk with the values replaced by their
// pointwise closure sensitivities.  Convection (frozen U), the pressure-gradient source
// and all boundary VALUES are θ-independent and drop out.
std::vector<std::vector<double>> SIMPLESolver::assembleResidualSensitivity(
        const FlowFields& state, const SSTCoefficients& theta,
        bool includeTransposeTheta) {
    const int nc  = mesh_.nCells();
    const int nIF = mesh_.nInternalFaces();
    SSTCoefficients saved = sst_.coeffs;
    sst_.coeffs = theta;

    FlowFields work = state;
    ScalarField Smag(mesh_, "Smag");
    recomputeClosure(work, Smag);

    // pointwise closure sensitivities per cell (Layer 1); the startup-only
    // nuT floor has released at the converged states this is evaluated on,
    // so the derivative carries no floor dead-zone
    const auto& wd = mesh_.wallDistance();
    std::vector<SSTClosureSensitivity> cs(nc);
    for (int ci = 0; ci < nc; ++ci)
        cs[ci] = sst_.closureSensitivity(work.k[ci], work.omega[ci], Smag[ci],
                                         wd[ci], nu_, work.CDkw[ci], 0.0);

    std::vector<std::vector<double>> dR(11, std::vector<double>(4 * nc, 0.0));
    const int BUX = 0, BUY = nc, BK = 2 * nc, BOM = 3 * nc;

    // blend coefficients φ = F1·φ1 + (1−F1)·φ2
    const double sk1 = sst_.coeffs.sigma_k1, sk2 = sst_.coeffs.sigma_k2;
    const double sw1 = sst_.coeffs.sigma_w1, sw2 = sst_.coeffs.sigma_w2;
    const double al1 = sst_.coeffs.alpha1,   al2 = sst_.coeffs.alpha2;
    const double be1 = sst_.coeffs.beta1,    be2 = sst_.coeffs.beta2;

    // ---- internal faces: convection (θ-indep) + diffusion (θ via nuT, F1) ----------
    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        const int o = face.owner, n = face.neighbor;
        const double w = face.weight;
        const double SfD = face.area / std::max(face.delta, 1e-20);
        const double nuTw = w * work.nuT[o] + (1.0 - w) * work.nuT[n];
        const double F1f  = w * work.F1[o]  + (1.0 - w) * work.F1[n];
        const double skf  = sk1 * F1f + sk2 * (1.0 - F1f);
        const double swf  = sw1 * F1f + sw2 * (1.0 - F1f);
        const double dUx = work.U[n].x - work.U[o].x;
        const double dUy = work.U[n].y - work.U[o].y;
        const double dk  = work.k[n]   - work.k[o];
        const double dw  = work.omega[n] - work.omega[o];

        for (int j = 0; j < 11; ++j) {
            const double dnuTw = w * cs[o].dnuT[j] + (1.0 - w) * cs[n].dnuT[j];
            const double dF1f  = w * cs[o].dF1[j]  + (1.0 - w) * cs[n].dF1[j];

            // momentum: nuEff_f = ν + nuTw
            const double dDfm = SfD * dnuTw;
            if (dDfm != 0.0) {
                dR[j][BUX + o] += dDfm * dUx;  dR[j][BUX + n] -= dDfm * dUx;
                dR[j][BUY + o] += dDfm * dUy;  dR[j][BUY + n] -= dDfm * dUy;
            }
            // k: nuEff_f = ν + σ_k(F1)·nuTw
            double dsk = (sk1 - sk2) * dF1f;
            if (j == 0) dsk += F1f; else if (j == 4) dsk += (1.0 - F1f);
            const double dDfk = SfD * (dsk * nuTw + skf * dnuTw);
            if (dDfk != 0.0) { dR[j][BK + o] += dDfk * dk; dR[j][BK + n] -= dDfk * dk; }
            // ω: nuEff_f = ν + σ_w(F1)·nuTw
            double dsw = (sw1 - sw2) * dF1f;
            if (j == 1) dsw += F1f; else if (j == 5) dsw += (1.0 - F1f);
            const double dDfw = SfD * (dsw * nuTw + swf * dnuTw);
            if (dDfw != 0.0) { dR[j][BOM + o] += dDfw * dw; dR[j][BOM + n] -= dDfw * dw; }
        }
    }

    // ---- boundary faces: both in/out branches give dDb·(φ_b − φ_o) (mFlux θ-indep) ---
    // wall faces carry a MOLECULAR momentum diffusion coefficient at resolved
    // walls (assembleMomentum), so their U-block theta-derivative is zero
    // there; the k/omega wall diffusion keeps its nuT dependence unchanged
    std::vector<char> isWallB(mesh_.nFaces(), 0);
    if (!settings_.useWallFunctions) {
        for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
            const Patch& pat = mesh_.patch(pi);
            if (pat.type != "wall") continue;
            for (FaceID wfi : pat.faces) isWallB[wfi] = 1;
        }
    }
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face& face = mesh_.face(fi);
        const int o = face.owner;
        const double SfD = face.area / std::max(face.delta, 1e-20);
        const double F1o = work.F1[o], nuTo = work.nuT[o];
        const double sko = sk1 * F1o + sk2 * (1.0 - F1o);
        const double swo = sw1 * F1o + sw2 * (1.0 - F1o);
        const double dUx = work.U.bface(fi).x - work.U[o].x;
        const double dUy = work.U.bface(fi).y - work.U[o].y;
        const double dk  = work.k.bface(fi)   - work.k[o];
        const double dw  = work.omega.bface(fi) - work.omega[o];

        for (int j = 0; j < 11; ++j) {
            const double dnuTo = cs[o].dnuT[j], dF1o = cs[o].dF1[j];
            const double dDbm = isWallB[fi] ? 0.0 : SfD * dnuTo;
            if (dDbm != 0.0) { dR[j][BUX + o] += dDbm * dUx; dR[j][BUY + o] += dDbm * dUy; }
            double dsk = (sk1 - sk2) * dF1o;
            if (j == 0) dsk += F1o; else if (j == 4) dsk += (1.0 - F1o);
            const double dDbk = SfD * (dsk * nuTo + sko * dnuTo);
            if (dDbk != 0.0) dR[j][BK + o] += dDbk * dk;
            double dsw = (sw1 - sw2) * dF1o;
            if (j == 1) dsw += F1o; else if (j == 5) dsw += (1.0 - F1o);
            const double dDbw = SfD * (dsw * nuTo + swo * dnuTo);
            if (dDbw != 0.0) dR[j][BOM + o] += dDbw * dw;
        }
    }

    // ---- transpose-stress theta-derivative ------------------------------------------
    // the momentum source carries div(nuT (grad U)^T) (StressOperators.hpp), whose
    // theta-dependence is through nuT alone (U is the frozen state):
    //   d(flux)/dtheta_j = dnuT_f (dU_./dx_i)_f . n A, mirrored face-by-face
    if (includeTransposeTheta) {
        VelocityGradients vgT = computeVelocityGradients(work.U);
        auto gcol = [&](int ci, int comp) -> Vec3 {
            if (comp == 0)
                return Vec3(vgT.dudx[ci].x, vgT.dvdx[ci].x, vgT.dwdx[ci].x);
            return Vec3(vgT.dudx[ci].y, vgT.dvdx[ci].y, vgT.dwdx[ci].y);
        };
        for (int fi = 0; fi < nIF; ++fi) {
            const Face& face = mesh_.face(fi);
            const int o = face.owner, n2 = face.neighbor;
            const double w = face.weight;
            const Vec3 gx = gcol(o, 0) * w + gcol(n2, 0) * (1.0 - w);
            const Vec3 gy = gcol(o, 1) * w + gcol(n2, 1) * (1.0 - w);
            const double fx = gx.dot(face.normal) * face.area;
            const double fy = gy.dot(face.normal) * face.area;
            for (int j = 0; j < 11; ++j) {
                const double dnuTf = w * cs[o].dnuT[j] + (1.0 - w) * cs[n2].dnuT[j];
                if (dnuTf == 0.0) continue;
                dR[j][BUX + o] += dnuTf * fx;  dR[j][BUX + n2] -= dnuTf * fx;
                dR[j][BUY + o] += dnuTf * fy;  dR[j][BUY + n2] -= dnuTf * fy;
            }
        }
        // boundary faces, patch-aware: WALL faces carry a zero transpose
        // coefficient in the assembly (the eddy viscosity vanishes at the
        // wall), so their theta-derivative is identically zero and they are
        // skipped here to keep the analytic dR matched to an FD of the
        // assembled residual
        for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
            const Patch& pat = mesh_.patch(pi);
            if (pat.type == "wall") continue;
            for (FaceID fi : pat.faces) {
                const Face& face = mesh_.face(fi);
                const int o = face.owner;
                const double fx = gcol(o, 0).dot(face.normal) * face.area;
                const double fy = gcol(o, 1).dot(face.normal) * face.area;
                for (int j = 0; j < 11; ++j) {
                    const double dnuTo = cs[o].dnuT[j];
                    if (dnuTo == 0.0) continue;
                    dR[j][BUX + o] += dnuTo * fx;
                    dR[j][BUY + o] += dnuTo * fy;
                }
            }
        }
    }

    // ---- volumetric sources -------------------------------------------------------
    for (int ci = 0; ci < nc; ++ci) {
        const double V   = mesh_.cell(ci).volume;
        const double F1  = work.F1[ci];
        const double omc = std::max(work.omega[ci], 1e-20);
        const double S   = Smag[ci];
        const double ko  = work.k[ci], wo = work.omega[ci];
        // corrected omega production q = min(S², 10 β* k ω / νT): the branch decision
        // must mirror productionOmega/the assembly exactly (same guards) so this
        // analytic derivative matches a FD of the assembled residual
        const double bS   = sst_.coeffs.betaStar;
        const double alB  = sst_.coeffs.alpha(F1);
        const double nuTc = std::max(work.nuT[ci], 1e-30);
        const double lim  = 10.0 * bS * std::max(ko, 0.0) * omc / nuTc;
        const bool   limActive = lim < S * S;   // std::min(S², lim) picks S² on ties
        const double q = limActive ? lim : S * S;
        for (int j = 0; j < 11; ++j) {
            // k production +V·∂Pk ;  k destruction −V·∂β*·ω·k (β* = idx 8)
            dR[j][BK + ci] += V * cs[ci].dPk[j];
            if (j == 8) dR[j][BK + ci] -= omc * V * ko;
            // ω production +V·∂[α·q]: ∂α·q always; the limiter branch adds
            // α·∂q with ∂q = lim·(δ_{j,β*}/β* − ∂νT/νT) (k, ω are frozen state;
            // the S² branch has ∂q = 0, reproducing the pre-correction ∂α·S²)
            double dal = (al1 - al2) * cs[ci].dF1[j];
            if (j == 3) dal += F1; else if (j == 7) dal += (1.0 - F1);
            double dq = 0.0;
            if (limActive)
                dq = (j == 8 ? lim / bS : 0.0) - lim * cs[ci].dnuT[j] / nuTc;
            dR[j][BOM + ci] += (dal * q + alB * dq) * V;
            // ω destruction −V·∂β·ω·ω
            double dbe = (be1 - be2) * cs[ci].dF1[j];
            if (j == 2) dbe += F1; else if (j == 6) dbe += (1.0 - F1);
            dR[j][BOM + ci] -= dbe * omc * V * wo;
            // ω cross-diffusion +V·[ −∂F1·CDkw + (1−F1)·∂CDkw ], UNCLIPPED per
            // SST-2003 (matches the corrected assembly; only F1's internal CDkw
            // is clipped, and that path is inside cs.dF1 already)
            double dcross = -cs[ci].dF1[j] * work.CDkw[ci]
                            + (1.0 - F1) * cs[ci].dCDkw[j];
            dR[j][BOM + ci] += V * dcross;
        }
    }

    sst_.coeffs = saved;
    return dR;
}
// Physics-based PRECONDITIONER blocks (Picard segregated operators) at the fixed state — see
// SIMPLESolver.hpp.  NOT the held ∂R/∂U assembly: each A_block ≈ −∂R_block/∂block (no cross
// coupling, no transpose).  Used to precondition the matrix-free coupled tangent.
void SIMPLESolver::assemblePreconBlocks(const FlowFields& state, const SSTCoefficients& theta,
                                        LinearSystem& Amom, LinearSystem& Ak, LinearSystem& Aom,
                                        std::vector<double>& aP) {
    const int nc = mesh_.nCells();
    SSTCoefficients saved = sst_.coeffs;
    sst_.coeffs = theta;

    FlowFields work = state;
    ScalarField Smag(mesh_, "Smag");
    recomputeClosure(work, Smag);            // closure (nuT/F1/.../floor) + frozen |S| from θ

    aP.assign(nc, 0.0);
    Amom = makeSystem(mesh_); assembleMomentum(Amom, work, 0, aP);   // x-momentum; aP = relaxed diag
    Ak   = makeSystem(mesh_); assembleKEquation(Ak, work);
    Aom  = makeSystem(mesh_); assembleOmegaEquation(Aom, work, Smag);

    sst_.coeffs = saved;
}
