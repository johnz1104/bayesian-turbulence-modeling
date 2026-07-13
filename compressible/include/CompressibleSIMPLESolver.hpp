#pragma once
#include "Mesh.hpp"
#include "Field.hpp"
#include "LinearSolver.hpp"
#include "SSTModel.hpp"
#include "FlowFields.hpp"        // SolverSettings
#include "CompressibleFlowFields.hpp"
#include "CompressibleBCs.hpp"
#include "IdealGasEOS.hpp"
#include <memory>
#include <vector>

// Pressure-based compressible SIMPLE solver for steady RANS with ideal gas.
//
// Algorithm (per outer iteration):
//   1. Update density from EOS:  ρ = p/(R T)
//   2. Update dynamic viscosity via Sutherland's law
//   3. Update SST turbulence model (nuT, F1, F2, Pk)
//   4. Solve momentum equations for U* (upwind convection, central diffusion)
//   5. Solve pressure-correction equation (Rhie-Chow + compressibility term)
//   6. Correct U, p, ρ
//   7. Solve energy equation: ∇·(ρ U Cp T) = ∇·(λ_eff ∇T)
//   8. Update T from energy, update ρ from EOS
//   9. Solve turbulence (k, ω) with density-scaled production
//
// Valid for Ma < ~0.8; higher Mach numbers require density-based schemes.
class CompressibleSIMPLESolver {
public:
    CompressibleSIMPLESolver(const Mesh& mesh,
                             const SSTModel& sst,
                             const CompressibleBoundaryConditions& bcs,
                             const IdealGasEOS& eos,
                             const SolverSettings& settings = {});

    CompressibleConvergenceHistory solve(CompressibleFlowFields& fields);

    void initUniform(CompressibleFlowFields& f,
                     const Vec3& Uinit,
                     double p_init,
                     double T_init,
                     double kInit,
                     double omegaInit);

private:
    const Mesh&                   mesh_;
    SSTModel                      sst_;
    CompressibleBoundaryConditions bcs_;
    IdealGasEOS                   eos_;
    SolverSettings                settings_;

    std::unique_ptr<ILinearSolver> pSolver_;
    std::unique_ptr<ILinearSolver> mSolver_;
    std::unique_ptr<ILinearSolver> tSolver_;
    std::unique_ptr<ILinearSolver> eSolver_;

    // Momentum diagonal coefficients for Rhie-Chow
    std::vector<double> aP_;
    std::vector<double> aPunrelaxed_;
    // Cell-centred dynamic viscosity (from Sutherland at current T)
    std::vector<double> mu_;


    double normUx0_ = 1, normUy0_ = 1, normP0_ = 1, normT0_ = 1;

    void updateViscosity(const CompressibleFlowFields& f);
    void updateDensity(CompressibleFlowFields& f);
    // The prescribed outlet pressure is a THERMODYNAMIC static value; the
    // field carries the mechanical working pressure, so every BC application
    // is followed by this conversion of the outlet boundary faces,
    // p_mech,b = p_out + (2/3) rho_o k_o with the owner-cell state standing
    // in for the boundary turbulence energy.
    void mechanicalizeOutletPressure(CompressibleFlowFields& f) const;

    void assembleMomentum(LinearSystem& sys, const CompressibleFlowFields& f,
                          int component, std::vector<double>& aP);
    void assemblePressureCorrection(LinearSystem& sys, const CompressibleFlowFields& f,
                                    const std::vector<double>& aP,
                                    ScalarField& pPrime);
    void assembleEnergy(LinearSystem& sys, const CompressibleFlowFields& f);
    void assembleKEquation(LinearSystem& sys, const CompressibleFlowFields& f);
    void assembleOmegaEquation(LinearSystem& sys, const CompressibleFlowFields& f,
                               const ScalarField& Smag);

    void correctVelocity(CompressibleFlowFields& f, const ScalarField& pPrime,
                         const std::vector<double>& aP);
    void correctPressure(CompressibleFlowFields& f, const ScalarField& pPrime);

    double computeResidual(const CompressibleFlowFields& f, int component);
    double computeScalarResidual(const ScalarField& phi_old,
                                 const ScalarField& phi_new);
};
