#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
#include "BoundaryCondition.hpp"
#include "LinearSolver.hpp"
#include "SSTModel.hpp"
#include <vector>
#include <string>
#include <memory>

// SIMPLE = Semi-Implicit Method for Pressure-Linked Equations
// Iteratively solves momentum and pressure-correction equations to enforce
// continuity and converge the steady incompressible RANS flow field.
class SIMPLESolver {
public:
    SIMPLESolver(const Mesh& mesh,
                 const SSTModel& sst,
                 const FlowBoundaryConditions& bcs,
                 double nu,
                 const SolverSettings& settings = {});

    ConvergenceHistory solve(FlowFields& fields);

    void initUniform(FlowFields& f, const Vec3& Uinit,
                     double pInit, double kInit, double omegaInit);

    // ---- ADJOINT GROUNDWORK (∂R/∂θ increment; non-held — IMPLEMENTATION_SUMMARY.md §8) ----
    // Discrete residual R(U,θ) of the momentum-x, -y, k and ω equations at a FIXED primary
    // state, with the SST closure (nuT,F1,F2,Pk,CDkw) recomputed from `theta` exactly as
    // the solver does (computeFields + 0.1ν nuT floor + frozen |S|).  No solve / no
    // iteration; reuses assembleMomentum/assembleKEquation/assembleOmegaEquation so the
    // discretisation is identical to solve() and R(U*,θ*) ≈ 0 by construction.  Returns
    // the stacked vector [Rux | Ruy | Rk | Rω], each block nCells long.
    std::vector<double> assembleResidual(const FlowFields& state,
                                         const SSTCoefficients& theta);

    // Exact analytic ∂R/∂θ — the derivative of assembleResidual at fixed state — as 11
    // residual-shaped vectors (one per coefficient, same [Rux|Ruy|Rk|Rω] layout).  Built
    // from the pointwise SSTModel::closureSensitivity blocks; differentiates ONLY w.r.t.
    // the 11 coefficients and NEVER forms ∂R/∂U (the held adjoint core).
    std::vector<std::vector<double>> assembleResidualSensitivity(const FlowFields& state,
                                                                 const SSTCoefficients& theta);

    // Assemble the (Picard) segregated block operators at a FIXED state with the closure from
    // `theta`: the momentum-x system A_mom (its relaxed diagonal returned in aP), the k system
    // A_k, and the ω system A_om.  These are used ONLY as a physics-based PRECONDITIONER for the
    // matrix-free coupled tangent (full within-block solves capture the stiff ω/k coupling that
    // a Jacobi diagonal misses) — they are NOT the held analytic ∂R/∂U assembly (no cross-block
    // coupling, no transpose; A ≈ −∂R_block/∂block per block).  Reuses assembleMomentum/K/Omega.
    void assemblePreconBlocks(const FlowFields& state, const SSTCoefficients& theta,
                              LinearSystem& Amom, LinearSystem& Ak, LinearSystem& Aom,
                              std::vector<double>& aP);

    int residualSize() const { return 4 * mesh_.nCells(); }   // length of one residual vector

private:
    const Mesh& mesh_;
    SSTModel sst_;
    FlowBoundaryConditions bcs_;
    double nu_;
    SolverSettings settings_;

    std::unique_ptr<ILinearSolver> pSolver_;
    std::unique_ptr<ILinearSolver> mSolver_;
    std::unique_ptr<ILinearSolver> tSolver_;

    // Diagonal momentum coefficients retained for Rhie-Chow interpolation.
    // aP_ is the relaxed diagonal (aP_raw/alphaU) used for velocity correction;
    // aPunrelaxed_ is the raw diagonal used for the pressure Laplacian.
    std::vector<double> aP_;
    std::vector<double> aPunrelaxed_;

    // Iter-0 residual norms for normalisation. Equations with negligible iter-0
    // residual (e.g. Uy in 2D) are skipped via safeNorm to avoid false divergence.
    double normUx0_ = 1, normUy0_ = 1, normP0_ = 1;

    void assembleMomentum(LinearSystem& sys, const FlowFields& f,
                          int component, std::vector<double>& aP);
    void assemblePressureCorrection(LinearSystem& sys, const FlowFields& f,
                                    const std::vector<double>& aP, ScalarField& pPrime);
    void assembleKEquation(LinearSystem& sys, const FlowFields& f);
    // Smag is frozen from pre-correction computeFields to prevent pressure-correction
    // oscillations from amplifying omega production and causing blow-up.
    void assembleOmegaEquation(LinearSystem& sys, const FlowFields& f,
                               const ScalarField& Smag);

    void correctVelocity(FlowFields& f, const ScalarField& pPrime,
                         const std::vector<double>& aP);
    void correctPressure(FlowFields& f, const ScalarField& pPrime);

    double computeResidual(const FlowFields& f, int component);

    // ADJOINT GROUNDWORK helper: recompute the SST closure fields of `work` from the
    // current sst_.coeffs (computeFields + 0.1ν nuT floor) and the frozen strain |S|,
    // exactly as the SIMPLE loop does on a turbulence-update iteration.
    void recomputeClosure(FlowFields& work, ScalarField& Smag);
};
