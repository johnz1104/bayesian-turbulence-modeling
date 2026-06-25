#pragma once

#include "Mesh.hpp"
#include "DBNSTypes.hpp"
#include "DBNSConfig.hpp"
#include "SSTModel.hpp"
#include "EvaluationTypes.hpp"
#include <vector>

// ---------------------------------------------------------------------------
// Density-based, shock-capturing finite-volume RANS solver carrying the
// Favre-averaged SST k-omega closure.
//
// Discretization:
//   - Convective flux: HLLC approximate Riemann solver (HLLCFlux).
//   - Reconstruction:  MUSCL linear with the Venkatakrishnan limiter (2nd
//     order, TVD); switchable to 1st order or unlimited (for MMS).
//   - Viscous flux:    corrected face gradients (averaged cell gradients with a
//     normal-difference correction), full stress tensor and heat conduction.
//   - Turbulence:      compressible SST sources with optional dilatational
//     (Sarkar/Zeman) dissipation; turbulent Prandtl number set by the EOS.
//   - Time march:      explicit SSP Runge-Kutta with local time stepping for
//     steady problems and a global step for unsteady (shock tube) problems.
//
// The solver classifies outcomes (Converged / Unconverged / Diverged) instead
// of throwing, matching the repo's forward-model discipline.
// ---------------------------------------------------------------------------

namespace dbns {

struct SolveReport {
    EvaluationStatus status = EvaluationStatus::Unknown;
    int    iterations = 0;
    double finalResidual = 0.0;
    double tFinal = 0.0;
    std::vector<double> residualHistory;   // density-equation L2 residual per report
};

class DBNSSolver {
public:
    DBNSSolver(const Mesh& mesh,
               const IdealGasEOS& eos,
               const SSTCoefficients& sst,
               const DBNSBoundaryConditions& bcs,
               const DBNSSettings& settings = {});

    // Initialise every cell to a uniform primitive state.
    void initUniform(const Primitive& V);

    // Initialise from a per-cell primitive field (e.g. an MMS field).
    void initField(const std::vector<Primitive>& V);

    // Advance the solution (steady iteration or unsteady march).
    SolveReport solve();

    // --- accessors for observation / extraction ---
    int nCells() const { return mesh_.nCells(); }
    Primitive primitive(int ci) const { return GasState::toPrimitive(W_[ci], eos_); }
    const StateVec& conserved(int ci) const { return W_[ci]; }
    double eddyViscosity(int ci) const { return muT_[ci]; }   // mu_t [Pa s]
    const Mesh& mesh() const { return mesh_; }
    const IdealGasEOS& eos() const { return eos_; }

    // --- MMS support ---
    // Inject a constant-in-time per-cell source S so the discrete residual
    // balances a manufactured solution: dW/dt = -(Res - S*V)/V.
    void setManufacturedSource(const std::vector<StateVec>& src) { mmsSource_ = src; }
    // Override the boundary ghost state per boundary face (for spatially varying
    // Dirichlet, e.g. the MMS exact solution).  Indexed by boundary-face index
    // (global face id - nInternalFaces).
    void setBoundaryOverride(const std::vector<Primitive>& ghost) { bndOverride_ = ghost; }

private:
    const Mesh&             mesh_;
    IdealGasEOS             eos_;
    SSTCoefficients         sstCoeffs_;
    SSTModel                sst_;            // for F1/F2 blending formulas
    DBNSBoundaryConditions  bcs_;
    DBNSSettings            settings_;

    std::vector<StateVec>   W_;              // conserved state per cell
    std::vector<double>     vol_;            // consistent per-unit-depth cell volume
    std::vector<double>     muLam_;          // laminar dynamic viscosity per cell
    std::vector<double>     muT_;            // turbulent (eddy) dynamic viscosity
    std::vector<StateVec>   mmsSource_;      // optional MMS source (per cell)
    std::vector<Primitive>  bndOverride_;    // optional per-boundary-face ghost

    // gradients of primitive variables (rho,u,v,p,k,omega) per cell
    std::vector<std::array<Vec3, 6>> grad_;
    std::vector<std::array<double, 6>> limiter_;  // Venkatakrishnan scalar per var

    // workspace
    std::vector<StateVec>   res_;            // residual accumulator
    std::vector<double>     dtCell_;         // local time step

    // pipeline stages
    void updateProperties();                 // muLam, muT, SST blending
    void computeGradients();
    void computeLimiters();
    Primitive ghostState(const Primitive& interior, int faceId,
                         const BoundarySpec& spec, int boundaryIdx) const;
    void computeResidual();                  // fills res_
    void addViscousFace(int faceId);         // internal-face viscous contribution
    void addBoundaryFlux(int faceId, int patchIdx);
    void addTurbulenceSources();
    void computeTimeStep();
    void clampPositivity();
    double rhoResidualNorm() const;

    // reconstruct primitive at a face from cell ci toward face center xf
    Primitive reconstruct(int ci, const Vec3& xf) const;

    // helper: kinematic viscosity floor for SST blending
    static double safeOmega(double w, double floor) { return w > floor ? w : floor; }
};

}  // namespace dbns
