#pragma once

#include "Mesh.hpp"
#include "DBNSTypes.hpp"
#include "DBNSConfig.hpp"
#include "SSTModel.hpp"
#include "EvaluationTypes.hpp"
#include <array>
#include <cstdint>
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

// Diagnostics of the model-form injection (mirrors the incompressible
// solver's record): the per-residual-evaluation realizability re-check of the
// injected anisotropy target, recorded rather than silently projected, so a
// study can report violations (the caller projects before setting).
struct InjectionDiagnostics {
    bool   active = false;
    int    checkedIters = 0;
    bool   allRealizable = true;
    double maxViolation = 0.0;   // worst negative barycentric margin seen
    double maxDb = 0.0;          // max |b_target| component magnitude
    double maxDq = 0.0;          // max |dq_target| component magnitude
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

    // Compute derived properties (gradients, laminar/eddy viscosity) for the
    // current state without advancing it.  Lets the observation operator be
    // exercised on a prescribed field independent of solver convergence.
    void prepareProperties() { computeGradients(); updateProperties(); }

    // --- accessors for observation / extraction ---
    int nCells() const { return mesh_.nCells(); }
    Primitive primitive(int ci) const { return GasState::toPrimitive(W_[ci], eos_); }
    const StateVec& conserved(int ci) const { return W_[ci]; }
    double eddyViscosity(int ci) const { return muT_[ci]; }    // mu_t [Pa s]
    double laminarViscosity(int ci) const { return muLam_[ci]; } // mu [Pa s] (honours constMu)
    const Mesh& mesh() const { return mesh_; }
    const IdealGasEOS& eos() const { return eos_; }

    // --- model-form injection (the deferred-correction coupling) ---
    // Set the sampled closure correction: per-cell target anisotropy b
    // (nCells x 6, ordered xx, yy, zz, xy, xz, yz) and per-cell turbulent
    // heat-flux correction dq (nCells x 2, x and y components, SOLVER units
    // of a Favre temperature flux, <rho u_i''T''>/<rho> [m/s K]). The stress
    // difference enters the momentum fluxes as an explicit deferred
    // correction against the Boussinesq term (running k, mu_t and strain, so
    // a target equal to the solver's own Boussinesq anisotropy reproduces the
    // baseline identically); with energyReach the consistent stress work and
    // the injected heat flux (against the implicit gradient-diffusion term)
    // enter the energy equation, the heat-flux reach of the pre-registered
    // scheme. Values are COPIED (no caller lifetime coupling). The caller
    // projects b into the realizable set first; the solver re-checks the
    // barycentric margin every residual evaluation and records violations in
    // the diagnostics rather than silently projecting.
    void setTargetCorrection(const std::vector<double>& db6,
                             const std::vector<double>& bDiag6,
                             const std::vector<double>& dq2,
                             bool energyReach = true,
                             const std::vector<std::uint8_t>& mask = {});
    void clearTargetCorrection();
    // per-cell omega-production limiter activation of the LAST residual
    // evaluation (the solver's own branch decision, so activation maps are
    // exact rather than a python recomputation)
    const std::vector<std::uint8_t>& limiterActive() const {
        return limiterActive_;
    }
    const InjectionDiagnostics& injectionDiagnostics() const { return injDiag_; }

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

    // patch-local ordinal of each boundary face (per-face boundary profiles)
    std::vector<int> patchLocalIdx_;

    // wall-adjacent cells (owners of no-slip faces) and their center-to-face
    // distances, for the per-iteration Menter omega re-pin
    std::vector<int>    wallAdjCell_;
    std::vector<double> wallAdjDelta_;

    // wall distance measured to the patches the BOUNDARY CONDITIONS mark as
    // no-slip walls (the mesh's own lazily-computed distance is empty unless
    // a caller populates it, and its patch types are the generator's guess;
    // the turbulent path needs the true viscous walls). Computed once in the
    // constructor; 1e10 everywhere when no no-slip patch exists.
    std::vector<double> wallDist_;

    // scale-aware positivity floors, captured at initialisation so the
    // rescue values are sane at any unit scale (nondimensional shock tubes
    // and dimensional wall flows share this solver)
    double rhoFloor_ = 1e-10;
    double pFloor_ = 1e-10;

    // pipeline stages
    void updateProperties();                 // muLam, muT, SST blending
    void computeGradients();
    void computeLimiters();
    Primitive ghostState(const Primitive& interior, int faceId,
                         const BoundarySpec& spec, int boundaryIdx) const;
    double wallOmegaGhost(const Primitive& interior, int faceId) const;
    void computeResidual();                  // fills res_
    void addViscousFace(int faceId);         // internal-face viscous contribution
    void addBoundaryFlux(int faceId, int patchIdx);
    void addTurbulenceSources();
    void computeTimeStep(double cfl, bool includeViscous);
    void clampPositivity();
    void clampPositivityCore();
    double rhoResidualNorm() const;
    // per-equation volume-scaled RMS residual norms (all NVAR conservative
    // equations); the completed implicit-steady convergence criterion gates
    // on EVERY live equation's relative decay, not density alone (a member
    // could otherwise classify converged while momentum, energy or the
    // turbulence equations are still moving, exactly the false-convergence
    // class the audit closed on the pressure-based solvers)
    std::array<double, NVAR> equationResidualNorms() const;
    // direct per-cell conservative-state validation: every component finite
    // and the decoded thermodynamic state admissible (max/min reductions
    // drop NaN operands, so only an explicit sweep can prove integrity)
    bool stateIsValid() const;

    // implicit (LU-SGS) steady march: matrix-free backward-Euler pseudo-time
    // with lower-upper symmetric Gauss-Seidel sweeps on the spectral-radius-
    // split first-order Jacobian, point-implicit SST destruction diagonal
    SolveReport solveImplicitSteady();
    void computeFaceSpectralRadii();         // fills lamFace_
    std::vector<double>   lamFace_;          // face spectral radius (conv + visc)
    std::vector<StateVec> dW_;               // LU-SGS state-increment workspace

    // model-form injection state (values owned by the solver)
    void addInjectionFluxes();               // internal-face deferred correction
    std::vector<double>   dbTarget6_;        // nCells x 6 stored discrepancy
    std::vector<double>   bTarget6_;         // nCells x 6 target anisotropy (diagnostics)
    std::vector<std::uint8_t> injMask_;      // per-cell injection activity
    double injRampScale_ = 1.0;              // current ramp factor on the fluxes
    std::vector<double> injRhoK0_;           // rho k at target-set time (frozen-k)
    std::vector<std::uint8_t> limiterActive_;  // omega-limiter branch record
    std::vector<double>   dqTarget2_;        // nCells x 2 heat-flux correction
    bool                  injectEnergyReach_ = true;
    InjectionDiagnostics  injDiag_;

    // reconstruct primitive at a face from cell ci toward face center xf
    Primitive reconstruct(int ci, const Vec3& xf) const;

};

}  // namespace dbns
