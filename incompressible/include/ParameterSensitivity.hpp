#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
#include "SSTModel.hpp"
#include "SIMPLESolver.hpp"
#include "ObservationOperator.hpp"
#include "BoundaryCondition.hpp"
#include "InferenceParameters.hpp"
#include "EvaluationTypes.hpp"
#include "MatrixFreeSolver.hpp"
#include <vector>

// ADJOINT GROUNDWORK orchestrator (∂R/∂θ + ∂g/∂θ increment; NON-HELD — see
// IMPLEMENTATION_SUMMARY.md §8 and the standing adjoint plan).
//
// Fixes one converged primary state q* = (U, p, k, ω) and exposes, at that FIXED state,
// the EXACT analytic parameter sensitivities of the discrete residual and of the
// observation operator, w.r.t. the 11 SST coefficients θ:
//
//   residual(θ)        → R(q*, θ)            (stacked [Rux|Ruy|Rk|Rω]; ≈0 at θ*)
//   dResidualDTheta(θ) → ∂R/∂θ   (11 × nState)   — analytic, reuses SIMPLESolver
//   observe(θ)         → g(q*, θ)            (closure recomputed from θ → predictions)
//   dObsDTheta(θ)      → ∂g/∂θ   (nObs × 11)     — analytic (∂g/∂nuT · ∂nuT/∂θ)
//
// The forward state q* is held fixed throughout; only θ varies and only the SST closure
// (nuT, F1, F2, Pk, CDkw) is recomputed.  NOTHING here differentiates w.r.t. the fields
// — the held (∂R/∂U)ᵀ adjoint core is untouched.  The Python FD machinery validates
// dResidualDTheta against a fixed-state FD of residual(θ), and dObsDTheta against a
// fixed-state FD of observe(θ).

// ---- RUNG 1 — semi-analytic true-model gradient result (NON-HELD) ------------------
// Output of etaJacobianTangent: the full observable Jacobian ∂η/∂θ (nObs × 11) plus the
// per-coefficient matrix-free-Krylov diagnostics and cost accounting.  ∂g/∂θ ≡ 0, so this
// Jacobian IS the implicit-coupling term  (∂g/∂U)·(−(∂R/∂U)⁻¹ ∂R/∂θ)  computed without
// forming the held analytic (∂R/∂U)ᵀ core — the tangent solves are matrix-free.
struct TangentGradientResult {
    std::vector<std::vector<double>> dObsDTheta;   // nObs × 11  (∂η/∂θ)
    std::vector<double> logLikGradient;            // 11 — ∂logL/∂θ (Gaussian); warm-FD fills this
    std::vector<int>    krylovIters;               // 11 — Krylov/outer iterations per coefficient
    std::vector<double> krylovRelRes;              // 11 — final ‖r‖/‖b‖ (or continuity) per coeff
    std::vector<int>    krylovConverged;           // 11 — 1/0 (int for a clean binding)
    int nResidualEvals = 0;                        // total residual assemblies / solve iterations
    int nColors        = 0;                        // colours used for the colored-FD diagonal
};

class ParameterSensitivity {
public:
    ParameterSensitivity(const Mesh& mesh,
                         const ObservationOperator& obs,
                         const FlowBoundaryConditions& bcs,
                         double nu,
                         const SolverSettings& settings = {},
                         const Vec3& Uinit = {1, 0, 0},
                         double pInit = 0.0, double kInit = 1e-4, double omegaInit = 1.0);

    // Solve once at θ (full 11-vector) and store the converged state as the fixed point.
    EvaluationStatus solveState(const std::vector<double>& theta11);

    bool hasState() const { return hasState_; }
    int  nState()  const { return 4 * mesh_.nCells(); }     // [Ux|Uy|k|ω] blocks
    int  nCells()  const { return mesh_.nCells(); }
    int  nObs()    const { return obs_.nObs(); }

    // R(q*, θ): stacked discrete residual at the fixed state, closure recomputed from θ.
    std::vector<double> residual(const std::vector<double>& theta11);
    // ∂R/∂θ: 11 residual-shaped vectors (analytic).
    std::vector<std::vector<double>> dResidualDTheta(const std::vector<double>& theta11);

    // g(q*, θ): observation predictions with the SST closure recomputed from θ.
    std::vector<double> observe(const std::vector<double>& theta11);
    // Scalar Gaussian log-likelihood at the fixed state (closure recomputed from θ).  With a
    // preceding solveState(θ) this is logL at the converged state — the WarmFDForwardModel loglik.
    double logLik(const std::vector<double>& theta11);
    // ∂g/∂θ: nObs × 11 (analytic, via ∂g/∂nuT · ∂nuT/∂θ).
    std::vector<std::vector<double>> dObsDTheta(const std::vector<double>& theta11);

    // ---- RUNG 1 — semi-analytic true-model observable gradient ∂η/∂θ (NON-HELD) -----
    // Full true-model dη/dθ at the fixed converged state, via the matrix-free tangent
    // solve  (∂R/∂U) w_j = −∂R/∂θ_j  (full turbulence coupling; frozen pressure, matching
    // the program's 4-block residual R = [Rux|Ruy|Rk|Rω]).  For each of the 11 coefficients
    // a column-scaled, Jacobi-preconditioned BiCGSTAB obtains w_j = dU/dθ_j matrix-free
    // (Jv ≈ [R(U*+εv) − R(U*−εv)]/(2ε)); ∂η_i/∂θ_j is then the directional derivative of
    // observable i along w_j.  NEVER assembles the held analytic (∂R/∂U)ᵀ core.  Requires
    // a converged fixed state (call solveState first); returns an all-zero Jacobian if not.
    TangentGradientResult etaJacobianTangent(const std::vector<double>& theta11,
                                             double krylovTol = 1e-8, int maxIter = 3000,
                                             double fdStep = 1e-6);

    // ---- RUNG 1 (PRESSURE-COUPLED) — the FULL true-model gradient ∂η/∂θ (NON-HELD) ---
    // Augments the tangent with the pressure/continuity coupling the frozen-pressure
    // etaJacobianTangent omits: the unknown is the 5-block [Ux|Uy|k|ω|p] and the residual
    // gains the discrete continuity row R_cont = ∇·U (the solver's plain mass-flux
    // divergence — no Rhie-Chow).  The momentum↔pressure (∇p source) and continuity↔velocity
    // coupling enter the matrix-free Jv automatically; a SIMPLE preconditioner (Jacobi on the
    // momentum/turbulence blocks + a pressure-Poisson Schur solve, reusing the forward
    // ∇·((V/aP)∇p') operator) makes the indefinite saddle tractable for BiCGSTAB.  This
    // recovers dp/dθ, so dη/dθ matches full FD (the frozen-pressure bias is removed).  Still
    // NEVER forms the held analytic (∂R/∂U)ᵀ core.  Requires a converged fixed state.
    TangentGradientResult etaJacobianTangentCoupled(const std::vector<double>& theta11,
                                                    double krylovTol = 1e-8, int maxIter = 3000,
                                                    double fdStep = 1e-6);

    // ---- RUNG 1 (WARM-FD) — the robust full true-model gradient ∂η/∂θ (NON-HELD) -------
    // Central finite difference of the WHOLE solve: re-converge SIMPLE at θ±h WARM-STARTED
    // from the fixed converged state (not a cold uniform init), then central-difference η.
    // It IS the full-FD true-model gradient (matches cold FD exactly — same fixed point, no
    // frozen-pressure bias, no Krylov-saddle fragility), only far cheaper because each
    // perturbed re-solve starts ~converged.  β*/a1 (the only coefficients nonlinear in the
    // closure) use a smaller step.  Reported as the robust default vs the semi-analytic
    // etaJacobianTangentCoupled.  Requires a converged fixed state (call solveState first).
    // warmMaxIter / warmTol override the solver's iteration cap / convergence tolerance for
    // the perturbed re-solves (0 ⇒ use settings).  A warm re-solve re-equilibrates the small
    // θ-perturbation in far fewer iterations than a cold solve, so a looser cap is both safe
    // and the source of the speedup (central FD: θ±h re-solve from the same warm state, so a
    // shared under-convergence cancels).
    TangentGradientResult etaJacobianWarmFD(const std::vector<double>& theta11,
                                            double hRel = 5e-4, double hFloor = 1e-7,
                                            int warmMaxIter = 0, double warmTol = 0.0);

private:
    const Mesh& mesh_;
    ObservationOperator    obs_;
    FlowBoundaryConditions bcs_;
    double nu_;
    SolverSettings settings_;
    Vec3   Uinit_;
    double pInit_, kInit_, omegaInit_;

    FlowFields state_;
    bool       hasState_ = false;

    SSTModel makeModel(const std::vector<double>& theta11) const;

    // Greedy distance-1 colouring of the cell face-adjacency graph: two face-neighbour
    // cells never share a colour, so a single perturbation of all same-colour cells reads
    // the exact tangent-operator diagonal ∂R_(i,f)/∂x_(i,f) at every probed cell (the
    // residual stencil is distance-1).  Used to build the Jacobi preconditioner.
    std::vector<int> cellColoring(int& nColors) const;

    // Per-cell discrete continuity residual Σ_f (U_f·n) S_f (the SIMPLE mass-flux divergence,
    // no Rhie-Chow).  `homogeneous` true ⇒ tangent boundary fluxes (fixed-velocity walls/inlet
    // contribute 0; outlet uses zero-gradient = owner increment); false ⇒ use the state's
    // boundary face velocities.  Operates on the supplied interior velocity components.
    std::vector<double> massFluxDivergence(const std::vector<double>& ux,
                                           const std::vector<double>& uy,
                                           bool homogeneous) const;

    // Assemble the SIMPLE pressure-Poisson Laplacian ∇·((V/aP)∇p') with outlet-Dirichlet
    // (p'=0), matching assemblePressureCorrection — the Schur operator of the SIMPLE
    // preconditioner.  Matrix only (zero source); `aP` is the momentum diagonal.
    LinearSystem assemblePressurePoisson(const std::vector<double>& aP) const;
    // Recompute the closure of `work` from θ (computeFields + 0.1ν floor); if `cs` is
    // non-null also fill the per-cell pointwise closure sensitivities (and |S|).
    void recompute(const std::vector<double>& theta11, FlowFields& work,
                   std::vector<SSTClosureSensitivity>* cs) const;

    int nearestWallFace(const std::string& patch, const Vec3& loc) const;
};
