#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include <cmath>
#include <algorithm>

// Container for SST k-omega turbulence model constants
struct SSTCoefficients {
    // k-omega model coefficients (near-wall layer)
    double sigma_k1  = 0.85;
    double sigma_w1  = 0.5;
    double beta1     = 0.075;
    double alpha1    = 5.0 / 9.0;   // ~0.5556

    // k-epsilon model coefficients (expressed in k-omega form) (away-from-wall layer)
    double sigma_k2  = 1.0;
    double sigma_w2  = 0.856;
    double beta2     = 0.0828;
    double alpha2    = 0.44;

    // general SST constants
    double betaStar  = 0.09;
    double a1        = 0.31;        // Bradshaw structural parameter
    double kappa     = 0.41;        // von Karman (usually fixed)

    // blended coefficient: phi = F1*phi1 + (1-F1)*phi2
    // F1 = 1: pure k-omega
    // F1 = 0: pure k-epsilon
    double blend(double phi1,   // k-omega coefficient 
                 double phi2,   // k-epsilon coefficient    
                 double F1)     // blending function
                 const {
        return F1 * phi1 + (1.0 - F1) * phi2;
    }
    double sigma_k(double F1) const { return blend(sigma_k1, sigma_k2, F1); }
    double sigma_w(double F1) const { return blend(sigma_w1, sigma_w2, F1); }
    double beta(double F1)    const { return blend(beta1, beta2, F1); }
    double alpha(double F1)   const { return blend(alpha1, alpha2, F1); }
};

// PHASE 3 (angle 7) — term-toggled closure variants for Bayesian model selection.
// Competing models are toggles in the SAME SST model, never new solvers:
//   Full      — standard Menter SST.
//   NoLimiter — drop the Bradshaw shear-stress limiter: nuT = a1 k/max(a1 w, S F2)
//               -> nuT = k/w  (standard k-w eddy viscosity).
//   KOmega    — force F1 = 1 everywhere (baseline k-w, no k-e blending).
enum class SSTVariant { Full = 0, NoLimiter = 1, KOmega = 2 };

// ADJOINT GROUNDWORK (∂R/∂θ increment; non-held, see IMPLEMENTATION_SUMMARY.md §8).
// Pointwise analytic sensitivities of the closure quantities that enter the discrete
// residual, w.r.t. each of the 11 SST coefficients θ, at FIXED primary state
// (k, ω, U).  These are the algebraic building blocks the eventual discrete adjoint
// consumes unchanged.  Indexing matches InferenceParameterSet:
//   0 σ_k1  1 σ_w1  2 β1  3 α1  4 σ_k2  5 σ_w2  6 β2  7 α2  8 β*  9 a1  10 κ
// Only nuT (floored, for diffusion), Pk, F1 and CDkw enter the assembly directly; F2
// is upstream of nuT and so appears only inside dnuT (the β* path).
struct SSTClosureSensitivity {
    double dnuT [11] = {0};   // ∂(floored nuT)/∂θ_j   — momentum/k/ω diffusion
    double dPk  [11] = {0};   // ∂Pk/∂θ_j              — k production (uses UNfloored nuT)
    double dF1  [11] = {0};   // ∂F1/∂θ_j              — σ_k/σ_w/α/β blends + cross-diff
    double dCDkw[11] = {0};   // ∂CDkw/∂θ_j (raw)      — ω cross-diffusion
};

// SSTModel
// Computes F1, F2, nuT, Pk, and source terms
class SSTModel {
public:
    SSTCoefficients coeffs;
    SSTVariant      variant = SSTVariant::Full;   // closure-structure toggle (Phase 3)

    // Constructors
    SSTModel() = default;
    explicit SSTModel(const SSTCoefficients& c) : coeffs(c) {}
    SSTModel(const SSTCoefficients& c, SSTVariant v) : coeffs(c), variant(v) {}

    // Pointwise functions - operate on one control volume (single cell)
    double computeF1(double k, double omega, double y, double nu, double CDkw_pos) const;       // primary blending function (detects if cell is in boundary layer)
    double computeF2(double k, double omega, double y, double nu) const;                        // second blending function (used in eddy viscosity limiter)
    double crossDiffusion(const Vec3& gradK, const Vec3& gradOmega, double omega) const;        // computes cross-diffusion term in omega equation
    double eddyViscosity(double k, double omega, double S, double F2) const;                    // computes turbulent viscosity
    double production(double nuT, double S, double k, double omega) const;                      // computes turbulence production term
    double sourceK(double Pk, double k, double omega) const;                                    // computes source term in k equation
    double sourceOmega(double Pk, double nuT, double omega, double F1, double CDkw) const;      // computes RHS of omega equation

    // ADJOINT GROUNDWORK — pointwise analytic ∂(closure)/∂θ at one cell, holding the
    // primary state (k, ω, U) fixed.  Re-derives the same branch decisions computeFields
    // makes (eddy-viscosity Bradshaw branch, F2/F1 min/max winners, Pk Menter limiter,
    // nuT floor) so the analytic derivative matches a FD of computeFields exactly.
    //   k,omega          — cell k and ω (raw; safeguarded internally as in computeFields)
    //   S                — strain-rate magnitude |S|
    //   y,nu             — wall distance, molecular viscosity
    //   CDkw             — raw cross-diffusion value (signed) for this cell
    //   nuTFloor         — eddy-viscosity floor the solver applies (0.1·ν); dnuT=0 below it
    SSTClosureSensitivity closureSensitivity(double k, double omega, double S,
                                             double y, double nu,
                                             double CDkw, double nuTFloor) const;

    // Computes for the entire mesh
    void computeFields(const Mesh& mesh,
                       const ScalarField& k, const ScalarField& omega,
                       const VectorField& U, double nu,
                       ScalarField& nuT, ScalarField& F1field,
                       ScalarField& F2field, ScalarField& Pk,
                       ScalarField& CDkwField) const;
};