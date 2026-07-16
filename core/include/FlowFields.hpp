#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include <vector>
#include <string>

// Primary solved flow variables and all turbulence model fields, stored cell-centred.
struct FlowFields {
    VectorField U;          // velocity
    ScalarField p;          // pressure
    ScalarField k;          // turbulent kinetic energy
    ScalarField omega;      // specific dissipation rate

    ScalarField nuT;        // eddy viscosity
    ScalarField F1, F2;     // SST blending functions
    ScalarField Pk;         // production of k
    ScalarField CDkw;       // cross-diffusion term

    // Turbulence-establishment marker carried WITH the state: false after a
    // cold (uniform) init, set true once a solve passes the startup floor
    // window. The startup-only nuT floor consults this so a WARM restart
    // (fields from a converged cache) never re-engages the floor at iter 0,
    // which would otherwise contaminate warm-FD gradients and warm-started
    // ensembles with a floor the base state does not carry.
    bool turbEstablished = false;

    FlowFields() = default;
    explicit FlowFields(const Mesh& mesh)
        : U(mesh, "U"), p(mesh, "p"), k(mesh, "k"), omega(mesh, "omega"),
          nuT(mesh, "nuT"), F1(mesh, "F1"), F2(mesh, "F2"),
          Pk(mesh, "Pk"), CDkw(mesh, "CDkw") {}
};

// Per-iteration residual norms for all solved equations.
struct ResidualEntry {
    int    iteration = 0;
    double Ux = 0, Uy = 0, Uz = 0;
    double p  = 0;
    double k  = 0, omega = 0;
};

// Full residual history from one SIMPLE solve.
struct ConvergenceHistory {
    std::vector<ResidualEntry> entries;
    bool converged = false;
    bool diverged  = false;
    int  finalIter = 0;
};

// Configuration for the SIMPLE solver: iteration limits, tolerances,
// relaxation factors, linear solver names, and turbulence scheduling.
struct SolverSettings {
    int    maxIterations   = 500;
    double convergenceTol  = 1e-5;
    double divergenceLimit = 1e6;

    double alphaU     = 0.7;
    double alphaP     = 0.3;
    double alphaK     = 0.5;
    double alphaOmega = 0.5;
    double alphaT     = 0.7;  // temperature under-relaxation (compressible solver)

    int    innerIterations = 200;
    double innerTolerance  = 1e-3;

    int turbStartIter      = 5;
    int turbUpdateInterval = 1;

    // Startup-only eddy-viscosity floor window: the 0.1*nu floor that guards
    // against early k-omega collapse stays active only while
    // iter < turbStartIter + nuTFloorIters, then releases to a plain
    // non-negativity clamp. SST requires nuT -> 0 at a resolved wall, so a
    // permanent floor would bias the converged near-wall solution and the
    // wall stress (about +10 percent where it binds); the converged state
    // must be floor-free. Widen per-case in config if a startup needs it.
    int nuTFloorIters = 500;

    // Under-relaxation of the explicit Reynolds-stress-injection source (the
    // deferred-correction body force is blended across outer iterations at
    // this factor; 1.0 disables blending). The explicit correction cancels a
    // first-order part of the implicit nuT stabilizer, so unrelaxed feedback
    // leaves a residual floor near the tolerance.
    double alphaInjection = 0.3;

    // Constant body force per unit volume (momentum source f_i * V per cell).
    // A streamwise-PERIODIC domain has no inlet to drive it: the mean pressure
    // gradient is represented by this force (e.g. bodyForce.x tuned so the
    // solved bulk velocity matches the target Reynolds number).
    Vec3 bodyForce = Vec3(0.0, 0.0, 0.0);

    double kMin     = 1e-10;
    double kMax     = 1e10;
    double omegaMin = 1e-6;

    int  reportInterval = 10;
    bool verbose        = true;

    std::string pressureSolver   = "AMG";
    std::string momentumSolver   = "BiCGSTAB";
    std::string turbulenceSolver = "BiCGSTAB";

    // ---- PHASE 7 — Wall functions / coarse-mesh mode ------------------
    // ``useWallFunctions = false`` keeps the legacy resolved-LES wall ω BC
    // (Menter low-Re form ω = 60ν/(β1 y²)), which requires y⁺ ≈ 1.
    // Enabling this flag activates Menter's automatic wall treatment that
    // blends the resolved value with the log-law form
    //   ω_log = u_τ / (√β* κ y)
    // via   ω_w = √(ω_res² + ω_log²)
    // and lets the solver run safely on coarser meshes (y⁺ ≥ 30).
    bool useWallFunctions = false;
    double vonKarman      = 0.41;   // κ
    double wallFnE        = 9.0;    // log-law coefficient

    // ---- PHASE 3 — closure-structure toggle for model selection (angle 7) ----
    // Maps to SSTVariant: 0 = Full (standard SST), 1 = NoLimiter (drop Bradshaw
    // shear-stress limiter), 2 = KOmega (force F1=1, baseline k-w).  Stored as int
    // to keep FlowFields.hpp free of an SSTModel.hpp dependency.
    int sstVariant = 0;
};
