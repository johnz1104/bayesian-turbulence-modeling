#pragma once

#include "DBNSTypes.hpp"
#include "Limiters.hpp"
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Configuration for the density-based shock-capturing solver: numerical
// settings and boundary-condition specification.
// ---------------------------------------------------------------------------

namespace dbns {

// Pseudo-time integration mode.
//   Steady:   local time stepping, iterate to residual convergence.
//   Unsteady: global (min) time step, march to settings.tEnd (e.g. shock tube).
enum class TimeMode { Steady, Unsteady };

// Compressibility (dilatational dissipation) correction model.
//   None:   incompressible SST dissipation.
//   Sarkar: eps_total = eps_s (1 + xi* M_t^2)              (Sarkar 1991)
//   Zeman:  eps_total = eps_s (1 + xi* F(M_t)), M_t0 cutoff (Wilcox/Zeman form)
enum class CompressibilityModel { None, Sarkar, Zeman };

struct DBNSSettings {
    TimeMode timeMode = TimeMode::Steady;
    double   cfl      = 0.5;        // Courant number (explicit stability)
    int      maxIterations = 20000; // steady iteration cap
    double   tEnd     = 0.2;        // unsteady end time
    double   convergenceTol = 1e-6; // steady density-residual L2 drop
    double   divergenceLimit = 1e12;

    // Spatial reconstruction: 1 = first order (constant), 2 = MUSCL linear.
    // limitReconstruction toggles the multidimensional Venkatakrishnan limiter;
    // turn it OFF only for smooth MMS order-of-accuracy verification.
    int    reconstructOrder = 2;
    bool   limitReconstruction = true;
    double venkatK = 5.0;           // Venkatakrishnan smoothing constant K

    bool   viscous    = true;       // include viscous + heat fluxes
    bool   turbulent  = false;      // integrate SST k-omega transport
    CompressibilityModel compressibility = CompressibilityModel::None;
    double turbMachCutoff = 0.25;   // M_t0 for the Zeman form
    double comprXiStar    = 1.0;    // xi* scaling of dilatational dissipation

    int    rkStages   = 3;          // SSP-RK stages (3 = SSP-RK3)
    int    reportInterval = 1000;
    bool   verbose    = false;

    double kFloor     = 1e-12;      // turbulence positivity floors
    double omegaFloor = 1e-3;
};

// Boundary-condition kinds.  Wall thermal condition is split into adiabatic and
// isothermal as the task requires.
enum class BoundaryKind {
    SupersonicInflow,    // all variables prescribed from the freestream
    Extrapolate,         // zero-gradient (supersonic outflow / far-field exit)
    SubsonicInflow,      // prescribe velocity + temperature, extrapolate pressure
    SubsonicOutflow,     // prescribe back pressure, extrapolate the rest
    SlipWall,            // inviscid wall / symmetry (reflect normal velocity)
    NoSlipAdiabatic,     // viscous wall, zero wall heat flux
    NoSlipIsothermal,    // viscous wall, fixed wall temperature
    FixedState           // Dirichlet to a per-patch uniform state (MMS / imposed shock)
};

// Per-patch boundary specification.  The prescribed values are interpreted
// according to kind; unused fields are ignored.
struct BoundarySpec {
    BoundaryKind kind = BoundaryKind::Extrapolate;
    Primitive    freestream;     // for inflow / fixed-state / imposed-shock
    double       wallTemp = 300; // for NoSlipIsothermal
    double       backPressure = 101325.0;  // for SubsonicOutflow
};

// Container mapping patch names to their boundary specs.
struct DBNSBoundaryConditions {
    std::unordered_map<std::string, BoundarySpec> specs;

    void set(const std::string& patch, const BoundarySpec& s) { specs[patch] = s; }

    const BoundarySpec& get(const std::string& patch) const {
        auto it = specs.find(patch);
        if (it == specs.end())
            throw std::runtime_error("DBNSBoundaryConditions: no spec for patch '" + patch + "'");
        return it->second;
    }

    bool has(const std::string& patch) const { return specs.count(patch) > 0; }
};

}  // namespace dbns
