#pragma once
#include "Mesh.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"   // reuse SolverSettings, ConvergenceHistory

// Compressible RANS flow fields: extends incompressible set with density and temperature.
struct CompressibleFlowFields {
    VectorField U;      // velocity [m/s]
    ScalarField p;      // MECHANICAL working pressure [Pa]: p_mech = p_thermo
                        // + (2/3) rho k (the Favre trace absorbed into the
                        // momentum working variable). Thermodynamic statics
                        // are recovered as p_thermo = p - (2/3) rho k; the
                        // observation shim and the python field export do
                        // this, and boundary/initialization inputs specified
                        // as thermodynamic values are converted on entry.
    ScalarField T;      // static temperature [K]
    ScalarField rho;    // density [kg/m³]  (derived: ρ = p_mech/(R T + 2k/3))

    // SST turbulence model fields (same as incompressible)
    ScalarField k;      // turbulent kinetic energy [m²/s²]
    ScalarField omega;  // specific dissipation rate [1/s]
    ScalarField nuT;    // eddy viscosity [m²/s]
    ScalarField F1, F2; // SST blending functions
    ScalarField Pk;     // turbulence production
    ScalarField CDkw;   // cross-diffusion

    CompressibleFlowFields() = default;
    explicit CompressibleFlowFields(const Mesh& mesh)
        : U(mesh,"U"), p(mesh,"p"), T(mesh,"T"), rho(mesh,"rho"),
          k(mesh,"k"), omega(mesh,"omega"),
          nuT(mesh,"nuT"), F1(mesh,"F1"), F2(mesh,"F2"),
          Pk(mesh,"Pk"), CDkw(mesh,"CDkw") {}
};

// Per-iteration residuals for the compressible solver.
struct CompressibleResidualEntry {
    int    iteration = 0;
    double Ux = 0, Uy = 0, Uz = 0;
    double p  = 0;
    double T  = 0;
    double k  = 0, omega = 0;
};

struct CompressibleConvergenceHistory {
    std::vector<CompressibleResidualEntry> entries;
    bool converged = false;
    bool diverged  = false;
    int  finalIter = 0;
};
