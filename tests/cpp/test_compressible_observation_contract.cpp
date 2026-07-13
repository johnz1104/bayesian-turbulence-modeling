// The generic drag adapter uses kinematic pressure and kinematic viscosity.
// CompressibleForwardModel carries dimensional pressure but passes a
// density-free FlowFields view to that adapter, so the combination must fail
// at construction rather than return a plausible mixed-units coefficient.

#include "Mesh.hpp"
#include "ObservationOperator.hpp"
#include "CompressibleForwardModel.hpp"
#include "CompressibleBCs.hpp"
#include "IdealGasEOS.hpp"
#include "InferenceParameters.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {

#define REQUIRE(cond, msg)                                                  \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n  required: %s\n",       \
                         __FILE__, __LINE__, (msg), #cond);                 \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

}  // namespace

int main() {
    Mesh mesh = Mesh::makeChannel2D(8, 6, 2.0, 1.0, 5000.0, 1.0);
    mesh.computeWallDistance();
    IdealGasEOS eos;
    const double T = 300.0, p = 101325.0, U = 10.0;
    auto bcs = CompressibleBoundaryConditions::channelDefaults(
        mesh, U, T, p, 1e-3, 10.0);
    InferenceParameterSet params = InferenceParameterSet::a1_betaStar();

    ObservationOperator drag;
    drag.addDrag("bottom", 0.0, 1.0, 1.0, U);
    bool rejected = false;
    try {
        CompressibleForwardModel unsupported(
            mesh, params, drag, bcs, eos, SolverSettings{},
            Vec3(U, 0.0, 0.0), p, T, 1e-3, 10.0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    REQUIRE(rejected,
            "compressible generic drag must be rejected at construction");

    // A referenced pressure tap remains supported through the thermodynamic
    // pressure shim and the explicit Cp normalization contract.
    ObservationOperator pressure;
    pressure.addPressureTap(Vec3(1.0, 0.5, 0.0), 0.0, 1.0, U, 0.0,
                            p, eos.density(p, T));
    bool accepted = true;
    try {
        CompressibleForwardModel supported(
            mesh, params, pressure, bcs, eos, SolverSettings{},
            Vec3(U, 0.0, 0.0), p, T, 1e-3, 10.0);
    } catch (...) {
        accepted = false;
    }
    REQUIRE(accepted, "referenced compressible pressure tap must stay supported");

    std::printf("test_compressible_observation_contract: all checks passed\n");
    return 0;
}
