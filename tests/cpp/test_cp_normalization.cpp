// Dimensional consistency of the pressure-coefficient observable.
//
// The audit found the compressible observation path mapping ABSOLUTE static
// pressure (about 1e5 Pa) into the incompressible p/(0.5 U^2) operator with
// no reference pressure and no density, so the "Cp" was dominated by the
// absolute-pressure offset. The operator now computes
// Cp = (p - p_ref)/(0.5 rho_ref U_ref^2). This test pins:
//   1. legacy incompressible taps (kinematic pressure, default references)
//      evaluate bit-identically to the old p/(0.5 U^2);
//   2. a compressible-style tap with SI references recovers the physically
//      correct coefficient from an absolute-pressure field;
//   3. omitting the references on an absolute-pressure field visibly produces
//      the pathological offset-dominated number the fix eliminates.

#include "Mesh.hpp"
#include "Field.hpp"
#include "FlowFields.hpp"
#include "ObservationOperator.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>

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
    Mesh mesh = Mesh::makeChannel2D(8, 6, 2.0, 1.0);
    mesh.computeWallDistance();
    FlowFields f(mesh);

    const Vec3 loc(1.0, 0.5, 0.0);

    // 1. legacy incompressible tap: kinematic pressure field, defaults
    {
        f.p.setUniform(0.32);                       // kinematic p (p/rho)
        ObservationOperator obs;
        obs.addPressureTap(loc, 0.0, 1.0, 2.0);     // refVel = 2
        double cp = obs.evaluate(mesh, f, 1e-3)[0];
        REQUIRE(std::fabs(cp - 0.32 / (0.5 * 2.0 * 2.0)) < 1e-15,
                "legacy kinematic tap must be unchanged");
    }

    // 2. compressible-style tap: absolute pressure with SI references
    {
        const double p_ref = 101325.0, rho_ref = 1.177, Uref = 34.7;
        const double dynP = 0.5 * rho_ref * Uref * Uref;
        f.p.setUniform(p_ref + 0.25 * dynP);        // true Cp = 0.25
        ObservationOperator obs;
        obs.addPressureTap(loc, 0.0, 1.0, Uref, 0.0, p_ref, rho_ref);
        double cp = obs.evaluate(mesh, f, 1e-3)[0];
        REQUIRE(std::fabs(cp - 0.25) < 1e-12,
                "referenced tap must recover the physical Cp");

        // 3. the pathological pre-fix reading: no references on the same field
        ObservationOperator bad;
        bad.addPressureTap(loc, 0.0, 1.0, Uref);
        double cp_bad = bad.evaluate(mesh, f, 1e-3)[0];
        REQUIRE(cp_bad > 100.0,
                "unreferenced absolute-pressure Cp is offset-dominated (the audit defect)");
    }

    // 4. drag on a wall patch shares the reference treatment: on an
    //    absolute-pressure field an unreferenced drag is offset-dominated
    //    (a wall patch is not a closed surface), the referenced one is not
    {
        const double p_ref = 101325.0, rho_ref = 1.177, Uref = 34.7;
        f.p.setUniform(p_ref);            // uniform absolute pressure: zero
        for (int ci = 0; ci < mesh.nCells(); ++ci)   // physical pressure drag
            f.U[ci] = Vec3(0.0, 0.0, 0.0);
        for (int fi = mesh.nInternalFaces(); fi < mesh.nFaces(); ++fi)
            f.U.bface(fi) = Vec3(0.0, 0.0, 0.0);

        ObservationOperator good;
        good.addDrag("bottom", 0.0, 1.0, 1.0, Uref, 0.0, p_ref, rho_ref);
        double cd = good.evaluate(mesh, f, 1e-3)[0];
        REQUIRE(std::fabs(cd) < 1e-12,
                "referenced drag on a uniform absolute-pressure field must vanish");

        ObservationOperator bad;
        bad.addDrag("bottom", 0.0, 1.0, 1.0, Uref);
        double cd_bad = bad.evaluate(mesh, f, 1e-3)[0];
        REQUIRE(std::fabs(cd_bad) > 1.0,
                "unreferenced absolute-pressure drag is offset-dominated (the defect)");
    }

    std::printf("test_cp_normalization: all checks passed\n");
    return 0;
}
