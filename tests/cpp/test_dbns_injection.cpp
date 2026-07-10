// Model-form injection verification for the density-based solver: the
// deferred-correction coupling with the heat-flux (energy-equation) reach.
//
//   1. Zero-correction identity: with a zero target anisotropy on a laminar
//      flow (mu_t = 0, k = 0) the injected stress difference vanishes
//      identically, so the injected solve must reproduce the baseline fields
//      exactly (the plumbing adds nothing when it should add nothing).
//   2. Heat-flux reach directionality: a constant positive wall-normal
//      heat-flux correction transports energy upward, so the converged
//      temperature field must become asymmetric (upper half warmer than the
//      symmetric baseline) by the physically expected sign.
//   3. Realizability recording: a target outside the barycentric set is
//      recorded in the diagnostics (violation magnitude, not silently
//      projected); a realizable target records clean.
//
// The turbulent (stress-reach) coupled validation at the interaction Mach
// runs with the study's attached baseline (gate A of the pre-registered
// scheme), where the SST inflow state exists; this file verifies the
// injection machinery itself.

#include "Mesh.hpp"
#include "DBNSSolver.hpp"
#include "IdealGasEOS.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dbns;

namespace {
#define REQUIRE(cond, msg)                                              \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n  required: %s\n",   \
                         __FILE__, __LINE__, (msg), #cond);             \
            std::exit(1);                                               \
        }                                                               \
    } while (0)

struct CouetteCase {
    IdealGasEOS eos;
    double T0 = 300.0, p0 = 101325.0, H = 0.01, L = 0.02, Uw = 300.0;
    double rho0, mu;
    Mesh mesh;
    DBNSBoundaryConditions bcs;
    DBNSSettings st;

    CouetteCase() : mesh(Mesh::makeChannel2D(4, 32, 0.02, 0.01, 1.0e4, 1.0)) {
        rho0 = p0 / (eos.R * T0);
        mu = rho0 * Uw * H / 1.0e4;
        BoundarySpec bottom; bottom.kind = BoundaryKind::NoSlipIsothermal;
        bottom.wallTemp = T0;
        BoundarySpec top; top.kind = BoundaryKind::NoSlipIsothermal;
        top.wallTemp = T0; top.wallVelocity = Vec3{Uw, 0.0, 0.0};
        BoundarySpec ends; ends.kind = BoundaryKind::Extrapolate;
        BoundarySpec anchor; anchor.kind = BoundaryKind::SubsonicOutflow;
        anchor.backPressure = p0;
        bcs.set("bottom", bottom); bcs.set("top", top);
        bcs.set("inlet", ends); bcs.set("outlet", anchor);

        st.timeMode = TimeMode::Steady;
        st.implicitSteady = true;
        st.cflImplicit = 1e4; st.cflRampStart = 2.0; st.cflRampIters = 200;
        st.viscous = true; st.turbulent = false; st.constMu = mu;
        st.reconstructOrder = 2; st.limitReconstruction = true;
        st.convergenceTol = 1e-8; st.maxIterations = 120000;
    }
};

// mean temperature of the upper minus the lower quarter at mid length
double thermalAsymmetry(const DBNSSolver& s, const CouetteCase& c) {
    const Mesh& m = s.mesh();
    double hi = 0.0, lo = 0.0; int nhi = 0, nlo = 0;
    for (int ci = 0; ci < m.nCells(); ++ci) {
        double x = m.cell(ci).center.x, y = m.cell(ci).center.y;
        if (x < 0.25 * c.L || x > 0.75 * c.L) continue;
        double T = GasState::temperature(s.primitive(ci), c.eos);
        if (y > 0.75 * c.H) { hi += T; ++nhi; }
        if (y < 0.25 * c.H) { lo += T; ++nlo; }
    }
    return hi / nhi - lo / nlo;
}
}  // namespace

static void test_zero_correction_identity() {
    CouetteCase c;
    DBNSSolver base(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    base.initUniform({c.rho0, 0.0, 0.0, c.p0, 0.0, 0.0});
    SolveReport rb = base.solve();
    REQUIRE(rb.status == EvaluationStatus::Converged, "baseline converges");

    DBNSSolver inj(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    inj.initUniform({c.rho0, 0.0, 0.0, c.p0, 0.0, 0.0});
    int nc = c.mesh.nCells();
    std::vector<double> b6(6 * nc, 0.0), dq2(2 * nc, 0.0);
    inj.setTargetCorrection(b6, dq2, true);
    SolveReport ri = inj.solve();
    REQUIRE(ri.status == EvaluationStatus::Converged, "injected converges");

    double maxdiff = 0.0;
    for (int ci = 0; ci < nc; ++ci) {
        Primitive Vb = base.primitive(ci), Vi = inj.primitive(ci);
        maxdiff = std::max(maxdiff, std::abs(Vb.u - Vi.u) / c.Uw);
        maxdiff = std::max(maxdiff,
                           std::abs(Vb.p - Vi.p) / c.p0);
    }
    std::printf("  [identity] max relative field difference %.3e\n", maxdiff);
    REQUIRE(maxdiff < 1e-10, "zero correction reproduces the baseline");
    REQUIRE(inj.injectionDiagnostics().active, "diagnostics active");
    REQUIRE(inj.injectionDiagnostics().allRealizable,
            "zero anisotropy is realizable");
    REQUIRE(inj.injectionDiagnostics().checkedIters > 0,
            "realizability re-checked in the running solve");
}

static void test_heat_flux_reach_direction() {
    CouetteCase c;
    DBNSSolver base(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    base.initUniform({c.rho0, 0.0, 0.0, c.p0, 0.0, 0.0});
    REQUIRE(base.solve().status == EvaluationStatus::Converged,
            "baseline converges");
    double asymBase = thermalAsymmetry(base, c);

    // an upward turbulent heat-flux correction with an interior divergence
    // (dq_y ~ sin(pi y/H)): a divergence-free constant flux would
    // short-circuit into the isothermal walls through the thin wall cells
    // and leave only a milli-Kelvin trace, so the test drives the interior,
    // where div(rho cp dq) cools the lower half and warms the upper half.
    // b = 0 stays inert on this laminar flow (the stress part carries k).
    DBNSSolver inj(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    inj.initUniform({c.rho0, 0.0, 0.0, c.p0, 0.0, 0.0});
    int nc = c.mesh.nCells();
    std::vector<double> b6(6 * nc, 0.0), dq2(2 * nc, 0.0);
    for (int ci = 0; ci < nc; ++ci) {
        double y = c.mesh.cell(ci).center.y;
        dq2[2 * ci + 1] = 1.0 * std::sin(M_PI * y / c.H);    // dq_y [m/s K]
    }
    inj.setTargetCorrection(b6, dq2, true);
    REQUIRE(inj.solve().status == EvaluationStatus::Converged,
            "injected converges");
    double asymInj = thermalAsymmetry(inj, c);

    std::printf("  [heat reach] upper-lower asymmetry: baseline %.3f K, "
                "injected %.3f K\n", asymBase, asymInj);
    // the baseline viscous-heating profile is close to symmetric between
    // isothermal walls (a sub-Kelvin outlet-anchor bias is tolerated); the
    // upward flux correction must warm the upper quarter against it
    REQUIRE(std::abs(asymBase) < 1.0, "baseline near-symmetric");
    REQUIRE(asymInj > asymBase + 0.5, "upward heat-flux reach warms the top");

    // the energy reach off: the same targets must leave the flow unchanged
    // on this laminar case (the stress part is identically zero)
    DBNSSolver noReach(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    noReach.initUniform({c.rho0, 0.0, 0.0, c.p0, 0.0, 0.0});
    noReach.setTargetCorrection(b6, dq2, false);
    REQUIRE(noReach.solve().status == EvaluationStatus::Converged,
            "anisotropy-only variant converges");
    double asymOff = thermalAsymmetry(noReach, c);
    std::printf("  [heat reach] energy reach off: asymmetry %.3f K\n", asymOff);
    REQUIRE(std::abs(asymOff - asymBase) < 0.2,
            "anisotropy-only variant carries no heat-flux correction");
}

static void test_realizability_recording() {
    CouetteCase c;
    c.st.maxIterations = 300;      // a short march suffices for the recording
    c.st.convergenceTol = 1e-30;
    DBNSSolver s(c.mesh, c.eos, SSTCoefficients{}, c.bcs, c.st);
    s.initUniform({c.rho0, 0.0, 0.0, c.p0, 0.0, 0.0});
    int nc = c.mesh.nCells();
    std::vector<double> b6(6 * nc, 0.0), dq2;
    b6[0] = 0.9;                   // b_xx beyond the one-component corner
    b6[1] = -0.45; b6[2] = -0.45;
    s.setTargetCorrection(b6, dq2, true);
    s.solve();
    const InjectionDiagnostics& d = s.injectionDiagnostics();
    std::printf("  [realizability] checked %d iters, all realizable %d, "
                "max violation %.3e\n", d.checkedIters,
                (int)d.allRealizable, d.maxViolation);
    REQUIRE(d.checkedIters > 0, "check ran");
    REQUIRE(!d.allRealizable, "violation recorded, not silently projected");
    REQUIRE(d.maxViolation > 0.0, "violation magnitude recorded");

    s.clearTargetCorrection();
    REQUIRE(!s.injectionDiagnostics().active, "clear resets the diagnostics");
}

int main() {
    std::printf("[dbns injection] zero-correction identity\n");
    test_zero_correction_identity();
    std::printf("[dbns injection] heat-flux reach directionality\n");
    test_heat_flux_reach_direction();
    std::printf("[dbns injection] realizability recording\n");
    test_realizability_recording();
    std::printf("[dbns injection] ALL PASSED\n");
    return 0;
}
