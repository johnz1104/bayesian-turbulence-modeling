// Unit tests for the density-based solver's foundational numerics:
//   - exact Riemann solver: Sod star-region pressure/velocity (known values)
//   - HLLC flux: consistency (L==R -> physical flux) and supersonic upwinding
//   - HLLC conservation: flux is single-valued across a face (anti-symmetry)
//   - barycentric realizability projection: projects unrealizable anisotropy in,
//     leaves realizable states fixed, and reproduces eigenvalue sums.
//
// No framework: each check is a REQUIRE; the program exits non-zero on failure.

#include "DBNSTypes.hpp"
#include "ExactRiemann.hpp"
#include "HLLCFlux.hpp"
#include "RealizabilityProjection.hpp"
#include "IdealGasEOS.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

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

bool close(double a, double b, double rtol, double atol = 1e-12) {
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}
}  // namespace

static void test_exact_riemann_sod() {
    // Classic Sod problem (Toro): L=(1,0,1), R=(0.125,0,0.1), gamma=1.4.
    // Accepted star-region values: p* = 0.30313, u* = 0.92745.
    ExactRiemann ex(1.4);
    RiemannState1D L{1.0, 0.0, 1.0};
    RiemannState1D R{0.125, 0.0, 0.1};
    double pStar, uStar;
    ex.starState(L, R, pStar, uStar);
    REQUIRE(close(pStar, 0.30313, 2e-4), "Sod p* mismatch");
    REQUIRE(close(uStar, 0.92745, 2e-4), "Sod u* mismatch");

    // sampling at x/t = 0 sits in the star region just left of the contact:
    // density there is the post-rarefaction value ~0.42632.
    RiemannState1D s = ex.sample(L, R, 0.0);
    REQUIRE(close(s.p, 0.30313, 2e-4), "Sod sampled p mismatch");
    REQUIRE(close(s.u, 0.92745, 2e-4), "Sod sampled u mismatch");
    REQUIRE(close(s.rho, 0.42632, 2e-3), "Sod sampled rho mismatch");
}

static void test_hllc_consistency() {
    IdealGasEOS eos;  // air
    // Identical left/right states -> HLLC must return the exact physical flux.
    Primitive V;
    V.rho = 1.2; V.u = 30.0; V.v = -5.0; V.p = 1.0e5; V.k = 0.0; V.omega = 0.0;
    double nx = 0.6, ny = 0.8;
    StateVec fh = HLLCFlux::flux(V, V, nx, ny, eos);
    StateVec fp = GasState::normalFlux(V, nx, ny, eos);
    for (int i = 0; i < NVAR; ++i)
        REQUIRE(close(fh[i], fp[i], 1e-10, 1e-6), "HLLC not consistent with physical flux");

    // Supersonic flow aligned with the normal -> all waves move right, flux=FL.
    Primitive Lsup;
    Lsup.rho = 1.0; Lsup.u = 1000.0; Lsup.v = 0.0; Lsup.p = 1.0e5;
    Primitive Rsup = Lsup; Rsup.rho = 0.9; Rsup.p = 0.8e5;
    StateVec fsup = HLLCFlux::flux(Lsup, Rsup, 1.0, 0.0, eos);
    StateVec fL   = GasState::normalFlux(Lsup, 1.0, 0.0, eos);
    for (int i = 0; i < NVAR; ++i)
        REQUIRE(close(fsup[i], fL[i], 1e-10, 1e-6), "HLLC supersonic should be pure upwind (FL)");
}

static void test_hllc_anti_symmetry() {
    // The flux through a face must be single-valued: F(L,R,n) = -F(R,L,-n).
    IdealGasEOS eos;
    Primitive L; L.rho = 1.0; L.u = 100.0; L.v = 20.0; L.p = 1.0e5;
    Primitive R; R.rho = 0.7; R.u = -50.0; R.v = 10.0; R.p = 0.6e5;
    double nx = 0.8, ny = -0.6;
    StateVec f1 = HLLCFlux::flux(L, R, nx, ny, eos);
    StateVec f2 = HLLCFlux::flux(R, L, -nx, -ny, eos);
    for (int i = 0; i < NVAR; ++i)
        REQUIRE(close(f1[i], -f2[i], 1e-9, 1e-4), "HLLC flux not anti-symmetric across face");

    // Conserved-state admissibility round-trip and positive pressure.
    StateVec W = GasState::toConserved(L, eos);
    REQUIRE(GasState::admissible(W, eos), "valid state flagged inadmissible");
    Primitive back = GasState::toPrimitive(W, eos);
    REQUIRE(close(back.rho, L.rho, 1e-12) && close(back.p, L.p, 1e-10, 1e-3),
            "conserved/primitive round-trip failed");
}

static void test_realizability_projection() {
    // Isotropic anisotropy (b = 0) is realizable and must be unchanged.
    Sym3 b0{};
    REQUIRE(RealizabilityProjection::isRealizable(b0), "isotropic b should be realizable");
    double dist = 0.0;
    Sym3 p0 = RealizabilityProjection::projectAnisotropy(b0, &dist);
    REQUIRE(std::abs(dist) < 1e-12, "isotropic projection should not move");
    REQUIRE(std::abs(p0.trace()) < 1e-12, "projected isotropic trace should be ~0");

    // A wildly unrealizable diagonal anisotropy with eigenvalues outside the
    // triangle (does not satisfy the barycentric constraints) must project in.
    Sym3 bad; bad.xx = 1.5; bad.yy = -0.9; bad.zz = -0.6;  // traceless but unrealizable
    REQUIRE(!RealizabilityProjection::isRealizable(bad), "construct an unrealizable b");
    double d2 = 0.0;
    Sym3 fixed = RealizabilityProjection::projectAnisotropy(bad, &d2);
    REQUIRE(d2 > 0.0, "projection of unrealizable b must move");
    REQUIRE(RealizabilityProjection::isRealizable(fixed, 1e-7),
            "projected anisotropy must be realizable");
    REQUIRE(std::abs(fixed.trace()) < 1e-9, "projected anisotropy must stay traceless");

    // Reynolds-stress wrapper: a near-one-component stress projects to a
    // realizable stress with the same trace (2k preserved).
    Sym3 R; R.xx = 1.9; R.yy = 0.05; R.zz = 0.05; R.xy = 0.0;
    double trBefore = R.trace();
    Sym3 Rp = RealizabilityProjection::projectReynoldsStress(R);
    REQUIRE(close(Rp.trace(), trBefore, 1e-9, 1e-9), "projection must preserve 2k");
    Sym3 bcheck;
    double invTwoK = 1.0 / Rp.trace();
    bcheck.xx = Rp.xx * invTwoK - 1.0/3.0;
    bcheck.yy = Rp.yy * invTwoK - 1.0/3.0;
    bcheck.zz = Rp.zz * invTwoK - 1.0/3.0;
    bcheck.xy = Rp.xy * invTwoK; bcheck.xz = Rp.xz * invTwoK; bcheck.yz = Rp.yz * invTwoK;
    REQUIRE(RealizabilityProjection::isRealizable(bcheck, 1e-7),
            "projected Reynolds stress must be realizable");
}

int main() {
    test_exact_riemann_sod();
    test_hllc_consistency();
    test_hllc_anti_symmetry();
    test_realizability_projection();
    std::printf("test_dbns_numerics: all checks passed\n");
    return 0;
}
