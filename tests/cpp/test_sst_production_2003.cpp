// SST-2003 omega-production correction: discriminating unit tests.
//
// The 2003 paper misprints the omega-equation production as alpha*S^2; the
// specification (NASA TMR SST page) corrects it to alpha*Pk_limited/nuT, i.e.
// alpha*min(S^2, 10*betaStar*k*omega/nuT). These tests pin the corrected form
// where it DIFFERS from the misprint (k-production limiter active), pin the
// regimes where the two coincide (limiter inactive: equilibrium attached
// flows), pin the singular nuT -> 0 limit, and pin the unclipped
// cross-diffusion of the omega source (only the F1-internal CDkw is clipped).
//
//   1. Limiter inactive: productionOmega == S^2 exactly (misprint and
//      corrected form coincide; committed attached-flow results unchanged).
//   2. Limiter active: productionOmega == 10*betaStar*k*omega/nuT and is
//      STRICTLY below S^2. This value discriminates: the alpha*S^2 misprint
//      would be larger by the tested ratio.
//   3. Bounded above: productionOmega <= S^2 across a parameter sweep (the
//      algebraic reason no production feedback is possible).
//   4. Singular limit: nuT -> 0 selects the S^2 branch (Pk/nuT -> S^2).
//   5. Consistency with the k-equation helper: productionOmega equals
//      production()/nuT for nuT bounded away from zero.
//   6. Dimensional (density-based solver) equivalence: the conservative-form
//      expression min(S2, 10 bStar rho k w / muT) equals
//      productionOmega(muT/rho, S, k, w), the identity the DBNS assembly
//      relies on.
//   7. sourceOmega passes negative cross-diffusion through UNCLIPPED, and
//      assembles exactly alpha*productionOmega - beta*w^2 + (1-F1)*CDkw.

#include "SSTModel.hpp"
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

bool close(double a, double b, double rtol) {
    double scale = std::max(std::abs(a), std::abs(b));
    return std::abs(a - b) <= rtol * std::max(scale, 1e-300);
}

}  // namespace

int main() {
    SSTModel sst;  // default Menter coefficients: betaStar = 0.09
    const double bStar = sst.coeffs.betaStar;

    // 1. Limiter inactive (equilibrium-like state): Pk_raw = nuT*S^2 well below
    //    10*betaStar*k*omega, so the corrected form reduces to S^2 exactly.
    {
        double nuT = 0.01, S = 1.0, k = 1.0, w = 100.0;
        REQUIRE(nuT * S * S < 10.0 * bStar * k * w,
                "state must be limiter-inactive for this check");
        double p = sst.productionOmega(nuT, S, k, w);
        REQUIRE(close(p, S * S, 1e-14), "inactive limiter must give exactly S^2");
    }

    // 2. Limiter active (high-strain state): the corrected form equals
    //    10*betaStar*k*omega/nuT and sits far below the S^2 misprint.
    {
        double nuT = 1.0, S = 100.0, k = 1.0, w = 1.0;
        double lim = 10.0 * bStar * k * w / nuT;          // = 0.9
        REQUIRE(lim < S * S, "state must be limiter-active for this check");
        double p = sst.productionOmega(nuT, S, k, w);
        REQUIRE(close(p, lim, 1e-14),
                "active limiter must give 10*betaStar*k*omega/nuT");
        REQUIRE(p < 1e-3 * S * S,
                "corrected production must sit far below the S^2 misprint here");
    }

    // 3. Bounded above by S^2 across a sweep (no-feedback argument).
    {
        const double nuTs[] = {1e-12, 1e-6, 1e-2, 0.5, 3.0};
        const double Ss[]   = {0.0, 0.3, 3.0, 300.0};
        const double ks[]   = {1e-8, 1e-2, 1.0, 50.0};
        const double ws[]   = {1e-3, 1.0, 1e4};
        for (double nuT : nuTs)
            for (double S : Ss)
                for (double k : ks)
                    for (double w : ws) {
                        double p = sst.productionOmega(nuT, S, k, w);
                        REQUIRE(p <= S * S + 1e-14,
                                "productionOmega must never exceed S^2");
                    }
    }

    // 4. Singular limit nuT -> 0: Pk/nuT -> S^2 (the min form selects S^2).
    {
        double S = 2.0, k = 0.5, w = 200.0;
        double p = sst.productionOmega(0.0, S, k, w);
        REQUIRE(close(p, S * S, 1e-14), "nuT = 0 must select the S^2 branch");
    }

    // 5. Consistency with the k-equation limited production for nuT > 0.
    {
        const double states[][4] = {  // nuT, S, k, w spanning both branches
            {0.01, 1.0, 1.0, 100.0},  // inactive
            {1.0, 100.0, 1.0, 1.0},   // active
            {0.2, 5.0, 2.0, 30.0},
        };
        for (const auto& s : states) {
            double byMin   = sst.productionOmega(s[0], s[1], s[2], s[3]);
            double byRatio = sst.production(s[0], s[1], s[2], s[3]) / s[0];
            REQUIRE(close(byMin, byRatio, 1e-12),
                    "productionOmega must equal production()/nuT for nuT > 0");
        }
    }

    // 6. Dimensional equivalence used by the density-based assembly:
    //    min(S2, 10 bStar rho k w / muT) == productionOmega(muT/rho, S, k, w).
    {
        double rho = 1.7, muT = 0.4, S = 40.0, k = 3.0, w = 8.0;
        double S2 = S * S;
        double conservative = std::min(S2, 10.0 * bStar * rho * k * w / muT);
        double kinematic    = sst.productionOmega(muT / rho, S, k, w);
        REQUIRE(close(conservative, kinematic, 1e-12),
                "conservative and kinematic omega-production forms must agree");
    }

    // 7. sourceOmega: exact assembly and UNCLIPPED cross-diffusion.
    {
        double S = 100.0, nuT = 1.0, k = 1.0, w = 1.0, F1 = 0.25;
        double alphaB = sst.coeffs.alpha(F1);
        double betaB  = sst.coeffs.beta(F1);
        double prod   = sst.productionOmega(nuT, S, k, w);

        double CDneg = -5.0;
        double got   = sst.sourceOmega(S, nuT, k, w, F1, CDneg);
        double want  = alphaB * prod - betaB * w * w + (1.0 - F1) * CDneg;
        REQUIRE(close(got, want, 1e-13),
                "sourceOmega must assemble alpha*prodOmega - beta*w^2 + (1-F1)*CDkw");

        // The negative cross-diffusion contribution must pass through in full:
        // a clipped source would show zero difference between CDkw = -5 and 0.
        double gotZero = sst.sourceOmega(S, nuT, k, w, F1, 0.0);
        REQUIRE(close(got - gotZero, (1.0 - F1) * CDneg, 1e-13),
                "negative cross-diffusion must enter the omega source unclipped");
    }

    std::printf("test_sst_production_2003: all checks passed\n");
    return 0;
}
