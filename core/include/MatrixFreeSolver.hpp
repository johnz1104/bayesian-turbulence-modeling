#pragma once

#include "LinearSolver.hpp"   // linalg::dot / linalg::norm
#include <vector>
#include <functional>
#include <cmath>

// ---- Rung-1 semi-analytic gradient: matrix-free Krylov core (NON-HELD) -------------
//
// BiCGSTAB that never forms a matrix: the operator A is supplied as a matvec functor
// y = A·x and the preconditioner as a precomputed inverse diagonal (Jacobi).  This is
// the linear engine for the Rung-1 tangent solve  (∂R/∂U) w_j = −∂R/∂θ_j , where the
// matvec is the directional residual difference  Jv ≈ [R(U*+εv) − R(U*−εv)]/(2ε)  built
// from SIMPLESolver::assembleResidual on a perturbed state.  It NEVER assembles the held
// analytic (∂R/∂U)ᵀ core — A is only ever touched through `matvec`.
//
// The algorithm is bit-for-bit the same recurrence as core BiCGSTABSolver (so the two
// agree on an explicit system); the only changes are (1) A.matvec → the functor and
// (2) linalg::jacobiPrecond(A,·) → component multiply by `invDiag`.  Starts from x = 0,
// so the initial residual is exactly b (one matvec saved).

struct MatrixFreeResult {
    int    iterations = 0;
    double finalRes   = 0.0;   // ‖b − A·x‖ / ‖b‖ at return
    double initialRes = 0.0;   // ‖b‖
    bool   converged  = false;
    int    matvecs    = 0;     // operator applications (cost accounting)
};

// MatVec: callable void(const std::vector<double>& x, std::vector<double>& y) writing y = A·x.
// invDiag: length-n inverse-diagonal Jacobi preconditioner (z_i = invDiag_i · r_i).  Pass a
// vector of 1.0 for the unpreconditioned solve.
// x is the initial guess AND the output: pass an all-zero x for a cold start (the initial
// matvec is skipped), or a previous solution to WARM-START (the outer SIMPLE-tangent loop
// re-solves with only the RHS changed, so warm-starting from the last increment is cheap).
inline MatrixFreeResult bicgstabMatrixFree(
        const std::function<void(const std::vector<double>&, std::vector<double>&)>& matvec,
        const std::vector<double>& b,
        std::vector<double>& x,
        const std::vector<double>& invDiag,
        int maxIter = 1000, double tol = 1e-8) {
    const int n = (int)b.size();
    MatrixFreeResult res;
    std::vector<double> r(n), rh(n), p(n, 0.0), v(n, 0.0);
    std::vector<double> s(n), t(n), ph(n), sh(n);

    auto precond = [&](const std::vector<double>& in, std::vector<double>& out) {
        for (int i = 0; i < n; ++i) out[i] = invDiag[i] * in[i];
    };

    // r0 = b − A·x0.  Cold start (x0 ≈ 0) skips the matvec; warm start pays one.
    if (linalg::norm(x) < 1e-300) {
        r = b;
    } else {
        matvec(x, r); ++res.matvecs;
        for (int i = 0; i < n; ++i) r[i] = b[i] - r[i];
    }
    double r0 = linalg::norm(r);
    res.initialRes = r0;
    if (r0 < 1e-300) { res.converged = true; return res; }

    rh = r;
    double rho = 1, alpha = 1, omega = 1;

    for (int it = 0; it < maxIter; ++it) {
        double rhoN = linalg::dot(rh, r);
        if (std::abs(rhoN) < 1e-300) break;
        double beta = (rhoN / (rho + 1e-300)) * (alpha / (omega + 1e-300));
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * (p[i] - omega * v[i]);

        precond(p, ph);
        matvec(ph, v); ++res.matvecs;
        double rv = linalg::dot(rh, v);
        if (std::abs(rv) < 1e-300) break;
        alpha = rhoN / rv;

        for (int i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];
        double sn = linalg::norm(s);
        if (sn / r0 < tol) {
            for (int i = 0; i < n; ++i) x[i] += alpha * ph[i];
            res.iterations = it + 1; res.finalRes = sn / r0; res.converged = true;
            return res;
        }

        precond(s, sh);
        matvec(sh, t); ++res.matvecs;
        double tt = linalg::dot(t, t);
        omega = (tt > 1e-300) ? linalg::dot(t, s) / tt : 0.0;

        for (int i = 0; i < n; ++i) x[i] += alpha * ph[i] + omega * sh[i];
        for (int i = 0; i < n; ++i) r[i] = s[i] - omega * t[i];

        double rn = linalg::norm(r);
        res.iterations = it + 1;
        res.finalRes   = rn / r0;
        if (res.finalRes < tol) { res.converged = true; return res; }
        if (std::abs(omega) < 1e-300) break;
        rho = rhoN;
    }
    return res;
}

// Restarted GMRES(m), matrix-free.  Unlike BiCGSTAB, GMRES minimises the residual over the
// Krylov subspace at every step ⇒ MONOTONE and robust on the indefinite saddle where BiCGSTAB
// breaks down.  `matvec` should already be the (left-)preconditioned operator M⁻¹A and `b` the
// preconditioned RHS M⁻¹·rhs, so the minimised residual is the balanced preconditioned one.
// x is the initial guess and the output.
inline MatrixFreeResult gmresMatrixFree(
        const std::function<void(const std::vector<double>&, std::vector<double>&)>& matvec,
        const std::vector<double>& b,
        std::vector<double>& x,
        int restart = 60, int maxIter = 2000, double tol = 1e-8) {
    const int n = (int)b.size();
    const int m = std::max(1, restart);
    MatrixFreeResult res;
    std::vector<double> Ax(n), w(n), r(n);

    double bnorm = linalg::norm(b);
    if (bnorm < 1e-300) { std::fill(x.begin(), x.end(), 0.0); res.converged = true; return res; }

    std::vector<std::vector<double>> V(m + 1, std::vector<double>(n));
    std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
    std::vector<double> cs(m, 0.0), sn(m, 0.0), g(m + 1, 0.0);

    int total = 0;
    while (total < maxIter) {
        if (linalg::norm(x) < 1e-300) r = b;                       // r = b − A·x
        else { matvec(x, Ax); ++res.matvecs; for (int i = 0; i < n; ++i) r[i] = b[i] - Ax[i]; }
        double beta = linalg::norm(r);
        res.initialRes = (res.initialRes == 0.0) ? beta : res.initialRes;
        res.finalRes = beta / bnorm;
        if (res.finalRes < tol) { res.converged = true; return res; }

        for (int i = 0; i < n; ++i) V[0][i] = r[i] / beta;
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;

        int k = 0;
        for (k = 0; k < m && total < maxIter; ++k, ++total) {
            matvec(V[k], w); ++res.matvecs;                        // Arnoldi
            for (int i = 0; i <= k; ++i) {
                H[i][k] = linalg::dot(w, V[i]);
                for (int p = 0; p < n; ++p) w[p] -= H[i][k] * V[i][p];
            }
            H[k + 1][k] = linalg::norm(w);
            if (H[k + 1][k] > 1e-300)
                for (int p = 0; p < n; ++p) V[k + 1][p] = w[p] / H[k + 1][k];
            for (int i = 0; i < k; ++i) {                          // apply prior Givens
                double t = cs[i] * H[i][k] + sn[i] * H[i + 1][k];
                H[i + 1][k] = -sn[i] * H[i][k] + cs[i] * H[i + 1][k];
                H[i][k] = t;
            }
            double denom = std::sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
            cs[k] = (denom > 1e-300) ? H[k][k] / denom : 1.0;
            sn[k] = (denom > 1e-300) ? H[k + 1][k] / denom : 0.0;
            H[k][k] = cs[k] * H[k][k] + sn[k] * H[k + 1][k];
            H[k + 1][k] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k] = cs[k] * g[k];
            res.iterations = total + 1;
            res.finalRes = std::abs(g[k + 1]) / bnorm;
            if (res.finalRes < tol) { ++k; break; }
        }
        // back-substitution y, update x += V[0..k-1]·y
        std::vector<double> y(k, 0.0);
        for (int i = k - 1; i >= 0; --i) {
            double s = g[i];
            for (int j = i + 1; j < k; ++j) s -= H[i][j] * y[j];
            y[i] = (std::abs(H[i][i]) > 1e-300) ? s / H[i][i] : 0.0;
        }
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < k; ++j) x[i] += V[j][i] * y[j];
        if (res.finalRes < tol) { res.converged = true; return res; }
    }
    return res;
}

// Flexible restarted FGMRES(m), matrix-free + RIGHT preconditioned.  Stores the preconditioned
// Krylov vectors Z[k] = M⁻¹·V[k] and updates x += Σ Z[k]·y[k], so the preconditioner M⁻¹ may
// VARY between applications (here it runs inner iterative block solves).  Right preconditioning
// minimises the TRUE residual ‖b − A·x‖ (so a singular/weak M⁻¹ cannot collapse the RHS into a
// false-converged zero — the GMRES failure mode).  `matvec` = A (the σ-scaled Jv); `precond` =
// M⁻¹ (the physics-based block-solve SIMPLE preconditioner).  x is the initial guess + output.
inline MatrixFreeResult fgmresMatrixFree(
        const std::function<void(const std::vector<double>&, std::vector<double>&)>& matvec,
        const std::function<void(const std::vector<double>&, std::vector<double>&)>& precond,
        const std::vector<double>& b,
        std::vector<double>& x,
        int restart = 60, int maxIter = 1500, double tol = 1e-8) {
    const int n = (int)b.size();
    const int m = std::max(1, restart);
    MatrixFreeResult res;
    std::vector<double> Ax(n), w(n), r(n);

    double bnorm = linalg::norm(b);
    if (bnorm < 1e-300) { std::fill(x.begin(), x.end(), 0.0); res.converged = true; return res; }

    std::vector<std::vector<double>> V(m + 1, std::vector<double>(n));
    std::vector<std::vector<double>> Z(m, std::vector<double>(n));
    std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
    std::vector<double> cs(m, 0.0), sn(m, 0.0), g(m + 1, 0.0);

    int total = 0;
    while (total < maxIter) {
        if (linalg::norm(x) < 1e-300) r = b;
        else { matvec(x, Ax); ++res.matvecs; for (int i = 0; i < n; ++i) r[i] = b[i] - Ax[i]; }
        double beta = linalg::norm(r);
        if (res.initialRes == 0.0) res.initialRes = beta;
        res.finalRes = beta / bnorm;
        if (res.finalRes < tol) { res.converged = true; return res; }

        for (int i = 0; i < n; ++i) V[0][i] = r[i] / beta;
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;

        int k = 0;
        for (k = 0; k < m && total < maxIter; ++k, ++total) {
            precond(V[k], Z[k]);                 // Z[k] = M⁻¹ V[k]  (flexible)
            matvec(Z[k], w); ++res.matvecs;      // w = A·Z[k]
            for (int i = 0; i <= k; ++i) {
                H[i][k] = linalg::dot(w, V[i]);
                for (int p = 0; p < n; ++p) w[p] -= H[i][k] * V[i][p];
            }
            H[k + 1][k] = linalg::norm(w);
            if (H[k + 1][k] > 1e-300)
                for (int p = 0; p < n; ++p) V[k + 1][p] = w[p] / H[k + 1][k];
            for (int i = 0; i < k; ++i) {
                double t = cs[i] * H[i][k] + sn[i] * H[i + 1][k];
                H[i + 1][k] = -sn[i] * H[i][k] + cs[i] * H[i + 1][k];
                H[i][k] = t;
            }
            double denom = std::sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
            cs[k] = (denom > 1e-300) ? H[k][k] / denom : 1.0;
            sn[k] = (denom > 1e-300) ? H[k + 1][k] / denom : 0.0;
            H[k][k] = cs[k] * H[k][k] + sn[k] * H[k + 1][k];
            H[k + 1][k] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k] = cs[k] * g[k];
            res.iterations = total + 1;
            res.finalRes = std::abs(g[k + 1]) / bnorm;
            if (res.finalRes < tol) { ++k; break; }
        }
        std::vector<double> y(k, 0.0);
        for (int i = k - 1; i >= 0; --i) {
            double s = g[i];
            for (int j = i + 1; j < k; ++j) s -= H[i][j] * y[j];
            y[i] = (std::abs(H[i][i]) > 1e-300) ? s / H[i][i] : 0.0;
        }
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < k; ++j) x[i] += Z[j][i] * y[j];   // flexible: use Z, not V
        if (res.finalRes < tol) { res.converged = true; return res; }
    }
    // recompute the TRUE residual on exit (the in-cycle |g| estimate is unreliable with a
    // flexible/iterative preconditioner — never trust it as the convergence verdict).
    matvec(x, Ax); ++res.matvecs;
    double rn = 0.0; for (int i = 0; i < n; ++i) { double e = b[i] - Ax[i]; rn += e * e; }
    res.finalRes = std::sqrt(rn) / bnorm;
    res.converged = res.finalRes < tol;
    return res;
}
