#include "ForwardModel.hpp"
#include <iostream>
#include <stdexcept>

// Constructor
ForwardModel::ForwardModel(const Mesh& mesh,
                           const InferenceParameterSet& paramSet,
                           const ObservationOperator& obsOp,
                           const FlowBoundaryConditions& bcs,
                           double nu,
                           const SolverSettings& settings,
                           const Vec3& Uinit,
                           double pInit, double kInit, double omegaInit)
    // initializer list
    : mesh_(mesh), paramSet_(paramSet), obsOp_(obsOp), bcs_(bcs),
      nu_(nu), settings_(settings),
      Uinit_(Uinit), pInit_(pInit), kInit_(kInit), omegaInit_(omegaInit),
      cache_(50, 0.5)
{}

// set the injected per-cell target anisotropy (validated length, stored by value)
void ForwardModel::setTargetAnisotropy(std::vector<double> b6) {
    if (static_cast<int>(b6.size()) != 6 * mesh_.nCells())
        throw std::runtime_error("setTargetAnisotropy: expected nCells*6 values");
    bTarget6_ = std::move(b6);
}

// evaluate: θ -> EvaluationResult
EvaluationResult ForwardModel::evaluate(const std::vector<double>& theta) {
    EvaluationResult result;

    // bounds check
    if (!paramSet_.inBounds(theta)) {
        result.status = EvaluationStatus::InvalidParameters;
        result.loglik = -1e30;
        return result;
    }

    try {
        // unpack θ -> SST coefficients
        SSTCoefficients coeffs = paramSet_.unpack(theta);
        SSTModel sst(coeffs);
        // Phase 3: closure-structure toggle (Full / NoLimiter / KOmega).
        sst.variant = static_cast<SSTVariant>(settings_.sstVariant);

        // set up solver
        SIMPLESolver solver(mesh_, sst, bcs_, nu_, settings_);
        if (!bTarget6_.empty()) solver.setTargetAnisotropy(&bTarget6_);
        FlowFields fields(mesh_);

        // warm start: check cache for nearby solution. Injected solves are
        // fully cache-independent (no lookup, no store): a warm start from a
        // nearby state makes the iter-0 residual normalisation, and with it
        // the convergence classification, depend on what ran before, and an
        // ensemble study needs every member reproducible in isolation.
        const WarmStartCache::CacheEntry* cached =
            bTarget6_.empty() ? cache_.findNearest(theta) : nullptr;
        if (cached) {
            // copy cached fields as initial condition
            fields = cached->fields;
        } else {
            solver.initUniform(fields, Uinit_, pInit_, kInit_, omegaInit_);
        }

        // solve
        ConvergenceHistory hist = solver.solve(fields);
        result.simpleIters = hist.finalIter;
        lastInjection_ = solver.injectionDiagnostics();

        // classify convergence
        if (hist.diverged) {
            result.status = EvaluationStatus::Diverged;
            result.loglik = -1e6;
            // INVALIDATE any previously retained fields: a diverged solve
            // leaves no valid state, and keeping the previous run's fields
            // visible let downstream QoI extraction silently inherit them
            // (two committed hills member records duplicated their
            // predecessor exactly this way)
            hasLastFields_ = false;
            return result;
        } else if (hist.converged) {
            result.status = EvaluationStatus::Converged;
            // cache the converged solution (plain solves only: injected states
            // must not pollute the theta -> solution warm-start cache)
            if (bTarget6_.empty()) cache_.store(theta, fields);
        } else {
            result.status = EvaluationStatus::Unconverged;
            // still evaluate - don't bias posterior toward easy regions
        }

        // retain fields for visualization / post-processing
        lastFields_ = fields;
        hasLastFields_ = true;

        // extract observables and log-likelihood
        result.predictions = obsOp_.evaluate(mesh_, fields, nu_);
        result.loglik = obsOp_.logLikelihood(mesh_, fields, nu_);

    } catch (const std::exception& e) {
        result.status = EvaluationStatus::Unknown;
        result.loglik = -1e6;
        if (settings_.verbose)
            std::cerr << "ForwardModel exception: " << e.what() << "\n";
    }
    return result;
}

// MCMC friendly wrapper
double ForwardModel::penalizedLogLikelihood(const std::vector<double>& theta) {
    EvaluationResult res = evaluate(theta);
    switch (res.status) {
        case EvaluationStatus::Converged:
        case EvaluationStatus::Unconverged:
            return res.loglik;
        case EvaluationStatus::Diverged:
        case EvaluationStatus::Unknown:
            return -1e6;
        case EvaluationStatus::InvalidParameters:
            return -1e30;   // effectively -infinity
    }
    return -1e30;
}