# research/shared - FROZEN interfaces (core-v1.0)

These are the minimal shared contracts every Thread A and Thread B agent builds
against. They are FROZEN: consume them, do not edit them in place from a
consuming branch. An interface change is escalated to a reviewed `core/` PR (root
CLAUDE.md, working rules). One planned revision is expected after Phase 1; until
then the surface stays as small as it is here.

The contracts are designed against the real C++/Python framework
(`rans_sst_py`, `python/`), and the SST baseline closure implements the full
inference handshake to prove the shape against something known-working. See
`tests/python/test_research_shared_handshake.py`.

## The five interfaces

- `inference/handshake.py` - the inference handshake (the core contract):
  - `ParameterSpec` - coefficient names, literature defaults, and the
    realizability + stability bounds (the truncation box).
  - `ClosurePrior` - truncated-normal prior over the coefficients PLUS the
    `FluctuationDissipationCoupling` hook. The hook is `None` for a memoryless
    closure (SST); a generalized Langevin closure uses it to tie the memory
    kernel to the noise autocorrelation via the second FD theorem,
    `K(s) ~ (F(s), F(0))`, so memory and noise are never specified independently.
  - `Likelihood.predict(theta) -> Prediction` - the likelihood hook mapping
    parameters to predicted statistics `eta = H(theta)`, which the KOH likelihood
    compares to data `y = eta + delta + epsilon`.
  - `EvaluationStatus` - Converged / Unconverged / DivergenceDetected /
    InvalidParameters / Unknown; failure is returned, never raised (mirrors the
    C++ ForwardModel).

- `closures/base.py` - the `Closure` ABC: `name`, `parameter_spec()`, `prior()`,
  `likelihood(benchmark)`, and the structural flags `has_memory`,
  `is_stochastic`, `is_nonlocal` (all `False` for SST). `closures/sst.py` is the
  SST reference adapter (imports the binding; import it explicitly).

- `constraints/base.py` - TWO separate entry points, never conflated:
  - `RealizabilityProjection` (reference `BarycentricRealizability`) projects the
    anisotropy eigenvalues into the barycentric triangle.
  - `GalileanInvariantBasis` (reference `IntegrityBasis`) gives the
    invariance-delivering integrity basis. The basis does NOT deliver
    realizability; conflating the two diverges the solver.

- `benchmarks/base.py` - `Benchmark` / `BenchmarkData`: observation locations,
  values, sigmas, optional fields, and provenance metadata keyed as in
  `data/README.md`.

- `metrics/base.py` - `Metric` (deterministic, `NormalizedRMSE`),
  `ProbabilisticMetric` (UQ, `GaussianNLL`), and `ood_gap` for the
  out-of-distribution generalization test.

## House style (holds for any code added here)

Class-based OOP; `@staticmethod` factories; no `@dataclass`; no `try`/`except`
(classify and return, as the forward model does); inline math comments; no em
dashes.
