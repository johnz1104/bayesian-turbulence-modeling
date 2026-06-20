"""SST baseline closure: the known-working reference that validates the frozen
inference handshake (foundation freeze, core-v1.0).

This adapts the existing C++/Python RANS-SST framework (rans_sst_py +
case_library + priors) to the research.shared Closure interface. It is the
MEMORYLESS, DETERMINISTIC, LOCAL zero-point of the program:

    has_memory = is_stochastic = is_nonlocal = False,

against which every later memory / noise / non-local closure is measured. Its
purpose here is to prove the handshake's shape on something that already works:
the param spec comes from the real InferenceParameterSet, the prior reproduces
the real priors.Prior truncated normal, and the likelihood hook drives the real
ForwardModel.

House style: class-based, no dataclass, no try/except; a failed evaluation is the
forward model's returned EvaluationStatus, not an exception.
"""

import sys
from pathlib import Path

import numpy as np

from .base import Closure
from ..inference.handshake import (
    ClosurePrior, EvaluationStatus, Likelihood, ParameterSpec, Prediction,
)
from ..benchmarks.base import BenchmarkData, InMemoryBenchmark

# Make the compiled binding and the existing python/ layer importable, mirroring
# the bootstrap in python/case_library.py so this works standalone too.
_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO / "build"), str(_REPO / "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rans_sst_py as rs            # noqa: E402  (after sys.path bootstrap)
import case_library as _cl          # noqa: E402


def _spec_from_param_set(param_set) -> ParameterSpec:
    """Build a ParameterSpec from a rans_sst_py.InferenceParameterSet.

    Defaults are the active components of the Menter (1994) coefficients; the
    [lower, upper] box is the physical positivity + stability region the C++
    layer enforces, which is exactly the realizability + stability bounds the
    prior truncates to.
    """
    names = list(param_set.active_names())
    defaults = list(param_set.pack(rs.SSTCoefficients()))
    lower = list(param_set.lower_bounds())
    upper = list(param_set.upper_bounds())
    return ParameterSpec(names, defaults, lower, upper)


def _gaussian_loglik(eta, values, sigmas) -> float:
    """Diagonal Gaussian log-likelihood of predicted statistics vs truth.

    A minimal stand-in for the Kennedy-O'Hagan likelihood (which adds the
    marginalised model-form discrepancy on top of this same residual r = y - eta).
    """
    eta = np.asarray(eta, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if eta.shape != values.shape or not np.all(np.isfinite(eta)):
        return -np.inf
    r = (values - eta) / sigmas
    return float(-0.5 * np.sum(r * r + np.log(2.0 * np.pi * sigmas ** 2)))


class _ForwardModelLikelihood(Likelihood):
    """Likelihood hook over a forward model.

    Works with a C++ rans_sst_py.ForwardModel (exposes evaluate ->
    {status, predictions, log_lik} and penalized_log_likelihood) and with any
    python ForwardModelBase (evaluate -> {predictions, converged, status}).
    """

    def __init__(self, forward_model, spec: ParameterSpec, benchmark=None):
        self._fm = forward_model
        self._spec = spec
        self._benchmark = benchmark

    def parameter_spec(self) -> ParameterSpec:
        return self._spec

    def predict(self, theta) -> Prediction:
        r = self._fm.evaluate(list(theta))
        preds = list(r.predictions) if getattr(r, "predictions", None) is not None else []
        status = getattr(r, "status", None)
        if status is None:
            status = EvaluationStatus.Converged if getattr(r, "converged", False) \
                else EvaluationStatus.Unconverged
        return Prediction(preds, status=status,
                          log_likelihood=getattr(r, "log_lik", None))

    def log_likelihood(self, theta) -> float:
        # Prefer the forward model's own penalized likelihood (the real SST path).
        pll = getattr(self._fm, "penalized_log_likelihood", None)
        if callable(pll):
            value = float(pll(list(theta)))
            return value if np.isfinite(value) else -np.inf
        # Otherwise score predicted statistics against the bound benchmark truth.
        pred = self.predict(theta)
        if not pred.converged:
            return -np.inf
        if self._benchmark is not None:
            data = self._benchmark.load()
            return _gaussian_loglik(pred.statistics, data.values, data.sigmas)
        if pred.log_likelihood is not None and np.isfinite(pred.log_likelihood):
            return float(pred.log_likelihood)
        return -np.inf


class SSTClosure(Closure):
    """Menter (1994) k-omega SST as a research.shared Closure (the baseline)."""

    has_memory = False
    is_stochastic = False
    is_nonlocal = False

    def __init__(self, case_spec=None, param_set=None):
        if case_spec is None:
            ps = param_set or rs.InferenceParameterSet.a1_betaStar()
            case_spec = _cl.build_channel(param_set=ps)
        self._case = case_spec
        self._spec = _spec_from_param_set(case_spec.param_set)

    # ---- frozen handshake -------------------------------------------------

    @property
    def name(self) -> str:
        return "sst_menter1994"

    def parameter_spec(self) -> ParameterSpec:
        return self._spec

    def prior(self) -> ClosurePrior:
        # Truncated normal on Menter defaults, truncated to the realizability +
        # stability box. Memoryless, so no fluctuation-dissipation coupling.
        # This reproduces priors.make_prior_from_param_set (relative_std = 0.15).
        return ClosurePrior(self._spec, relative_std=0.15, fd_coupling=None)

    def likelihood(self, benchmark=None) -> Likelihood:
        bench = benchmark if benchmark is not None else self.benchmark()
        return _ForwardModelLikelihood(self._case.fm, self._spec, benchmark=bench)

    # ---- truth + convenience ---------------------------------------------

    def benchmark(self) -> InMemoryBenchmark:
        """The bound case's observations as an in-memory benchmark (truth)."""
        cs = self._case
        data = BenchmarkData(
            name=cs.name, observable=cs.obs_kind,
            locations=cs.obs_locations, values=cs.obs_values, sigmas=cs.obs_sigmas,
            metadata={"fidelity": "synthetic_analytic", "geometry": cs.name,
                      "source": "case_library anchor correlation"},
        )
        return InMemoryBenchmark(data)

    @staticmethod
    def channel(param_set=None, nx: int = 40, ny: int = 30) -> "SSTClosure":
        ps = param_set or rs.InferenceParameterSet.a1_betaStar()
        return SSTClosure(case_spec=_cl.build_channel(param_set=ps, nx=nx, ny=ny))

    @staticmethod
    def bfs(param_set=None) -> "SSTClosure":
        ps = param_set or rs.InferenceParameterSet.a1_betaStar()
        return SSTClosure(case_spec=_cl.build_bfs(param_set=ps))
