"""
POST-PHASE 4: Black-box forward-model interface.

Provides ForwardModelBase — a unified ABC so that BayesianInferenceKOH can
work transparently with any of:
  - InternalIncompressibleForwardModel  (C++ rans_sst_py.ForwardModel)
  - InternalCompressibleForwardModel    (C++ rans_sst_py.CompressibleForwardModel)
  - PrecomputedEnsembleForwardModel     (θ_i, H(θ_i)) pairs from NPZ or dict
  - ExternalSolverForwardModel          (queue-based stub for async CFD)
  - SurrogateForwardModel               (trained MultiOutputSurrogate)

BayesianInferenceKOH calls forward_model.evaluate(theta) → result.predictions
for every ensemble member.  All concrete classes satisfy this protocol.

Optional short-circuit
----------------------
If forward_model.precomputed_ensemble() returns (X, Y) arrays,
BayesianInferenceKOH.run_ensemble() can use them directly instead of doing its
own LHC sampling.  Add the following guard at the top of run_ensemble():

    pre = self.forward_model.precomputed_ensemble()
    if pre is not None:
        self.ensemble_X, self.ensemble_Y = pre
        return self.ensemble_X, self.ensemble_Y
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Result of a single forward-model evaluation.

    Compatible with the protocol expected by BayesianInferenceKOH:
        result.predictions  → list[float]
        result.status       → str
    """
    predictions: List[float]
    converged: bool = True
    status: str = "ok"
    metadata: Dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.converged and bool(self.predictions)


# ---------------------------------------------------------------------------
# ForwardModelBase
# ---------------------------------------------------------------------------

class ForwardModelBase(ABC):
    """
    Abstract base class for all forward-model implementations.

    Subclasses must implement:
        evaluate(theta)     → EvaluationResult
        parameter_names()   → list[str]

    Optionally override:
        precomputed_ensemble() → (X, Y) or None
    """

    @abstractmethod
    def evaluate(self, theta) -> EvaluationResult:
        """Evaluate the forward model at parameter vector ``theta``."""

    @abstractmethod
    def parameter_names(self) -> List[str]:
        """Names of the input parameters (length == ndim of theta)."""

    def precomputed_ensemble(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Return (X, Y) arrays if this model holds a precomputed ensemble,
        else return None.

        X : (n_samples, d_theta)
        Y : (n_samples, n_obs)
        """
        return None


# ---------------------------------------------------------------------------
# Internal C++ wrappers
# ---------------------------------------------------------------------------

class InternalIncompressibleForwardModel(ForwardModelBase):
    """
    Thin wrapper around rans_sst_py.ForwardModel (C++ incompressible SIMPLE).

    Parameters
    ----------
    cpp_forward_model : rans_sst_py.ForwardModel
        Fully initialised C++ forward model.  The Python owner must remain
        in scope for the lifetime of this wrapper (C++ stores references).
    param_set : rans_sst_py.InferenceParameterSet or dict
        Describes the active SST parameters.
    """

    def __init__(self, cpp_forward_model, param_set):
        self._fm = cpp_forward_model
        self._param_set = param_set

    def evaluate(self, theta) -> EvaluationResult:
        result = self._fm.evaluate(list(theta))
        preds = list(result.predictions) if result.predictions else []
        converged = bool(preds) and all(np.isfinite(p) for p in preds)
        return EvaluationResult(
            predictions=preds,
            converged=converged,
            status=str(result.status),
        )

    def parameter_names(self) -> List[str]:
        return _extract_param_names(self._param_set)


class InternalCompressibleForwardModel(ForwardModelBase):
    """
    Thin wrapper around rans_sst_py.CompressibleForwardModel (C++ Ma<~0.5).

    Parameters
    ----------
    cpp_forward_model : rans_sst_py.CompressibleForwardModel
    param_set : rans_sst_py.InferenceParameterSet or dict
    """

    def __init__(self, cpp_forward_model, param_set):
        self._fm = cpp_forward_model
        self._param_set = param_set

    def evaluate(self, theta) -> EvaluationResult:
        result = self._fm.evaluate(list(theta))
        preds = list(result.predictions) if result.predictions else []
        converged = bool(preds) and all(np.isfinite(p) for p in preds)
        return EvaluationResult(
            predictions=preds,
            converged=converged,
            status=str(result.status),
        )

    def parameter_names(self) -> List[str]:
        return _extract_param_names(self._param_set)


# ---------------------------------------------------------------------------
# PrecomputedEnsembleForwardModel
# ---------------------------------------------------------------------------

class PrecomputedEnsembleForwardModel(ForwardModelBase):
    """
    Forward model backed by a precomputed (θ, H(θ)) ensemble.

    evaluate(theta) returns the prediction for the nearest stored θ (L2 in
    normalised parameter space).  The full ensemble is also available via
    precomputed_ensemble() for direct injection into BayesianInferenceKOH.

    Construction
    ------------
    Build from arrays::

        fm = PrecomputedEnsembleForwardModel(X, Y, param_names=["a1", "betaStar"])

    Load from an NPZ file produced by :func:`save_ensemble_npz`::

        fm = PrecomputedEnsembleForwardModel.from_npz("ensemble.npz")

    Parameters
    ----------
    X : array-like, shape (n, d_theta)
        Parameter vectors used during the ensemble run.
    Y : array-like, shape (n, n_obs)
        Corresponding forward-model predictions.
    param_names : list[str], optional
        Names of the theta components.  Inferred from NPZ metadata if absent.
    """

    def __init__(self, X, Y, param_names: Optional[List[str]] = None):
        self._X = np.asarray(X, dtype=float)          # (n, d)
        self._Y = np.asarray(Y, dtype=float)          # (n, n_obs)
        n, d = self._X.shape
        self._param_names: List[str] = (
            list(param_names) if param_names
            else [f"theta_{i}" for i in range(d)]
        )
        # Normalise columns for distance computation
        self._x_std = self._X.std(axis=0)
        self._x_std[self._x_std < 1e-12] = 1.0

    # ---- construction helpers --------------------------------------------

    @classmethod
    def from_npz(cls, path) -> "PrecomputedEnsembleForwardModel":
        """Load from an NPZ file written by :func:`save_ensemble_npz`."""
        data = np.load(str(path), allow_pickle=True)
        X = data["X"]
        Y = data["Y"]
        names: Optional[List[str]] = None
        if "param_names" in data:
            names = list(data["param_names"])
        return cls(X, Y, param_names=names)

    @classmethod
    def from_dict(cls, mapping: Dict) -> "PrecomputedEnsembleForwardModel":
        """Build from a plain dict with keys ``'X'``, ``'Y'``, ``'param_names'``."""
        X = np.asarray(mapping["X"])
        Y = np.asarray(mapping["Y"])
        names = mapping.get("param_names")
        return cls(X, Y, param_names=names)

    # ---- ForwardModelBase interface --------------------------------------

    def evaluate(self, theta) -> EvaluationResult:
        theta = np.asarray(theta, dtype=float)
        idx = self._nearest(theta)
        preds = self._Y[idx].tolist()
        return EvaluationResult(predictions=preds, converged=True, status="precomputed")

    def parameter_names(self) -> List[str]:
        return list(self._param_names)

    def precomputed_ensemble(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, Y) directly so BayesianInferenceKOH skips LHC sampling."""
        return self._X.copy(), self._Y.copy()

    # ---- helpers ---------------------------------------------------------

    def _nearest(self, theta: np.ndarray) -> int:
        delta = (self._X - theta) / self._x_std
        dists = np.sum(delta ** 2, axis=1)
        return int(np.argmin(dists))

    @property
    def n_samples(self) -> int:
        return len(self._X)

    @property
    def n_outputs(self) -> int:
        return self._Y.shape[1]


def save_ensemble_npz(
    forward_model: ForwardModelBase,
    bounds: Tuple[np.ndarray, np.ndarray],
    n_samples: int,
    path,
    rng_seed: Optional[int] = None,
    verbose: bool = True,
) -> Path:
    """
    Latin-hypercube sample ``forward_model`` and save (X, Y) to an NPZ file.

    This is the offline pre-computation step.  Call once; load the result
    with :meth:`PrecomputedEnsembleForwardModel.from_npz` thereafter.

    Parameters
    ----------
    forward_model : ForwardModelBase
        Any concrete forward model (typically an internal C++ solver).
    bounds : (lower, upper)
        Parameter bounds, each shape (d_theta,).
    n_samples : int
        Number of LHC samples to evaluate.
    path : str or Path
        Output file path (will have ``.npz`` appended if absent).
    rng_seed : int, optional
    verbose : bool

    Returns
    -------
    Path
        Path to the saved NPZ file.
    """
    lower, upper = np.asarray(bounds[0]), np.asarray(bounds[1])
    d = len(lower)
    rng = np.random.default_rng(rng_seed)

    # Latin hypercube
    X_lhc = _latin_hypercube(n_samples, d, lower, upper, rng)

    valid_X, valid_Y = [], []
    t0 = time.time()
    for i, theta in enumerate(X_lhc):
        result = forward_model.evaluate(theta)
        preds = result.predictions
        if result.converged and preds and all(np.isfinite(p) for p in preds):
            valid_X.append(theta)
            valid_Y.append(preds)
        if verbose and (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  ensemble {i+1}/{n_samples}  valid={len(valid_X)}"
                  f"  [{time.time()-t0:.1f}s]")

    X = np.array(valid_X)
    Y = np.array(valid_Y)
    names = np.array(forward_model.parameter_names())

    path = Path(path)
    np.savez(str(path), X=X, Y=Y, param_names=names)
    if verbose:
        print(f"  saved {len(X)} valid samples → {path}")
    return path


# ---------------------------------------------------------------------------
# ExternalSolverForwardModel  (queue-based stub)
# ---------------------------------------------------------------------------

class ExternalSolverForwardModel(ForwardModelBase):
    """
    Stub for an external or queued CFD solver.

    Writes each evaluate() call to a ``queue_dir`` as a JSON file
    (``<uuid>.json``) and polls ``results_dir`` for a matching result file.
    Returns an unconverged EvaluationResult immediately if no result is found
    within ``poll_timeout_s`` seconds.

    This is the hook for asynchronous high-Mach CFD (e.g. an HPC solver that
    cannot run in-process).  Active learning can queue new theta samples even
    when actual CFD evaluation is external.

    Parameters
    ----------
    queue_dir : str or Path
        Directory where pending theta files are written.
    results_dir : str or Path
        Directory where the external solver deposits result files.
    param_names : list[str]
        Names of the theta components.
    poll_timeout_s : float
        Seconds to wait for a result before returning unconverged.
    poll_interval_s : float
        Polling interval in seconds.
    n_obs : int
        Expected number of observables (used only to size failed results).
    """

    def __init__(
        self,
        queue_dir,
        results_dir,
        param_names: List[str],
        poll_timeout_s: float = 0.0,
        poll_interval_s: float = 0.5,
        n_obs: int = 0,
    ):
        self._queue_dir   = Path(queue_dir)
        self._results_dir = Path(results_dir)
        self._queue_dir.mkdir(parents=True, exist_ok=True)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._param_names  = list(param_names)
        self._poll_timeout = poll_timeout_s
        self._poll_interval = poll_interval_s
        self._n_obs = n_obs
        self._pending: Dict[str, np.ndarray] = {}  # job_id → theta

    def evaluate(self, theta) -> EvaluationResult:
        """Queue theta and poll for result; return unconverged if not ready."""
        import uuid
        job_id = str(uuid.uuid4())
        theta = np.asarray(theta, dtype=float)
        payload = {
            "job_id": job_id,
            "theta": theta.tolist(),
            "param_names": self._param_names,
        }
        queue_file = self._queue_dir / f"{job_id}.json"
        queue_file.write_text(json.dumps(payload))
        self._pending[job_id] = theta

        result_file = self._results_dir / f"{job_id}.json"
        deadline = time.monotonic() + self._poll_timeout
        while time.monotonic() < deadline:
            if result_file.exists():
                break
            time.sleep(self._poll_interval)

        if result_file.exists():
            try:
                data = json.loads(result_file.read_text())
                preds = [float(v) for v in data["predictions"]]
                self._pending.pop(job_id, None)
                return EvaluationResult(
                    predictions=preds,
                    converged=data.get("converged", True),
                    status=data.get("status", "external"),
                    metadata={"job_id": job_id},
                )
            except Exception as exc:
                return EvaluationResult([], converged=False,
                                        status=f"parse_error: {exc}")

        return EvaluationResult(
            predictions=[],
            converged=False,
            status="queued_pending",
            metadata={"job_id": job_id},
        )

    def parameter_names(self) -> List[str]:
        return list(self._param_names)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def deliver_result(self, job_id: str, predictions: List[float],
                       converged: bool = True) -> None:
        """
        Write a result file to results_dir (used in tests and by the external
        solver shim to signal completion).
        """
        result_file = self._results_dir / f"{job_id}.json"
        result_file.write_text(json.dumps({
            "job_id": job_id,
            "predictions": [float(p) for p in predictions],
            "converged": converged,
            "status": "external",
        }))


# ---------------------------------------------------------------------------
# SurrogateForwardModel
# ---------------------------------------------------------------------------

class SurrogateForwardModel(ForwardModelBase):
    """
    Forward model backed by a trained MultiOutputSurrogate.

    Allows BayesianInferenceKOH to use a cheap GP surrogate in place of a
    full CFD solve during exploratory MCMC or active-learning iterations.

    Parameters
    ----------
    surrogate : MultiOutputSurrogate
        A fully trained surrogate (``surrogate.trained`` must be True).
    param_names : list[str]
        Names of the theta components.
    """

    def __init__(self, surrogate, param_names: Optional[List[str]] = None):
        if not surrogate.trained:
            raise ValueError("SurrogateForwardModel requires a trained surrogate.")
        self._surrogate = surrogate
        self._param_names: List[str] = (
            list(param_names) if param_names
            else [f"theta_{i}" for i in range(surrogate.gps[0].X.shape[1])]
        )

    def evaluate(self, theta) -> EvaluationResult:
        mu, _var = self._surrogate.predict(np.asarray(theta, dtype=float))
        preds = mu.tolist()
        converged = all(np.isfinite(p) for p in preds)
        return EvaluationResult(predictions=preds, converged=converged,
                                status="surrogate")

    def parameter_names(self) -> List[str]:
        return list(self._param_names)


# ---------------------------------------------------------------------------
# Factory / convenience
# ---------------------------------------------------------------------------

def load_forward_model(config) -> ForwardModelBase:
    """
    Factory: load a ForwardModelBase from a config dict or JSON/NPZ path.

    Supported config keys:
        type: "precomputed_npz"  → PrecomputedEnsembleForwardModel.from_npz(path)
        type: "precomputed_dict" → PrecomputedEnsembleForwardModel.from_dict(config)

    Parameters
    ----------
    config : dict or str or Path
        If a path is given, its extension determines the loader:
          ``.npz`` → PrecomputedEnsembleForwardModel.from_npz
          ``.json`` → read JSON, then dispatch on "type" key
    """
    if isinstance(config, (str, Path)):
        p = Path(config)
        if p.suffix == ".npz":
            return PrecomputedEnsembleForwardModel.from_npz(p)
        if p.suffix == ".json":
            config = json.loads(p.read_text())
        else:
            raise ValueError(f"Unsupported file extension: {p.suffix}")

    kind = config.get("type", "precomputed_dict")
    if kind == "precomputed_npz":
        return PrecomputedEnsembleForwardModel.from_npz(config["path"])
    if kind == "precomputed_dict":
        return PrecomputedEnsembleForwardModel.from_dict(config)
    raise ValueError(f"Unknown forward model type: {kind!r}")


# ---------------------------------------------------------------------------
# ActiveLearner bridge
# ---------------------------------------------------------------------------

def forward_model_to_callable(fm: ForwardModelBase):
    """
    Wrap a ForwardModelBase for use with the existing ActiveLearner.

    ActiveLearner expects ``forward(theta) -> (predictions_array, ok_flag)``.

    Parameters
    ----------
    fm : ForwardModelBase

    Returns
    -------
    callable
        A function ``forward(theta) -> (np.ndarray | None, bool)``
    """
    def _forward(theta):
        result = fm.evaluate(theta)
        preds = np.array(result.predictions) if result.predictions else None
        ok = bool(result.converged
                  and preds is not None
                  and np.all(np.isfinite(preds)))
        return (preds if ok else None, ok)
    return _forward


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_param_names(param_set) -> List[str]:
    if hasattr(param_set, "active_names"):
        return list(param_set.active_names())
    if isinstance(param_set, dict) and "names" in param_set:
        return list(param_set["names"])
    n = param_set.n_active() if hasattr(param_set, "n_active") else 2
    return [f"theta_{i}" for i in range(n)]


def _latin_hypercube(n: int, d: int, lower: np.ndarray,
                     upper: np.ndarray, rng) -> np.ndarray:
    samples = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        samples[:, j] = lower[j] + (upper[j] - lower[j]) * (perm + rng.random(n)) / n
    return samples
