"""
External CFD Queue Dry Run
==============================
Phase 10 infrastructure hardening.

Demonstrates the queue-based workflow for asynchronous CFD integration using
ExternalSolverForwardModel.  In this dry run, "external CFD" is simulated by
the local Python process writing result files to a results directory.

When connected to a real external solver (SU2, OpenFOAM, FUN3D, DNS pipeline):
1. The queue writer (this script) writes theta JSON files to queue_dir.
2. The external solver reads them, runs CFD, writes result JSON to results_dir.
3. This script reloads completed outputs and updates the surrogate.

Queue file format
-----------------
    queue_dir/<uuid>.json
    {
        "job_id":      "<uuid>",
        "theta":       [a1, betaStar],
        "param_names": ["a1", "betaStar"]
    }

Result file format
------------------
    results_dir/<uuid>.json
    {
        "job_id":     "<uuid>",
        "predictions": [<float>, ...],
        "converged":   true,
        "status":      "external"
    }

Active-learning loop
--------------------
1. Start with a small synthetic ensemble.
2. Train a surrogate on it.
3. Use active learning to recommend new theta values.
4. Write candidates to queue_dir.
5. Simulate external completion (write synthetic result files).
6. Reload completed outputs and update the ensemble.

Failed-run simulation
---------------------
- One candidate returns invalid predictions (all zeros → surrogate rejects).
- One candidate returns NaN predictions (excluded from ensemble update).
- One candidate succeeds.

Usage
-----
    python external_cfd_queue_dry_run.py --quick -o /tmp/queue_dry_run
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic forward model (analytic SBLI)
# ---------------------------------------------------------------------------

_OBS_X = np.linspace(0.0, 1.0, 6)
X_SEP, X_REAT = 0.35, 0.65


def _synthetic_predictions(theta) -> Optional[List[float]]:
    a1, bs = float(theta[0]), float(theta[1])
    cf = (0.002 * bs / 0.09
          - 0.0015 * np.exp(-((_OBS_X - X_SEP)  / 0.06) ** 2)
          + 0.0010 * np.exp(-((_OBS_X - X_REAT) / 0.08) ** 2))
    cp = ((0.3 + 0.4 * a1 / 0.31)
          * 0.5 * (1 + np.tanh((_OBS_X - 0.40) / 0.05))
          - 0.05 * np.exp(-((_OBS_X - X_SEP) / 0.05) ** 2))
    preds = np.concatenate([cp, cf]).tolist()
    return preds


# ---------------------------------------------------------------------------
# Queue simulation helpers
# ---------------------------------------------------------------------------

class QueueSimulator:
    """
    Simulates the role of an external CFD solver:
    reads pending jobs from queue_dir and writes synthetic results to results_dir.

    In production, replace this class with the actual solver shim.
    """

    def __init__(self, queue_dir: Path, results_dir: Path,
                 fail_mode: Optional[str] = None, seed: int = 0):
        self._queue_dir   = queue_dir
        self._results_dir = results_dir
        self._fail_mode   = fail_mode   # "invalid", "nan", or None (success)
        self._rng         = np.random.default_rng(seed)

    def process_pending(self) -> int:
        """Process all pending queue files; return count processed."""
        processed = 0
        for job_file in sorted(self._queue_dir.glob("*.json")):
            payload = json.loads(job_file.read_text())
            job_id  = payload["job_id"]
            theta   = payload["theta"]

            result_file = self._results_dir / f"{job_id}.json"
            if result_file.exists():
                continue  # already processed

            if self._fail_mode == "nan":
                preds = [float("nan")] * 12
                ok    = True
            elif self._fail_mode == "invalid":
                preds = [0.0] * 12
                ok    = False
            else:
                preds = _synthetic_predictions(theta)
                noise = self._rng.standard_normal(len(preds)) * 0.001
                preds = [p + n for p, n in zip(preds, noise)]
                ok    = True

            result_file.write_text(json.dumps({
                "job_id":      job_id,
                "predictions": preds,
                "converged":   ok,
                "status":      "external",
            }))
            processed += 1

        return processed


def _queue_candidate(queue_dir: Path, theta: List[float],
                     param_names: List[str]) -> str:
    """Write a candidate theta to the queue; return job_id."""
    job_id   = str(uuid.uuid4())
    job_file = queue_dir / f"{job_id}.json"
    job_file.write_text(json.dumps({
        "job_id":      job_id,
        "theta":       [float(t) for t in theta],
        "param_names": param_names,
    }))
    return job_id


def _collect_results(results_dir: Path, job_ids: List[str],
                     n_obs: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """Collect completed results; filter NaN and unconverged rows."""
    xs, ys, statuses = [], [], {}

    for jid in job_ids:
        rfile = results_dir / f"{jid}.json"
        if not rfile.exists():
            statuses[jid] = "timeout_not_found"
            continue
        try:
            data  = json.loads(rfile.read_text())
            preds = data.get("predictions", [])
            ok    = data.get("converged", False)
        except Exception as e:
            statuses[jid] = f"parse_error: {e}"
            continue

        if not ok:
            statuses[jid] = "not_converged"
            continue

        if len(preds) != n_obs:
            statuses[jid] = f"shape_mismatch: {len(preds)} vs {n_obs}"
            continue

        if any(not np.isfinite(p) for p in preds):
            statuses[jid] = "nan_in_predictions"
            continue

        statuses[jid] = "ok"

    ok_jobs = [j for j, s in statuses.items() if s == "ok"]
    # We don't have theta for each job stored here — caller must zip
    return ok_jobs, statuses


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_demo(out_dir: Path, quick: bool = False):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from bayesian_inference import MultiOutputSurrogate, Prior

    out_dir.mkdir(parents=True, exist_ok=True)
    queue_dir   = out_dir / "queue"
    results_dir = out_dir / "results"
    queue_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    param_names = ["a1", "betaStar"]
    lower  = np.array([0.22, 0.06])
    upper  = np.array([0.45, 0.13])
    n_obs  = 12   # 6 stations × 2 (Cp + Cf)

    n_init  = 15 if quick else 40
    n_query = 5  if quick else 15

    rng = np.random.default_rng(7)

    # ------------------------------------------------------------------
    # 1. Initial ensemble (LHS, processed synchronously)
    # ------------------------------------------------------------------
    print(f"\n[1/6] Building initial LHS ensemble (n={n_init}) ...")
    X_init = np.empty((n_init, 2))
    for j in range(2):
        perm = rng.permutation(n_init)
        X_init[:, j] = (lower[j]
                         + (upper[j] - lower[j]) * (perm + rng.random(n_init)) / n_init)

    # Write and immediately process
    init_jobs = {}
    for theta in X_init:
        jid = _queue_candidate(queue_dir, theta.tolist(), param_names)
        init_jobs[jid] = theta.tolist()

    sim = QueueSimulator(queue_dir, results_dir, fail_mode=None, seed=99)
    sim.process_pending()

    valid_X, valid_Y = [], []
    for jid, theta in init_jobs.items():
        rfile = results_dir / f"{jid}.json"
        if not rfile.exists():
            continue
        data  = json.loads(rfile.read_text())
        preds = data.get("predictions", [])
        if (data.get("converged", False)
                and len(preds) == n_obs
                and all(np.isfinite(p) for p in preds)):
            valid_X.append(theta)
            valid_Y.append(preds)

    X_ens = np.array(valid_X)
    Y_ens = np.array(valid_Y)
    print(f"      {len(X_ens)}/{n_init} valid samples")

    # ------------------------------------------------------------------
    # 2. Train initial surrogate
    # ------------------------------------------------------------------
    print("[2/6] Training initial surrogate ...")
    surrogate = MultiOutputSurrogate()
    surrogate.train(X_ens, Y_ens)
    print(f"      Surrogate trained on {len(X_ens)} samples, {n_obs} outputs")

    # ------------------------------------------------------------------
    # 3. Active learning — recommend new theta via max_var
    # ------------------------------------------------------------------
    print(f"[3/6] Active-learning: recommending {n_query} candidates ...")
    x_std  = X_ens.std(axis=0)
    x_std[x_std < 1e-12] = 1.0

    # Candidate pool (random samples from prior)
    candidates = rng.uniform(lower, upper, size=(200, 2))
    scores = []
    for cand in candidates:
        mu, var = surrogate.predict(cand.reshape(1, -1))
        scores.append(float(var.sum()))

    top_idx     = np.argsort(scores)[-n_query:][::-1]
    recommended = candidates[top_idx]

    print(f"      Selected {len(recommended)} max-variance candidates")

    # Write case_config.json documenting the queue format
    (out_dir / "case_config.json").write_text(json.dumps({
        "description":      "External CFD queue configuration",
        "param_names":      param_names,
        "param_bounds":     {"a1": [0.22, 0.45], "betaStar": [0.06, 0.13]},
        "queue_dir":        str(queue_dir),
        "results_dir":      str(results_dir),
        "n_obs_expected":   n_obs,
        "obs_description":  "6 Cp values + 6 Cf values at x=[0,0.2,...,1.0]",
        "external_solver":  "REPLACE_WITH_REAL_SOLVER_PATH",
        "notes": [
            "Write result JSON to results_dir/<job_id>.json after CFD completes.",
            "Set converged=false if the run diverged.",
            "Set all predictions to NaN if outputs are unphysical.",
        ],
    }, indent=2))

    # Write status.txt
    (out_dir / "status.txt").write_text(
        f"Queued {n_query} candidates at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        f"Awaiting external CFD results.\n"
    )

    # ------------------------------------------------------------------
    # 4. Write candidates to queue
    # ------------------------------------------------------------------
    print("[4/6] Writing candidate thetas to queue ...")
    candidate_jobs: Dict[str, List[float]] = {}
    for theta in recommended:
        jid = _queue_candidate(queue_dir, theta.tolist(), param_names)
        candidate_jobs[jid] = theta.tolist()

    (out_dir / "candidate_theta.json").write_text(json.dumps({
        "n_candidates": len(recommended),
        "param_names":  param_names,
        "candidates":   [{"job_id": jid, "theta": th}
                          for jid, th in candidate_jobs.items()],
    }, indent=2))

    # ------------------------------------------------------------------
    # 5. Simulate external completion — including failure modes
    # ------------------------------------------------------------------
    print("[5/6] Simulating external CFD (with failure cases) ...")
    job_ids  = list(candidate_jobs.keys())
    n_jobs   = len(job_ids)

    # Simulate: first job → invalid output, second → NaN, rest → success
    def _sim_job(jid, theta, mode):
        if mode == "invalid":
            results_dir.joinpath(f"{jid}.json").write_text(json.dumps({
                "job_id": jid, "predictions": [0.0] * n_obs,
                "converged": False, "status": "diverged",
            }))
        elif mode == "nan":
            results_dir.joinpath(f"{jid}.json").write_text(json.dumps({
                "job_id": jid, "predictions": [float("nan")] * n_obs,
                "converged": True, "status": "nan_output",
            }))
        else:
            preds = _synthetic_predictions(theta)
            results_dir.joinpath(f"{jid}.json").write_text(json.dumps({
                "job_id": jid, "predictions": preds,
                "converged": True, "status": "external",
            }))

    modes = (["invalid", "nan"]
             + ["success"] * (n_jobs - 2)) if n_jobs >= 3 else ["success"] * n_jobs
    for jid, mode in zip(job_ids, modes):
        _sim_job(jid, candidate_jobs[jid], mode)
        print(f"      {mode:9s}: {jid[:8]}...")

    # ------------------------------------------------------------------
    # 6. Reload and update surrogate
    # ------------------------------------------------------------------
    print("[6/6] Collecting results and updating surrogate ...")
    ok_jids, statuses = _collect_results(results_dir, job_ids, n_obs)

    n_success = sum(1 for s in statuses.values() if s == "ok")
    n_failed  = len(statuses) - n_success
    print(f"      {n_success} successful, {n_failed} failed/excluded")

    # Update ensemble with successful new runs
    for jid in ok_jids:
        theta = candidate_jobs[jid]
        preds = json.loads((results_dir / f"{jid}.json").read_text())["predictions"]
        X_ens = np.vstack([X_ens, theta])
        Y_ens = np.vstack([Y_ens, preds])

    print(f"      Ensemble now: {len(X_ens)} samples")
    surrogate.train(X_ens, Y_ens)
    print("      Surrogate updated successfully")

    # ------------------------------------------------------------------
    # Save log
    # ------------------------------------------------------------------
    run_log = {
        "initial_ensemble_size":  n_init,
        "valid_initial_samples":  len(valid_X),
        "n_candidates_queued":    n_jobs,
        "candidate_results":      statuses,
        "n_successful_new_runs":  n_success,
        "n_failed_new_runs":      n_failed,
        "final_ensemble_size":    len(X_ens),
        "note": (
            "DRY RUN — external CFD simulated locally using the analytic SBLI model. "
            "Connect to a real solver by writing theta JSON files to queue_dir and "
            "having the solver deposit result JSON files in results_dir."
        ),
    }
    (out_dir / "run_log.json").write_text(json.dumps(run_log, indent=2))
    (out_dir / "status.txt").write_text(
        f"Completed at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        f"Final ensemble: {len(X_ens)} samples\n"
        f"Successful new runs: {n_success}/{n_jobs}\n"
    )

    print(f"\nDry-run complete. Outputs in: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}")
    print("\nNOTE: SYNTHETIC — replace QueueSimulator with a real solver shim")
    print("      to connect to SU2 / OpenFOAM / FUN3D / DNS pipeline.")


def main():
    parser = argparse.ArgumentParser(
        description="External CFD queue dry run (synthetic CFD simulation)"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Reduced ensemble for CI")
    parser.add_argument("-o", "--out-dir",
                        default="outputs/external_cfd_queue_dry_run",
                        help="Output directory")
    args = parser.parse_args()

    run_demo(out_dir=Path(args.out_dir), quick=args.quick)


if __name__ == "__main__":
    main()
