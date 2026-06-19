"""
Acceptance: shared-θ recovery on a synthetic multi-case problem.

Three synthetic cases share one θ; each constrains it only partially, but jointly they
identify it.  ``MultiCaseCalibration`` must recover the planted θ_true within 2σ of the
joint posterior (the §6.2 multi-case acceptance).  No CFD (synthetic linear forward
models), so this locks the contract cheaply and deterministically.
"""

from __future__ import annotations

import numpy as np

from multi_case_calibration import MultiCaseCalibration, Case


class _Res:
    def __init__(self, eta):
        self.predictions = list(eta)
        self.log_lik = 0.0
        self.status = "Converged"


class _LinearFM:
    """η(θ) = A·θ — a synthetic case that constrains θ along the rows of A."""

    def __init__(self, A):
        self.A = np.asarray(A, float)

    def evaluate(self, theta_list):
        return _Res(self.A @ np.asarray(theta_list, float))


def test_shared_theta_recovered_within_2sigma(rs):
    np.random.seed(0)
    ps = rs.InferenceParameterSet.a1_betaStar()      # θ = (a1, betaStar)
    theta_true = np.array([0.31, 0.09])              # Menter defaults

    # three cases with different sensitivities (jointly identify θ)
    As = [np.array([[1.0, 0.0], [0.5, 0.5]]),
          np.array([[0.0, 1.0], [1.0, -0.3]]),
          np.array([[0.8, 0.2], [0.2, 0.8]])]
    mcc = MultiCaseCalibration(ps)
    for i, A in enumerate(As):
        fm = _LinearFM(A)
        y = A @ theta_true + 0.002 * np.random.randn(A.shape[0])    # synthetic obs
        mcc.add_case(Case(name=f"case{i}", forward_model=fm,
                          obs_locations=np.arange(A.shape[0]),
                          obs_values=y, obs_sigmas=np.full(A.shape[0], 0.01)))

    mcc.run_ensemble(n_samples=60, verbose=False, rng_seed=0)
    mcc.train_surrogates(verbose=False)
    mcc.run_mcmc(n_steps=1500, burn_in=500, verbose=False, rng_seed=0)

    summ = mcc.posterior_summary()
    names = list(summ.keys())
    means = np.array([summ[n]["mean"] for n in names])
    stds = np.array([summ[n]["std"] for n in names])

    # recovered within 2σ of the joint posterior
    z = np.abs(means - theta_true) / stds
    assert np.all(z < 2.0), f"shared-θ not recovered within 2σ: z={z}, means={means}"
