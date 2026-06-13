"""
Synthetic-data calibration harnesses (pipeline validation studies).

Research-specific: these build particular cases with synthetic observations
to verify the inference pipeline recovers known parameters.  Extracted from
bayesian_inference.py (2026-06-12 modularization).
"""

import numpy as np

from solver_bindings import _rs
from bfs_reference import DS1985, _bfs_solver_settings
from calibration import BayesianInference
from koh_calibration import BayesianInferenceKOH
from koh_likelihood import KOHLikelihood


class BFSSyntheticCalibration:
    """
    Synthetic Bayesian calibration on a backward-facing step.

    Builds the BFS mesh + forward model at Menter defaults, generates
    synthetic observations (reattachment length + Cf profile) with
    Gaussian noise, then runs the full ensemble→GP→MCMC pipeline to
    verify the posterior recovers the generating parameters.

    Parameters
    ----------
    nx_up, nx_down : int
        Cells in x for upstream / downstream region.
    ny_up, ny_down : int
        Cells in y for upper (h_s..H) / lower (0..h_s) block.
    param_set : InferenceParameterSet or dict
        Which SST coefficients to calibrate.  Defaults to a1_betaStar.
    """

    def __init__(self, nx_up=20, nx_down=40, ny_up=20, ny_down=15,
                 param_set=None, yPlusTarget=1.0):
        rs = _rs()
        h_s   = DS1985['h_s']
        H_up  = DS1985['H_up']
        H     = DS1985['H']
        Lu    = DS1985['Lu']
        Ld    = DS1985['Ld']
        Re_h  = DS1985['Re_h']

        Ub = 1.0
        nu = Ub * h_s / Re_h

        self.mesh = rs.Mesh.make_backward_facing_step_2d(
            nx_up, nx_down, ny_up, ny_down,
            Lu, Ld, h_s, H,
            Re=Re_h, yPlusTarget=yPlusTarget
        )
        self.mesh.compute_wall_distance()

        Tu   = 0.05
        kIn  = 1.5 * (Ub * Tu) ** 2
        omIn = kIn / (nu * 100.0)

        self.bcs = rs.FlowBoundaryConditions.bfs_defaults(self.mesh, Ub, kIn, omIn)
        self.nu  = nu
        self.kIn = kIn
        self.omIn = omIn
        self.Ub  = Ub

        if param_set is None:
            param_set = rs.InferenceParameterSet.a1_betaStar()
        self.param_set = param_set

        # Build observation operator at Menter defaults to get synthetic truth
        self._obs_truth = self._build_obs(Re_h=Re_h, xr_obs=DS1985['xr_h'],
                                          xr_sigma=0.3, Ub=Ub)

        settings = _bfs_solver_settings(rs)
        self.forward_model = rs.ForwardModel(
            self.mesh, param_set, self._obs_truth, self.bcs, nu, settings,
            rs.Vec3(Ub, 0, 0), 0.0, kIn, omIn
        )

    def _build_obs(self, Re_h, xr_obs, xr_sigma, Ub):
        rs = _rs()
        obs = rs.ObservationOperator()
        # Reattachment length (primary observable for a1 sensitivity)
        obs.add_reattachment_length(
            'bottom_wall_down', xr_obs, sigma=xr_sigma
        )
        # Cf at 3 downstream stations (primary observable for betaStar)
        for x_over_h, cf_obs, cf_std in DS1985['Cf_stations'][3:6]:
            obs.add_skin_friction(
                wall_patch='bottom_wall_down',
                location=rs.Vec3(x_over_h * DS1985['h_s'], 0.0, 0.5),
                cf_obs=cf_obs,
                sigma=cf_std,
                ref_vel=Ub,
            )
        return obs

    def generate_synthetic_truth(self, verbose=True):
        """
        Run forward model at Menter defaults → extract clean predictions.

        Returns
        -------
        theta_true : ndarray
            Menter default values for the active parameters.
        obs_clean : ndarray
            Clean forward model predictions at the defaults.
        """
        rs = _rs()
        defaults = np.array(self.param_set.pack(rs.SSTCoefficients()))
        result   = self.forward_model.evaluate(defaults.tolist())
        if verbose:
            print(f"  Truth solve: {result.status}")
            print(f"  Predictions: xr={result.predictions[0]:.3f}h, "
                  f"Cf@x4={result.predictions[1]:.5f}, "
                  f"Cf@x5={result.predictions[2]:.5f}, "
                  f"Cf@x6={result.predictions[3]:.5f}")
        return defaults, np.array(result.predictions)

    def run(self, n_ensemble=80, n_steps=1000, noise_frac=0.05,
            verbose=True, use_koh=False):
        """
        Full synthetic calibration pipeline.

        1. Solve at Menter defaults to get clean observations.
        2. Add Gaussian noise (noise_frac * |obs|).
        3. Build a new ForwardModel with noisy observations.
        4. Run BayesianInference (ensemble → GP → MCMC).
           If use_koh=True, runs BayesianInferenceKOH instead, which adds
           Kennedy-O'Hagan model-form discrepancy hyperparameters [log σ_δ, log l_δ].

        Returns
        -------
        bi : BayesianInference or BayesianInferenceKOH
        theta_true : ndarray
        """
        rs = _rs()

        theta_true, obs_clean = self.generate_synthetic_truth(verbose=verbose)

        # Noisy observations
        noise = noise_frac * np.abs(obs_clean)
        noise = np.maximum(noise, 1e-6)
        obs_noisy = obs_clean + noise * np.random.randn(len(obs_clean))

        # Rebuild obs operator with noisy data
        obs_noisy_op = rs.ObservationOperator()
        obs_noisy_op.add_reattachment_length(
            'bottom_wall_down', float(obs_noisy[0]), sigma=float(noise[0])
        )
        cf_stations = DS1985['Cf_stations'][3:6]
        for k, (x_over_h, _, cf_std) in enumerate(cf_stations):
            obs_noisy_op.add_skin_friction(
                wall_patch='bottom_wall_down',
                location=rs.Vec3(x_over_h * DS1985['h_s'], 0.0, 0.5),
                cf_obs=float(obs_noisy[1 + k]),
                sigma=float(noise[1 + k]),
                ref_vel=self.Ub,
            )

        fm_noisy = rs.ForwardModel(
            self.mesh, self.param_set, obs_noisy_op, self.bcs,
            self.nu, _bfs_solver_settings(rs),
            rs.Vec3(self.Ub, 0, 0), 0.0, self.kIn, self.omIn
        )

        n_walkers = max(16, 2 * (self.param_set.n_active() + (2 if use_koh else 0)))
        burn_in   = max(50, n_steps // 5)

        if use_koh:
            # Observation locations (x/h): reattachment + Cf stations
            xr_loc = float(obs_noisy[0])   # reattachment position itself
            cf_locs = [x_over_h for x_over_h, _, _ in cf_stations]
            obs_locations = np.array([xr_loc] + cf_locs)

            koh = KOHLikelihood(
                obs_locations=obs_locations,
                obs_values=obs_noisy,
                obs_sigmas=noise,
            )

            if verbose:
                print(f"\n  BFS KOH calibration: {n_ensemble} ensemble, "
                      f"{n_steps} MCMC steps, {n_walkers} walkers")
                print(f"  Obs locations (x/h): {obs_locations.tolist()}")

            bi = BayesianInferenceKOH(fm_noisy, self.param_set, koh)
            bi.run_ensemble(n_samples=n_ensemble, verbose=verbose)
            bi.train_surrogate(verbose=verbose)
            bi.run_mcmc(n_walkers=n_walkers, n_steps=n_steps,
                        burn_in=burn_in, thin=1, verbose=verbose)
        else:
            if verbose:
                print(f"\n  BFS synthetic calibration: {n_ensemble} ensemble, "
                      f"{n_steps} MCMC steps, {n_walkers} walkers")

            bi = BayesianInference(fm_noisy, self.param_set)
            bi.run_ensemble(n_samples=n_ensemble, verbose=verbose)
            bi.train_surrogate(verbose=verbose)
            bi.run_mcmc(n_walkers=n_walkers, n_steps=n_steps,
                        burn_in=burn_in, thin=1, verbose=verbose)

        return bi, theta_true


#  Synthetic data generation (pipeline validation)

class SyntheticCalibration:
    """
    Generate synthetic calibration data for testing the inference pipeline.

    Picks "true" SST coefficients (perturbations of Menter defaults),
    runs the forward model, extracts observables, adds Gaussian noise,
    and returns the truth + noisy observations for use in calibration.

    This validates that the pipeline can recover known parameters
    before applying it to real/DNS data.
    """

    def __init__(self, forward_model, param_set):
        self.forward_model = forward_model
        self.param_set = param_set

    def generate(self, perturbation=0.10, noise_std=0.001, verbose=True):
        """
        Create synthetic observations from perturbed SST coefficients.

        Parameters
        ----------
        perturbation : float
            Fractional perturbation from Menter defaults (e.g. 0.10 = 10%).
        noise_std : float
            Additive Gaussian noise standard deviation on observables.

        Returns
        -------
        theta_true : ndarray
            The "true" parameter vector used to generate data.
        obs_noisy : ndarray
            Noisy synthetic observations.
        """
        if hasattr(self.param_set, 'pack'):
            defaults = np.array(self.param_set.pack(_rs().SSTCoefficients()))
            lo = np.array(self.param_set.lower_bounds())
            hi = np.array(self.param_set.upper_bounds())
        else:
            # dict fallback
            defaults = np.array(self.param_set['defaults'])
            lo = np.array(self.param_set['lower'])
            hi = np.array(self.param_set['upper'])

        # perturb defaults
        theta_true = defaults * (1.0 + perturbation * np.random.randn(len(defaults)))
        theta_true = np.clip(theta_true, lo, hi)

        # run forward model at true parameters
        result = self.forward_model.evaluate(theta_true.tolist())

        if verbose:
            print(f"  Synthetic truth:    θ = {theta_true}")
            print(f"  Forward model:      {result.status}")
            print(f"  Clean predictions:  {result.predictions}")

        # add noise
        obs_clean = np.array(result.predictions)
        obs_noisy = obs_clean + noise_std * np.random.randn(len(obs_clean))

        if verbose:
            print(f"  Noisy observations: {obs_noisy}")
            print(f"  Noise std:          {noise_std}")

        return theta_true, obs_noisy
