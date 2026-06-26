"""Investigate the generalized-Bayes reliability bow above the diagonal."""
import sys, json
import numpy as np
from scipy import stats
import os
_R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(_R, "build")); sys.path.insert(0, os.path.join(_R, "python"))
from UQ.datasets import ChannelDNS
from UQ.datasets.channel_calibration import ChannelCalibration
from UQ.reproduce_channel import CONFIG, ensemble_path
from UQ import evaluation as ev

CASES = (180, 550, 1000, 2000, 5200)
levels = np.linspace(0.1, 0.9, 9)
all_pit, all_z = [], []
emp_temp = np.zeros(len(levels)); emp_conf = np.zeros(len(levels))
nstd = 0
for n in CASES:
    dns = ChannelDNS.load(n)
    c = ChannelCalibration(dns, param_set="a1_betaStar", n_stations=CONFIG["n_stations"],
                           cfg=CONFIG["cfg"], sigma_floor=CONFIG["sigma_floor"])
    c.load_cache(dict(np.load(ensemble_path(n, "a1_betaStar"))))
    post1 = c.sample_posterior(eta=1.0, n_steps=1500, burn_in=400, seed=0)
    eta = c.calibrate_eta(post1, np.arange(c.n_qoi))
    post_t = c.sample_posterior(eta=eta, n_steps=1500, burn_in=400, seed=0)
    pp = c.posterior_predictive(post_t, eta=eta, seed=1)        # (n_theta, n_qoi)
    truth = c.qoi_truth
    # PIT + reliability for the Gaussian (generalized-Bayes) predictive
    all_pit.append(ev.pit_values(truth, pp.T))
    for i, lv in enumerate(levels):
        cov, _ = ev.coverage_from_samples(truth, pp.T, level=lv)
        emp_temp[i] += cov
    # standardized residual = (truth - predictive mean)/predictive std
    mu = pp.mean(0); sd = pp.std(0)
    z = (truth - mu) / np.maximum(sd, 1e-12)
    all_z.append(z); nstd += 1
    # conformal reliability across levels (distribution-free), random station split
    rng = np.random.default_rng(0); idx = np.arange(c.n_qoi)
    cal = np.sort(rng.choice(idx, c.n_qoi // 2, replace=False))
    test = np.array([i for i in idx if i not in set(cal)])
    from UQ import conformal as cf
    pc = c.point_prediction(post1, cal); pt = c.point_prediction(post1, test)
    for i, lv in enumerate(levels):
        lo, hi = cf.split_conformal_intervals(pc, c.qoi_truth[cal], pt, alpha=1.0 - lv)
        emp_conf[i] += ev.empirical_coverage(c.qoi_truth[test], lo, hi)

emp_temp /= len(CASES); emp_conf /= len(CASES)
pit = np.concatenate(all_pit); z = np.concatenate(all_z)
print("nominal   emp(genBayes)  bow(emp-nom)   emp(conformal)")
for i, lv in enumerate(levels):
    print(f"  {lv:.1f}      {emp_temp[i]:.3f}        {emp_temp[i]-lv:+.3f}        {emp_conf[i]:.3f}")
print(f"\nPIT: mean={pit.mean():.3f} (ideal 0.5)  std={pit.std():.3f} (ideal {1/np.sqrt(12):.3f})")
print(f"  PIT in central half [0.25,0.75]: {np.mean((pit>=0.25)&(pit<=0.75)):.3f} (ideal 0.50)")
print(f"  KS p-value vs uniform: {stats.kstest(pit,'uniform').pvalue:.3f}")
print(f"standardized residual z: excess kurtosis={stats.kurtosis(z):.2f} (Gaussian 0)  "
      f"Shapiro p={stats.shapiro(z).pvalue:.1e}")
print(f"  fraction |z|<0.5: {np.mean(np.abs(z)<0.5):.3f} (Gaussian 0.383)  "
      f"|z|>2: {np.mean(np.abs(z)>2):.3f} (Gaussian 0.046)")
json.dump({"levels":levels.tolist(),"emp_tempered":emp_temp.tolist(),
           "emp_conformal":emp_conf.tolist(),"pit_excess_kurtosis":float(stats.kurtosis(z)),
           "pit_central_half_frac":float(np.mean((pit>=0.25)&(pit<=0.75)))},
          open(os.path.join(_R, "results/channel/reliability_investigation.json"),"w"), indent=2)
print("INVESTIGATION DONE")
