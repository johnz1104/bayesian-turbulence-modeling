# viz - figures from real QBTM output

Every figure here is generated from real solver / inference output produced on
this machine, never from synthetic or hardcoded data. The data-producing scripts
write artifacts into `viz/artifacts/` (gitignored, regenerable); the plotting
scripts read those artifacts and write PNGs into `viz/figures/`.

## Reproduce

```bash
# from the repo root, with the project built (see top-level README)
export PYTHONPATH=build:python

# 1. produce real artifacts
python3 viz/run_cpp_validation.py                 # solver Cf vs Dean/Schoenherr
python3 viz/run_calibration.py a1_betaStar         # 2-coef channel calibration (centrepiece)
python3 viz/run_calibration.py near_wall4          # 4-coef calibration (sensitivity/identifiability)
python3 viz/run_surrogate_benchmark.py             # surrogate-vs-CFD timing

# 2. render every figure from whatever artifacts exist
python3 viz/make_all_figures.py
```

`run_calibration.py` runs the actual pipeline: a Latin-hypercube ensemble of C++
SIMPLE solves, a GP surrogate fit to the resulting log-likelihoods, and emcee
sampling on the surrogate. Seeds are fixed, so the artifacts and figures
reproduce. The `near_wall4` run uses `--reuse-ensemble` to re-sample without
repeating the expensive CFD ensemble.

## Data-producing scripts

| script | reads | writes (artifact) |
|---|---|---|
| `run_cpp_validation.py` | `build/rans_sst` CLI | `artifacts/cpp_validation.json` |
| `run_calibration.py` | C++ forward model | `artifacts/<set>/{ensemble,chain,surrogate_holdout}.npz`, `summary.json` |
| `run_surrogate_benchmark.py` | cached ensemble + C++ solves | `artifacts/surrogate_benchmark.json` |

## Figure scripts

| figure | script | source artifact |
|---|---|---|
| `validation_cf.png` | `plot_validation.py` | `cpp_validation.json` |
| `posterior_corner_<set>.png` | `plot_posterior_corner.py` | `<set>/chain.npz` |
| `convergence_<set>.png` | `plot_convergence.py` | `<set>/chain.npz` |
| `surrogate_<set>.png` | `plot_surrogate.py` | `<set>/surrogate_holdout.npz` |
| `prior_posterior_<set>.png` | `plot_prior_posterior.py` | `<set>/chain.npz` + `summary.json` |
| `sensitivity_<set>.png` | `plot_sensitivity.py` | `<set>/summary.json` (ARD lengthscales) |
| `identifiability_<set>.png` | `plot_identifiability.py` | `<set>/summary.json` (posterior correlation) |
