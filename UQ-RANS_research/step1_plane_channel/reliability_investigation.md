# Note: why generalized Bayes bows above the diagonal in the reliability diagram

Investigation of an observation on `figures/reliability.png`: the generalized-Bayes
curve sits slightly above the ideal diagonal at low nominal levels (0.1-0.2) and
more so across the middle (0.3-0.7), converging back to the diagonal by 0.8-0.9.
Question: is this a defect or expected? Answer: expected, and informative. It is
the signature of a Gaussian correction applied to a non-Gaussian discrepancy, and
it is exactly the gap the generative model-form (Steps 3+) is meant to close. The
headline 90 percent coverage claim is unaffected.

## What the diagram shows (quantified, pooled over the five Re)

| nominal | empirical (generalized Bayes) | empirical minus nominal |
|---|---|---|
| 0.1 | 0.133 | +0.033 |
| 0.2 | 0.210 | +0.010 |
| 0.3 | 0.305 | +0.005 |
| 0.4 | 0.467 | +0.067 |
| 0.5 | 0.571 | +0.071 |
| 0.6 | 0.676 | +0.076 |
| 0.7 | 0.733 | +0.033 |
| 0.8 | 0.790 | -0.010 |
| 0.9 | 0.905 | +0.005 |

The deviation is an over-coverage that peaks in the middle (about +0.07 near
nominal 0.5-0.6) and vanishes at the 90 percent level where the method is
calibrated by construction.

## Mechanism

The generalized-Bayes predictive is Gaussian: the multi-output surrogate mean
propagated over the (tempered) coefficient posterior, plus observation noise
inflated to variance sigma^2 / eta, with the learning rate eta moment-matched to
the empirical residual variance. Moment-matching fixes the second moment, so the
interval whose width is set by the variance (the ~90 percent interval) is
calibrated, but it says nothing about the shape of the predictive at intermediate
quantiles.

The actual model-misfit residuals are not Gaussian. Standardising them by the
predictive standard deviation (z = (truth - mean) / std, pooled over all stations
and Reynolds numbers):

- Shapiro-Wilk p = 0.024: normality is rejected.
- excess kurtosis = +0.53: the distribution is mildly leptokurtic (more peaked
  than Gaussian, with thinner tails).
- 43.8 percent of residuals have |z| < 0.5, versus 38.3 percent for a Gaussian:
  there is excess mass near zero.
- 3.8 percent have |z| > 2, versus 4.6 percent for a Gaussian: the tails are
  slightly thin.

A Gaussian predictive whose variance is matched to a more-peaked-than-Gaussian
residual distribution is too wide in the middle: the central intervals contain
more truth than their nominal level, which is the bow, while the variance-setting
90 percent interval stays on target. The probability-integral transform confirms
the same thing from the data side: 57.1 percent of the PIT values fall in the
central half [0.25, 0.75] versus 50 percent for a calibrated predictor (the PIT is
under-dispersed), and a Kolmogorov-Smirnov test against uniform gives p = 0.003.

A secondary, smaller effect: the PIT mean is 0.41 rather than 0.5, a mild mean
bias (the predictive median sits a little above the DNS on average), from the
residual being slightly asymmetric and the developing-channel forward map. This
shifts the curve but is small next to the shape effect.

## Why this is expected and not a defect

This is the documented limitation of a Gaussian or moment-matched model-form
correction under a non-Gaussian discrepancy (compressible_research.md, "Why a
stochastic, generative discrepancy specifically": a Gaussian Kennedy-O'Hagan term
cannot match a discrepancy that is heteroscedastic, skewed, or multimodal, so the
predictive distribution is misshapen even when its mean and its variance are
reasonable). On the attached channel the non-Gaussianity is mild (excess kurtosis
only +0.53), so the bow is small (at most +0.076) and the 90 percent coverage is
restored cleanly. The effect is expected to be larger on the separated (Step 3)
and compressible-SBLI (Step 5) cases, where the discrepancy is strongly
non-Gaussian, and that is precisely where the conditional generative model-form,
which fits the full conditional density rather than a variance, is designed to
remove it. In that sense the bow here is a small, in-distribution preview of the
argument for the generative spine.

## What does not fix it, and what does

- Conformal is distribution-free and so is not tied to a Gaussian shape, but at
  this per-Re station count (about 20 stations, split for calibration) its
  across-level curve is finite-sample noisy; it is reliable at the single headline
  level (0.9, where the full-split conformal gives 0.91 in the main finding) but is
  not a clean fix for the whole curve at this sample size.
- The principled fix is a predictive that matches the discrepancy shape, not just
  its variance: the generative conditional model-form. Establishing that on a case
  with a strongly non-Gaussian discrepancy is the Step 3 finding.

## Reproduce

`reliability_investigation.py` (scratch) loads the cached ensembles, recomputes
the tempered posterior predictive, and prints the table above plus the PIT and
residual-shape statistics; numbers are also in
`results/channel/reliability_investigation.json` (gitignored, regenerable).
