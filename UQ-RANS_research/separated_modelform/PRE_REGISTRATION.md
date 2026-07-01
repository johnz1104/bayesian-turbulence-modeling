# Separated-flow generative model-form: pre-registration

Fixed 2026-07-01, before any model-form result was computed on the real separated
DNS. This file records the success and null criteria in advance, so they are
timestamped ahead of every result and are not tuned toward afterwards (the
no-tuning-toward-the-number rule, CLAUDE.md sections 2 and 3). It is committed first.

## The work

The first test of the generative, realizability-constrained model-form on flows where the Boussinesq
hypothesis genuinely fails, and the incompressible precursor to the compressible
shock-boundary-layer work. The channel coverage-correction study and the cross-flow
generalization study are complete and merged. This work replaces the global
correction (generalized Bayes and conformal) with a feature-conditioned generative
discrepancy law and propagates it through the solver to predicted quantities of
interest.

## Scope decisions taken at the start (confirmed 2026-07-01)

1. A-posteriori, not a-priori only. The sampled, realizable model-form is propagated
   through the solver to predicted quantities of interest (reattachment length,
   bubble size, Cf, Cp). The a-priori discrepancy fit and its coverage are the
   precursor; the primary result is the a-posteriori one, because the
   error reduction must be shown on predicted (not frozen) quantities and the
   eigenspace-perturbation baseline is itself a-posteriori, so a fair comparison runs
   both through the solver.
2. Datasets: backward-facing step (Le, Moin and Kim 1997, Re_h = 5100, sparse
   profiles) and periodic hills (Xiao, Wu, Laizet, Duan 2020, five-case slope family,
   dense field). The backward-facing step is built first (existing inlet/outlet mesh
   factory); periodic hills follows with streamwise-periodic boundary conditions added
   to the incompressible SIMPLE solver, so it runs a-posteriori as well and supplies
   the dense-field coverage plus the cross-geometry out-of-distribution axis.
3. Injection formulation: explicit deferred-correction. The sampled anisotropy
   b_target = b_baseline + db enters the momentum equation as an explicit body force
   equal to the divergence of (2 k b_target minus the Boussinesq-modeled deviatoric
   stress), with the eddy-viscosity diffusion retained implicitly as the stabilizer.
   Every sampled b_target is projected into the barycentric realizable set and the
   projection is re-asserted each outer iteration. This avoids the known
   ill-conditioning of full explicit Reynolds-stress substitution (Wu et al. 2019).
   Convergence of a single coupled run is verified before any ensemble.

## The question (the gate)

Does the conditional generative model-form, trained on the real separated discrepancy
and projected into the realizable set, reduce quantity-of-interest error and deliver
calibrated coverage over the eigenspace-perturbation and Gaussian Kennedy-O'Hagan
baselines, when propagated through the solver.

### Pre-registered positive shape

Propagated through the solver, all four hold:

1. Coverage and accuracy. The 90 percent predictive band for the reattachment length
   contains the DNS truth (backward-facing step x_r/h about 6.28; periodic hills the
   per-case reattachment from the dense field), while the baseline SST point error is
   reduced.
2. Proper scoring rules. CRPS and the multivariate energy score on the
   reattachment-region quantities (reattachment length, bubble size, Cf and Cp along
   the recovery region) are no worse than, and in aggregate better than, both the
   eigenspace-perturbation envelope and the Gaussian Kennedy-O'Hagan baseline.
3. Realizability in the running solve. Every sampled closure stays realizable in the
   coupled solve (the barycentric check passes each outer iteration), as a check
   separate from the Galilean-invariant feature construction.
4. Cross-geometry generalization. Train on one geometry and predict the other, with
   the coverage gap reported honestly. Graceful degradation is expected; a silent
   collapse is a negative signal.

### Pre-registered null or negative shape (equally reportable)

The generative model-form does not beat the eigenspace-perturbation envelope on the
proper scores, or coverage is not restored. Per CLAUDE.md section 2, this is a real
result that reshapes the compressible direction, reported plainly, not engineered
around.

### Thresholds

Shape-based, consistent with the channel and Couette pre-registrations ("contains the
truth", "no worse than the baselines on the proper scores", "graceful, not silent,
degradation"). No numeric threshold is fixed in advance, and no learning rate, conformal
score, feature set, or model is tuned toward any test number.

## Baselines the result is measured against

- Deterministic baseline SST (the point-accuracy reference).
- Eigenspace-perturbation model-form UQ (Emory, Larsson and Iaccarino 2013;
  Iaccarino, Mishra and Gorle 2017): barycentric eigenvalue perturbation to the 1C,
  2C and 3C corners, propagated a-posteriori. The dominant model-form method.
- Gaussian Kennedy-O'Hagan model-form UQ, propagated a-posteriori.

## Metrics

Empirical coverage of nominal intervals, sharpness at matched coverage, CRPS, the
multivariate energy score for the joint Reynolds stress, reliability and PIT, and
quantity-of-interest point error versus DNS with percent improvement over baseline
SST, reported by region.

## Data attribution

Both datasets are third-party DNS, not produced by this project, cited where used.
Periodic hills: Xiao, Wu, Laizet and Duan (2020), Comput. Fluids 200, 104431
(github.com/xiaoh/para-database-for-PIML; no explicit license stated in the source
repository, treated as research-use with citation). Backward-facing step: Le, Moin and
Kim (1997), J. Fluid Mech. 330, 349-374. Provenance is in DNS_data/README.md; raw
fields stay local and gitignored.
