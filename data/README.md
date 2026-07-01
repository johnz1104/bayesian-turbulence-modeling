# Data provenance manifest

Dataset CONTENTS in this directory are gitignored and kept local; only this
manifest is tracked (see `.gitignore`: `data/*` ignored, `data/README.md`
excepted). Record every dataset here so any consumer can fetch or regenerate it.

There is NO real hypersonic / SBLI data in this repository yet. Mark synthetic
data clearly, and never present a synthetic or incompressible result as a
hypersonic one (root CLAUDE.md, cross-cutting constraints).

The field keys below match `research/shared/benchmarks/base.py`
(`PROVENANCE_FIELDS`), so a loader's `BenchmarkData.metadata` and this manifest
cannot drift. `docs/data_acquisition.md` holds the candidate databases and the
acceptance bar these entries must clear.

## Per-dataset entry template

Copy this block per dataset. Use `n/a` where a field genuinely does not apply
(for example a ramp angle for a channel), never to skip a field that matters.

```
## <dataset-name>
- source:               author/lab, year, paper or DOI, and the access URL
- fidelity:             DNS | wall-resolved LES | experiment | synthetic-analytic
- geometry:             e.g. compression ramp | impinging-shock SWBLI | double cone |
                        flat plate | channel | backward-facing step | periodic hill
- mach:                 freestream Mach number (and edge Mach if different)
- ramp_or_wedge_angle:  deg (or shock-generator angle for impinging-shock cases); n/a
- reynolds:             value AND its definition (Re_theta, Re_x, Re_delta2, Re_h, unit Re)
- normalization:        how lengths/velocities/stresses are nondimensionalized
- reference_state:      freestream vs boundary-layer-edge vs wall units (state which)
- separation_def:       criterion used (e.g. mean Cf = 0; sign change of wall shear)
- reattachment_def:     criterion used (e.g. mean Cf = 0 downstream; x_r location)
- coordinate_origin:    origin placement, esp. SHOCK-RELATIVE (x measured from the
                        nominal inviscid impingement / corner, sign convention)
- available:            mean fields? Reynolds stresses + anisotropy? surface Cf / Cp /
                        St (heat flux)? list exactly what is present
- license:              license and redistribution terms
- access_status:        public | on-request | embargoed | local-synthetic
- version_date:         dataset version or retrieval date
- checksum_sha256:      sha256 of the stored artifact
- fetch_or_regenerate:  exact command to fetch or regenerate the artifact
- used_by:              which experiment/thread consumes it
- notes:                synthetic? caveats? known issues?
```

<!-- Example (synthetic Thread A truth; deterministic given the seed)
## lorenz96_F8_truth
- source:               generated locally
- fidelity:             synthetic-analytic
- geometry:             n/a (Lorenz-96 dynamical system, K=40, F=8)
- mach:                 n/a
- ramp_or_wedge_angle:  n/a
- reynolds:             n/a
- normalization:        none (nondimensional model variables)
- reference_state:      n/a
- separation_def:       n/a
- reattachment_def:     n/a
- coordinate_origin:    n/a
- available:            full state trajectory; long-time statistics
- license:              n/a (locally generated)
- access_status:        local-synthetic
- version_date:         2026-06-18
- checksum_sha256:      <sha256 of the .npz>
- fetch_or_regenerate:  python research/experiments/thread_a_chaotic/lorenz96.py --seed 0
- used_by:              research/experiments/thread_a_chaotic
- notes:                synthetic truth, deterministic given the seed
-->
