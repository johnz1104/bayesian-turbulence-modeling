# Data provenance manifest

Dataset CONTENTS in this directory are gitignored and kept local; only this
manifest is tracked (see .gitignore: `data/*` ignored, `data/README.md` excepted).
Record every dataset here so any consumer can fetch or regenerate it. There is no
real hypersonic / SBLI data in this repository; mark synthetic data clearly.

Add one entry per dataset:

## <dataset-name>
- Source or URL:
- Version / date:
- Checksum (sha256):
- Fetch or regenerate command:
- Used by:
- Notes (synthetic? license?):

<!-- Example
## lorenz96_F8_truth
- Source or URL: generated locally
- Version / date: 2026-06-18
- Checksum (sha256): <sha256 of the .npz>
- Fetch or regenerate command: python research/shared/benchmarks/lorenz96.py --seed 0
- Used by: research/experiments/thread_a_chaotic
- Notes: synthetic truth, deterministic given the seed
-->
