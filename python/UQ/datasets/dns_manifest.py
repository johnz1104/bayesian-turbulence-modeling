"""Deterministic content digests of the raw DNS source datasets.

The transitive cache lineage previously started at the solver-side caches:
an extraction identity hashed its baseline fields but not the DNS record the
discrepancy was measured against, and the far-transfer pools read the
attached datasets directly with no identity edge at all. Changing
QBTM_DNS_DATA, replacing a source file, or correcting a dataset could then
silently reuse an incompatible extraction or far-transfer result (the
review's raw-input lineage finding). This module closes the input side:

- dataset_digest(name): a deterministic digest of one registered dataset
  directory, sha256 over the sorted (relative path, content hash, size)
  listing of every regular file (hidden files and OS droppings excluded),
  truncated to 16 hex characters and memoized per process. Granularity is
  the DATASET DIRECTORY: any change inside a dataset invalidates every
  identity bound to it, which over-invalidates relative to per-case
  tracking but can never under-invalidate (source-data changes are rare,
  adjudicated events).
- The digests are BOUND INTO identities by the consumers: interaction
  extraction identities, the a-posteriori target configurations (member
  targets consume the record's reference scales and mask spans directly),
  and the a-priori partial and numbers identities (the far-transfer pools
  are direct DNS consumers).
- write_manifest / verify_manifest: the checksummed manifest record for
  the baseline and freeze provenance. The drivers verify it at start and
  refuse to run against silently changed data; regenerating the manifest
  is an explicit operator action after adjudicating the change.

A missing dataset digests to None, which embeds into identities and can
never match a real digest (fail closed).
"""
import hashlib
import json
import os

from . import _common
from .. import cache_fingerprint as cfp

# registered datasets: name -> subdirectory under the DNS data root
DATASETS = {
    "interaction_adiabatic": "shock_wave_BLI",
    "interaction_heated": "heat_transfer_SBLI",
    "gv_channel": os.path.join("compressible_channel_gv", "GV_TPC_MB_AIR0"),
    "supersonic_tbl": "supersonic_turbulent_BL",
    "zdc_plate": "sup_hypersonic_plate_flow",
}

# the dataset sets each consumer binds (interaction legs read the records
# directly; the far pools additionally read the attached sources)
INTERACTION_SET = ("interaction_adiabatic", "interaction_heated")
FAR_SET = INTERACTION_SET + ("gv_channel", "supersonic_tbl", "zdc_plate")

_MEMO = {}


def _dataset_dir(name, root=None):
    return os.path.join(_common.data_root(root), DATASETS[name])


def dataset_digest(name, root=None):
    """Content digest of one registered dataset directory (memoized per
    process; None when the directory is absent)."""
    base = os.path.abspath(_dataset_dir(name, root))
    if base in _MEMO:
        return _MEMO[base]
    if not os.path.isdir(base):
        _MEMO[base] = None
        return None
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, base)
            h.update(rel.encode())
            h.update(str(os.path.getsize(p)).encode())
            h.update(cfp.file_sha(p).encode())
    d = h.hexdigest()[:16]
    _MEMO[base] = d
    return d


def digests(names, root=None):
    """name -> digest map for a consumer's dataset set."""
    return {n: dataset_digest(n, root) for n in names}


def _stats(name, root=None):
    base = _dataset_dir(name, root)
    n_files, n_bytes = 0, 0
    if os.path.isdir(base):
        for dirpath, dirnames, filenames in os.walk(base):
            for fn in filenames:
                if fn.startswith("."):
                    continue
                n_files += 1
                n_bytes += os.path.getsize(os.path.join(dirpath, fn))
    return n_files, n_bytes


def write_manifest(path, names=tuple(DATASETS), root=None):
    """The checksummed DNS manifest record (freeze and baseline
    provenance): per-dataset digest, file count and byte count, carrying
    the standard identity block."""
    entries = {}
    for n in names:
        n_files, n_bytes = _stats(n, root)
        entries[n] = {"digest": dataset_digest(n, root),
                      "subdir": DATASETS[n],
                      "n_files": n_files, "bytes": n_bytes}
    rec = {"datasets": entries}
    ident = {"kind": "dns-manifest",
             "manifest": {n: entries[n]["digest"] for n in sorted(entries)}}
    cfp.json_atomic(path, cfp.attach_json(rec, ident))
    return rec


def verify_manifest(path, root=None):
    """Compare the recorded manifest against the live data. Returns
    (ok, reason); a missing manifest is reported as such (the caller
    decides whether first-run bootstrap writes it)."""
    if not os.path.isfile(path):
        return False, "manifest absent"
    rec = json.load(open(path))
    for name, entry in rec.get("datasets", {}).items():
        live = dataset_digest(name, root)
        if live != entry.get("digest"):
            return False, (f"dataset {name} changed: manifest "
                           f"{entry.get('digest')}, live {live}")
    return True, ""
