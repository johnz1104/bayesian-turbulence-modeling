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

Verification is CLOSED over the registered set, not over the record: an
earlier form compared only the entries the file happened to carry, so a
manifest written against absent data (every digest None) and a manifest
hand-reduced to an empty dataset block both verified successfully, and the
per-process memo could rule on a digest computed before the data changed
(the review's fail-open finding, reproduced before the fix). verify_manifest
now requires exactly the registered datasets, requires each of them to be
present on disk with content, validates the record's own identity block, and
drops the memo for the verified set so the ruling is always a fresh pass.
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
# the attached joint-flux training pool of the far-transfer dq legs is the
# committed channel matrix alone (the only attached source carrying the full
# flux vector), so a consumer of that pool binds this set and not FAR_SET
ATTACHED_DQ_SET = ("gv_channel",)

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


def forget(names, root=None):
    """Drop the memoized digests of a dataset set, so the next lookup is a
    fresh content pass. The memo exists because identity construction asks
    for the same digests many times inside one process; a VERIFICATION must
    never rule on a value cached before the data moved (a digest taken
    while a dataset was absent memoizes None, and a digest taken before an
    in-place edit memoizes the old content), so the gate drops it first."""
    for n in names:
        _MEMO.pop(os.path.abspath(_dataset_dir(n, root)), None)


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


def manifest_ident(entries):
    """Identity configuration of a manifest record: the recorded digest map
    itself. Rebuilt from the loaded entries at verification, so removing,
    adding or editing an entry no longer matches the stored fingerprint."""
    return {"kind": "dns-manifest",
            "manifest": {n: entries[n].get("digest") for n in sorted(entries)}}


def write_manifest(path, names=tuple(DATASETS), root=None):
    """The checksummed DNS manifest record (freeze and baseline
    provenance): per-dataset digest, file count and byte count, carrying
    the standard identity block. Digests are taken fresh (the memo is
    dropped first) so a record can never be written from a digest cached
    before the data moved."""
    forget(names, root)
    entries = {}
    for n in names:
        n_files, n_bytes = _stats(n, root)
        entries[n] = {"digest": dataset_digest(n, root),
                      "subdir": DATASETS[n],
                      "n_files": n_files, "bytes": n_bytes}
    rec = {"datasets": entries}
    cfp.json_atomic(path, cfp.attach_json(rec, manifest_ident(entries)))
    return rec


TOKEN_FIELDS = ("manifest", "digests", "writer_pid")


def token_ident(body):
    """Identity configuration of a run token: the whole recorded body."""
    return {"kind": "dns-run-token", "token": body}


def write_run_token(token_path, manifest_path, required=tuple(DATASETS)):
    """Record that THIS process has just fresh-verified the manifest.

    Verification hashes every registered dataset, which is the right cost
    once per run and the wrong cost once per worker: an orchestrated matrix
    spawns a worker per member, and each one re-reading the whole source
    corpus turns a correctness gate into hours of pure hashing. The parent
    verifies, writes this token, and passes it to the workers it spawns; a
    standalone worker gets no token and verifies in full. Only a caller that
    has just verified may write one."""
    rec = json.load(open(manifest_path))
    body = {"manifest": cfp.file_sha(manifest_path),
            "digests": {n: rec["datasets"][n]["digest"] for n in sorted(required)},
            "writer_pid": os.getpid()}
    cfp.json_atomic(token_path, cfp.attach_json(body, token_ident(body)))
    return body


def verify_run_token(token_path, manifest_path, required=tuple(DATASETS)):
    """The worker-side gate: the token must be self-consistent and must name
    the manifest that is on disk right now, digest for digest. Cheap by
    construction (no dataset is re-read), and never a substitute for the
    parent's fresh pass, which is what actually bound the manifest to the
    live data. Returns (ok, reason)."""
    if not os.path.isfile(token_path):
        return False, "run token absent"
    tok = json.load(open(token_path))
    if not isinstance(tok, dict) or set(TOKEN_FIELDS) - set(tok):
        return False, "run token is missing fields"
    body = {k: tok[k] for k in TOKEN_FIELDS}
    status, why = cfp.check_json(tok, token_ident(body))
    if status != "match":
        return False, f"run token identity {status} ({why})"
    if not os.path.isfile(manifest_path):
        return False, "manifest absent"
    if body["manifest"] != cfp.file_sha(manifest_path):
        return False, ("the manifest changed since the run token was "
                       "written; re-verify")
    rec = json.load(open(manifest_path))
    entries = rec.get("datasets")
    if not isinstance(entries, dict):
        return False, "manifest carries no datasets block"
    live = {n: entries[n].get("digest") for n in entries}
    if body["digests"] != live:
        return False, "run token digests do not match the manifest record"
    if set(body["digests"]) != set(required):
        return False, ("run token does not cover the registered datasets "
                       f"{sorted(set(required) - set(body['digests']))}")
    return True, ""


def verify_manifest(path, root=None, required=tuple(DATASETS)):
    """Rule the recorded manifest against the live data, CLOSED over the
    required dataset set. Returns (ok, reason); a missing manifest is
    reported as such (the caller decides whether first-run bootstrap writes
    it).

    A formal run passes only when the record covers exactly `required`,
    carries an unmodified identity block, and every required dataset is a
    non-empty directory on disk whose fresh content digest equals the
    recorded one. `required` is narrowed only by hermetic tests, never by a
    production driver: a subset would reinstate the fail-open hole (an
    absent dataset would simply drop out of the comparison)."""
    if not os.path.isfile(path):
        return False, "manifest absent"
    rec = json.load(open(path))
    entries = rec.get("datasets")
    if not isinstance(entries, dict):
        return False, "manifest carries no datasets block"
    status, why = cfp.check_json(rec, manifest_ident(entries))
    if status != "match":
        return False, f"manifest record identity {status} ({why})"
    want, have = set(required), set(entries)
    if want - have:
        return False, (f"manifest does not cover the registered datasets "
                       f"{sorted(want - have)}")
    if have - want:
        return False, (f"manifest carries unrequired datasets "
                       f"{sorted(have - want)}")
    forget(required, root)
    for name in sorted(want):
        recorded = entries[name].get("digest")
        if recorded is None:
            return False, (f"dataset {name} was manifested with no digest "
                           f"(absent when the record was written)")
        if not os.path.isdir(_dataset_dir(name, root)):
            return False, f"dataset {name} directory absent"
        n_files, _ = _stats(name, root)
        if n_files == 0:
            return False, f"dataset {name} directory holds no files"
        live = dataset_digest(name, root)
        if live != recorded:
            return False, (f"dataset {name} changed: manifest "
                           f"{recorded}, live {live}")
    return True, ""
