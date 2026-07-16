"""Configuration fingerprints for ensemble/surrogate caches.

The reproduce pipelines cache expensive solver ensembles as .npz files and
reuse them across runs. A cache is only valid for the configuration that built
it (ensemble size, grid, iteration caps, observation sigma, seed, ...), and
quick-mode runs deliberately shrink that configuration, so a full run must
never silently accept a quick-mode cache. Filenames stay stable (downstream
stages key on them); identity lives INSIDE the file:

  - fingerprint(config): stable short hash of the scientifically relevant
    configuration (canonical JSON, sorted keys, numpy types normalised).
  - attach(arrays, config): the arrays plus the fingerprint and the full
    config JSON, ready for np.savez.
  - check(loaded, config): classify a loaded cache against the expected
    configuration: "match" (safe to reuse), "mismatch" (REFUSE and
    regenerate; the reason names both configs), or "legacy" (pre-fingerprint
    cache). No exceptions; callers branch on the classification, matching the
    house failure-handling convention.

Legacy policy: a legacy cache is reused ONLY under the explicit opt-in
environment variable QBTM_ALLOW_LEGACY_CACHE=1 (legacy_reuse_allowed());
the default is refuse-and-regenerate. Blind reuse of unstamped caches is
exactly the failure mode fingerprinting exists to close, so acceptance has
to be a deliberate, per-run decision by whoever knows the cache's provenance.

Code provenance versus physics identity: attach() also records the git
revision that wrote the cache (code_rev()). The revision is PROVENANCE, not
identity (any commit, including one that never touches the solver, would
otherwise invalidate every ensemble). What IS identity is the producing
model: every cache configuration carries a "physics" schema token (for
example "channel-rans-v3") that the owning reproduce script bumps exactly
when the physics of the producing model changes, so caches built by older
solver physics can never be reused automatically on configuration match
alone. Regeneration decisions are therefore config-plus-physics driven, with
the git revision surfaced for the reader and the --regen flags as the manual
override.
"""
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np

SCHEMA_VERSION = 1

FINGERPRINT_KEY = "cache_fingerprint"
CONFIG_KEY = "cache_config_json"
CODE_REV_KEY = "cache_code_rev"


def code_rev():
    """Short git revision of the working tree that writes a cache.

    Provenance metadata only (see module docstring); "unknown" when git or a
    repository is unavailable, never an error.
    """
    if shutil.which("git") is None:
        return "unknown"
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.abspath(__file__)))
    if r.returncode != 0:
        return "unknown"
    return r.stdout.strip()


def legacy_reuse_allowed():
    """The explicit opt-in for reusing pre-fingerprint caches."""
    return os.environ.get("QBTM_ALLOW_LEGACY_CACHE", "") == "1"


def _canonical(obj):
    """Recursively normalise numpy scalars/arrays so JSON is deterministic."""
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _canonical(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def config_json(config):
    """Canonical JSON for a configuration dict (schema version included)."""
    payload = _canonical(config)
    payload["_schema"] = SCHEMA_VERSION
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def fingerprint(config):
    """Short stable hash of the scientifically relevant configuration."""
    return hashlib.sha256(config_json(config).encode()).hexdigest()[:16]


def attach(arrays, config):
    """Arrays plus identity keys, ready for np.savez(path, **attach(...))."""
    out = dict(arrays)
    out[FINGERPRINT_KEY] = np.array(fingerprint(config))
    out[CONFIG_KEY] = np.array(config_json(config))
    out[CODE_REV_KEY] = np.array(code_rev())
    return out


def check(loaded, config):
    """Classify a loaded cache dict against the expected configuration.

    Returns (status, reason): status is "match", "mismatch", or "legacy".
    "legacy" means the cache predates fingerprinting; it is reused only under
    the QBTM_ALLOW_LEGACY_CACHE=1 opt-in (legacy_reuse_allowed()), refused
    otherwise. A "match" whose stored code revision differs from the current
    one is still a match (provenance, not identity), with the drift named in
    the reason so the caller can print it.
    """
    if FINGERPRINT_KEY not in loaded:
        return ("legacy", "cache carries no fingerprint (built before cache "
                "identity); set QBTM_ALLOW_LEGACY_CACHE=1 to reuse it")
    stored = str(np.asarray(loaded[FINGERPRINT_KEY])[()])
    expected = fingerprint(config)
    if stored == expected:
        reason = ""
        if CODE_REV_KEY in loaded:
            rev_stored = str(np.asarray(loaded[CODE_REV_KEY])[()])
            rev_now = code_rev()
            if rev_stored != rev_now:
                reason = (f"built at code revision {rev_stored}, current "
                          f"{rev_now} (config identical)")
        return "match", reason
    stored_cfg = "<missing>"
    if CONFIG_KEY in loaded:
        stored_cfg = str(np.asarray(loaded[CONFIG_KEY])[()])
    return ("mismatch",
            f"cache built with a different configuration: stored {stored_cfg}, "
            f"expected {config_json(config)}")
