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
    cache: reusable only because its provenance is known, and the caller must
    say so loudly). No exceptions; callers branch on the classification,
    matching the house failure-handling convention.
"""
import hashlib
import json

import numpy as np

SCHEMA_VERSION = 1

FINGERPRINT_KEY = "cache_fingerprint"
CONFIG_KEY = "cache_config_json"


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
    return out


def check(loaded, config):
    """Classify a loaded cache dict against the expected configuration.

    Returns (status, reason): status is "match", "mismatch", or "legacy".
    "legacy" means the cache predates fingerprinting; reuse is the caller's
    explicit, printed decision, never a silent default for new caches.
    """
    if FINGERPRINT_KEY not in loaded:
        return "legacy", "cache carries no fingerprint (built before cache identity)"
    stored = str(np.asarray(loaded[FINGERPRINT_KEY])[()])
    expected = fingerprint(config)
    if stored == expected:
        return "match", ""
    stored_cfg = "<missing>"
    if CONFIG_KEY in loaded:
        stored_cfg = str(np.asarray(loaded[CONFIG_KEY])[()])
    return ("mismatch",
            f"cache built with a different configuration: stored {stored_cfg}, "
            f"expected {config_json(config)}")
