"""Shared helpers for applying and restoring environment variable overrides."""

from __future__ import annotations

import os
from typing import Mapping, MutableMapping, Optional


EnvironmentMapping = MutableMapping[str, str]
SnapshotMapping = Mapping[str, Optional[str]]


def _current_env(env: Optional[EnvironmentMapping] = None) -> EnvironmentMapping:
    return env if env is not None else os.environ  # type: ignore[return-value]


def restore_environment(snapshot: SnapshotMapping, *, env: Optional[EnvironmentMapping] = None) -> None:
    """Restore environment variables from a previous snapshot."""

    if not snapshot:
        return
    target = _current_env(env)
    for key, previous in snapshot.items():
        if previous is None:
            target.pop(key, None)
        else:
            target[key] = previous


def apply_environment(
    overrides: Mapping[str, str], *, env: Optional[EnvironmentMapping] = None
) -> dict[str, Optional[str]]:
    """Apply overrides and return the previous values for future restoration."""

    if not overrides:
        return {}

    target = _current_env(env)
    snapshot: dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        snapshot[key] = target.get(key)
        target[key] = value
    return snapshot


__all__ = ["apply_environment", "restore_environment"]
