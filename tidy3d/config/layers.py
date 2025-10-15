"""Layer management utilities for the Tidy3D configuration system."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from .loader import deep_merge


class ConfigLayers:
    """Maintain layered configuration data per profile.
    Layers (lowest → highest):
      builtin(profile)  +  base  +  profile(profile)  +  runtime(profile)  [+ env]
    """

    def __init__(self) -> None:
        self._base: dict[str, Any] = {}
        self._builtin: dict[str, dict[str, Any]] = {}
        self._profiles: dict[str, dict[str, Any]] = {}
        self._runtime: dict[str, dict[str, Any]] = defaultdict(dict)
        self._env: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Layer setters (store defensive copies)
    # ------------------------------------------------------------------
    def set_base(self, data: dict[str, Any]) -> None:
        self._base = deepcopy(data)

    def set_builtin(self, profile: str, data: dict[str, Any]) -> None:
        self._builtin[profile] = deepcopy(data)

    def set_profile(self, profile: str, data: dict[str, Any]) -> None:
        self._profiles[profile] = deepcopy(data)

    def set_env(self, data: dict[str, Any]) -> None:
        self._env = deepcopy(data)

    # ------------------------------------------------------------------
    # Layer getters (defensive copies)
    # ------------------------------------------------------------------
    def base(self) -> dict[str, Any]:
        return deepcopy(self._base)

    def builtin(self, profile: str) -> dict[str, Any]:
        return deepcopy(self._builtin.get(profile, {}))

    def profile(self, profile: str) -> dict[str, Any]:
        return deepcopy(self._profiles.get(profile, {}))

    def env(self) -> dict[str, Any]:
        return deepcopy(self._env)

    # ------------------------------------------------------------------
    # Runtime management
    # ------------------------------------------------------------------
    def runtime_snapshot(self, profile: str) -> dict[str, Any]:
        return deepcopy(self._runtime.get(profile, {}))

    def restore_runtime(self, profile: str, snapshot: dict[str, Any]) -> None:
        self._runtime[profile] = snapshot

    def reset_runtime(self) -> None:
        self._runtime = defaultdict(dict)

    def update_runtime(self, profile: str, section: str, updates: dict[str, Any]) -> None:
        """Shallow-update a section under runtime(profile)."""
        if not updates:
            return
        leaf = self._ensure_path(self._runtime.setdefault(profile, {}), section)
        if not isinstance(leaf, dict):
            # If the leaf was a non-dict before, replace it with a dict to keep shape consistent.
            parent = self._get_parent(self._runtime[profile], section)
            parent[section.split(".")[-1]] = leaf = {}
        leaf.update(updates)

    # ------------------------------------------------------------------
    # Composition helpers
    # ------------------------------------------------------------------
    def raw(self, profile: str) -> dict[str, Any]:
        """Static merged config without runtime or environment overrides."""
        return deep_merge(
            self._builtin.get(profile, {}),
            self._base,
            self._profiles.get(profile, {}),
        )

    def compose(self, profile: str, *, include_env: bool = True) -> dict[str, Any]:
        """Fully composed config for a profile: raw + runtime (+ env)."""
        merged = deep_merge(self.raw(profile), self._runtime.get(profile, {}))
        return deep_merge(merged, self._env) if include_env else merged

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_path(root: dict[str, Any], dotted: str) -> dict[str, Any] | Any:
        """Ensure intermediate dicts exist for a dotted path; return the leaf value."""
        node: dict[str, Any] = root
        parts = dotted.split(".")
        for seg in parts[:-1]:
            nxt = node.get(seg)
            if not isinstance(nxt, dict):
                nxt = {}
                node[seg] = nxt
            node = nxt
        leaf_key = parts[-1]
        if leaf_key not in node:
            node[leaf_key] = {}
        return node[leaf_key]

    @staticmethod
    def _get_parent(root: dict[str, Any], dotted: str) -> dict[str, Any]:
        """Return the parent dict of the dotted path."""
        node: dict[str, Any] = root
        for seg in dotted.split(".")[:-1]:
            node = node[seg]  # by contract of _ensure_path, these exist and are dicts
        return node


__all__ = ["ConfigLayers"]
