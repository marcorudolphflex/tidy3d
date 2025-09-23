"""Central configuration manager implementation."""

from __future__ import annotations

import os
import shutil
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, get_args, get_origin

from pydantic import BaseModel

from tidy3d.log import log

from .loader import (
    ConfigLoader,
    deep_diff,
    deep_merge,
    load_environment_overrides,
)
from .profiles import BUILTIN_PROFILES
from .registry import attach_manager, get_handlers, get_sections


def normalize_profile_name(name: str) -> str:
    """Return a canonical profile name for builtin profiles."""

    normalized = name.strip()
    lowered = normalized.lower()
    if lowered in BUILTIN_PROFILES:
        return lowered
    return normalized


class SectionAccessor:
    """Attribute proxy that routes assignments back through the manager."""

    def __init__(self, manager: ConfigManager, path: str):
        self._manager = manager
        self._path = path

    def __getattr__(self, name: str) -> Any:
        model = self._manager._get_model(self._path)
        if model is None:
            raise AttributeError(f"Section '{self._path}' is not available")
        return getattr(model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._manager.update_section(self._path, **{name: value})

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        model = self._manager._get_model(self._path)
        return f"SectionAccessor({self._path}={model!r})"

    def dict(self, *args, **kwargs):  # type: ignore[override]
        model = self._manager._get_model(self._path)
        if model is None:
            return {}
        return model.model_dump(*args, **kwargs)


class PluginsAccessor:
    """Provides access to registered plugin configurations."""

    def __init__(self, manager: ConfigManager):
        self._manager = manager

    def __getattr__(self, plugin: str) -> SectionAccessor:
        if plugin not in self._manager._plugin_models:
            raise AttributeError(f"Plugin '{plugin}' is not registered")
        return SectionAccessor(self._manager, f"plugins.{plugin}")

    def list(self) -> Iterable[str]:
        return sorted(self._manager._plugin_models.keys())


class ProfilesAccessor:
    """Read-only profile helper."""

    def __init__(self, manager: ConfigManager):
        self._manager = manager

    def list(self) -> dict[str, list[str]]:
        return self._manager.list_profiles()

    def __getattr__(self, profile: str) -> dict[str, Any]:
        return self._manager.preview_profile(profile)


class ConfigManager:
    """High-level orchestrator for tidy3d configuration."""

    def __init__(
        self,
        profile: Optional[str] = None,
        config_dir: Optional[os.PathLike[str]] = None,
    ):
        loader_path = None if config_dir is None else Path(config_dir)
        self._loader = ConfigLoader(loader_path)
        self._runtime_overrides: dict[str, dict[str, Any]] = defaultdict(dict)
        self._plugin_models: dict[str, BaseModel] = {}
        self._section_models: dict[str, BaseModel] = {}
        self._profile = self._resolve_initial_profile(profile)
        self._builtin_data: dict[str, Any] = {}
        self._base_data: dict[str, Any] = {}
        self._profile_data: dict[str, Any] = {}
        self._raw_tree: dict[str, Any] = {}
        self._effective_tree: dict[str, Any] = {}
        self._env_overrides: dict[str, Any] = load_environment_overrides()

        attach_manager(self)
        self._reload()
        self._apply_handlers()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def profile(self) -> str:
        return self._profile

    @property
    def config_dir(self):
        return self._loader.config_dir

    @property
    def plugins(self) -> PluginsAccessor:
        return PluginsAccessor(self)

    @property
    def profiles(self) -> ProfilesAccessor:
        return ProfilesAccessor(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_section(self, name: str, **updates: Any) -> None:
        if not updates:
            return
        segments = name.split(".")
        overrides = self._runtime_overrides[self._profile]
        previous = deepcopy(overrides)
        node = overrides
        for segment in segments[:-1]:
            node = node.setdefault(segment, {})
        section_key = segments[-1]
        section_payload = node.setdefault(section_key, {})
        for key, value in updates.items():
            section_payload[key] = _serialize_value(value)
        try:
            self._reload()
        except Exception:
            self._runtime_overrides[self._profile] = previous
            raise
        self._apply_handlers(section=name)

    def switch_profile(self, profile: str) -> None:
        if not profile:
            raise ValueError("Profile name cannot be empty")
        normalized = normalize_profile_name(profile)
        if not normalized:
            raise ValueError("Profile name cannot be empty")
        self._profile = normalized
        self._reload()
        self._apply_handlers()

    def save(self, include_defaults: bool = False) -> None:
        base_without_env = self._filter_persisted(self._compose_without_env())
        if include_defaults:
            defaults = self._filter_persisted(self._default_tree())
            base_without_env = deep_merge(defaults, base_without_env)

        if self._profile == "default":
            self._loader.save_base(base_without_env)
        else:
            baseline = self._filter_persisted(deep_merge(self._builtin_data, self._base_data))
            diff = deep_diff(baseline, base_without_env)
            self._loader.save_profile(self._profile, diff)
        # refresh cached base/profile data after saving
        self._base_data = self._loader.load_base()
        self._profile_data = self._loader.load_user_profile(self._profile)
        self._reload()

    def reset_to_defaults(self, *, include_profiles: bool = True) -> None:
        """Reset configuration files to their default annotated state."""

        self._runtime_overrides = defaultdict(dict)
        defaults = self._filter_persisted(self._default_tree())
        self._loader.save_base(defaults)

        if include_profiles:
            profiles_dir = self._loader.profile_path("_dummy").parent
            if profiles_dir.exists():
                shutil.rmtree(profiles_dir)
            loader_docs = getattr(self._loader, "_docs", {})
            for path in list(loader_docs.keys()):
                try:
                    path.relative_to(profiles_dir)
                except ValueError:
                    continue
                loader_docs.pop(path, None)
            self._profile = "default"

        self._reload()
        self._apply_handlers()

    def list_profiles(self) -> dict[str, list[str]]:
        profiles_dir = self._loader.config_dir / "profiles"
        user_profiles = []
        if profiles_dir.exists():
            for path in profiles_dir.glob("*.toml"):
                user_profiles.append(path.stem)
        built_in = sorted(name for name in BUILTIN_PROFILES.keys())
        return {"built_in": built_in, "user": sorted(user_profiles)}

    def preview_profile(self, profile: str) -> dict[str, Any]:
        builtin = self._loader.get_builtin_profile(profile)
        base = self._loader.load_base()
        overrides = self._loader.load_user_profile(profile)
        view = deep_merge(builtin, base, overrides)
        return deepcopy(view)

    def get_section(self, name: str) -> BaseModel:
        model = self._get_model(name)
        if model is None:
            raise AttributeError(f"Section '{name}' is not available")
        return model

    def as_dict(self, include_env: bool = True) -> dict[str, Any]:
        if include_env:
            return deepcopy(self._effective_tree)
        return self._compose_without_env()

    # ------------------------------------------------------------------
    # Registry callbacks
    # ------------------------------------------------------------------
    def on_section_registered(self, section: str) -> None:  # pragma: no cover - simple hook
        self._reload()
        self._apply_handlers(section=section)

    def on_handler_registered(self, section: str) -> None:  # pragma: no cover - simple hook
        self._apply_handlers(section=section)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_initial_profile(self, profile: Optional[str]) -> str:
        if profile:
            return normalize_profile_name(str(profile))

        candidate = (
            os.getenv("TIDY3D_CONFIG_PROFILE")
            or os.getenv("TIDY3D_PROFILE")
            or os.getenv("TIDY3D_ENV")
            or "default"
        )
        return normalize_profile_name(candidate)

    def _reload(self) -> None:
        self._env_overrides = load_environment_overrides()
        self._builtin_data = deepcopy(self._loader.get_builtin_profile(self._profile))
        self._base_data = deepcopy(self._loader.load_base())
        self._profile_data = deepcopy(self._loader.load_user_profile(self._profile))
        self._raw_tree = deep_merge(self._builtin_data, self._base_data, self._profile_data)

        runtime = deepcopy(self._runtime_overrides.get(self._profile, {}))
        effective = deep_merge(self._raw_tree, runtime, self._env_overrides)
        self._effective_tree = effective
        self._build_models()

    def _build_models(self) -> None:
        sections = get_sections()
        self._section_models.clear()
        self._plugin_models.clear()

        for name, schema in sections.items():
            if name.startswith("plugins."):
                plugin_name = name.split(".", 1)[1]
                plugin_data = _deep_get(self._effective_tree, ("plugins", plugin_name)) or {}
                try:
                    self._plugin_models[plugin_name] = schema(**plugin_data)
                except Exception as exc:  # pragma: no cover - validation guard
                    log.error(f"Failed to load configuration for plugin '{plugin_name}': {exc}")
                    raise
                continue
            if name == "plugins":
                continue
            section_data = self._effective_tree.get(name, {})
            try:
                self._section_models[name] = schema(**section_data)
            except Exception as exc:  # pragma: no cover
                log.error(f"Failed to load configuration for section '{name}': {exc}")
                raise

    def _get_model(self, name: str) -> Optional[BaseModel]:
        if name.startswith("plugins."):
            plugin = name.split(".", 1)[1]
            return self._plugin_models.get(plugin)
        return self._section_models.get(name)

    def _apply_handlers(self, section: Optional[str] = None) -> None:
        handlers = get_handlers()
        targets = [section] if section else handlers.keys()
        for target in targets:
            handler = handlers.get(target)
            if handler is None:
                continue
            model = self._get_model(target)
            if model is None:
                continue
            try:
                handler(model)
            except Exception as exc:
                log.error(f"Failed to apply configuration handler for '{target}': {exc}")

    def _compose_without_env(self) -> dict[str, Any]:
        runtime = self._runtime_overrides.get(self._profile, {})
        return deep_merge(self._raw_tree, runtime)

    def _default_tree(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for name, schema in get_sections().items():
            if name.startswith("plugins."):
                plugin = name.split(".", 1)[1]
                defaults.setdefault("plugins", {})[plugin] = _model_dict(schema())
            elif name == "plugins":
                defaults.setdefault("plugins", {})
            else:
                defaults[name] = _model_dict(schema())
        return defaults

    def _filter_persisted(self, tree: dict[str, Any]) -> dict[str, Any]:
        sections = get_sections()
        filtered: dict[str, Any] = {}
        plugins_source = tree.get("plugins", {})
        plugin_filtered: dict[str, Any] = {}

        for name, schema in sections.items():
            if name == "plugins":
                continue
            if name.startswith("plugins."):
                plugin_name = name.split(".", 1)[1]
                plugin_data = plugins_source.get(plugin_name, {})
                if not isinstance(plugin_data, dict):
                    continue
                persisted_plugin = _extract_persisted(schema, plugin_data)
                if persisted_plugin:
                    plugin_filtered[plugin_name] = persisted_plugin
                continue

            section_data = tree.get(name, {})
            if not isinstance(section_data, dict):
                continue
            persisted_section = _extract_persisted(schema, section_data)
            if persisted_section:
                filtered[name] = persisted_section

        if plugin_filtered:
            filtered["plugins"] = plugin_filtered
        return filtered

    # ------------------------------------------------------------------
    # Python protocol hooks
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name in self._section_models:
            return SectionAccessor(self, name)
        if name == "plugins":
            return self.plugins
        raise AttributeError(f"Config has no section '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name in self._section_models:
            if isinstance(value, BaseModel):
                payload = value.model_dump(exclude_unset=False)
            else:
                payload = value
            self.update_section(name, **payload)
            return
        object.__setattr__(self, name, value)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _deep_get(tree: dict[str, Any], path: Iterable[str]) -> Optional[dict[str, Any]]:
    node: Any = tree
    for segment in path:
        if not isinstance(node, dict):
            return None
        node = node.get(segment)
        if node is None:
            return None
    return node if isinstance(node, dict) else None


def _resolve_model_type(annotation: Any) -> Optional[type[BaseModel]]:
    """Return the first BaseModel subclass found in an annotation (if any)."""

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation

    origin = get_origin(annotation)
    if origin is None:
        return None

    for arg in get_args(annotation):
        nested = _resolve_model_type(arg)
        if nested is not None:
            return nested
    return None


def _serialize_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_unset=False)
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return value


def _model_dict(model: BaseModel) -> dict[str, Any]:
    data = model.model_dump(exclude_unset=False)
    for key, value in list(data.items()):
        if hasattr(value, "get_secret_value"):
            data[key] = value.get_secret_value()
    return data


def _extract_persisted(schema: type[BaseModel], data: dict[str, Any]) -> dict[str, Any]:
    persisted: dict[str, Any] = {}
    for field_name, field in schema.model_fields.items():
        schema_extra = field.json_schema_extra or {}
        annotation = field.annotation
        persist = bool(schema_extra.get("persist")) if isinstance(schema_extra, dict) else False
        if not persist:
            continue
        if field_name not in data:
            continue
        value = data[field_name]
        if value is None:
            persisted[field_name] = None
            continue

        nested_type = _resolve_model_type(annotation)
        if nested_type is not None:
            nested_source = value if isinstance(value, dict) else {}
            nested_persisted = _extract_persisted(nested_type, nested_source)
            if nested_persisted:
                persisted[field_name] = nested_persisted
            continue

        if hasattr(value, "get_secret_value"):
            persisted[field_name] = value.get_secret_value()
        else:
            persisted[field_name] = deepcopy(value)

    return persisted


__all__ = [
    "ConfigManager",
    "PluginsAccessor",
    "ProfilesAccessor",
    "SectionAccessor",
    "normalize_profile_name",
]
