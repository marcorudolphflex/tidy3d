"""Central configuration manager implementation."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, TypeAlias, get_args, get_origin

from pydantic import BaseModel

from tidy3d.log import log

from .layers import ConfigLayers
from .loader import ConfigLoader, deep_diff, deep_merge, load_environment_overrides
from .profiles import BUILTIN_PROFILES
from .registry import attach_manager, get_handlers, get_sections

Tree: TypeAlias = dict[str, Any]
PLUGINS_KEY = "plugins"
PLUGIN_PREFIX = f"{PLUGINS_KEY}."


def normalize_profile_name(name: str) -> str:
    """Return a canonical profile name for builtin profiles."""
    normalized = name.strip()
    lowered = normalized.lower()
    return lowered if lowered in BUILTIN_PROFILES else normalized


def _is_plugin_section(name: str) -> bool:
    return name.startswith(PLUGIN_PREFIX)


def _plugin_name(name: str) -> str:
    return name.split(".", 1)[1]


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
        return SectionAccessor(self._manager, f"{PLUGINS_KEY}.{plugin}")

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
    """High-level orchestrator for tidy3d configuration.

    Responsibilities:
      - Hold the active profile.
      - Build/cache Pydantic models from the composed tree.
      - Apply section handlers on changes.
      - Persist diffs via ConfigLoader.
      - Expose convenient attribute accessors.
    """

    def __init__(
        self,
        profile: Optional[str] = None,
        config_dir: Optional[os.PathLike[str]] = None,
    ):
        self._loader = ConfigLoader(None if config_dir is None else Path(config_dir))
        self._layers = ConfigLayers()
        self._plugin_models: dict[str, BaseModel] = {}
        self._section_models: dict[str, BaseModel] = {}
        self._profile = self._resolve_initial_profile(profile)

        attach_manager(self)
        self._reload()            # load files + env into layers
        self._apply_handlers()    # apply handlers on initial models

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def profile(self) -> str:
        return self._profile

    @property
    def config_dir(self) -> Path:
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
        serialized = {k: _serialize_value(v) for k, v in updates.items()}
        snapshot = self._layers.runtime_snapshot(self._profile)
        try:
            self._layers.update_runtime(self._profile, name, serialized)
            self._reload()
        except Exception:
            self._layers.restore_runtime(self._profile, snapshot)
            self._reload()
            raise
        self._apply_handlers(section=name)

    def switch_profile(self, profile: str) -> None:
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
            self._loader.load_user_profile(self._profile)
            self._loader.save_profile(self._profile, base_without_env)

        # refresh cached layers after saving
        self._layers.set_base(self._loader.load_base())
        self._layers.set_profile(self._profile, self._loader.load_user_profile(self._profile))
        self._reload()

    def reset_to_defaults(self, *, include_profiles: bool = True) -> None:
        """Reset configuration files to their default annotated state."""
        self._layers.reset_runtime()
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
        user_profiles = sorted(p.stem for p in profiles_dir.glob("*.toml")) if profiles_dir.exists() else []
        built_in = sorted(BUILTIN_PROFILES.keys())
        return {"built_in": built_in, "user": user_profiles}

    def preview_profile(self, profile: str) -> dict[str, Any]:
        # Preview is “static”: builtin + base + user profile (no runtime/env)
        return deepcopy(
            deep_merge(
                self._loader.get_builtin_profile(profile),
                self._loader.load_base(),
                self._loader.load_user_profile(profile),
            )
        )

    def get_section(self, name: str) -> BaseModel:
        model = self._get_model(name)
        if model is None:
            raise AttributeError(f"Section '{name}' is not available")
        return model

    def as_dict(self, include_env: bool = True) -> Tree:
        return deepcopy(
            self._layers.compose(self._profile, include_env=include_env)
        )

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
        self._layers.set_builtin(self._profile, self._loader.get_builtin_profile(self._profile))
        self._layers.set_base(self._loader.load_base())
        self._layers.set_profile(self._profile, self._loader.load_user_profile(self._profile))

        self._layers.set_env(load_environment_overrides())
        self._build_models()

    def _build_models(self) -> None:
        sections = get_sections()
        self._section_models.clear()
        self._plugin_models.clear()

        tree = self._layers.compose(self._profile, include_env=True)

        for name, schema in sections.items():
            if name == PLUGINS_KEY:
                # container key; no direct model
                continue

            if _is_plugin_section(name):
                plugin = _plugin_name(name)
                plugin_data = _deep_get(tree, (PLUGINS_KEY, plugin)) or {}
                self._plugin_models[plugin] = self._build_one_model(schema, plugin_data, f"plugin '{plugin}'")
            else:
                section_data = tree.get(name, {}) or {}
                self._section_models[name] = self._build_one_model(schema, section_data, f"section '{name}'")

    def _build_one_model(self, schema: type[BaseModel], data: dict[str, Any], label: str) -> BaseModel:
        try:
            return schema(**data)
        except Exception as exc:  # pragma: no cover - validation guard
            log.error(f"Failed to load configuration for {label}: {exc}")
            raise

    def _get_model(self, name: str) -> Optional[BaseModel]:
        if _is_plugin_section(name):
            return self._plugin_models.get(_plugin_name(name))
        return self._section_models.get(name)

    def _apply_handlers(self, section: Optional[str] = None) -> None:
        handlers = get_handlers()
        targets = [section] if section else list(handlers.keys())
        for target in targets:
            handler = handlers.get(target)
            if handler is None:
                continue
            model = self._get_model(target)
            if model is None:
                continue
            try:
                handler(model)
            except Exception as exc:  # keep handlers non-fatal
                log.error(f"Failed to apply configuration handler for '{target}': {exc}")

    def _compose_without_env(self) -> Tree:
        return self._layers.compose(self._profile, include_env=False)

    def _default_tree(self) -> Tree:
        defaults: Tree = {}
        for name, schema in get_sections().items():
            if name == PLUGINS_KEY:
                defaults.setdefault(PLUGINS_KEY, {})
                continue
            if _is_plugin_section(name):
                plugin = _plugin_name(name)
                defaults.setdefault(PLUGINS_KEY, {})[plugin] = _model_dict(schema())
            else:
                defaults[name] = _model_dict(schema())
        return defaults

    def _filter_persisted(self, tree: Tree) -> Tree:
        sections = get_sections()
        filtered: Tree = {}
        plugins_source = tree.get(PLUGINS_KEY, {}) if isinstance(tree, dict) else {}
        plugin_filtered: Tree = {}

        for name, schema in sections.items():
            if name == PLUGINS_KEY:
                continue

            if _is_plugin_section(name):
                plugin_name = _plugin_name(name)
                plugin_data = plugins_source.get(plugin_name, {})
                if isinstance(plugin_data, dict):
                    persisted = _extract_persisted(schema, plugin_data)
                    if persisted:
                        plugin_filtered[plugin_name] = persisted
                continue

            section_data = tree.get(name, {})
            if isinstance(section_data, dict):
                persisted = _extract_persisted(schema, section_data)
                if persisted:
                    filtered[name] = persisted

        if plugin_filtered:
            filtered[PLUGINS_KEY] = plugin_filtered
        return filtered

    # ------------------------------------------------------------------
    # Python protocol hooks
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name in self._section_models:
            return SectionAccessor(self, name)
        if name == PLUGINS_KEY:
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
def _deep_get(tree: Tree, path: Iterable[str]) -> Optional[dict[str, Any]]:
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
