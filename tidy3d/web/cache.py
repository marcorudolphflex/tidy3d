"""Local simulation cache manager."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from tidy3d import config
from tidy3d.components.types.workflow import WorkflowDataType, WorkflowType
from tidy3d.log import log
from tidy3d.web.api.tidy3d_stub import Tidy3dStub
from tidy3d.web.core.constants import TaskId
from tidy3d.web.core.http_util import get_version as _get_protocol_version

DEFAULT_CACHE_RELATIVE_DIR = Path(".tidy3d") / "cache" / "simulations"
CACHE_ARTIFACT_NAME = "simulation_data.hdf5"
CACHE_METADATA_NAME = "metadata.json"

ENV_ENABLE = "TIDY3D_CACHE_ENABLED"
ENV_DIRECTORY = "TIDY3D_CACHE_DIR"
ENV_MAX_SIZE = "TIDY3D_CACHE_MAX_SIZE_GB"
ENV_MAX_ENTRIES = "TIDY3D_CACHE_MAX_ENTRIES"

TMP_PREFIX = "tidy3d-cache-"
TMP_BATCH_PREFIX = "tmp_batch"


_CONFIG_LOCK = threading.RLock()


@dataclass(frozen=True)
class SimulationCacheConfig:
    """Configuration for the simulation cache."""

    enabled: bool = False
    directory: Path = field(default_factory=lambda: Path.home() / DEFAULT_CACHE_RELATIVE_DIR)
    max_size_gb: float = 8.0
    max_entries: int = 32


def _coerce_bool(value: str) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _coerce_float(value: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: str) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    enabled_env = _coerce_bool(os.getenv(ENV_ENABLE))
    if enabled_env is not None:
        overrides["enabled"] = enabled_env

    directory_env = os.getenv(ENV_DIRECTORY)
    if directory_env:
        overrides["directory"] = directory_env

    size_env = _coerce_float(os.getenv(ENV_MAX_SIZE))
    if size_env is not None:
        overrides["max_size_gb"] = size_env

    entries_env = _coerce_int(os.getenv(ENV_MAX_ENTRIES))
    if entries_env is not None:
        overrides["max_entries"] = entries_env

    return overrides


def _load_effective_config() -> SimulationCacheConfig:
    """
    Build the initial, global cache config at import-time.

    Precedence for fields (lowest → highest):
      1) library defaults (disabled, ~/.tidy3d/cache/simulations, limits)
      2) persisted app config (config.simulation_cache_settings), if present
      3) environment overrides (TIDY3D_CACHE_*)

    Note: per-call `use_cache` is *not* applied here; that’s handled in
    resolve_simulation_cache(...), which can reconfigure the singleton later.
    """
    sim_cache_settings = config.simulation_cache

    cfg = SimulationCacheConfig(
        enabled=sim_cache_settings.enabled,
        directory=sim_cache_settings.directory,
        max_size_gb=sim_cache_settings.max_size_gb,
        max_entries=sim_cache_settings.max_entries,
    )

    env_overrides = _load_env_overrides()
    if env_overrides:
        allowed = {k: v for k, v in env_overrides.items() if v is not None}
        if allowed:
            cfg = replace(cfg, **allowed)

    if cfg.directory:
        cfg = replace(cfg, directory=Path(cfg.directory).expanduser().resolve())

    return cfg


_CACHE_CONFIG: SimulationCacheConfig = _load_effective_config()


def get_cache_config() -> SimulationCacheConfig:
    """Thread-safe snapshot copy of the active global cache configuration."""
    with _CONFIG_LOCK:
        return replace(_CACHE_CONFIG)


def configure_cache(new_config: SimulationCacheConfig) -> None:
    """Swap the active global config and reset the cache singleton."""
    global _CACHE_CONFIG
    with _CONFIG_LOCK:
        _CACHE_CONFIG = new_config
    get_cache.cache_clear()


@lru_cache
def get_cache() -> SimulationCache:
    """
    Return the singleton SimulationCache built from the *current* global config.

    This is automatically refreshed whenever `configure_cache(...)` is called,
    because that function clears this LRU entry.
    """
    cfg = get_cache_config()
    return SimulationCache(cfg)


def _merge_from_tidy3d_config() -> SimulationCacheConfig:
    """Overlay app-level persisted settings (if any) onto the current global config snapshot."""
    simulation_cache_settings = config.simulation_cache
    return SimulationCacheConfig(
        enabled=simulation_cache_settings.enabled,
        directory=simulation_cache_settings.directory,
        max_size_gb=simulation_cache_settings.max_size_gb,
        max_entries=simulation_cache_settings.max_entries,
    )


def _apply_overrides(
    cfg: SimulationCacheConfig, overrides: dict[str, Any]
) -> SimulationCacheConfig:
    """Apply dict-based overrides (enabled/directory/max_size_gb/max_entries)."""
    if not overrides:
        return cfg
    # Filter to fields that exist on the dataclass and are not None
    allowed = {k: v for k, v in overrides.items() if v is not None and hasattr(cfg, k)}
    return replace(cfg, **allowed) if allowed else cfg


def resolve_simulation_cache(use_cache: Optional[bool] = None) -> Optional[SimulationCache]:
    """
    Return a SimulationCache configured from:
      1) persisted config (directory/limits + default enabled),
      2) environment overrides (enabled + directory/limits),
      3) per-call 'use_cache' (enabled only, highest precedence).

    If effective config differs from the active global config, reconfigure the singleton.
    Returns None if final 'enabled' is False.
    """
    current = get_cache_config()
    desired = _load_effective_config()

    if use_cache is not None:
        if desired.directory != current.directory:
            get_cache().clear(hard=True)
        desired = replace(desired, enabled=use_cache)

    if desired != current:
        configure_cache(desired)

    if not desired.enabled:
        return None

    try:
        return get_cache()
    except Exception as err:
        log.debug("Simulation cache unavailable: %s", err)
        return None


@dataclass
class CacheEntry:
    """Internal representation of a cache entry."""

    key: str
    root: Path
    metadata: dict[str, Any]

    @property
    def path(self) -> Path:
        return self.root / self.key

    @property
    def artifact_path(self) -> Path:
        return self.path / CACHE_ARTIFACT_NAME

    @property
    def metadata_path(self) -> Path:
        return self.path / CACHE_METADATA_NAME

    def exists(self) -> bool:
        return self.path.exists() and self.artifact_path.exists() and self.metadata_path.exists()

    def verify(self) -> bool:
        if not self.exists():
            return False
        checksum = self.metadata.get("checksum")
        if not checksum:
            return False
        try:
            actual_checksum, file_size = _copy_and_hash(self.artifact_path, None)
        except FileNotFoundError:
            return False
        if checksum != actual_checksum:
            log.warning(
                "Simulation cache checksum mismatch for key '%s'. Removing stale entry.", self.key
            )
            return False
        if int(self.metadata.get("file_size", file_size)) != file_size:
            self.metadata["file_size"] = file_size
            _write_metadata(self.metadata_path, self.metadata)
        return True

    def materialize(self, target: Path) -> Path:
        """Copy cached artifact to ``target`` and return the resulting path."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.artifact_path, target)
        return target


class SimulationCache:
    """Manages storing and retrieving cached simulation artifacts."""

    def __init__(self, config: SimulationCacheConfig):
        self._config = config
        self._root = Path(config.directory).expanduser().resolve()
        self._lock = threading.RLock()
        if config.enabled:
            self._root.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> SimulationCacheConfig:
        return self._config

    @property
    def root(self) -> Path:
        return self._root

    def list(self) -> list[dict[str, Any]]:
        """Return metadata for all cache entries."""
        with self._lock:
            return [entry.metadata for entry in self._iter_entries()]

    def clear(self, hard=False) -> None:
        """Remove all cache contents."""
        with self._lock:
            if self._root.exists():
                try:
                    shutil.rmtree(self._root)
                    if not hard:
                        self._root.mkdir(parents=True, exist_ok=True)
                except (FileNotFoundError, OSError):
                    pass

    def _fetch(self, key: str) -> Optional[CacheEntry]:
        """Retrieve an entry by key, verifying checksum."""
        with self._lock:
            entry = self._load_entry(key)
            if not entry or not entry.exists():
                return None
            if not entry.verify():
                self._remove_entry(entry)
                return None
            self._touch(entry)
            return entry

    def fetch_by_task(self, task_id: str) -> Optional[CacheEntry]:
        """Retrieve an entry by task id."""
        with self._lock:
            for entry in self._iter_entries():
                metadata = entry.metadata
                task_ids = metadata.get("task_ids", [])
                if task_id in task_ids and entry.exists():
                    if not entry.verify():
                        self._remove_entry(entry)
                        return None
                    self._touch(entry)
                    return entry
        return None

    def __len__(self) -> int:
        """Return number of valid cache entries."""
        with self._lock:
            return sum(1 for _ in self._iter_entries())

    def _store(
        self, key: str, task_id: Optional[str], source_path: Path, metadata: dict[str, Any]
    ) -> Optional[CacheEntry]:
        """Store a new cache entry from ``source_path``.

        Parameters
        ----------
        key : str
            Cache key computed from simulation hash and runtime context.
        task_id : str, optional
            Server task id associated with this artifact.
        source_path : Path
            Location of the artifact to cache.
        metadata : dict[str, Any]
            Additional metadata to persist alongside artifact.

        Returns
        -------
        CacheEntry
            Representation of the stored cache entry.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Cannot cache missing artifact: {source_path}")
        os.makedirs(self._root, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=TMP_PREFIX, dir=self._root))
        tmp_artifact = tmp_dir / CACHE_ARTIFACT_NAME
        tmp_meta = tmp_dir / CACHE_METADATA_NAME
        os.makedirs(tmp_dir, exist_ok=True)

        checksum, file_size = _copy_and_hash(source_path, tmp_artifact)
        now_iso = _now()
        metadata = dict(metadata)
        metadata.setdefault("cache_key", key)
        metadata.setdefault("created_at", now_iso)
        metadata["last_used"] = now_iso
        metadata["checksum"] = checksum
        metadata["file_size"] = file_size
        if task_id:
            task_ids = list(metadata.get("task_ids", []))
            if task_id not in task_ids:
                task_ids.append(task_id)
            metadata["task_ids"] = task_ids

        _write_metadata(tmp_meta, metadata)
        try:
            with self._lock:
                self._root.mkdir(parents=True, exist_ok=True)
                self._ensure_limits(file_size)
                final_dir = self._root / key
                backup_dir: Optional[Path] = None

                try:
                    if final_dir.exists():
                        backup_dir = final_dir.with_name(
                            f"{final_dir.name}.bak.{_timestamp_suffix()}"
                        )
                        os.replace(final_dir, backup_dir)
                    # move tmp_dir into place
                    os.replace(tmp_dir, final_dir)
                except Exception:
                    # restore backup if needed
                    if backup_dir and backup_dir.exists():
                        os.replace(backup_dir, final_dir)
                    raise
                else:
                    entry = CacheEntry(key=key, root=self._root, metadata=metadata)
                    if backup_dir and backup_dir.exists():
                        shutil.rmtree(backup_dir, ignore_errors=True)
                    log.debug("Stored simulation cache entry '%s' (%d bytes).", key, file_size)
                    return entry
        finally:
            try:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except FileNotFoundError:
                pass

    def invalidate(self, key: str) -> None:
        with self._lock:
            entry = self._load_entry(key)
            if entry:
                self._remove_entry(entry)

    def _ensure_limits(self, incoming_size: int) -> None:
        max_entries = max(self._config.max_entries, 0)
        max_size_bytes = int(max(0.0, self._config.max_size_gb) * (1024**3))

        entries = list(self._iter_entries())
        if max_entries and len(entries) >= max_entries:
            self._evict(entries, keep=max_entries - 1)
            entries = list(self._iter_entries())

        if not max_size_bytes:
            return

        existing_size = sum(int(e.metadata.get("file_size", 0)) for e in entries)
        allowed_size = max(max_size_bytes - incoming_size, 0)
        if existing_size > allowed_size:
            self._evict_by_size(entries, existing_size, allowed_size)

    def _evict(self, entries: Iterable[CacheEntry], keep: int) -> None:
        sorted_entries = sorted(entries, key=lambda e: e.metadata.get("last_used", ""))
        to_remove = sorted_entries[: max(0, len(sorted_entries) - keep)]
        for entry in to_remove:
            self._remove_entry(entry)

    def _evict_by_size(
        self, entries: Iterable[CacheEntry], current_size: int, allowed_size: float
    ) -> None:
        if allowed_size < 0:
            allowed_size = 0
        sorted_entries = sorted(entries, key=lambda e: e.metadata.get("last_used", ""))
        reclaimed = 0
        for entry in sorted_entries:
            if current_size - reclaimed <= allowed_size:
                break
            size = int(entry.metadata.get("file_size", 0))
            self._remove_entry(entry)
            reclaimed += size
            log.info(f"Simulation cache evicted entry '{entry.key}' to reclaim {size} bytes.")

    def _iter_entries(self) -> Iterable[CacheEntry]:
        if not self._root.exists():
            return []
        entries: list[CacheEntry] = []
        for child in self._root.iterdir():
            if child.name.startswith(TMP_PREFIX) or child.name.startswith(TMP_BATCH_PREFIX):
                continue
            meta_path = child / CACHE_METADATA_NAME
            if not meta_path.exists():
                continue
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
            entries.append(CacheEntry(key=child.name, root=self._root, metadata=metadata))
        return entries

    def _load_entry(self, key: str) -> Optional[CacheEntry]:
        entry = CacheEntry(key=key, root=self._root, metadata={})
        if not entry.metadata_path.exists() or not entry.artifact_path.exists():
            return None
        try:
            metadata = json.loads(entry.metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        entry.metadata = metadata
        return entry

    def _touch(self, entry: CacheEntry) -> None:
        entry.metadata["last_used"] = _now()
        _write_metadata(entry.metadata_path, entry.metadata)

    def _remove_entry(self, entry: CacheEntry) -> None:
        if entry.path.exists():
            shutil.rmtree(entry.path, ignore_errors=True)

    def try_fetch(
        self,
        simulation: WorkflowType,
        verbose: bool = False,
    ) -> Optional[CacheEntry]:
        """
        Attempt to resolve and fetch a cached result entry for the given simulation context.
        On miss or any cache error, returns None (the caller should proceed with upload/run).

        Notes
        -----
        - Mirrors the exact cache key/context computation from `run`.
        - Safe to call regardless of `use_cache` value; will no-op if cache is disabled.
        """
        try:
            simulation_hash = simulation._hash_self()
            workflow_type = Tidy3dStub(simulation=simulation).get_type()

            versions = _get_protocol_version()

            cache_key = build_cache_key(
                simulation_hash=simulation_hash,
                workflow_type=workflow_type,
                version=versions,
            )

            entry = self._fetch(cache_key)
            if not entry:
                return None
                # self._store(key=cache_key, task_id=task_id, source_path=path, metadata={})
            if verbose:
                log.info(
                    "Simulation cache hit for workflow '%s'; using local results.", workflow_type
                )

            return entry
        except Exception as e:
            log.error("Failed to fetch cache results." + str(e))

    def store_result(
        self,
        stub_data: WorkflowDataType,
        task_id: TaskId,
        path: str,
        workflow_type: str,
    ) -> None:
        """
        After we have the data (postprocess done), store it in the cache using the
        canonical key (simulation hash + workflow type + environment + version).
        Also records the task_id mapping for legacy lookups.
        """
        try:
            simulation_obj = getattr(stub_data, "simulation", None)
            simulation_hash = simulation_obj._hash_self() if simulation_obj is not None else None
            if not simulation_hash:
                return

            version = _get_protocol_version()

            cache_key = build_cache_key(
                simulation_hash=simulation_hash,
                workflow_type=workflow_type,
                version=version,
            )

            metadata = build_entry_metadata(
                simulation_hash=simulation_hash,
                workflow_type=workflow_type,
                runtime_context={
                    "task_id": task_id,
                },
                version=version,
                extras={"path": str(Path(path))},
            )

            self._store(
                key=cache_key,
                task_id=task_id,  # keeps a reverse link for legacy fetch_by_task
                source_path=Path(path),
                metadata=metadata,
            )
        except Exception:
            log.error("Could not store cache entry.")


def _copy_and_hash(
    source: Path, dest: Optional[Path], existing_hash: Optional[str] = None
) -> tuple[str, int]:
    """Copy ``source`` to ``dest`` while computing SHA256 checksum.

    Parameters
    ----------
    source : Path
        Source file path.
    dest : Path or None
        Destination file path. If ``None``, no copy is performed.
    existing_hash : str, optional
        If provided alongside ``dest`` and ``dest`` already exists, skip copying when hashes match.

    Returns
    -------
    tuple[str, int]
        The hexadecimal digest and file size in bytes.
    """
    source = Path(source)
    if dest is not None:
        dest = Path(dest)
    sha256 = _Hasher()
    size = 0
    with source.open("rb") as src:
        if dest is None:
            while chunk := src.read(1024 * 1024):
                sha256.update(chunk)
                size += len(chunk)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
                    sha256.update(chunk)
                    size += len(chunk)
    return sha256.hexdigest(), size


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")


class _Hasher:
    def __init__(self):
        self._hasher = hashlib.sha256()

    def update(self, data: bytes) -> None:
        self._hasher.update(data)

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


def clear() -> None:
    """Remove all cache entries."""
    get_cache().clear()


def _canonicalize(value: Any) -> Any:
    """Convert value into a JSON-serializable object for hashing/metadata."""

    if isinstance(value, dict):
        return {
            str(k): _canonicalize(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, set):
        return sorted(_canonicalize(v) for v in value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def build_cache_key(
    *,
    simulation_hash: str,
    workflow_type: str,
    version: str,
) -> str:
    """Construct a deterministic cache key."""

    payload = {
        "simulation_hash": simulation_hash,
        "workflow_type": workflow_type,
        "versions": _canonicalize(version),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_entry_metadata(
    *,
    simulation_hash: str,
    workflow_type: str,
    runtime_context: dict[str, Any],
    version: str,
    extras: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create metadata dictionary for a cache entry."""

    metadata: dict[str, Any] = {
        "simulation_hash": simulation_hash,
        "workflow_type": workflow_type,
        "runtime_context": _canonicalize(runtime_context),
        "versions": _canonicalize(version),
        "task_ids": [],
    }
    if extras:
        metadata.update(_canonicalize(extras))
    return metadata
