from __future__ import annotations

import uuid
from pathlib import Path

import tidy3d as td
from tests.test_plugins.test_adjoint import use_emulated_run
from tests.utils import run_emulated
from tidy3d.web.api import webapi as web
from tidy3d.web.cache import (
    CACHE_ARTIFACT_NAME,
    SimulationCache,
    SimulationCacheConfig,
    get_cache,
)
import pytest


import os
import toml
import tempfile
from pathlib import Path

import pytest

from tidy3d.web.cache import (
    SimulationCacheConfig,
    configure_cache,
    get_cache_config,
    _apply_updates,
    _load_env_overrides,
    _load_cli_cache_settings,
)
from tidy3d.web import run_async


MOCK_TASK_ID = "task-xyz"



class _FakeStubData:
    def __init__(self, simulation: td.Simulation):
        self.simulation = simulation

@pytest.fixture(autouse=True)
def isolated_cache(tmp_path_factory):
    """Force every test into its own unique simulation cache directory."""
    # make per-test unique dir with uuid
    cache_dir = tmp_path_factory.mktemp(f"tidy3d_cache_{uuid.uuid4().hex}")

    # save original config (singleton)
    original = get_cache()._config if get_cache() else None

    # point global cache at fresh unique directory
    cfg = SimulationCacheConfig(
        enabled=True,
        directory=cache_dir,
        max_size_gb=1.0,
        max_entries=10,
    )
    configure_cache(cfg)
    get_cache().clear()

    yield cache_dir

    # restore previous config or clear
    if original is not None:
        configure_cache(original)
    get_cache().clear()

@pytest.fixture
def basic_simulation():
    pulse = td.GaussianPulse(freq0=200e12, fwidth=20e12)
    pt_dipole = td.PointDipole(source_time=pulse, polarization="Ex")
    return td.Simulation(
        size=(1, 1, 1),
        grid_spec=td.GridSpec.auto(wavelength=1.0),
        run_time=1e-12,
        sources=[pt_dipole],
    )


@pytest.fixture(autouse=True)
def fake_data(monkeypatch, basic_simulation):
    """Patch postprocess to return predictable stub data and track invocations."""
    calls = {"postprocess": 0}

    def _fake_postprocess(path: str):
        calls["postprocess"] += 1
        return _FakeStubData(basic_simulation)

    monkeypatch.setattr(web.Tidy3dStubData, "postprocess", staticmethod(_fake_postprocess))
    return calls


def _patch_run_pipeline(monkeypatch, tmp_path):
    """Patch upload, start, monitor, and download to avoid network calls."""
    counters = {"upload": 0, "start": 0, "monitor": 0, "download": 0}

    def _fake_upload(**kwargs):
        counters["upload"] += 1
        return MOCK_TASK_ID

    def _fake_start(task_id, **kwargs):
        counters["start"] += 1

    def _fake_monitor(task_id, verbose=True):
        counters["monitor"] += 1

    def _fake_download(*, task_id, path, **kwargs):
        counters["download"] += 1
        Path(path).write_text(f"payload:{task_id}")

    monkeypatch.setattr(web, "upload", _fake_upload)
    monkeypatch.setattr(web, "start", _fake_start)
    monkeypatch.setattr(web, "monitor", _fake_monitor)
    monkeypatch.setattr(web, "download", _fake_download)
    monkeypatch.setattr(
        web,
        "get_info",
        lambda task_id, verbose=True: type(
            "_Info", (), {"solverVersion": "solver-1", "taskType": "FDTD"}
        )(),
    )
    return counters


def _reset_counters(counters: dict[str, int]) -> None:
    for key in counters:
        counters[key] = 0


@pytest.mark.serial
def _test_run_cache_hit(monkeypatch, tmp_path, basic_simulation, fake_data):
    counters = _patch_run_pipeline(monkeypatch, tmp_path)
    out_path = tmp_path / "result.hdf5"
    get_cache().clear()

    data = web.run(basic_simulation, task_name="demo", path=str(out_path), use_cache=True)
    assert isinstance(data, _FakeStubData)
    assert counters == {"upload": 1, "start": 1, "monitor": 1, "download": 1}

    _reset_counters(counters)
    data2 = web.run(basic_simulation, task_name="demo", path=str(out_path), use_cache=True)
    assert isinstance(data2, _FakeStubData)
    assert counters == {"upload": 0, "start": 0, "monitor": 0, "download": 0}



@pytest.mark.serial
def test_run_cache_hit_async(use_emulated_run, monkeypatch, tmp_path, basic_simulation, fake_data):
    counters = _patch_run_pipeline(monkeypatch, tmp_path)
    out_path = tmp_path / "result.hdf5"
    get_cache().clear()

    data = web.run(basic_simulation, task_name="demo", path=str(out_path), use_cache=True)
    assert isinstance(data, _FakeStubData)
    assert counters == {"upload": 1, "start": 1, "monitor": 1, "download": 1}

    _reset_counters(counters)
    data = run_async({"task1": basic_simulation}, use_cache=True)


@pytest.mark.serial
@pytest.mark.xdist_group("serial")
def _test_load_cache_hit(monkeypatch, tmp_path, basic_simulation, fake_data):
    get_cache().clear()
    counters = _patch_run_pipeline(monkeypatch, tmp_path)
    out_path = tmp_path / "load.hdf5"

    web.run(basic_simulation, task_name="demo", path=str(out_path), use_cache=True)
    assert counters["download"] == 1

    _reset_counters(counters)
    data = web.load(MOCK_TASK_ID, path=str(out_path), use_cache=True)
    assert isinstance(data, _FakeStubData)
    assert counters["download"] == 0  # served from cache


@pytest.mark.serial
@pytest.mark.xdist_group("serial")
def _test_checksum_mismatch_triggers_refresh(monkeypatch, tmp_path, basic_simulation):
    counters = _patch_run_pipeline(monkeypatch, tmp_path)
    out_path = tmp_path / "checksum.hdf5"

    web.run(basic_simulation, task_name="demo", path=str(out_path), use_cache=True)

    cache = get_cache()
    metadata = cache.list()[0]
    corrupted_path = cache.root / metadata["cache_key"] / CACHE_ARTIFACT_NAME
    corrupted_path.write_text("corrupted")

    _reset_counters(counters)
    web.load(MOCK_TASK_ID, path=str(out_path), use_cache=True)
    assert counters["download"] == 1


@pytest.mark.serial
@pytest.mark.xdist_group("serial")
def _test_cache_eviction_by_entries(tmp_path_factory, basic_simulation):
    cache = SimulationCache(SimulationCacheConfig(enabled=True, max_size_gb=10.0, max_entries=1))

    file1 = tmp_path_factory.mktemp("art1") / CACHE_ARTIFACT_NAME
    file1.write_text("a" * 10)
    cache.store_result(_FakeStubData(basic_simulation), MOCK_TASK_ID, str(file1), "FDTD")
    assert len(cache) == 1

    sim2 = basic_simulation.updated_copy(normalize_index=0.1)
    file2 = tmp_path_factory.mktemp("art2") / CACHE_ARTIFACT_NAME
    file2.write_text("b" * 10)
    cache.store_result(_FakeStubData(sim2), MOCK_TASK_ID, str(file2), "FDTD")

    entries = cache.list()
    assert len(entries) == 1
    assert entries[0]["simulation_hash"] == sim2._hash_self()


@pytest.mark.serial
@pytest.mark.xdist_group("serial")
def _test_cache_eviction_by_size(tmp_path_factory, basic_simulation):
    cache = SimulationCache(SimulationCacheConfig(enabled=True, max_size_gb=1e-5, max_entries=10))

    file1 = tmp_path_factory.mktemp("art1") / CACHE_ARTIFACT_NAME
    file1.write_text("a" * 12_000)
    cache.store_result(_FakeStubData(basic_simulation), MOCK_TASK_ID, str(file1), "FDTD")
    assert len(cache) == 1

    sim2 = basic_simulation.updated_copy(normalize_index=0.2)
    file2 = tmp_path_factory.mktemp("art2") / CACHE_ARTIFACT_NAME
    file2.write_text("b" * 12_000)
    cache.store_result(_FakeStubData(sim2), MOCK_TASK_ID, str(file2), "FDTD")

    entries = cache.list()
    assert len(entries) == 1
    assert entries[0]["simulation_hash"] == sim2._hash_self()


@pytest.mark.xdist_group("serial")
def test_cache_end_to_end(monkeypatch, tmp_path, tmp_path_factory, basic_simulation, fake_data):
    """Run all critical cache tests in sequence to ensure end-to-end stability."""
    _test_run_cache_hit(monkeypatch, tmp_path, basic_simulation, fake_data)
    _test_load_cache_hit(monkeypatch, tmp_path, basic_simulation, fake_data)
    _test_checksum_mismatch_triggers_refresh(monkeypatch, tmp_path, basic_simulation)
    _test_cache_eviction_by_entries(tmp_path_factory, basic_simulation)
    _test_cache_eviction_by_size(tmp_path_factory, basic_simulation)

@pytest.mark.serial
def test_configure_cache_roundtrip(tmp_path):
    new_cfg = SimulationCacheConfig(enabled=True, directory=tmp_path, max_size_gb=1.23, max_entries=5)
    configure_cache(new_cfg)
    cfg = get_cache_config()
    assert cfg.enabled is True
    assert cfg.directory == tmp_path
    assert cfg.max_size_gb == 1.23
    assert cfg.max_entries == 5

@pytest.mark.serial
def test_env_var_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("TIDY3D_CACHE_ENABLED", "true")
    monkeypatch.setenv("TIDY3D_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TIDY3D_CACHE_MAX_SIZE_GB", "0.5")
    monkeypatch.setenv("TIDY3D_CACHE_MAX_ENTRIES", "7")

    overrides = _load_env_overrides()
    assert overrides == {
        "enabled": True,
        "directory": str(tmp_path),
        "max_size_gb": 0.5,
        "max_entries": 7,
    }

@pytest.mark.serial
def test_cli_config_overrides(tmp_path, monkeypatch):
    # Build fake toml config file
    cli_config_file = tmp_path / "config.toml"
    monkeypatch.setenv("TIDY3D_CLI_CONFIG", str(cli_config_file))  # if your code reads via constant adjust
    content = {
        "simulation_cache": {
            "enabled": True,
            "directory": str(tmp_path / "cli_dir"),
            "max_size_gb": 2.5,
            "max_entries": 99,
        }
    }
    cli_config_file.write_text(toml.dumps(content))

    # Patch constant so _load_cli_cache_settings sees our file
    from tidy3d.web import cache as cache_mod
    monkeypatch.setattr(cache_mod, "CLI_CONFIG_FILE", str(cli_config_file))

    settings = _load_cli_cache_settings()
    assert settings["enabled"] is True
    assert Path(settings["directory"]).name == "cli_dir"
    assert settings["max_size_gb"] == 2.5
    assert settings["max_entries"] == 99

@pytest.mark.serial
def test_apply_updates_invalid_values(tmp_path, caplog):
    base = SimulationCacheConfig()
    updates = {
        "enabled": "notbool",
        "directory": tmp_path,
        "max_size_gb": "-5",  # invalid
        "max_entries": "-10",  # invalid
        "irrelevant": 123,
    }
    cfg = _apply_updates(base, updates)
    # directory should be updated, invalid numbers ignored
    assert cfg.directory == tmp_path
    assert cfg.max_size_gb == base.max_size_gb
    assert cfg.max_entries == base.max_entries

@pytest.mark.serial
def test_effective_config_cli_then_env(monkeypatch, tmp_path):
    """CLI settings should apply first, then environment overrides take precedence."""

    # --- Step 1: fake CLI config ---
    cli_config_file = tmp_path / "config.toml"
    cli_settings = {
        "simulation_cache": {
            "enabled": False,  # will be overridden by env
            "directory": str(tmp_path / "cli_dir"),
            "max_size_gb": 2.5,
            "max_entries": 99,
        }
    }
    cli_config_file.write_text(toml.dumps(cli_settings))
    from tidy3d.web import cache as cache_mod
    monkeypatch.setattr(cache_mod, "CLI_CONFIG_FILE", str(cli_config_file))

    # --- Step 2: env vars override CLI ---
    env_dir = tmp_path / "env_dir"
    monkeypatch.setenv("TIDY3D_CACHE_ENABLED", "true")
    monkeypatch.setenv("TIDY3D_CACHE_DIR", str(env_dir))
    monkeypatch.setenv("TIDY3D_CACHE_MAX_SIZE_GB", "0.75")
    monkeypatch.setenv("TIDY3D_CACHE_MAX_ENTRIES", "7")

    # --- Step 3: load effective config ---
    from tidy3d.web.cache import _load_effective_config
    cfg = _load_effective_config()

    # --- Step 4: assertions ---
    # Env overrides should win over CLI
    assert cfg.enabled is True               # env overrides False
    assert cfg.directory == env_dir          # env overrides cli_dir
    assert cfg.max_size_gb == 0.75           # env overrides 2.5
    assert cfg.max_entries == 7              # env overrides 99