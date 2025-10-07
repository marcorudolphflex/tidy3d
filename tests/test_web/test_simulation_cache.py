from __future__ import annotations

from pathlib import Path

import pytest

import tidy3d as td
from tidy3d import config
from tidy3d.web import Job, run_async
from tidy3d.web.api import webapi as web
from tidy3d.web.cache import (
    CACHE_ARTIFACT_NAME,
    get_cache,
    resolve_simulation_cache,
)

MOCK_TASK_ID = "task-xyz"
# --- Fake pipeline global maps / queue ---
TASK_TO_SIM: dict[str, td.Simulation] = {}  # task_id -> Simulation
PATH_TO_SIM: dict[str, td.Simulation] = {}  # artifact path -> Simulation
SIM_ORDER: list[td.Simulation] = []  # fallback queue when upload isn't called


def _reset_fake_maps():
    TASK_TO_SIM.clear()
    PATH_TO_SIM.clear()
    SIM_ORDER.clear()


class _FakeStubData:
    def __init__(self, simulation: td.Simulation):
        self.simulation = simulation


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
    """Patch postprocess to return stub data bound to the correct simulation."""
    calls = {"postprocess": 0}

    def _fake_postprocess(path: str, lazy: bool = False):
        calls["postprocess"] += 1
        p = Path(path)
        sim = PATH_TO_SIM.get(str(p))
        if sim is None:
            # Try to recover task_id from file payload written by _fake_download
            try:
                txt = p.read_text()
                if "payload:" in txt:
                    task_id = txt.split("payload:", 1)[1].strip()
                    sim = TASK_TO_SIM.get(task_id)
            except Exception:
                pass
        if sim is None:
            # Last-resort fallback (keeps tests from crashing even if mapping failed)
            sim = basic_simulation
        return _FakeStubData(sim)

    monkeypatch.setattr(web.Tidy3dStubData, "postprocess", staticmethod(_fake_postprocess))
    return calls


def _patch_run_pipeline(monkeypatch):
    """Patch upload, start, monitor, and download to avoid network calls and map sims."""
    counters = {"upload": 0, "start": 0, "monitor": 0, "download": 0}
    _reset_fake_maps()  # isolate between tests

    def _extract_simulation(kwargs):
        """Extract the first td.Simulation object from upload kwargs."""
        if "simulation" in kwargs and isinstance(kwargs["simulation"], td.Simulation):
            return kwargs["simulation"]
        if "simulations" in kwargs:
            sims = kwargs["simulations"]
            if isinstance(sims, dict):
                for sim in sims.values():
                    if isinstance(sim, td.Simulation):
                        return sim
            elif isinstance(sims, (list, tuple)):
                for sim in sims:
                    if isinstance(sim, td.Simulation):
                        return sim
        return None

    def _fake_upload(**kwargs):
        counters["upload"] += 1
        task_id = f"{MOCK_TASK_ID}{counters['upload']}"
        sim = _extract_simulation(kwargs)
        if sim is None and SIM_ORDER:
            # Upload wasn't given the sim (or async path differs) -> fallback
            sim = SIM_ORDER.pop(0)
        if sim is not None:
            TASK_TO_SIM[task_id] = sim
        return task_id

    def _fake_start(task_id, **kwargs):
        counters["start"] += 1

    def _fake_monitor(task_id, verbose=True):
        counters["monitor"] += 1

    def _fake_download(*, task_id, path, **kwargs):
        counters["download"] += 1
        # Ensure we have a simulation for this task id (even if upload wasn't called)
        sim = TASK_TO_SIM.get(task_id)
        if sim is None and SIM_ORDER:
            sim = SIM_ORDER.pop(0)
            TASK_TO_SIM[task_id] = sim
        Path(path).write_text(f"payload:{task_id}")
        if sim is not None:
            PATH_TO_SIM[str(Path(path))] = sim

    def _fake_status(self):
        return "success"

    monkeypatch.setattr(web, "upload", _fake_upload)
    monkeypatch.setattr(web, "start", _fake_start)
    monkeypatch.setattr(web, "monitor", _fake_monitor)
    monkeypatch.setattr(web, "download", _fake_download)
    monkeypatch.setattr(web, "estimate_cost", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(Job, "status", property(_fake_status))
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


def _test_run_cache_hit(monkeypatch, tmp_path, basic_simulation, fake_data):
    counters = _patch_run_pipeline(monkeypatch)
    out_path = tmp_path / "result.hdf5"
    get_cache().clear()

    data = web.run(basic_simulation, task_name="demo", path=str(out_path), use_cache=True)
    assert isinstance(data, _FakeStubData)
    assert counters == {"upload": 1, "start": 1, "monitor": 1, "download": 1}

    _reset_counters(counters)
    data2 = web.run(basic_simulation, task_name="demo", path=str(out_path), use_cache=True)
    assert isinstance(data2, _FakeStubData)
    assert counters == {"upload": 0, "start": 0, "monitor": 0, "download": 0}


def _test_run_cache_hit_async(monkeypatch, basic_simulation):
    counters = _patch_run_pipeline(monkeypatch)
    monkeypatch.setattr(config.simulation_cache, "max_entries", 128)
    monkeypatch.setattr(config.simulation_cache, "max_size_gb", 10)
    cache = resolve_simulation_cache(use_cache=True)
    cache.clear()

    _reset_counters(counters)
    sim2 = basic_simulation.updated_copy(shutoff=1e-4)
    sim3 = basic_simulation.updated_copy(shutoff=1e-3)
    SIM_ORDER[:] = [basic_simulation, sim2, sim3]

    data = run_async({"task1": basic_simulation, "task2": sim2}, use_cache=True)
    data_task1 = data["task1"]  # access to store in cache
    data_task2 = data["task2"]  # access to store in cache
    assert counters["download"] == 2
    assert isinstance(data_task1, _FakeStubData)
    assert isinstance(data_task2, _FakeStubData)
    assert len(cache) == 2

    _reset_counters(counters)
    run_async({"task1": basic_simulation, "task2": sim2}, use_cache=True)
    assert counters["download"] == 0
    assert isinstance(data_task1, _FakeStubData)
    assert len(cache) == 2

    _reset_counters(counters)
    data = run_async({"task1": basic_simulation, "task3": sim3}, use_cache=True)

    data_task1 = data["task1"]
    data_task2 = data["task3"]  # access to store in cache
    assert counters["download"] == 1  # sim3 is new
    assert isinstance(data_task1, _FakeStubData)
    assert isinstance(data_task2, _FakeStubData)
    assert len(cache) == 3


def _test_load_cache_hit(monkeypatch, tmp_path, basic_simulation, fake_data):
    get_cache().clear()
    counters = _patch_run_pipeline(monkeypatch)
    out_path = tmp_path / "load.hdf5"

    cache = get_cache()

    web.run(basic_simulation, task_name="demo", path=str(out_path), use_cache=True)
    assert counters["download"] == 1
    assert len(cache) == 1

    _reset_counters(counters)
    data = web.load(None, path=str(out_path), from_cache=True)
    assert isinstance(data, _FakeStubData)
    assert counters["download"] == 0  # served from cache
    assert len(cache) == 1  # still 1 item in cache


def _test_checksum_mismatch_triggers_refresh(monkeypatch, tmp_path, basic_simulation):
    out_path = tmp_path / "checksum.hdf5"
    get_cache().clear()

    web.run(basic_simulation, task_name="demo", path=str(out_path), use_cache=True)

    cache = get_cache()
    metadata = cache.list()[0]
    corrupted_path = cache.root / metadata["cache_key"] / CACHE_ARTIFACT_NAME
    corrupted_path.write_text("corrupted")

    cache._fetch(metadata["cache_key"])
    assert len(cache) == 0


def _test_cache_eviction_by_entries(monkeypatch, tmp_path_factory, basic_simulation):
    monkeypatch.setattr(config.simulation_cache, "max_entries", 1)
    cache = resolve_simulation_cache(use_cache=True)
    cache.clear()

    file1 = tmp_path_factory.mktemp("art1") / CACHE_ARTIFACT_NAME
    file1.write_text("a")
    cache.store_result(_FakeStubData(basic_simulation), MOCK_TASK_ID, str(file1), "FDTD")
    assert len(cache) == 1

    sim2 = basic_simulation.updated_copy(shutoff=1e-4)
    file2 = tmp_path_factory.mktemp("art2") / CACHE_ARTIFACT_NAME
    file2.write_text("b")
    cache.store_result(_FakeStubData(sim2), MOCK_TASK_ID, str(file2), "FDTD")

    entries = cache.list()
    assert len(entries) == 1
    assert entries[0]["simulation_hash"] == sim2._hash_self()


def _test_cache_eviction_by_size(monkeypatch, tmp_path_factory, basic_simulation):
    monkeypatch.setattr(config.simulation_cache, "max_size_gb", float(10_000 * 1e-9))
    cache = resolve_simulation_cache(use_cache=True)
    cache.clear()

    file1 = tmp_path_factory.mktemp("art1") / CACHE_ARTIFACT_NAME
    file1.write_text("a" * 8_000)
    cache.store_result(_FakeStubData(basic_simulation), MOCK_TASK_ID, str(file1), "FDTD")
    assert len(cache) == 1

    sim2 = basic_simulation.updated_copy(shutoff=1e-4)
    file2 = tmp_path_factory.mktemp("art2") / CACHE_ARTIFACT_NAME
    file2.write_text("b" * 8_000)
    cache.store_result(_FakeStubData(sim2), MOCK_TASK_ID, str(file2), "FDTD")

    entries = cache.list()
    print("len(entries)", len(entries))
    assert len(cache) == 1
    assert entries[0]["simulation_hash"] == sim2._hash_self()


def test_configure_cache_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(config.simulation_cache, "enabled", True)
    monkeypatch.setattr(config.simulation_cache, "directory", tmp_path)
    monkeypatch.setattr(config.simulation_cache, "max_size_gb", 1.23)
    monkeypatch.setattr(config.simulation_cache, "max_entries", 5)

    cfg = resolve_simulation_cache().config
    assert cfg.enabled is True
    assert cfg.directory == tmp_path
    assert cfg.max_size_gb == 1.23
    assert cfg.max_entries == 5


def test_env_var_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("TIDY3D_CACHE_ENABLED", "true")
    monkeypatch.setenv("TIDY3D_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TIDY3D_CACHE_MAX_SIZE_GB", "0.5")

    monkeypatch.setattr(config.simulation_cache, "max_entries", 5)
    monkeypatch.setenv("TIDY3D_CACHE_MAX_ENTRIES", "7")

    cfg = resolve_simulation_cache().config
    assert cfg.enabled is True
    assert cfg.directory == tmp_path
    assert cfg.max_size_gb == 0.5
    assert cfg.max_entries == 7


def test_cache_end_to_end(monkeypatch, tmp_path, tmp_path_factory, basic_simulation, fake_data):
    """Run all critical cache tests in sequence to ensure end-to-end stability."""
    _test_run_cache_hit(monkeypatch, tmp_path, basic_simulation, fake_data)
    _test_load_cache_hit(monkeypatch, tmp_path, basic_simulation, fake_data)
    _test_checksum_mismatch_triggers_refresh(monkeypatch, tmp_path, basic_simulation)
    _test_cache_eviction_by_entries(monkeypatch, tmp_path_factory, basic_simulation)
    _test_cache_eviction_by_size(monkeypatch, tmp_path_factory, basic_simulation)
    _test_run_cache_hit_async(monkeypatch, basic_simulation)