from __future__ import annotations

from pathlib import Path

from playground import make_sim_playground
from tidy3d.web.api import webapi as web
from tidy3d.web.cache import (
    SimulationCacheConfig,
    configure_cache,
    get_cache,
)


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def reset_cache():
    # cache_dir = tmp_dir / "cache"
    # if cache_dir.exists():
    #     shutil.rmtree(cache_dir)
    cfg = SimulationCacheConfig(
        enabled=True,
        # directory=cache_dir,
        max_size_gb=1.0,
        max_entries=10,
    )
    configure_cache(cfg)
    get_cache.cache_clear()  # <--- important!
    get_cache().clear()
    return


def disp_cache_entries():
    cache = get_cache()
    print(f"Cache has {len(cache.list())}")  # ,entries:\n{pprint.pformat(cache.list())}")


# -----------------------------------------------------------
# Manual test driver
# -----------------------------------------------------------
if __name__ == "__main__":
    tmp_dir = Path("debug_cache_tmp").resolve()
    # disp_cache_entries()
    # reset_cache()
    disp_cache_entries()

    sim = make_sim_playground()
    out_path = tmp_dir / "result.hdf5"
    out_path2 = tmp_dir / "result2.hdf5"

    # print("\n=== First run (should upload/download, then cache) ===")
    # data1 = web.run(sim, task_name="demo", path=str(out_path), use_cache=True)
    # print(f"Got data type: {type(data1)}")
    # disp_cache_entries()
    # exit()
    print("\n=== Second run (should hit cache, skip pipeline) ===")
    data2 = web.run(sim, task_name="demo", path=str(out_path), use_cache=True)
    print(f"Got data type: {type(data2)}")
    disp_cache_entries()
    # exit()
    print("\n=== Load by task id (should also hit cache) ===")
    # task_id = "fdve-dc37255d-77a5-4101-8964-97954fa3bde3"  # your monkeypatch or real task id
    task_id = "fdve-f56d0ebf-ec88-439e-bcbf-1316ab0ed43d"  # your monkeypatch or real task id
    data3 = web.load(task_id, path=str(out_path), use_cache=True)
    print(f"Got data type: {type(data3)}")
    disp_cache_entries()

    print("\n=== Manually corrupting artifact to trigger re-download ===")
    entries = get_cache().list()
    # if entries:
    #     key = entries[0]["cache_key"]
    #
    #     # get the actual cache directory from config and expand '~'
    #     cache_dir = get_cache_config().directory.expanduser()
    #
    #     artifact_path = cache_dir / key / "artifact.hdf5"
    #     artifact_path.parent.mkdir(parents=True, exist_ok=True)
    #
    #     # overwrite with junk bytes (binary-safe)
    #     artifact_path.write_bytes(b"corrupted")
    #     print(
    #         f"[debug] Corrupted artifact at {artifact_path} (exists={artifact_path.exists()}, size={artifact_path.stat().st_size} bytes)")
    #
    #     data4 = web.load(task_id, path=str(out_path), use_cache=True)
    #     print(f"After corruption -> Got data type: {type(data4)}")
    #     disp_cache_entries()
    #
    # print("\nDone. Inspect log output above to debug pipeline vs cache paths.")
