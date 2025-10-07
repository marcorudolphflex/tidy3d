from __future__ import annotations

from tidy3d import config
from tidy3d.web.cache import resolve_simulation_cache

# config.simulation_cache.max_size_gb = float(10_000 * 1e-9)
# cache = resolve_simulation_cache(use_cache=True)
# print(cache.config)
tmp_path = "dfsd"

config.simulation_cache.enabled = True
config.simulation_cache.directory = tmp_path
config.simulation_cache.max_size_gb = 1.23
config.simulation_cache.max_entries = 5
cfg = resolve_simulation_cache().config
assert cfg.enabled is True
assert cfg.directory == tmp_path
assert cfg.max_size_gb == 1.23
assert cfg.max_entries == 5
