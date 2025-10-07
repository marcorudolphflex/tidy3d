"""Sets the configuration of the script, can be changed with `td.config.config_name = new_val`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pydantic.v1 as pd

from .log import DEFAULT_LEVEL, LogLevel, set_log_suppression, set_logging_level
_DEFAULT_CACHE_DIR = Path.home() / ".tidy3d" / "cache" / "simulations"


class SimulationCacheSettings(pd.BaseModel):
    """Settings controlling the optional local simulation cache."""

    enabled: bool = pd.Field(
        False,
        description="Enable or disable the local simulation cache.",
    )
    directory: Path = pd.Field(
        _DEFAULT_CACHE_DIR,
        description="Directory where cached simulation artifacts are stored.",
    )
    max_size_gb: float = pd.Field(
        10.0,
        description="Maximum cache size in gigabytes. Set to 0 for no size limit.",
        ge=0.0,
    )
    max_entries: int = pd.Field(
        128,
        description="Maximum number of cache entries. Set to 0 for no limit.",
        ge=0,
    )

    @pd.validator("directory", pre=True, always=True)
    def _validate_directory(cls, value):
        return Path(value).expanduser()

class Tidy3dConfig(pd.BaseModel):
    """configuration of tidy3d"""

    class Config:
        """Config of the config."""

        arbitrary_types_allowed = False
        validate_all = True
        extra = "forbid"
        validate_assignment = True
        allow_population_by_field_name = True
        frozen = False

    logging_level: LogLevel = pd.Field(
        DEFAULT_LEVEL,
        title="Logging Level",
        description="The lowest level of logging output that will be displayed. "
        'Can be "DEBUG", "SUPPORT", "USER", INFO", "WARNING", "ERROR", or "CRITICAL". '
        'Note: "SUPPORT" and "USER" levels are only used in backend solver logging.',
    )

    log_suppression: bool = pd.Field(
        True,
        title="Log suppression",
        description="Enable or disable suppression of certain log messages when they are repeated "
        "for several elements.",
    )

    suppress_rf_license_warning: bool = pd.Field(
        False,
        title="Suppress RF License Warning",
        description="Enable or disable the RF/microwave license warning message when "
        "instantiating microwave components.",
    )

    use_local_subpixel: Optional[bool] = pd.Field(
        None,
        title="Whether to use local subpixel averaging. If 'None', local subpixel "
        "averaging will be used if 'tidy3d-extras' is installed and not used otherwise.",
    )

    simulation_cache: SimulationCacheSettings = pd.Field(
        default_factory=SimulationCacheSettings,
        title="Simulation Cache",
        description="Configuration for the optional local simulation cache.",
    )

    @pd.validator("logging_level", pre=True, always=True)
    def _set_logging_level(cls, val):
        """Set the logging level if logging_level is changed."""
        set_logging_level(val)
        return val

    @pd.validator("log_suppression", pre=True, always=True)
    def _set_log_suppression(cls, val):
        """Control log suppression when log_suppression is changed."""
        set_log_suppression(val)
        return val


# instance of the config that can be modified.
config = Tidy3dConfig()
