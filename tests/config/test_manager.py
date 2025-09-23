from __future__ import annotations

import numpy as np
import pytest

from tidy3d.config import Env, get_manager, reload_config


def test_default_web_settings(config_manager):
    web = config_manager.get_section("web")
    assert str(web.api_endpoint) == "https://tidy3d-api.simulation.cloud"
    assert str(web.website_endpoint) == "https://tidy3d.simulation.cloud"
    assert web.ssl_verify is True


def test_update_section_runtime_overlay(config_manager):
    config_manager.update_section("logging", level="DEBUG", suppression=False)
    logging_section = config_manager.get_section("logging")
    assert logging_section.level == "DEBUG"
    assert logging_section.suppression is False


def test_runtime_isolated_per_profile(config_manager):
    config_manager.update_section("web", timeout=45)
    config_manager.switch_profile("customer")
    assert config_manager.get_section("web").timeout == 120
    config_manager.switch_profile("default")
    assert config_manager.get_section("web").timeout == 45


def test_environment_variable_precedence(monkeypatch, config_manager):
    monkeypatch.setenv("TIDY3D_LOGGING__LEVEL", "WARNING")
    config_manager.switch_profile(config_manager.profile)
    config_manager.update_section("logging", level="DEBUG")
    logging_section = config_manager.get_section("logging")
    # env var should still take precedence
    assert logging_section.level == "WARNING"


@pytest.mark.parametrize("profile", ["dev", "uat"])
def test_builtin_profiles(profile, config_manager):
    config_manager.switch_profile(profile)
    web = config_manager.get_section("web")
    assert web.s3_region is not None


def test_uppercase_profile_normalization(monkeypatch):
    monkeypatch.setenv("TIDY3D_ENV", "DEV")
    try:
        reload_config()
        manager = get_manager()
        assert manager.profile == "dev"
        web = manager.get_section("web")
        assert str(web.api_endpoint) == "https://tidy3d-api.dev-simulation.cloud"
        assert Env.current.name == "dev"
    finally:
        reload_config(profile="default")


def test_autograd_defaults(config_manager):
    autograd = config_manager.get_section("autograd")
    assert autograd.min_wvl_fraction == pytest.approx(5e-2)
    assert autograd.points_per_wavelength == 10
    assert autograd.monitor_interval_poly == (1, 1, 1)
    assert autograd.quadrature_sample_fraction == pytest.approx(0.4)
    assert autograd.gauss_quadrature_order == 7
    assert autograd.edge_clip_tolerance == pytest.approx(1e-9)
    assert autograd.minimum_spacing_fraction == pytest.approx(1e-2)
    assert autograd.gradient_precision == "single"
    assert autograd.max_traced_structures == 500
    assert autograd.max_adjoint_per_fwd == 10


def test_autograd_update_section(config_manager):
    config_manager.update_section(
        "autograd",
        min_wvl_fraction=0.08,
        points_per_wavelength=12,
        solver_freq_chunk_size=3,
        gradient_precision="double",
        minimum_spacing_fraction=0.02,
        gauss_quadrature_order=5,
        edge_clip_tolerance=2e-9,
        max_traced_structures=600,
        max_adjoint_per_fwd=7,
    )
    autograd = config_manager.get_section("autograd")
    assert autograd.min_wvl_fraction == pytest.approx(0.08)
    assert autograd.points_per_wavelength == 12
    assert autograd.solver_freq_chunk_size == 3
    assert autograd.gauss_quadrature_order == 5
    assert autograd.edge_clip_tolerance == pytest.approx(2e-9)
    assert autograd.minimum_spacing_fraction == pytest.approx(0.02)
    assert autograd.gradient_precision == "double"
    assert autograd.max_traced_structures == 600
    assert autograd.max_adjoint_per_fwd == 7

    assert autograd.gradient_dtype_float is np.float64
    assert autograd.gradient_dtype_complex is np.complex128
