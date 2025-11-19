# Test autograd gradients for scaled spheres represented as both native Sphere and TriangleMesh
# geometries, comparing to finite differences for validation.
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import value_and_grad
from matplotlib import pyplot as plt

import tidy3d as td
import tidy3d.web as web
from tests.test_components.autograd.numerical.test_autograd_box_polyslab_numerical import (
    angled_overlap_deg,
)
from tidy3d import config
from tidy3d.components.autograd import get_static

config.local_cache.enabled = True
config.logging.level = "DEBUG"

WL_UM = 0.65
SPHERE_RADIUS_UM = 0.5 * WL_UM
SCALE_FACTORS = (0.2, 1.0, 5.0)
SCALE_AXES = (0, 1, 2)

FREQ0 = td.C_0 / WL_UM
INFINITE_DIM_SIZE_UM = 0.1
SRC_OFFSET = -2.5
MONITOR_OFFSET = 2.5
N_MAT = 2
PERMITTIVITY = N_MAT**2
TARGET_EDGE_LENGTH = WL_UM / 10
LOCAL_GRADIENT = True
VERBOSE = False
SHOW_PRINT_STATEMENTS = True
COMPARE_TO_FINITE_DIFFERENCE = True
SAVE_OUTPUT_DATA = True
ANGLE_OVERLAP_FD_ADJ_THRESH_DEG = 10.0
VERTEX_FD_STEP = 1e-3
FINITE_DIFF_STEP = 1e-3
td.config.adjoint.points_per_wavelength = 1

measure_flux_spec = False

freqs = td.C_0 / np.linspace(0.6, 0.7, 101)

if SHOW_PRINT_STATEMENTS:
    sys.stdout = sys.stderr


def make_base_simulation(
    radii: list[float], *, extra_structures: Sequence[td.Structure] | None = None
) -> tuple[td.Simulation, callable]:
    sim_size_3d = [
        2 * radii[0] + 2 * WL_UM,
        2 * radii[1] + 2 * WL_UM,
        (MONITOR_OFFSET - SRC_OFFSET) + 2 * WL_UM + 2 * radii[2],
    ]

    plane_wave = td.PlaneWave(
        center=(0.0, 0.0, SRC_OFFSET),
        size=(*sim_size_3d[:2], 0.0),
        source_time=td.GaussianPulse(freq0=FREQ0, fwidth=0.2 * FREQ0),
        direction="+",
    )

    flux_monitors = [
        td.FieldMonitor(
            center=(0.0, 0.0, MONITOR_OFFSET),
            size=(*sim_size_3d[:2], 0.0),
            freqs=FREQ0,
            name="field",
        )
    ]
    if measure_flux_spec:
        flux_monitors.append(
            td.FieldMonitor(
                center=(0.0, 0.0, MONITOR_OFFSET),
                size=(*sim_size_3d[:2], 0.0),
                freqs=freqs,
                name="field_spectrum",
            )
        )

    boundary_spec_3d = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    base_sim = td.Simulation(
        center=(0.0, 0.0, 0.0),
        size=tuple(sim_size_3d),
        monitors=flux_monitors,
        sources=[plane_wave],
        structures=list(extra_structures) if extra_structures else [],
        run_time=2e-11,
        boundary_spec=boundary_spec_3d,
        grid_spec=td.GridSpec(
            grid_x=td.UniformGrid(dl=WL_UM / 40),
            grid_y=td.UniformGrid(dl=WL_UM / 40),
            grid_z=td.UniformGrid(dl=WL_UM / 40),
        ),
    )

    def fom(sim_data):
        flux = sim_data["field"].flux.values  # shape: (N_freq,)
        # flux_spectrum = sim_data["field_spectrum"].flux.values  # shape: (N_freq,)
        # plt.plot(freqs, flux_spectrum)
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Flux")
        # plt.title("Flux Spectrum")
        # plt.grid(True)
        # plt.savefig("flux_spectrum.png")
        # exit()
        return flux

    return base_sim, fom


def make_overlap_cube_structure(radii: Sequence[float]) -> td.Structure:
    radii_arr = np.asarray(radii, dtype=float)
    size_x = float(radii_arr[0])
    size_y = float(2.0 * radii_arr[1])
    size_z = float(2.0 * radii_arr[2])
    cube_center = (size_x / 2.0, 0.0, 0.0)
    cube = td.Box(center=cube_center, size=(size_x, size_y, size_z))
    cube_medium = td.Medium(permittivity=PERMITTIVITY / 2.0)
    return td.Structure(geometry=cube, medium=cube_medium)


def run_parameter_simulations(
    parameter_sets: list[anp.ndarray],
    make_geometry,
    box_center,
    tag: str,
    base_sim: td.Simulation,
    fom,
    artifact_dir: Path,
    *,
    local_gradient: bool,
):
    simulation_dict = {}
    output_dir = artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, param_values in enumerate(parameter_sets):
        geometry = make_geometry(param_values, box_center)
        structure = td.Structure(
            geometry=geometry,
            medium=td.Medium(permittivity=PERMITTIVITY),
        )

        base_structures = list(getattr(base_sim, "structures", ()))
        structures = [structure, *base_structures]
        grid_spec = td.GridSpec.auto(min_steps_per_wvl=20, override_structures=[structure])
        sim = base_sim.updated_copy(structures=structures, grid_spec=grid_spec, validate=True)
        # import matplotlib.pyplot as plt
        # sim.plot(z=0)
        # plt.savefig("sphere_simx.png")
        # sim.plot(x=0)
        # plt.savefig("sphere_simy.png")
        # sim.plot(y=0)
        # plt.savefig("sphere_simz.png")
        # exit()
        task_name = f"{tag}_idx{idx}"
        simulation_dict[task_name] = sim

    if len(simulation_dict) == 1:
        key, sim = next(iter(simulation_dict.items()))
        result_path = output_dir / f"{sim._hash_self()}.hdf5"
        sim_data = web.run(
            sim,
            task_name=key,
            path=str(result_path),
            local_gradient=local_gradient,
            verbose=VERBOSE,
        )
        return fom(sim_data)

    sim_data_map = web.run_async(
        simulation_dict,
        path_dir=str(output_dir),
        local_gradient=local_gradient,
        verbose=VERBOSE,
    )

    return [fom(sim_data_map[key]) for key in simulation_dict]


def make_sphere_triangle_geometry(
    params: anp.ndarray,
    center: Sequence[float],
    scale_factor: float,
    scale_axis: int,
    target_edge_length: float = TARGET_EDGE_LENGTH,
) -> td.Geometry:
    radii = anp.array(params, dtype=float)
    mean_radius = radii.mean()
    unit_sphere_triangles = td.Sphere.unit_sphere_triangles(
        target_edge_length=target_edge_length / mean_radius
    )
    triangles = anp.array(unit_sphere_triangles)
    triangles = triangles * radii
    axis_selector = anp.equal(anp.arange(3), scale_axis)
    scale_vec = anp.where(axis_selector, scale_factor, 1.0)
    triangles = triangles * scale_vec
    center_arr = anp.array(center, dtype=float)
    triangles = triangles + center_arr
    mesh = td.TriangleMesh.from_triangles(triangles)
    return mesh


def make_mesh_sphere_from_radius(params: anp.ndarray, center: Sequence[float]) -> td.Geometry:
    radius = params[0]
    radii = anp.full(3, radius)
    return make_sphere_triangle_geometry(radii, center, scale_factor=1.0, scale_axis=0)


def make_native_sphere_geometry(params: anp.ndarray, center: Sequence[float]) -> td.Geometry:
    return td.Sphere(center=tuple(center), radius=params[0])


def _make_unique_vertices(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertex_map: dict[tuple[float, float, float], int] = {}
    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []
    for tri in triangles:
        face_idx: list[int] = []
        for vertex in tri:
            key = tuple(np.asarray(vertex, dtype=float))
            idx = vertex_map.get(key)
            if idx is None:
                idx = len(vertices)
                vertex_map[key] = idx
                vertices.append(np.asarray(vertex, dtype=float))
            face_idx.append(idx)
        faces.append(face_idx)

    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int32)


def finite_difference_params(objective, params: anp.ndarray, finite_diff_step) -> np.ndarray:
    step = np.full_like(np.asarray(params, dtype=float), finite_diff_step, dtype=float)
    perturbations = []
    valid_indices = []

    for idx in range(params.size):
        params_up = anp.array(params)
        params_down = anp.array(params)
        params_up = params_up.copy()
        params_down = params_down.copy()
        params_up[idx] += step[idx]
        params_down[idx] -= step[idx]
        perturbations.extend([params_up, params_down])
        valid_indices.append(idx)

    objectives = objective(anp.stack(perturbations))
    objectives = np.asarray(objectives, dtype=float)
    fd = np.zeros_like(np.asarray(params, dtype=float))
    for pair_idx, param_idx in enumerate(valid_indices):
        obj_up = objectives[2 * pair_idx]
        obj_down = objectives[2 * pair_idx + 1]
        fd[param_idx] = float((obj_up - obj_down) / (2.0 * step[param_idx]))

    return fd


def make_objective(
    make_geometry: Callable[[anp.ndarray, Sequence[float]], td.Geometry],
    center: Sequence[float],
    tag: str,
    base_sim: td.Simulation,
    fom: Callable,
    tmp_path,
    *,
    local_gradient: bool,
):
    def objective(parameters):
        return run_parameter_simulations(
            parameters,
            make_geometry,
            center,
            tag,
            base_sim,
            fom,
            tmp_path,
            local_gradient=local_gradient,
        )

    return objective


# @pytest.mark.numerical
@pytest.mark.parametrize("scale_factor", SCALE_FACTORS)
@pytest.mark.parametrize("scale_axis", SCALE_AXES)
# @pytest.mark.parametrize("scale_factor", (1,))
# @pytest.mark.parametrize("scale_axis", (0,))
@pytest.mark.parametrize("overlap_cube", (False, True))
def test_sphere_triangles_match_fd(
    scale_factor, scale_axis, overlap_cube, tmp_path, numerical_case_dir
):
    if scale_factor == 1 and scale_axis > 0:
        pytest.skip("Skipping duplicate test.")

    initial_params = [SPHERE_RADIUS_UM, SPHERE_RADIUS_UM, SPHERE_RADIUS_UM]
    params0 = anp.array(initial_params)

    radii = initial_params.copy()
    radii[scale_axis] *= scale_factor
    extra_structures = [make_overlap_cube_structure(radii)] if overlap_cube else []
    base_sim, fom = make_base_simulation(radii=radii, extra_structures=extra_structures)

    center = [0.0, 0.0, 0.0]

    part_make_geom = lambda p, c: make_sphere_triangle_geometry(p, c, scale_factor, scale_axis)

    triangle_objective = make_objective(
        part_make_geom,
        center,
        f"sphere_mesh_{scale_factor}_axis_{scale_axis}_cube_{overlap_cube}",
        base_sim,
        fom,
        tmp_path,
        local_gradient=LOCAL_GRADIENT,
    )
    triangle_objective_fd = make_objective(
        part_make_geom,
        center,
        f"sphere_mesh_fd_{scale_factor}_axis_{scale_axis}_cube_{overlap_cube}",
        base_sim,
        fom,
        tmp_path,
        local_gradient=False,
    )

    _, triangle_grad = value_and_grad(triangle_objective)([params0])
    assert triangle_grad is not None

    triangle_grad = np.squeeze(np.asarray(triangle_grad, dtype=float))

    fd_grad = finite_difference_params(triangle_objective_fd, params0, FINITE_DIFF_STEP)

    print("scale", scale_factor, "axis", scale_axis, "overlap_cube", overlap_cube)
    print("triangle_grad\t", triangle_grad.tolist())
    print("fd_grad\t\t", fd_grad.tolist())

    if COMPARE_TO_FINITE_DIFFERENCE:
        mesh_fd_overlap = angled_overlap_deg(triangle_grad, fd_grad)
        print(
            f"TriangleMesh FD vs. Adjoint angle overlap: {mesh_fd_overlap:.3f}° "
            f"(threshold = {ANGLE_OVERLAP_FD_ADJ_THRESH_DEG}°)"
        )
        assert mesh_fd_overlap < ANGLE_OVERLAP_FD_ADJ_THRESH_DEG, (
            f"FD–adjoint angle overlap too large: {mesh_fd_overlap:.3f}° "
            f"(threshold {ANGLE_OVERLAP_FD_ADJ_THRESH_DEG}°, "
        )

    if SAVE_OUTPUT_DATA:
        np.savez(
            numerical_case_dir
            / f"sphere_gradients_mesh_scale_{scale_factor}_axis_{scale_axis}_cube_{overlap_cube}.npz",
            triangle_grad=triangle_grad,
            fd_grad=fd_grad,
        )


# @pytest.mark.numerical
@pytest.mark.parametrize("radius_scale", (1, 2, 3))
@pytest.mark.parametrize("overlap_cube", (False,))
def test_native_sphere_match_fd(radius_scale, overlap_cube, tmp_path, numerical_case_dir):
    radius = SPHERE_RADIUS_UM * radius_scale
    params0 = anp.array([radius])

    radii = [radius, radius, radius]
    extra_structures = [make_overlap_cube_structure(radii)] if overlap_cube else []
    base_sim, fom = make_base_simulation(radii=radii, extra_structures=extra_structures)

    center = [0.0, 0.0, 0.0]

    native_objective = make_objective(
        make_native_sphere_geometry,
        center,
        f"native_sphere_scale_{radius_scale}_cube_{overlap_cube}",
        base_sim,
        fom,
        tmp_path,
        local_gradient=LOCAL_GRADIENT,
    )
    native_objective_fd = make_objective(
        make_native_sphere_geometry,
        center,
        f"native_sphere_fd_scale_{radius_scale}_cube_{overlap_cube}",
        base_sim,
        fom,
        tmp_path,
        local_gradient=False,
    )

    _, native_grad = value_and_grad(native_objective)([params0])
    native_grad = np.squeeze(np.asarray(native_grad, dtype=float))

    fd_grad = finite_difference_params(native_objective_fd, params0, FINITE_DIFF_STEP)

    print("native radius scale", radius_scale, "overlap_cube", overlap_cube)
    print("native_grad\t", native_grad.tolist())
    print("fd_grad\t\t", fd_grad.tolist())

    if COMPARE_TO_FINITE_DIFFERENCE:
        abs_diff = float(np.abs(native_grad - fd_grad))
        rel_err = abs_diff / max(np.abs(native_grad), np.abs(fd_grad), 1e-12)
        print(
            f"Native sphere FD vs. Adjoint absolute diff: {abs_diff:.3e}, "
            f"relative error: {float(get_static(rel_err)):.3e}"
        )
        assert rel_err < 1e-1, (
            f"Native sphere gradients mismatch: abs_diff={abs_diff:.3e}, "
            f"rel_err={float(get_static(rel_err)):.3e}, native_grad={native_grad.tolist()}, "
            f"fd_grad={fd_grad.tolist()}"
        )

    if SAVE_OUTPUT_DATA:
        np.savez(
            numerical_case_dir
            / f"native_sphere_gradients_scale_{radius_scale}_cube_{overlap_cube}.npz",
            native_grad=native_grad,
            fd_grad=fd_grad,
        )


@pytest.mark.numerical
@pytest.mark.parametrize("scale_factor", SCALE_FACTORS)
@pytest.mark.parametrize("scale_axis", SCALE_AXES)
@pytest.mark.parametrize("overlap_cube", (False, True))
def test_sphere_fd_step_sweep(tmp_path, scale_factor, scale_axis, overlap_cube, numerical_case_dir):
    initial_params = [SPHERE_RADIUS_UM, SPHERE_RADIUS_UM, SPHERE_RADIUS_UM]
    params0 = anp.array(initial_params)

    radii = initial_params.copy()
    radii[scale_axis] *= scale_factor
    extra_structures = [make_overlap_cube_structure(radii)] if overlap_cube else []
    base_sim, fom = make_base_simulation(radii=radii, extra_structures=extra_structures)

    center = [0.0, 0.0, 0.0]

    part_make_geom = lambda p, c: make_sphere_triangle_geometry(p, c, scale_factor, scale_axis)

    triangle_objective_fd = make_objective(
        part_make_geom,
        center,
        "sphere_mesh_fd_step_sweep",
        base_sim,
        fom,
        tmp_path,
        local_gradient=False,
    )

    steps = np.logspace(-6, -3, num=4)
    fd_grads = []
    for step in steps:
        grad = finite_difference_params(triangle_objective_fd, params0, step)
        fd_grads.append(grad)
        print(f"finite difference step {step:.1e}: gradient {grad.tolist()} cube={overlap_cube}")

    fd_grads = np.asarray(fd_grads, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, label in enumerate(["radius_x", "radius_y", "radius_z"]):
        ax.plot(steps, fd_grads[:, idx], marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Finite difference step (µm)")
    ax.set_ylabel("Gradient value")
    ax.set_title("Finite difference gradients vs. step size")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    fig_path = (
        numerical_case_dir
        / f"fd_step_sweep_scale_{scale_factor}_axis_{scale_axis}_cube_{overlap_cube}.png"
    )

    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    np.savez(
        numerical_case_dir
        / f"fd_step_sweep_scale_{scale_factor}_axis_{scale_axis}_cube_{overlap_cube}.npz",
        steps=steps,
        gradients=fd_grads,
    )


# @pytest.mark.numerical
@pytest.mark.parametrize("radius_scale", (0.25, 0.5, 1, 1.5, 2, 2.5, 3))
# @pytest.mark.parametrize("radius_scale", (1.5,))
@pytest.mark.parametrize("overlap_cube", (False,))
def test_native_sphere_fd_step_sweep(tmp_path, radius_scale, overlap_cube, numerical_case_dir):
    radius = SPHERE_RADIUS_UM * radius_scale
    params0 = anp.array([radius])

    radii = [radius, radius, radius]
    extra_structures = [make_overlap_cube_structure(radii)] if overlap_cube else []
    base_sim, fom = make_base_simulation(radii=radii, extra_structures=extra_structures)

    center = [0.0, 0.0, 0.0]

    native_objective_fd = make_objective(
        make_native_sphere_geometry,
        center,
        f"native_sphere_fd_step_sweep_{radius_scale}_cube_{overlap_cube}",
        base_sim,
        fom,
        tmp_path,
        local_gradient=False,
    )
    # min_log = -8
    # max_log = -1
    min_log = -6
    max_log = -2
    n = max_log - min_log + 1
    steps = np.logspace(min_log, max_log, num=n)
    fd_grads = []
    for step in steps:
        grad = finite_difference_params(native_objective_fd, params0, step)
        fd_grads.append(grad)
        print(
            f"native finite difference step {step:.1e}: gradient {grad.tolist()} cube={overlap_cube}"
        )

    fd_grads = np.asarray(fd_grads, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, fd_grads[:, 0], marker="o", label="radius")
    ax.set_xscale("log")
    ax.set_xlabel("Finite difference step (µm)")
    ax.set_ylabel("Gradient value")
    ax.set_title("Native sphere finite difference gradient vs. step size")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    fig_path = (
        numerical_case_dir / f"native_fd_step_sweep_scale_{radius_scale}_cube_{overlap_cube}.png"
    )

    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    np.savez(
        numerical_case_dir / f"native_fd_step_sweep_scale_{radius_scale}_cube_{overlap_cube}.npz",
        steps=steps,
        gradients=fd_grads,
    )


@pytest.mark.numerical
@pytest.mark.parametrize("scale_factor", (1,))
@pytest.mark.parametrize("scale_axis", (1,))
@pytest.mark.parametrize("unit_sphere_subdivisions", (0,))
def test_sphere_vertex_gradient_visualization(
    tmp_path, scale_factor, scale_axis, unit_sphere_subdivisions, numerical_case_dir
):
    unit_triangles = td.Sphere.unit_sphere_triangles(subdivisions=unit_sphere_subdivisions)
    base_vertices, faces = _make_unique_vertices(unit_triangles)

    radii = np.array([SPHERE_RADIUS_UM, SPHERE_RADIUS_UM, SPHERE_RADIUS_UM], dtype=float)
    base_vertices = base_vertices * radii
    axis_selector = np.arange(3) == scale_axis
    scale_vec = np.where(axis_selector, scale_factor, 1.0)
    base_vertices = base_vertices * scale_vec
    vertex_shape = base_vertices.shape
    params0 = anp.reshape(anp.array(base_vertices, dtype=float), (-1,))

    radii_scaled = radii.copy()
    radii_scaled[scale_axis] *= scale_factor
    base_sim, fom = make_base_simulation(radii=radii_scaled.tolist())

    center = [0.0, 0.0, 0.0]

    center_arr = np.asarray(center, dtype=float)
    domain_half_lengths = np.asarray(base_sim.size) / 2.0

    faces_arr = anp.asarray(faces)

    def make_geometry_from_vertices(param_vec, geom_center):
        vertices = anp.reshape(param_vec, vertex_shape)
        tris = vertices[faces_arr]
        center_arr = anp.array(geom_center, dtype=float)
        tris = tris + center_arr
        return td.TriangleMesh.from_triangles(tris)

    triangle_objective = make_objective(
        make_geometry_from_vertices,
        center,
        "sphere_vertex_grad",
        base_sim,
        fom,
        tmp_path,
        local_gradient=LOCAL_GRADIENT,
    )
    triangle_objective_fd = make_objective(
        make_geometry_from_vertices,
        center,
        "sphere_vertex_grad_fd",
        base_sim,
        fom,
        tmp_path,
        local_gradient=False,
    )

    _, triangle_grad = value_and_grad(triangle_objective)([params0])
    triangle_grad = np.squeeze(np.asarray(triangle_grad, dtype=float))
    fd_grad = finite_difference_params(triangle_objective_fd, params0, VERTEX_FD_STEP)

    triangle_grad = triangle_grad.reshape(vertex_shape)
    fd_grad = fd_grad.reshape(vertex_shape)

    vertex_positions = base_vertices + center_arr
    outside_mask = np.any(np.abs(vertex_positions) > domain_half_lengths + 1e-9, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    component_labels = ["x", "y", "z"]
    for comp, ax in enumerate(axes):
        fd_comp = fd_grad[:, comp]
        adj_comp = triangle_grad[:, comp]
        inside_fd = fd_comp[~outside_mask]
        inside_adj = adj_comp[~outside_mask]
        outside_fd = fd_comp[outside_mask]
        outside_adj = adj_comp[outside_mask]

        ax.scatter(
            inside_fd,
            inside_adj,
            s=10,
            alpha=1,
            label="inside" if comp == 0 else None,
        )
        if outside_mask.any():
            ax.scatter(
                outside_fd,
                outside_adj,
                s=12,
                alpha=1,
                marker="x",
                color="tab:red",
                label="outside" if comp == 0 else None,
            )
        data_min = min(fd_comp.min(), adj_comp.min())
        data_max = max(fd_comp.max(), adj_comp.max())
        if data_min == data_max:
            data_min -= 1e-6
            data_max += 1e-6
        ax.plot([data_min, data_max], [data_min, data_max], "k--", linewidth=1.0)
        ax.set_xlabel(f"FD grad ({component_labels[comp]})")
        ax.set_ylabel(f"Adjoint grad ({component_labels[comp]})")
        ax.set_title(f"Vertex component {component_labels[comp]}")
        ax.grid(True, ls=":")

    if outside_mask.any():
        axes[0].legend(loc="best")
    fig.tight_layout()
    fig_path = (
        numerical_case_dir
        / f"vertex_gradient_comparison_scale_{scale_factor}_axis_{scale_axis}_subdiv_{unit_sphere_subdivisions}.png"
    )
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    np.savez(
        numerical_case_dir
        / f"vertex_gradient_comparison_scale_{scale_factor}_axis_{scale_axis}_subdiv_{unit_sphere_subdivisions}.npz",
        triangle_grad=triangle_grad,
        fd_grad=fd_grad,
    )
