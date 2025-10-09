from __future__ import annotations

from tidy3d.web.api.autograd.autograd import run as run_autograd, run_async
from tidy3d.web.api.autograd.constants import LOCAL_GRADIENT
from tidy3d.web.api.webapi import _modesolver_patch

import typing
from tidy3d.components.autograd.constants import (
    MAX_NUM_ADJOINT_PER_FWD,
)
from tidy3d.components.types.workflow import WorkflowDataType, WorkflowType
from tidy3d.web.core.types import PayType

RunInput: typing.TypeAlias = typing.Union[
    WorkflowType,
    list["RunInput"],
    tuple["RunInput", ...],
    dict[typing.Hashable, "RunInput"],
]

RunOutput: typing.TypeAlias = typing.Union[
    WorkflowDataType,
    list["WorkflowDataType"],
    tuple["WorkflowDataType", ...],
    dict[typing.Hashable, "WorkflowDataType"],
]

def _collect_by_hash(
    node: RunInput,
    found: dict[str, WorkflowType] | None = None,
) -> dict[str, WorkflowType]:
    """Traversiert die Struktur und sammelt alle Simulationen in {hash: sim}.
    Die letzte Sicht auf denselben Hash überschreibt – ok, da identische Objekte."""
    if found is None:
        found = {}
    if isinstance(node, WorkflowType):
        found[str(hash(node))] = node
        return found
    if isinstance(node, (list, tuple)):
        for v in node:
            _collect_by_hash(v, found)
        return found
    if isinstance(node, dict):
        if any(isinstance(k, WorkflowType) for k in node.keys()):
            raise ValueError("Dict keys must not be simulations.")
        for v in node.values():
            _collect_by_hash(v, found)
        return found
    raise TypeError(f"Unsupported element in container: {type(node)!r}")


def _reconstruct_by_hash(node: RunInput, h2data: dict[str, WorkflowDataType]) -> RunOutput:
    """Ersetzt jedes Blatt (Simulation) durch sein Data-Objekt anhand des Hashes."""
    if isinstance(node, WorkflowType):
        return h2data[str(hash(node))]
    if isinstance(node, tuple):
        return tuple(_reconstruct_by_hash(v, h2data) for v in node)
    if isinstance(node, list):
        return list(_reconstruct_by_hash(v, h2data) for v in node)
    if isinstance(node, dict):
        return {k: _reconstruct_by_hash(v, h2data) for k, v in node.items()}
    raise TypeError(f"Unsupported element in reconstruction: {type(node)!r}")


def run(
    simulation: RunInput,
    task_name: typing.Optional[str] = None,
    folder_name: str = "default",
    path: str = "simulation_data",
    callback_url: typing.Optional[str] = None,
    verbose: bool = True,
    progress_callback_upload: typing.Optional[typing.Callable[[float], None]] = None,
    progress_callback_download: typing.Optional[typing.Callable[[float], None]] = None,
    solver_version: typing.Optional[str] = None,
    worker_group: typing.Optional[str] = None,
    simulation_type: str = "tidy3d",
    parent_tasks: typing.Optional[list[str]] = None,
    local_gradient: bool = LOCAL_GRADIENT,
    max_num_adjoint_per_fwd: int = MAX_NUM_ADJOINT_PER_FWD,
    reduce_simulation: typing.Literal["auto", True, False] = "auto",
    pay_type: typing.Union[PayType, str] = PayType.AUTO,
    priority: typing.Optional[int] = None,
    max_workers: typing.Optional[int] = None,
    lazy: typing.Optional[bool] = None,
) -> RunOutput:
    """
    Submit one or many simulations and return results in the same container shape.

    This is a convenience wrapper around the autograd runners that accepts a single
    :class:`WorkflowType` **or** an arbitrarily nested container of simulations
    (`list`, `tuple`, or `dict` values). Internally, all simulations are collected,
    deduplicated by object hash, executed either synchronously (single) or
    asynchronously (batch), and the returned data objects are reassembled to mirror
    the input structure.

    **Path behavior**
      - **Single simulation:** results are downloaded to ``f"{path}.hdf5"``.
      - **Multiple simulations:** ``path`` is treated as a **directory**, and each
        task will write its own results file inside that directory.

    **Lazy loading**
      - If ``lazy`` is *not* specified: single runs default to ``False`` (eager load);
        batch runs default to ``True`` (proxy objects that load on first access).

    Parameters
    ----------
    simulation : Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`] | list | tuple | dict
        A simulation or a container whose leaves are simulations.
        Supported containers are ``list``, ``tuple``, and ``dict`` (values only).
        Dict **keys must not** be simulations.
    task_name : Optional[str], default None
        Optional name for a single run. Ignored for batch runs (hash strings are used).
    folder_name : str, default "default"
        Folder shown on the web UI.
    path : str, default "simulation_data"
        Output path. File stem for single runs (``.hdf5`` is appended), or a
        directory for batch runs.
    callback_url : Optional[str], default None
        Optional HTTP PUT endpoint to receive completion events.
    verbose : bool, default True
        If ``True``, print status and progress; otherwise run quietly.
    progress_callback_upload : Optional[Callable[[float], None]], default None
        Callback invoked with byte counts during upload (single-run path only).
    progress_callback_download : Optional[Callable[[float], None]], default None
        Callback invoked with byte counts during download (single-run path only).
    solver_version : Optional[str], default None
        Target solver version.
    worker_group : Optional[str], default None
        Worker group to target.
    simulation_type : str, default "tidy3d"
        Simulation type label passed through to the runners.
    parent_tasks : Optional[List[str]], default None
        Parent task IDs, if any.
    local_gradient : bool, default ``LOCAL_GRADIENT``
        Compute gradients locally (more downloads; useful for experimental features).
    max_num_adjoint_per_fwd : int, default ``MAX_NUM_ADJOINT_PER_FWD``
        Maximum number of adjoint simulations allowed per forward run.
    reduce_simulation : {"auto", True, False}, default "auto"
        Whether to reduce structures to the simulation domain (mode solver only).
    pay_type : Union[PayType, str], default PayType.AUTO
        Payment method selection.
    priority : Optional[int], default None
        Queue priority for vGPU (1 = lowest, 10 = highest).
    max_workers : Optional[int], default None
        Maximum parallel submissions for batch runs. ``None`` submits all at once.
    lazy : Optional[bool], default None
        If provided, overrides the lazy/eager behavior described above.

    Returns
    -------
    RunOutput
        A data object (or nested container of data objects) matching the input
        container shape. Leaves are instances of the corresponding
        :class:`WorkflowDataType`.

    Notes
    -----
    - Simulations are indexed by ``hash(sim)``. If the *same object* appears multiple
      times in the input, it is executed once and its data is reused at all positions.
      The *last* occurrence wins if duplicates with the same hash are encountered.
    - For each simulation, a mode-solver compatibility patch is applied so that
      the returned data exposes expected convenience attributes.
    - ``progress_callback_*`` are only used in the single-run code path.

    Raises
    ------
    ValueError
        If no simulations are found in ``simulation``.
    TypeError
        If an unsupported container element is encountered, or if a dict key is a
        simulation object.

    Examples
    --------
    Single run (eager by default)::

        sim_data = run(sim, task_name="wg_bend", path="out/bend")
        # writes: "out/bend.hdf5"

    Batch run with nested structure (lazy by default)::

        sims = {
            "coarse": [sim_a, sim_b],
            "fine": (sim_c, sim_d),
        }
        data = run(sims, path="out/batch_dir", max_workers=4)

        # 'data' mirrors 'sims' structure:
        # data["coarse"][0] -> data for sim_a, etc.

    See Also
    --------
    tidy3d.web.api.autograd.autograd.run
        Underlying autograd single-run implementation.
    tidy3d.web.api.autograd.autograd.run_async
        Underlying autograd batch submission implementation.
    """
    h2sim: dict[str, WorkflowType] = _collect_by_hash(simulation)
    if not h2sim:
        raise ValueError("No simulation data found in simulation input.")

    if len(h2sim) == 1:
        hash_key, sim = next(iter(h2sim.items()))
        data = {hash_key: run_autograd(
            simulation=sim,
            task_name=task_name,
            folder_name=folder_name,
            path=f"{path}.hdf5",
            callback_url=callback_url,
            verbose=verbose,
            progress_callback_upload=progress_callback_upload,
            progress_callback_download=progress_callback_download,
            solver_version=solver_version,
            worker_group=worker_group,
            simulation_type=simulation_type,
            parent_tasks=parent_tasks,
            local_gradient=local_gradient,
            max_num_adjoint_per_fwd=max_num_adjoint_per_fwd,
            reduce_simulation=reduce_simulation,
            pay_type=pay_type,
            priority=priority,
            lazy=lazy or False,
        )}
    else:
        data = run_async(
            simulations=h2sim,
            folder_name=folder_name,
            path_dir=path,
            callback_url=callback_url,
            num_workers=max_workers,
            verbose=verbose,
            simulation_type=simulation_type,
            solver_version=solver_version,
            parent_tasks=parent_tasks,
            local_gradient = LOCAL_GRADIENT,
            max_num_adjoint_per_fwd = MAX_NUM_ADJOINT_PER_FWD,
            reduce_simulation=reduce_simulation,
            pay_type=pay_type,
            priority=priority,
            lazy=lazy or True,
        )

    h2data: dict[str, WorkflowDataType] = {}
    for h, sim_obj in h2sim.items():
        sim_data: WorkflowDataType = data[h]
        _modesolver_patch(sim_obj, sim_data)
        h2data[h] = sim_data
        print(f"type data {type(sim_data)}")

    return _reconstruct_by_hash(simulation, h2data)