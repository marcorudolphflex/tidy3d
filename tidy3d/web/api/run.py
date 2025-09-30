from typing import TypeAlias, Union, Optional, Callable, Literal

from tidy3d.components.types.workflow import WorkflowDataType, WorkflowType
from tidy3d.web.api.container import BatchData, Batch
from tidy3d.web.api.connect_util import wait_for_connection
from tidy3d.web.api.webapi import _modesolver_patch, upload, start, monitor, load
from tidy3d.web.core.types import PayType

RunInput: TypeAlias = Union[
    WorkflowType,
    list["RunInput"],
    tuple["RunInput", ...],
    dict[str, "RunInput"],
]
RunOutput: TypeAlias = Union[
    WorkflowDataType,
    list["WorkflowDataType"],
    tuple["WorkflowDataType", ...],
    dict[str, "WorkflowDataType"],
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
    if isinstance(node, list) or isinstance(node, tuple):
        for v in node:
            _collect_by_hash(v, found)
        return found
    if isinstance(node, dict):
        for v in node.values():
            _collect_by_hash(v, found)
        return found
    raise TypeError(f"Unsupported element in container: {type(node)!r}")


def _reconstruct_by_hash(node: RunInput, h2data: dict[str, WorkflowDataType]) -> RunOutput:
    """Ersetzt jedes Blatt (Simulation) durch sein Data-Objekt anhand des Hashes."""
    if isinstance(node, WorkflowType):
        return h2data[str(hash(node))]
    if isinstance(node, (list, tuple)):
        return tuple(_reconstruct_by_hash(v, h2data) for v in node)
    if isinstance(node, dict):
        return {k: _reconstruct_by_hash(v, h2data) for k, v in node.items()}
    raise TypeError(f"Unsupported element in reconstruction: {type(node)!r}")



@wait_for_connection
def run(
    simulation: RunInput,
    task_name: Optional[str] = None,
    folder_name: str = "default",
    path: str = "simulation_data.hdf5",
    callback_url: Optional[str] = None,
    verbose: bool = True,
    progress_callback_upload: Optional[Callable[[float], None]] = None,  # wird im Batch nicht genutzt
    progress_callback_download: Optional[Callable[[float], None]] = None, # wird im Batch nicht genutzt
    solver_version: Optional[str] = None,
    worker_group: Optional[str] = None,  # Batch-level, falls unterstützt
    simulation_type: str = "tidy3d",
    parent_tasks: Optional[dict[str, list[str]]] = None,  # falls du Parent-Graph per Namen hast
    reduce_simulation: Literal["auto", True, False] = "auto",
    pay_type: Union[PayType, str] = PayType.AUTO,
    priority: Optional[int] = None,
) -> RunOutput | WorkflowDataType:
    """
    Submits a :class:`.Simulation` to server, starts running, monitors progress, downloads,
    and loads results as a :class:`.WorkflowDataType` object.

    Parameters
    ----------
    simulation : Union[:class:`.Simulation`, :class:`.HeatSimulation`, :class:`.EMESimulation`]
        Simulation to upload to server.
    task_name : Optional[str] = None
        Name of task. If not provided, a default name will be generated.
    folder_name : str = "default"
        Name of folder to store task on web UI.
    path : str = "simulation_data.hdf5"
        Path to download results file (.hdf5), including filename.
    callback_url : str = None
        Http PUT url to receive simulation finish event. The body content is a json file with
        fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.
    verbose : bool = True
        If ``True``, will print progressbars and status, otherwise, will run silently.
    simulation_type : str = "tidy3d"
        Type of simulation being uploaded.
    progress_callback_upload : Callable[[float], None] = None
        Optional callback function called when uploading file with ``bytes_in_chunk`` as argument.
    progress_callback_download : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.
    solver_version: str = None
        target solver version.
    worker_group: str = None
        worker group
    reduce_simulation : Literal["auto", True, False] = "auto"
        Whether to reduce structures in the simulation to the simulation domain only. Note: currently only implemented for the mode solver.
    pay_type: Union[PayType, str] = PayType.AUTO
        Which method to pay the simulation.
    priority: int = None
        Priority of the simulation in the Virtual GPU (vGPU) queue (1 = lowest, 10 = highest).
        It affects only simulations from vGPU licenses and does not impact simulations using FlexCredits.
    Returns
    -------
    Union[:class:`.SimulationData`, :class:`.HeatSimulationData`, :class:`.EMESimulationData`]
        Object containing solver results for the supplied simulation.

    Notes
    -----

        Submitting a simulation to our cloud server is very easily done by a simple web API call.

        .. code-block:: python

            sim_data = tidy3d.web.api.webapi.run(simulation, task_name='my_task', path='out/data.hdf5')

        The :meth:`tidy3d.web.api.webapi.run()` method shows the simulation progress by default.  When uploading a
        simulation to the server without running it, you can use the :meth:`tidy3d.web.api.webapi.monitor`,
        :meth:`tidy3d.web.api.container.Job.monitor`, or :meth:`tidy3d.web.api.container.Batch.monitor` methods to
        display the progress of your simulation(s).

    Examples
    --------

        To access the original :class:`.Simulation` object that created the simulation data you can use:

        .. code-block:: python

            # Run the simulation.
            sim_data = web.run(simulation, task_name='task_name', path='out/sim.hdf5')

            # Get a copy of the original simulation object.
            sim_copy = sim_data.simulation

    See Also
    --------

    :meth:`tidy3d.web.api.webapi.monitor`
        Print the real time task progress until completion.

    :meth:`tidy3d.web.api.container.Job.monitor`
        Monitor progress of running :class:`Job`.

    :meth:`tidy3d.web.api.container.Batch.monitor`
        Monitor progress of each of the running tasks.
    """
    if isinstance(simulation, WorkflowType):
        task_id = upload(
            simulation=simulation,
            task_name=task_name,
            folder_name=folder_name,
            callback_url=callback_url,
            verbose=verbose,
            progress_callback=progress_callback_upload,
            simulation_type=simulation_type,
            parent_tasks=parent_tasks,
            solver_version=solver_version,
            reduce_simulation=reduce_simulation,
        )
        start(
            task_id,
            verbose=verbose,
            solver_version=solver_version,
            worker_group=worker_group,
            pay_type=pay_type,
            priority=priority,
        )
        monitor(task_id, verbose=verbose)
        data = load(
            task_id=task_id, path=path, verbose=verbose, progress_callback=progress_callback_download
        )
        _modesolver_patch(simulation, data)
    else:
        h2sim: dict[str, WorkflowType] = _collect_by_hash(simulation)
        if not h2sim:
            raise ValueError("No simulation data found in simulation input.")

        name2sim: dict[str, WorkflowType] = {h: s for h, s in h2sim.items()}

        batch = Batch(
            simulations=name2sim,
            folder_name=folder_name,
            callback_url=callback_url,
            verbose=verbose,
            simulation_type=simulation_type,
            solver_version=solver_version,
            parent_tasks=parent_tasks,
            reduce_simulation=reduce_simulation,
            pay_type=pay_type,
        )
        batch_data: BatchData = batch.run(path_dir=path, priority=priority)

        h2data: dict[str, WorkflowDataType] = {}
        for h, sim_obj in h2sim.items():
            sim_data: WorkflowDataType = batch_data[h]
            _modesolver_patch(sim_obj, sim_data)
            h2data[h] = sim_data
            print(f"type data {type(sim_data)}")

        return _reconstruct_by_hash(simulation, h2data)
