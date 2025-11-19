"""Mesh-defined geometry."""

from __future__ import annotations

import time
from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import autograd.numpy as anp
import numpy as np
import pydantic.v1 as pydantic
from numpy.typing import NDArray
from pydantic.v1 import PrivateAttr

from tidy3d.components.autograd import AutogradFieldMap, get_static
from tidy3d.components.autograd.derivative_utils import DerivativeInfo
from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import DATA_ARRAY_MAP, TriangleMeshDataArray
from tidy3d.components.data.dataset import TriangleMeshDataset
from tidy3d.components.data.validators import validate_no_nans
from tidy3d.components.types import Ax, Bound, Coordinate, MatrixReal4x4, Shapely
from tidy3d.components.viz import add_ax_if_none, equal_aspect
from tidy3d.config import config
from tidy3d.constants import fp_eps, inf
from tidy3d.exceptions import DataError, ValidationError
from tidy3d.log import log
from tidy3d.packaging import verify_packages_import

from . import base

if TYPE_CHECKING:
    from trimesh import Trimesh

AREA_SIZE_THRESHOLD = 1e-36


class TriangleMesh(base.Geometry, ABC):
    """Custom surface geometry given by a triangle mesh, as in the STL file format.

    Example
    -------
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    >>> stl_geom = TriangleMesh.from_vertices_faces(vertices, faces)
    """

    mesh_dataset: Optional[TriangleMeshDataset] = pydantic.Field(
        ...,
        title="Surface mesh data",
        description="Surface mesh data.",
    )

    _no_nans_mesh = validate_no_nans("mesh_dataset")

    _barycentric_cache: dict[tuple[int, int, tuple[int, ...]], np.ndarray] = PrivateAttr(
        default_factory=dict
    )

    @pydantic.root_validator(pre=True)
    @verify_packages_import(["trimesh"])
    def _validate_trimesh_library(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Check if the trimesh package is imported as a validator."""
        return values

    @pydantic.validator("mesh_dataset", pre=True, always=True)
    def _warn_if_none(cls, val: TriangleMeshDataset) -> TriangleMeshDataset:
        """Warn if the Dataset fails to load."""
        if isinstance(val, dict):
            if any((v in DATA_ARRAY_MAP for _, v in val.items() if isinstance(v, str))):
                log.warning("Loading 'mesh_dataset' without data.")
                return None
        return val

    @pydantic.validator("mesh_dataset")
    def debug(cls, val: Any) -> Any:
        """Warn if the Dataset fails to load."""
        return val

    @pydantic.validator("mesh_dataset", always=True)
    @verify_packages_import(["trimesh"])
    def _check_mesh(cls, val: TriangleMeshDataset) -> TriangleMeshDataset:
        """Check that the mesh is valid."""
        if val is None:
            return None

        import trimesh

        surface_mesh = val.surface_mesh
        triangles = get_static(surface_mesh.data)
        mesh = cls._triangles_to_trimesh(triangles)
        if not all(np.array(mesh.area_faces) > AREA_SIZE_THRESHOLD):
            old_tol = trimesh.tol.merge
            trimesh.tol.merge = np.sqrt(2 * AREA_SIZE_THRESHOLD)
            new_mesh = mesh.process(validate=True)
            trimesh.tol.merge = old_tol
            val = TriangleMesh.from_trimesh(new_mesh).mesh_dataset
            log.warning(
                f"The provided mesh has triangles with near zero area < {AREA_SIZE_THRESHOLD}. "
                "Triangles which have one edge of their 2D oriented bounding box shorter than "
                f"'sqrt(2*{AREA_SIZE_THRESHOLD}) are being automatically removed.'"
            )
            if not all(np.array(new_mesh.area_faces) > AREA_SIZE_THRESHOLD):
                raise ValidationError(
                    f"The provided mesh has triangles with near zero area < {AREA_SIZE_THRESHOLD}. "
                    "The automatic removal of these triangles has failed. You can try "
                    "using numpy-stl's 'from_file' import with 'remove_empty_areas' set "
                    "to True and a suitable 'AREA_SIZE_THRESHOLD' to remove them."
                )
        if not mesh.is_watertight:
            log.warning(
                "The provided mesh is not watertight. "
                "This can lead to incorrect permittivity distributions, "
                "and can also cause problems with plotting and mesh validation. "
                "You can try 'TriangleMesh.fill_holes', which attempts to repair the mesh. "
                "Otherwise, the mesh may require manual repair. You can use a "
                "'PermittivityMonitor' to check if the permittivity distribution is correct. "
                "You can see which faces are broken using 'trimesh.repair.broken_faces'."
            )
        if not mesh.is_winding_consistent:
            log.warning(
                "The provided mesh does not have consistent winding (face orientations). "
                "This can lead to incorrect permittivity distributions, "
                "and can also cause problems with plotting and mesh validation. "
                "You can try 'TriangleMesh.fix_winding', which attempts to repair the mesh. "
                "Otherwise, the mesh may require manual repair. You can use a "
                "'PermittivityMonitor' to check if the permittivity distribution is correct. "
            )
        if not mesh.is_volume:
            log.warning(
                "The provided mesh does not represent a valid volume, possibly due to "
                "incorrect normal vector orientation. "
                "This can lead to incorrect permittivity distributions, "
                "and can also cause problems with plotting and mesh validation. "
                "You can try 'TriangleMesh.fix_normals', "
                "which attempts to fix the normals to be consistent and outward-facing. "
                "Otherwise, the mesh may require manual repair. You can use a "
                "'PermittivityMonitor' to check if the permittivity distribution is correct."
            )

        return val

    @verify_packages_import(["trimesh"])
    def fix_winding(self) -> TriangleMesh:
        """Try to fix winding in the mesh."""
        import trimesh

        mesh = TriangleMesh._triangles_to_trimesh(self.mesh_dataset.surface_mesh)
        trimesh.repair.fix_winding(mesh)
        return TriangleMesh.from_trimesh(mesh)

    @verify_packages_import(["trimesh"])
    def fill_holes(self) -> TriangleMesh:
        """Try to fill holes in the mesh. Can be used to repair non-watertight meshes."""
        import trimesh

        mesh = TriangleMesh._triangles_to_trimesh(self.mesh_dataset.surface_mesh)
        trimesh.repair.fill_holes(mesh)
        return TriangleMesh.from_trimesh(mesh)

    @verify_packages_import(["trimesh"])
    def fix_normals(self) -> TriangleMesh:
        """Try to fix normals to be consistent and outward-facing."""
        import trimesh

        mesh = TriangleMesh._triangles_to_trimesh(self.mesh_dataset.surface_mesh)
        trimesh.repair.fix_normals(mesh)
        return TriangleMesh.from_trimesh(mesh)

    @classmethod
    @verify_packages_import(["trimesh"])
    def from_stl(
        cls,
        filename: str,
        scale: float = 1.0,
        origin: tuple[float, float, float] = (0, 0, 0),
        solid_index: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[TriangleMesh, base.GeometryGroup]:
        """Load a :class:`.TriangleMesh` directly from an STL file.
        The ``solid_index`` parameter can be used to select a single solid from the file.
        Otherwise, if the file contains a single solid, it will be loaded as a
        :class:`.TriangleMesh`; if the file contains multiple solids,
        they will all be loaded as a :class:`.GeometryGroup`.

        Parameters
        ----------
        filename : str
            The name of the STL file containing the surface geometry mesh data.
        scale : float = 1.0
            The length scale for the loaded geometry (um).
            For example, a scale of 10.0 means that a vertex (1, 0, 0) will be placed at
            x = 10 um.
        origin : Tuple[float, float, float] = (0, 0, 0)
            The origin of the loaded geometry, in units of ``scale``.
            Translates from (0, 0, 0) to this point after applying the scaling.
        solid_index : int = None
            If set, read a single solid with this index from the file.

        Returns
        -------
        Union[:class:`.TriangleMesh`, :class:`.GeometryGroup`]
            The geometry or geometry group from the file.
        """
        import trimesh

        from tidy3d.components.types.third_party import TrimeshType

        def process_single(mesh: TrimeshType) -> TriangleMesh:
            """Process a single 'trimesh.Trimesh' using scale and origin."""
            mesh.apply_scale(scale)
            mesh.apply_translation(origin)
            return cls.from_trimesh(mesh)

        scene = trimesh.load(filename, **kwargs)
        meshes = []
        if isinstance(scene, trimesh.Trimesh):
            meshes = [scene]
        elif isinstance(scene, trimesh.Scene):
            meshes = scene.dump()
        else:
            raise ValidationError(
                "Invalid trimesh type in file. Supported types are 'trimesh.Trimesh' "
                "and 'trimesh.Scene'."
            )

        if solid_index is None:
            if isinstance(scene, trimesh.Trimesh):
                return process_single(scene)
            if isinstance(scene, trimesh.Scene):
                geoms = [process_single(mesh) for mesh in meshes]
                return base.GeometryGroup(geometries=geoms)

        if solid_index < len(meshes):
            return process_single(meshes[solid_index])
        raise ValidationError("No solid found at 'solid_index' in the stl file.")

    @classmethod
    @verify_packages_import(["trimesh"])
    def from_trimesh(cls, mesh: trimesh.Trimesh) -> TriangleMesh:
        """Create a :class:`.TriangleMesh` from a ``trimesh.Trimesh`` object.

        Parameters
        ----------
        trimesh : ``trimesh.Trimesh``
            The Trimesh object containing the surface geometry mesh data.

        Returns
        -------
        :class:`.TriangleMesh`
            The custom surface mesh geometry given by the ``trimesh.Trimesh`` provided.
        """
        return cls.from_vertices_faces(mesh.vertices, mesh.faces)

    @classmethod
    def from_triangles(cls, triangles: NDArray) -> TriangleMesh:
        """Create a :class:`.TriangleMesh` from a numpy array
        containing the triangles of a surface mesh.

        Parameters
        ----------
        triangles : ``np.ndarray``
            A numpy array of shape (N, 3, 3) storing the triangles of the surface mesh.
            The first index labels the triangle, the second index labels the vertex
            within a given triangle, and the third index is the coordinate (x, y, or z).

        Returns
        -------
        :class:`.TriangleMesh`
            The custom surface mesh geometry given by the triangles provided.

        """
        triangles = np.array(triangles)
        if len(triangles.shape) != 3 or triangles.shape[1] != 3 or triangles.shape[2] != 3:
            raise ValidationError(
                f"Provided 'triangles' must be an N x 3 x 3 array, given {triangles.shape}."
            )
        num_faces = len(triangles)
        coords = {
            "face_index": np.arange(num_faces),
            "vertex_index": np.arange(3),
            "axis": np.arange(3),
        }
        vertices = TriangleMeshDataArray(triangles, coords=coords)
        mesh_dataset = TriangleMeshDataset(surface_mesh=vertices)
        return TriangleMesh(mesh_dataset=mesh_dataset)

    @classmethod
    @verify_packages_import(["trimesh"])
    def from_vertices_faces(cls, vertices: NDArray, faces: NDArray) -> TriangleMesh:
        """Create a :class:`.TriangleMesh` from numpy arrays containing the data
        of a surface mesh. The first array contains the vertices, and the second array contains
        faces formed from triples of the vertices.

        Parameters
        ----------
        vertices: ``np.ndarray``
            A numpy array of shape (N, 3) storing the vertices of the surface mesh.
            The first index labels the vertex, and the second index is the coordinate
            (x, y, or z).
        faces : ``np.ndarray``
            A numpy array of shape (M, 3) storing the indices of the vertices of each face
            in the surface mesh. The first index labels the face, and the second index
            labels the vertex index within the ``vertices`` array.

        Returns
        -------
        :class:`.TriangleMesh`
            The custom surface mesh geometry given by the vertices and faces provided.

        """
        import trimesh

        vertices = np.array(vertices)
        faces = np.array(faces)
        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValidationError(
                f"Provided 'vertices' must be an N x 3 array, given {vertices.shape}."
            )
        if len(faces.shape) != 2 or faces.shape[1] != 3:
            raise ValidationError(f"Provided 'faces' must be an M x 3 array, given {faces.shape}.")
        return cls.from_triangles(trimesh.Trimesh(vertices, faces).triangles)

    @classmethod
    @verify_packages_import(["trimesh"])
    def _triangles_to_trimesh(
        cls, triangles: NDArray
    ) -> Trimesh:  # -> We need to get this out of the classes and into functional methods operating on a class (maybe still referenced to the class)
        """Convert an (N, 3, 3) numpy array of triangles to a ``trimesh.Trimesh``."""
        import trimesh

        # ``triangles`` may contain autograd ``ArrayBox`` entries when differentiating
        # geometry parameters. ``trimesh`` expects plain ``float`` values, so strip any
        # tracing information before constructing the mesh.
        triangles = np.array(triangles)
        if triangles.dtype == np.object_:
            triangles = anp.array(triangles.tolist())
        triangles = np.asarray(get_static(triangles), dtype=np.float64)
        return trimesh.Trimesh(**trimesh.triangles.to_kwargs(get_static(triangles)))

    @classmethod
    def from_height_grid(
        cls,
        axis: Ax,
        direction: Literal["-", "+"],
        base: float,
        grid: tuple[np.ndarray, np.ndarray],
        height: NDArray,
    ) -> TriangleMesh:
        """Construct a TriangleMesh object from grid based height information.

        Parameters
        ----------
        axis : Ax
            Axis of extrusion.
        direction : Literal["-", "+"]
            Direction of extrusion.
        base : float
            Coordinate of the base surface along the geometry's axis.
        grid : Tuple[np.ndarray, np.ndarray]
            Tuple of two one-dimensional arrays representing the sampling grid (XY, YZ, or ZX
            corresponding to values of axis)
        height : NDArray
            Height values sampled on the given grid. Can be 1D (raveled) or 2D (matching grid mesh).

        Returns
        -------
        TriangleMesh
            The resulting TriangleMesh geometry object.
        """

        x_coords = grid[0]
        y_coords = grid[1]

        nx = len(x_coords)
        ny = len(y_coords)
        nt = nx * ny

        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")

        sign = 1
        if direction == "-":
            sign = -1

        flat_height = np.ravel(height)
        if flat_height.shape[0] != nt:
            raise ValueError(
                f"Shape of flattened height array {flat_height.shape} does not match "
                f"the number of grid points {nt}."
            )

        if np.any(flat_height < 0):
            raise ValueError("All height values must be non-negative.")

        max_h = np.max(flat_height)
        min_h_clip = fp_eps * max_h
        flat_height = np.clip(flat_height, min_h_clip, inf)

        vertices_raw_list = [
            [np.ravel(x_mesh), np.ravel(y_mesh), base + sign * flat_height],  # Alpha surface
            [np.ravel(x_mesh), np.ravel(y_mesh), base * np.ones(nt)],
        ]

        if direction == "-":
            vertices_raw_list = vertices_raw_list[::-1]

        vertices = np.hstack(vertices_raw_list).T
        vertices = np.roll(vertices, shift=axis - 2, axis=1)

        q0 = (np.arange(nx - 1)[:, None] * ny + np.arange(ny - 1)[None, :]).ravel()
        q1 = (np.arange(1, nx)[:, None] * ny + np.arange(ny - 1)[None, :]).ravel()
        q2 = (np.arange(1, nx)[:, None] * ny + np.arange(1, ny)[None, :]).ravel()
        q3 = (np.arange(nx - 1)[:, None] * ny + np.arange(1, ny)[None, :]).ravel()

        q0_b = nt + q0
        q1_b = nt + q1
        q2_b = nt + q2
        q3_b = nt + q3

        top_quads = np.stack((q0, q1, q2, q3), axis=-1)
        bottom_quads = np.stack((q0_b, q3_b, q2_b, q1_b), axis=-1)

        s1_q0 = (0 * ny + np.arange(ny - 1)).ravel()
        s1_q1 = (0 * ny + np.arange(1, ny)).ravel()
        s1_q2 = (nt + 0 * ny + np.arange(1, ny)).ravel()
        s1_q3 = (nt + 0 * ny + np.arange(ny - 1)).ravel()
        side1_quads = np.stack((s1_q0, s1_q1, s1_q2, s1_q3), axis=-1)

        s2_q0 = ((nx - 1) * ny + np.arange(ny - 1)).ravel()
        s2_q1 = (nt + (nx - 1) * ny + np.arange(ny - 1)).ravel()
        s2_q2 = (nt + (nx - 1) * ny + np.arange(1, ny)).ravel()
        s2_q3 = ((nx - 1) * ny + np.arange(1, ny)).ravel()
        side2_quads = np.stack((s2_q0, s2_q1, s2_q2, s2_q3), axis=-1)

        s3_q0 = (np.arange(nx - 1) * ny + 0).ravel()
        s3_q1 = (nt + np.arange(nx - 1) * ny + 0).ravel()
        s3_q2 = (nt + np.arange(1, nx) * ny + 0).ravel()
        s3_q3 = (np.arange(1, nx) * ny + 0).ravel()
        side3_quads = np.stack((s3_q0, s3_q1, s3_q2, s3_q3), axis=-1)

        s4_q0 = (np.arange(nx - 1) * ny + ny - 1).ravel()
        s4_q1 = (np.arange(1, nx) * ny + ny - 1).ravel()
        s4_q2 = (nt + np.arange(1, nx) * ny + ny - 1).ravel()
        s4_q3 = (nt + np.arange(nx - 1) * ny + ny - 1).ravel()
        side4_quads = np.stack((s4_q0, s4_q1, s4_q2, s4_q3), axis=-1)

        all_quads = np.vstack(
            (top_quads, bottom_quads, side1_quads, side2_quads, side3_quads, side4_quads)
        )

        triangles_list = [
            np.stack((all_quads[:, 0], all_quads[:, 1], all_quads[:, 3]), axis=-1),
            np.stack((all_quads[:, 3], all_quads[:, 1], all_quads[:, 2]), axis=-1),
        ]
        tri_faces = np.vstack(triangles_list)

        return cls.from_vertices_faces(vertices=vertices, faces=tri_faces)

    @classmethod
    def from_height_function(
        cls,
        axis: Ax,
        direction: Literal["-", "+"],
        base: float,
        center: tuple[float, float],
        size: tuple[float, float],
        grid_size: tuple[int, int],
        height_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> TriangleMesh:
        """Construct a TriangleMesh object from analytical expression of height function.
        The height function should be vectorized to accept 2D meshgrid arrays.

        Parameters
        ----------
        axis : Ax
            Axis of extrusion.
        direction : Literal["-", "+"]
            Direction of extrusion.
        base : float
            Coordinate of the base rectangle along the geometry's axis.
        center : Tuple[float, float]
            Center of the base rectangle in the plane perpendicular to the extrusion axis
            (XY, YZ, or ZX corresponding to values of axis).
        size : Tuple[float, float]
            Size of the base rectangle in the plane perpendicular to the extrusion axis
            (XY, YZ, or ZX corresponding to values of axis).
        grid_size : Tuple[int, int]
            Number of grid points for discretization of the base rectangle
            (XY, YZ, or ZX corresponding to values of axis).
        height_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
            Vectorized function to compute height values from 2D meshgrid coordinate arrays.
            It should take two ndarrays (x_mesh, y_mesh) and return an ndarray of heights.

        Returns
        -------
        TriangleMesh
            The resulting TriangleMesh geometry object.
        """
        x_lin = np.linspace(center[0] - 0.5 * size[0], center[0] + 0.5 * size[0], grid_size[0])
        y_lin = np.linspace(center[1] - 0.5 * size[1], center[1] + 0.5 * size[1], grid_size[1])

        x_mesh, y_mesh = np.meshgrid(x_lin, y_lin, indexing="ij")

        height_values = height_func(x_mesh, y_mesh)

        if not (isinstance(height_values, np.ndarray) and height_values.shape == x_mesh.shape):
            raise ValueError(
                f"The 'height_func' must return a NumPy array with shape {x_mesh.shape}, "
                f"but got shape {getattr(height_values, 'shape', type(height_values))}."
            )

        return cls.from_height_grid(
            axis=axis,
            direction=direction,
            base=base,
            grid=(x_lin, y_lin),
            height=height_values,
        )

    @cached_property
    @verify_packages_import(["trimesh"])
    def trimesh(
        self,
    ) -> Trimesh:  # -> We need to get this out of the classes and into functional methods operating on a class (maybe still referenced to the class)
        """A ``trimesh.Trimesh`` object representing the custom surface mesh geometry."""
        return self._triangles_to_trimesh(self.triangles)

    @cached_property
    def triangles(self) -> np.ndarray:
        """The triangles of the surface mesh as an ``np.ndarray``."""
        if self.mesh_dataset is None:
            raise DataError("Can't get triangles as 'mesh_dataset' is None.")
        return self.mesh_dataset.surface_mesh.to_numpy()

    def _surface_area(self, bounds: Bound) -> float:
        """Returns object's surface area within given bounds."""
        # currently ignores bounds
        return self.trimesh.area

    def _volume(self, bounds: Bound) -> float:
        """Returns object's volume within given bounds."""
        # currently ignores bounds
        return self.trimesh.volume

    @cached_property
    def bounds(self) -> Bound:
        """Returns bounding box min and max coordinates.

        Returns
        -------
        Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        """
        if self.mesh_dataset is None:
            return ((-inf, -inf, -inf), (inf, inf, inf))
        return self.trimesh.bounds

    def intersections_tilted_plane(
        self, normal: Coordinate, origin: Coordinate, to_2D: MatrixReal4x4
    ) -> list[Shapely]:
        """Return a list of shapely geometries at the plane specified by normal and origin.

        Parameters
        ----------
        normal : Coordinate
            Vector defining the normal direction to the plane.
        origin : Coordinate
            Vector defining the plane origin.
        to_2D : MatrixReal4x4
            Transformation matrix to apply to resulting shapes.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentation <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """
        section = self.trimesh.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            return []
        path, _ = section.to_2D(to_2D=to_2D)
        return path.polygons_full

    def intersections_plane(
        self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None
    ) -> list[Shapely]:
        """Returns list of shapely geometries at plane specified by one non-None value of x,y,z.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.

        Returns
        -------
        List[shapely.geometry.base.BaseGeometry]
            List of 2D shapes that intersect plane.
            For more details refer to
            `Shapely's Documentaton <https://shapely.readthedocs.io/en/stable/project.html>`_.
        """

        if self.mesh_dataset is None:
            return []

        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)

        origin = self.unpop_axis(position, (0, 0), axis=axis)
        normal = self.unpop_axis(1, (0, 0), axis=axis)

        mesh = self.trimesh

        try:
            section = mesh.section(plane_origin=origin, plane_normal=normal)

            if section is None:
                return []

            # homogeneous transformation matrix to map to xy plane
            mapping = np.eye(4)

            # translate to origin
            mapping[3, :3] = -np.array(origin)

            # permute so normal is aligned with z axis
            # and (y, z), (x, z), resp. (x, y) are aligned with (x, y)
            identity = np.eye(3)
            permutation = self.unpop_axis(identity[2], identity[0:2], axis=axis)
            mapping[:3, :3] = np.array(permutation).T

            section2d, _ = section.to_2D(to_2D=mapping)
            return list(section2d.polygons_full)

        except ValueError as e:
            if not mesh.is_watertight:
                log.warning(
                    "Unable to compute 'TriangleMesh.intersections_plane' "
                    "because the mesh was not watertight. Using bounding box instead. "
                    "This may be overly strict; consider using 'TriangleMesh.fill_holes' "
                    "to repair the non-watertight mesh."
                )
            else:
                log.warning(
                    "Unable to compute 'TriangleMesh.intersections_plane'. "
                    "Using bounding box instead."
                )
            log.warning(f"Error encountered: {e}")
            return self.bounding_box.intersections_plane(x=x, y=y, z=z)

    def inside(self, x: NDArray[float], y: NDArray[float], z: NDArray[float]) -> np.ndarray[bool]:
        """For input arrays ``x``, ``y``, ``z`` of arbitrary but identical shape, return an array
        with the same shape which is ``True`` for every point in zip(x, y, z) that is inside the
        volume of the :class:`Geometry`, and ``False`` otherwise.

        Parameters
        ----------
        x : NDArray[float]
            Array of point positions in x direction.
        y : NDArray[float]
            Array of point positions in y direction.
        z : NDArray[float]
            Array of point positions in z direction.

        Returns
        -------
        np.ndarray[bool]
            ``True`` for every point that is inside the geometry.
        """

        arrays = tuple(map(np.array, (x, y, z)))
        self._ensure_equal_shape(*arrays)
        arrays_flat = map(np.ravel, arrays)
        arrays_stacked = np.stack(tuple(arrays_flat), axis=-1)
        inside = self.trimesh.contains(arrays_stacked)
        return inside.reshape(arrays[0].shape)

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        ax: Ax = None,
        **patch_kwargs: Any,
    ) -> Ax:
        """Plot geometry cross section at single (x,y,z) coordinate.

        Parameters
        ----------
        x : float = None
            Position of plane in x direction, only one of x,y,z can be specified to define plane.
        y : float = None
            Position of plane in y direction, only one of x,y,z can be specified to define plane.
        z : float = None
            Position of plane in z direction, only one of x,y,z can be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.
        **patch_kwargs
            Optional keyword arguments passed to the matplotlib patch plotting of structure.
            For details on accepted values, refer to
            `Matplotlib's documentation <https://tinyurl.com/2nf5c2fk>`_.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        log.warning(
            "Plotting a 'TriangleMesh' may give inconsistent results "
            "if the mesh is not unionized. We recommend unionizing all meshes before import. "
            "A 'PermittivityMonitor' can be used to check that the mesh is loaded correctly."
        )

        return base.Geometry.plot(self, x=x, y=y, z=z, ax=ax, **patch_kwargs)

    def _compute_derivatives(self, derivative_info: DerivativeInfo) -> AutogradFieldMap:
        """Compute adjoint derivatives for a ``TriangleMesh`` geometry."""

        start_time = time.perf_counter()  # TODO remove
        vjps: AutogradFieldMap = {}

        if not self.mesh_dataset:
            raise DataError("Can't compute derivatives without mesh data.")

        valid_paths = {("mesh_dataset", "surface_mesh")}
        for path in derivative_info.paths:
            if path not in valid_paths:
                raise ValueError(f"No derivative defined w.r.t. 'TriangleMesh' field '{path}'.")

        if ("mesh_dataset", "surface_mesh") not in derivative_info.paths:
            return vjps

        triangles = np.asarray(self.triangles, dtype=config.adjoint.gradient_dtype_float)

        # early exit if geometry is completely outside simulation bounds
        sim_min, sim_max = map(np.asarray, derivative_info.simulation_bounds)
        mesh_min, mesh_max = map(np.asarray, self.bounds)
        if np.any(mesh_max < sim_min) or np.any(mesh_min > sim_max):
            log.warning(
                "'TriangleMesh' lies completely outside the simulation domain.",
                log_once=True,
            )
            zeros = np.zeros_like(triangles, dtype=config.adjoint.gradient_dtype_float)
            vjps[("mesh_dataset", "surface_mesh")] = zeros
            return vjps

        # gather surface samples within the simulation bounds
        dx = derivative_info.adaptive_vjp_spacing()
        print("dx = ", dx)
        dx = dx / 6
        samples = self._collect_surface_samples(
            triangles=triangles,
            spacing=dx,
            sim_min=sim_min,
            sim_max=sim_max,
        )

        if samples["points"].shape[0] == 0:
            zeros = np.zeros_like(triangles, dtype=config.adjoint.gradient_dtype_float)
            vjps[("mesh_dataset", "surface_mesh")] = zeros
            return vjps

        interpolators = derivative_info.interpolators
        if interpolators is None:
            interpolators = derivative_info.create_interpolators(
                dtype=config.adjoint.gradient_dtype_float
            )

        g = derivative_info.evaluate_gradient_at_points(
            samples["points"],
            samples["normals"],
            samples["perps1"],
            samples["perps2"],
            interpolators,
        )

        # accumulate per-vertex contributions using barycentric weights
        weights = (samples["weights"] * g).real
        normals = samples["normals"]
        faces = samples["faces"]
        bary = samples["barycentric"]

        contrib_vec = weights[:, None] * normals

        triangle_grads = np.zeros_like(triangles, dtype=config.adjoint.gradient_dtype_float)
        for vertex_idx in range(3):
            scaled = contrib_vec * bary[:, vertex_idx][:, None]
            np.add.at(triangle_grads[:, vertex_idx, :], faces, scaled)

        vjps[("mesh_dataset", "surface_mesh")] = triangle_grads
        duration = time.perf_counter() - start_time  # TODO REMOVE
        print(f"TriangleMesh._compute_derivatives runtime: {duration:.3f}s")
        return vjps

    def _collect_surface_samples(
        self,
        triangles: NDArray,
        spacing: float,
        sim_min: NDArray,
        sim_max: NDArray,
    ) -> dict[str, np.ndarray]:
        """Deterministic per-triangle sampling used historically."""

        dtype = config.adjoint.gradient_dtype_float
        tol = config.adjoint.edge_clip_tolerance

        sim_min = np.asarray(sim_min, dtype=dtype)
        sim_max = np.asarray(sim_max, dtype=dtype)

        points_list: list[NDArray] = []
        normals_list: list[NDArray] = []
        perps1_list: list[NDArray] = []
        perps2_list: list[NDArray] = []
        weights_list: list[NDArray] = []
        faces_list: list[NDArray] = []
        bary_list: list[NDArray] = []

        spacing = max(float(spacing), np.finfo(float).eps)
        triangles_arr = np.asarray(triangles, dtype=dtype)

        sim_extents = sim_max - sim_min
        collapsed_axes = np.isclose(sim_extents, 0.0, atol=tol)
        per_unit_scale = 1.0
        if np.any(collapsed_axes):
            coords = triangles_arr.reshape(-1, 3)
            geom_min = np.min(coords, axis=0)
            geom_max = np.max(coords, axis=0)
            geom_extents = geom_max - geom_min
            collapsed_extents = np.maximum(geom_extents[collapsed_axes], tol)
            per_unit_scale = float(np.prod(collapsed_extents))

        for face_index, tri in enumerate(triangles_arr):
            area, normal = self._triangle_area_and_normal(tri)
            if area <= AREA_SIZE_THRESHOLD:
                continue

            perps = self._triangle_tangent_basis(tri, normal)
            if perps is None:
                continue
            perp1, perp2 = perps

            barycentric = self._get_barycentric_samples(tri, spacing, dtype)
            num_samples = barycentric.shape[0]
            base_weight = area / num_samples
            if per_unit_scale != 1.0:
                base_weight /= per_unit_scale

            sample_points = barycentric @ tri

            valid_axes = np.abs(sim_max - sim_min) > tol
            inside_mask = np.all(
                sample_points[:, valid_axes] >= (sim_min - tol)[valid_axes], axis=1
            ) & np.all(sample_points[:, valid_axes] <= (sim_max + tol)[valid_axes], axis=1)
            if not np.any(inside_mask):
                continue

            sample_points = sample_points[inside_mask]
            bary_inside = barycentric[inside_mask]
            n_samples_inside = sample_points.shape[0]

            normal_tile = np.repeat(normal[None, :], n_samples_inside, axis=0)
            perp1_tile = np.repeat(perp1[None, :], n_samples_inside, axis=0)
            perp2_tile = np.repeat(perp2[None, :], n_samples_inside, axis=0)
            weights_tile = np.full(n_samples_inside, base_weight, dtype=dtype)
            faces_tile = np.full(n_samples_inside, face_index, dtype=int)

            points_list.append(sample_points)
            normals_list.append(normal_tile)
            perps1_list.append(perp1_tile)
            perps2_list.append(perp2_tile)
            weights_list.append(weights_tile)
            faces_list.append(faces_tile)
            bary_list.append(bary_inside)

        if not points_list:
            return {
                "points": np.zeros((0, 3), dtype=dtype),
                "normals": np.zeros((0, 3), dtype=dtype),
                "perps1": np.zeros((0, 3), dtype=dtype),
                "perps2": np.zeros((0, 3), dtype=dtype),
                "weights": np.zeros((0,), dtype=dtype),
                "faces": np.zeros((0,), dtype=int),
                "barycentric": np.zeros((0, 3), dtype=dtype),
            }

        return {
            "points": np.concatenate(points_list, axis=0),
            "normals": np.concatenate(normals_list, axis=0),
            "perps1": np.concatenate(perps1_list, axis=0),
            "perps2": np.concatenate(perps2_list, axis=0),
            "weights": np.concatenate(weights_list, axis=0),
            "faces": np.concatenate(faces_list, axis=0),
            "barycentric": np.concatenate(bary_list, axis=0),
        }

    @staticmethod
    def _triangle_area_and_normal(triangle: NDArray) -> tuple[float, np.ndarray]:
        """Return area and outward normal of the provided triangle."""

        edge01 = triangle[1] - triangle[0]
        edge02 = triangle[2] - triangle[0]
        cross = np.cross(edge01, edge02)
        norm = np.linalg.norm(cross)
        if norm <= 0.0:
            return 0.0, np.zeros(3, dtype=triangle.dtype)
        normal = (cross / norm).astype(triangle.dtype, copy=False)
        area = 0.5 * norm
        return area, normal

    @staticmethod
    def _edge_heights(triangle: NDArray) -> tuple[float, float, float]:
        """Return heights from each vertex to its opposing edge."""
        tri = np.asarray(triangle, float)
        heights: list[float] = []

        for i in range(3):
            v = tri[i]
            e0 = tri[(i + 1) % 3]
            e1 = tri[(i + 2) % 3]

            edge = e1 - e0
            L = float(np.linalg.norm(edge))

            if L < 1e-12:
                heights.append(0.0)
                continue

            h = float(np.linalg.norm(np.cross(edge, v - e0)) / L)
            heights.append(h)

        return tuple(heights)

    @classmethod
    def _subdivision_count(
        cls,
        area: float,
        spacing: float,
        edge_lengths: Optional[tuple[float, float, float]] = None,
        triangle: Optional[NDArray] = None,
    ) -> tuple[int, int, int]:
        """Determine per-vertex subdivisions based on orthogonal heights."""

        spacing = max(float(spacing), np.finfo(float).eps)

        if triangle is not None:
            heights = cls._edge_heights(triangle)
        elif edge_lengths is not None:
            heights = edge_lengths
        else:
            target = np.sqrt(max(area, 0.0))
            heights = (target, target, target)

        counts = []
        for height in heights:
            if height <= 0.0:
                counts.append(1)
            else:
                counts.append(max(1, int(np.ceil(height / spacing))))
        return tuple(counts)

    def _get_barycentric_samples(
        self,
        triangle: NDArray,
        spacing: float,
        dtype: np.dtype,
    ) -> np.ndarray:
        """Return barycentric sample coordinates for a triangle.

        This uses uniform geometric spacing along rows aligned to the
        largest height direction, with no points on the triangle edges.
        Results are cached per (row layout, primary bary index).
        """

        tri = np.asarray(triangle, float)
        spacing = max(float(spacing), np.finfo(float).eps)

        # 1) measure heights and choose primary index
        heights = self._edge_heights(tri)
        # primary bary index = vertex with largest height
        primary = int(np.argmax(heights))
        h_max = heights[primary]

        # number of rows in that direction
        height_count = max(1, int(np.ceil(h_max / spacing)))

        # 2) for each row, compute physical row length and how many points we want
        row_counts: list[int] = []
        other0 = (primary + 1) % 3
        other1 = (primary + 2) % 3

        for hi in range(height_count):
            lam_p = (hi + 0.5) / height_count
            lam_p = min(lam_p, 1.0 - 1e-9)
            rem = max(1e-9, 1.0 - lam_p)

            lam_start = np.zeros(3)
            lam_end = np.zeros(3)

            lam_start[primary] = lam_p
            lam_start[other0] = 0.0
            lam_start[other1] = rem

            lam_end[primary] = lam_p
            lam_end[other0] = rem
            lam_end[other1] = 0.0

            p0 = lam_start @ tri
            p1 = lam_end @ tri
            L = float(np.linalg.norm(p1 - p0))

            if L < 1e-12:
                # degenerate row; still keep a single point to avoid dropping it
                row_counts.append(1)
            else:
                npoints = max(1, int(np.ceil(L / spacing)))
                row_counts.append(npoints)

        row_counts_t = tuple(row_counts)
        key = (primary, height_count, row_counts_t)

        cache = self._barycentric_cache
        if key not in cache:
            cache[key] = self._build_barycentric_pattern(
                primary=primary,
                height_count=height_count,
                row_counts=row_counts_t,
            )

        return cache[key].astype(dtype, copy=False)

    @staticmethod
    def _build_barycentric_pattern(
        primary: int,
        height_count: int,
        row_counts: tuple[int, ...],
    ) -> np.ndarray:
        """Construct barycentric sampling points for a given row layout.

        Parameters
        ----------
        primary:
            Barycentric index used as the 'height' direction (0, 1, or 2).
        height_count:
            Number of rows in the primary direction.
        row_counts:
            Number of points per row (same length as height_count).

        Returns
        -------
        np.ndarray
            Array of shape (N, 3) with unique barycentric coordinates,
            each strictly inside the triangle (no component exactly 0 or 1).
        """

        if len(row_counts) != height_count:
            raise ValueError(
                f"row_counts length ({len(row_counts)}) must equal height_count ({height_count})."
            )

        bary: list[tuple[float, float, float]] = []

        other0 = (primary + 1) % 3
        other1 = (primary + 2) % 3

        for hi in range(height_count):
            npoints = max(1, int(row_counts[hi]))

            # row position in barycentric primary coordinate (interior-only)
            lam_p = (hi + 0.5) / height_count
            lam_p = min(lam_p, 1.0 - 1e-9)
            rem = max(1e-9, 1.0 - lam_p)

            lam_start = np.zeros(3)
            lam_end = np.zeros(3)

            lam_start[primary] = lam_p
            lam_start[other0] = 0.0
            lam_start[other1] = rem

            lam_end[primary] = lam_p
            lam_end[other0] = rem
            lam_end[other1] = 0.0

            # symmetric interior-only positions along the row
            ts = (np.arange(npoints, dtype=float) + 0.5) / float(npoints)

            for t in ts:
                lam = lam_start * (1.0 - t) + lam_end * t
                bary.append((float(lam[0]), float(lam[1]), float(lam[2])))

        arr = np.asarray(bary, dtype=float)
        return arr

    @staticmethod
    def subdivide_faces(vertices: NDArray, faces: NDArray) -> tuple[np.ndarray, np.ndarray]:
        """Uniformly subdivide each triangular face by inserting edge midpoints."""

        midpoint_cache: dict[tuple[int, int], int] = {}
        verts_list = [np.asarray(v, dtype=float) for v in vertices]

        def midpoint(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            if key in midpoint_cache:
                return midpoint_cache[key]
            vm = 0.5 * (verts_list[i] + verts_list[j])
            verts_list.append(vm)
            idx = len(verts_list) - 1
            midpoint_cache[key] = idx
            return idx

        new_faces: list[tuple[int, int, int]] = []
        for tri in faces:
            a = midpoint(tri[0], tri[1])
            b = midpoint(tri[1], tri[2])
            c = midpoint(tri[2], tri[0])
            new_faces.extend(((tri[0], a, c), (tri[1], b, a), (tri[2], c, b), (a, b, c)))

        verts_arr = np.asarray(verts_list, dtype=float)
        return verts_arr, np.asarray(new_faces, dtype=int)

    @staticmethod
    def _triangle_tangent_basis(
        triangle: NDArray, normal: NDArray
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Compute orthonormal tangential vectors for a triangle."""

        tol = np.finfo(triangle.dtype).eps
        edges = [triangle[1] - triangle[0], triangle[2] - triangle[0], triangle[2] - triangle[1]]

        edge = None
        for candidate in edges:
            length = np.linalg.norm(candidate)
            if length > tol:
                edge = (candidate / length).astype(triangle.dtype, copy=False)
                break

        if edge is None:
            return None

        perp1 = edge
        perp2 = np.cross(normal, perp1)
        perp2_norm = np.linalg.norm(perp2)
        if perp2_norm <= tol:
            return None
        perp2 = (perp2 / perp2_norm).astype(triangle.dtype, copy=False)
        return perp1, perp2

    @staticmethod
    def _barycentric_from_points(
        triangles: NDArray, points: NDArray, dtype: np.dtype
    ) -> np.ndarray:
        """Compute barycentric coordinates for points relative to their triangles."""

        v0 = triangles[:, 1] - triangles[:, 0]
        v1 = triangles[:, 2] - triangles[:, 0]
        v2 = points - triangles[:, 0]

        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)

        denom = d00 * d11 - d01 * d01
        tol = np.finfo(dtype).eps
        denom_safe = np.where(np.abs(denom) <= tol, 1.0, denom)

        v = (d11 * d20 - d01 * d21) / denom_safe
        w = (d00 * d21 - d01 * d20) / denom_safe
        u = 1.0 - v - w

        bary = np.stack([u, v, w], axis=1).astype(dtype, copy=False)
        degenerate = np.abs(denom) <= tol
        if np.any(degenerate):
            bary[degenerate] = 1.0 / 3.0
        return bary
