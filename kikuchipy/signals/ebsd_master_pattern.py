# Copyright 2019-2022 The kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.

import copy
import gc
import sys
from typing import Tuple, Union
import warnings

import dask.array as da
from dask.diagnostics import ProgressBar
from hyperspy._lazy_signals import LazySignal2D
from hyperspy._signals.signal2d import Signal2D
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.projections import StereographicProjection
from orix.quaternion import Rotation
from orix.vector import Vector3d
from skimage.util.dtype import dtype_range
from scipy.interpolate import interpn
from tqdm import tqdm

from kikuchipy.detectors.ebsd_detector import EBSDDetector
from kikuchipy.signals import LazyEBSD, EBSD
from kikuchipy.signals._common_image import CommonImage
from kikuchipy.signals.util._dask import get_chunking
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_for_single_pc_from_detector,
    _lambert2vector,
    _project_patterns_from_master_pattern,
)


class EBSDMasterPattern(CommonImage, Signal2D):
    """Simulated Electron Backscatter Diffraction (EBSD) master pattern.

    This class extends HyperSpy's Signal2D class for EBSD master
    patterns. Methods inherited from HyperSpy can be found in the
    HyperSpy user guide. See the docstring of
    :class:`hyperspy.signal.BaseSignal` for a list of additional
    attributes.

    Attributes
    ----------
    projection : str
        Which projection the pattern is in, ``"stereographic"`` or
        ``"lambert"``.
    hemisphere : str
        Which hemisphere the data contains: ``"upper"`` (previously
        ``"north"``), ``"lower"`` (previously ``"south"``) or
        ``"both"``.
    phase : orix.crystal_map.Phase
        Phase describing the crystal structure used in the master
        pattern simulation.

    """

    _signal_type = "EBSDMasterPattern"
    _alias_signal_types = ["ebsd_master_pattern", "master_pattern"]
    _lazy = False

    # ---------------------- Custom properties ----------------------- #

    phase = Phase()
    projection = None
    _hemisphere = None

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
        self.phase = kwargs.pop("phase", Phase())
        self.projection = kwargs.pop("projection", None)
        self._hemisphere = kwargs.pop("hemisphere", None)

    @property
    def _has_multiple_energies(self):
        return "energy" in [i.name for i in self.axes_manager.navigation_axes]

    @property
    def hemisphere(self) -> str:
        if self._hemisphere in ["upper", "north"]:
            return "upper"
        elif self._hemisphere in ["lower", "south"]:
            return "lower"
        else:
            return self._hemisphere

    @hemisphere.setter
    def hemisphere(self, value: str):
        if value in ["north", "south"]:
            # TODO: Remove warning after 0.6 is released
            warnings.warn(
                (
                    "`hemisphere` parameter options 'north' and 'south' are deprecated "
                    "and will raise an error in version 0.7, use 'upper' and 'lower'"
                    " instead. Changed to 'upper' or 'lower'."
                ),
                np.VisibleDeprecationWarning,
            )
            if value == "north":
                value = "upper"
            else:
                value = "lower"
        self._hemisphere = value

    def get_patterns(
        self,
        rotations: Rotation,
        detector: EBSDDetector,
        energy: Union[int, float],
        dtype_out: Union[type, np.dtype] = np.float32,
        compute: bool = False,
        **kwargs,
    ) -> Union[EBSD, LazyEBSD]:
        """Return a dictionary of EBSD patterns projected onto a
        detector from a master pattern in the square Lambert
        projection :cite:`callahan2013dynamical`, for a set of crystal
        rotations relative to the EDAX TSL sample reference frame (RD,
        TD, ND) and a fixed detector-sample geometry.

        Parameters
        ----------
        rotations
            Crystal rotations to get patterns from. The shape of this
            instance, a maximum of two dimensions, determines the
            navigation shape of the output signal.
        detector
            EBSD detector describing the detector dimensions and the
            detector-sample geometry with a single, fixed
            projection/pattern center.
        energy
            Acceleration voltage, in kV, used to simulate the desired
            master pattern to create a dictionary from. If only a single
            energy is present in the signal, this will be returned no
            matter its energy.
        dtype_out
            Data type of the returned patterns, by default np.float32.
        compute
            Whether to return a lazy result, by default False. For more
            information see :func:`~dask.array.Array.compute`.
        kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.signals.util.get_chunking` to control the
            number of chunks the dictionary creation and the output data
            array is split into. Only `chunk_shape`, `chunk_bytes` and
            `dtype_out` (to `dtype`) are passed on.

        Returns
        -------
        EBSD or LazyEBSD
            Signal with navigation and signal shape equal to the
            rotation instance and detector shape, respectively.

        Notes
        -----
        If the master pattern phase has a non-centrosymmetric point
        group, both the upper and lower hemispheres must be provided.
        For more details regarding the reference frame visit the
        reference frame user guide.
        """
        self._is_suitable_for_projection(raise_if_not=True)

        if len(detector.pc) > 1:
            raise NotImplementedError(
                "Detector must have exactly one projection center"
            )

        # Get suitable chunks when iterating over the rotations. Signal
        # axes are not chunked.
        nav_shape = rotations.shape
        nav_dim = len(nav_shape)
        if nav_dim > 2:
            raise ValueError(
                "`rotations` can only have one or two dimensions, but an instance with "
                f"{nav_dim} dimensions was passed"
            )
        data_shape = nav_shape + detector.shape
        chunks = get_chunking(
            data_shape=data_shape,
            nav_dim=nav_dim,
            sig_dim=len(detector.shape),
            chunk_shape=kwargs.pop("chunk_shape", None),
            chunk_bytes=kwargs.pop("chunk_bytes", None),
            dtype=dtype_out,
        )

        # Whether to rescale pattern intensities after projection
        if dtype_out != self.data.dtype:
            rescale = True
            if isinstance(dtype_out, np.dtype):
                dtype_out = dtype_out.type
            out_min, out_max = dtype_range[dtype_out]
        else:
            rescale = False
            # Cannot be None due to Numba, so they are set to something
            # here. Values aren't used unless `rescale` is True.
            out_min, out_max = 1, 2

        # Get direction cosines for each detector pixel relative to the
        # source point
        direction_cosines = _get_direction_cosines_for_single_pc_from_detector(detector)

        # Get dask array from rotations
        rot_da = da.from_array(rotations.data, chunks=chunks[:nav_dim] + (-1,))

        # Which axes to drop and add when iterating over the rotations
        # dask array to produce the EBSD signal array, i.e. drop the
        # (4,)-shape quaternion axis and add detector shape axes, e.g.
        # (60, 60)
        if nav_dim == 1:
            drop_axis = 1
            new_axis = (1, 2)
        else:  # nav_dim == 2
            drop_axis = 2
            new_axis = (2, 3)

        master_upper, master_lower = self._get_master_pattern_arrays_from_energy(energy)

        # Project simulated patterns onto detector
        npx, npy = self.axes_manager.signal_shape
        scale = (npx - 1) / 2
        simulated = rot_da.map_blocks(
            _project_patterns_from_master_pattern,
            direction_cosines=direction_cosines,
            master_upper=master_upper,
            master_lower=master_lower,
            npx=int(npx),
            npy=int(npy),
            scale=float(scale),
            dtype_out=dtype_out,
            rescale=rescale,
            out_min=out_min,
            out_max=out_max,
            drop_axis=drop_axis,
            new_axis=new_axis,
            chunks=chunks,
            dtype=dtype_out,
        )

        # Add crystal map and detector to keyword arguments
        kwargs = dict(
            xmap=CrystalMap(phase_list=PhaseList(self.phase), rotations=rotations),
            detector=detector,
        )

        # Specify navigation and signal axes for signal initialization
        names = ["y", "x", "dy", "dx"]
        scales = np.ones(4)
        ndim = simulated.ndim
        if ndim == 3:
            names = names[1:]
            scales = scales[1:]
        axes = [
            dict(
                size=data_shape[i],
                index_in_array=i,
                name=names[i],
                scale=scales[i],
                offset=0.0,
                units="px",
            )
            for i in range(ndim)
        ]

        if compute:
            patterns = np.zeros(shape=simulated.shape, dtype=simulated.dtype)
            with ProgressBar():
                print(
                    f"Creating a dictionary of {nav_shape} simulated patterns:",
                    file=sys.stdout,
                )
                simulated.store(patterns, compute=True)
            out = EBSD(patterns, axes=axes, **kwargs)
        else:
            out = LazyEBSD(simulated, axes=axes, **kwargs)
        gc.collect()

        return out

    def plot_spherical(
        self,
        energy: Union[int, float, None] = None,
        return_figure: bool = False,
        style: str = "surface",
        plotter_kwargs: Union[dict] = None,
        show_kwargs: Union[dict] = None,
    ):
        """Plot the master pattern sphere.

        This requires the master pattern to be in the stereographic
        projection and both hemispheres to be present.

        Parameters
        ----------
        energy
            Acceleration voltage in kV used to simulate the master
            pattern to plot. If not passed, the highest energy is used.
        return_figure
            Whether to return the :class:`pyvista.Plotter` instance for
            further modification and then plotting. Default is
            ``False``. If ``True``, the figure is not plotted.
        style
            Visualization style of the mesh, either ``"surface"``
            (default), ``"wireframe"`` or ``"points"``. In general,
            ``"surface"`` is recommended when zoomed out, while
            ``"points"`` is recommended when zoomed in. See
            :meth:`pyvista.Plotter.add_mesh` for details.
        plotter_kwargs
            Dictionary of keyword arguments passed to
            :class:`pyvista.Plotter`.
        show_kwargs
            Dictionary of keyword arguments passed to
            :meth:`pyvista.Plotter.show` if ``return_figure`` is
            ``False``.

        Returns
        -------
        pl : pyvista.Plotter
            Only returned if ``return_figure`` is ``True``.

        Notes
        -----
        Requires :mod:`pyvista`.

        Examples
        --------
        >>> import kikuchipy as kp
        >>> mp = kp.data.nickel_ebsd_master_pattern_small(projection="stereographic")
        >>> mp.plot_spherical()  # doctest: +SKIP
        """
        from kikuchipy import _pyvista_installed

        if not _pyvista_installed:  # pragma: no cover
            raise ImportError(
                "`pyvista` is required, see the installation guide for more information"
            )
        else:
            from orix.projections import InverseStereographicProjection
            from orix.vector import Vector3d
            import pyvista as pv

        if self.projection != "stereographic" or (
            self.hemisphere != "both" and not self.phase.point_group.contains_inversion
        ):
            raise ValueError(
                "Master pattern must be in the stereographic projection, and have both "
                "hemispheres present if the phase is non-centrosymmetric"
            )

        mp_upper, mp_lower = self._get_master_pattern_arrays_from_energy(energy)

        # Remove data outside equator and combine into a 1D array
        keep = mp_upper != 0
        data = np.ravel(np.stack((mp_upper[keep], mp_lower[keep])))

        # Get vectors for upper and lower hemispheres into a 1D array
        size = mp_upper.shape[0]
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        x = x[keep]
        y = y[keep]
        stereo2sphere = InverseStereographicProjection()
        v1 = stereo2sphere.xy2vector(x.ravel(), y.ravel())
        stereo2sphere.pole = 1
        v2 = stereo2sphere.xy2vector(x.ravel(), y.ravel())
        v3 = Vector3d.stack((v1, v2)).flatten()

        grid = pv.StructuredGrid(v3.x, v3.y, v3.z)
        grid.point_data["Intensity"] = data

        if plotter_kwargs is None:
            plotter_kwargs = {}
        pl = pv.Plotter(**plotter_kwargs)
        pl.add_mesh(pv.Sphere(radius=0.99), color="gray", lighting=False)
        pl.add_mesh(grid, style=style, scalar_bar_args=dict(color="k"), cmap="gray")
        pl.add_axes(color="k", xlabel="e1", ylabel="e2", zlabel="e3")
        pl.set_background("#fafafa")
        pl.set_viewup((0, 1, 0))

        if return_figure:
            return pl
        else:
            if show_kwargs is None:
                show_kwargs = {}
            pl.show(**show_kwargs)

    def as_lambert(self):
        """Return a new master pattern in the Lambert projection.

        Only implemented for non-lazy signals.

        Returns
        -------
        EBSDMasterPattern
            Master pattern in the Lambert projection with the same data
            shape but in 32-bit floating point data dtype.

        Examples
        --------
        >>> import kikuchipy as kp
        >>> mp_sp = kp.data.nickel_ebsd_master_pattern_small()
        >>> mp_sp.projection
        'stereographic'
        >>> mp_lp = mp_sp.as_lambert()
        >>> mp_lp.projection
        'lambert'
        """
        if self.projection == "lambert":
            warnings.warn(
                "Already in the Lambert projection, returning a deepcopy", UserWarning
            )
            return self.deepcopy()

        if self._lazy is True:
            raise NotImplementedError("Only implemented for non-lazy signals")

        # Set up square arrays
        sig_shape = self.axes_manager.signal_shape[::-1]
        arr = np.linspace(-1, 1, sig_shape[0], dtype=np.float64)
        x_lambert, y_lambert = np.meshgrid(arr, arr)
        x_lambert_flat = x_lambert.ravel()
        y_lambert_flat = y_lambert.ravel()

        # Get unit vectors per array coordinate, and then the
        # corresponding (X, Y) coordinate in the stereographic
        # projection
        xyz_upper = _lambert2vector(x_lambert_flat, y_lambert_flat)
        v = Vector3d(xyz_upper)
        sp = StereographicProjection()
        x_stereo, y_stereo = sp.vector2xy(v)
        x_stereo += 1
        y_stereo += 1

        # Keyword arguments for interpolation
        kwargs = {
            "points": (arr + 1, arr + 1),
            "xi": (y_stereo, x_stereo),
            "method": "splinef2d",
        }

        nav_shape = self.axes_manager.navigation_shape
        data_out = np.zeros(self.data.shape, dtype=np.float32)

        n_iterations = self.axes_manager.navigation_size
        if n_iterations == 0:
            n_iterations = 1

        for idx in tqdm(np.ndindex(nav_shape[::-1]), total=n_iterations):
            data_i = interpn(values=self.data[idx], **kwargs)
            data_out[idx] = data_i.reshape(sig_shape)

        return self.__class__(
            data_out,
            axes=list(self.axes_manager.as_dictionary().values()),
            phase=self.phase.deepcopy(),
            projection="lambert",
            hemisphere=copy.deepcopy(self.hemisphere),
        )

    # ------ Methods overwritten from hyperspy.signals.Signal2D ------ #

    def deepcopy(self):
        new = super().deepcopy()
        new.phase = self.phase.deepcopy()
        new.projection = copy.deepcopy(self.projection)
        new.hemisphere = copy.deepcopy(self.hemisphere)
        return new

    # ------------------------ Private methods ----------------------- #

    def _get_master_pattern_arrays_from_energy(
        self, energy: Union[int, float, None] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return upper and lower master patterns created with a single,
        given energy.

        Parameters
        ----------
        energy
            Acceleration voltage in kV. If only a single energy is
            present in the signal, this will be returned no matter its
            energy. If not given, the highest energy is used.

        Returns
        -------
        master_upper, master_lower
            Upper and lower hemispheres of master pattern.
        """
        if self._has_multiple_energies:
            if energy is None:
                energy = self.axes_manager["energy"].axis[-1]
            master_patterns = self.inav[float(energy)].data
        else:  # Assume a single energy
            master_patterns = self.data
        if self.hemisphere == "both":
            master_upper, master_lower = master_patterns
        else:
            master_upper = master_lower = master_patterns
        return master_upper, master_lower

    def _is_suitable_for_projection(self, raise_if_not: bool = False):
        """Check whether the master pattern is suitable for projection
        onto an EBSD detector and return a bool or raise an error
        message if desired.
        """
        suitable = True
        error = None
        if self.projection != "lambert":
            error = NotImplementedError(
                "Master pattern must be in the square Lambert projection"
            )
            suitable = False
        pg = self.phase.point_group
        if pg is None:
            error = AttributeError(
                "Master pattern `phase` attribute must have a valid point group"
            )
            suitable = False
        elif self.hemisphere != "both" and not pg.contains_inversion:
            error = AttributeError(
                "For point groups without inversion symmetry, like the current "
                f"{pg.name}, both hemispheres must be present in the master pattern "
                "signal"
            )
            suitable = False
        if not suitable and raise_if_not:
            raise error
        else:
            return suitable


class LazyEBSDMasterPattern(EBSDMasterPattern, LazySignal2D):
    """Lazy implementation of the :class:`EBSDMasterPattern` class.

    This class extends HyperSpy's LazySignal2D class for EBSD master
    patterns. Methods inherited from HyperSpy can be found in the
    HyperSpy user guide. See docstring of :class:`EBSDMasterPattern`
    for attributes and methods.

    """

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
