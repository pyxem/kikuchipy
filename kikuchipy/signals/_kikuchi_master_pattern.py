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

from copy import deepcopy
import logging
from typing import Any, Optional, Tuple, Union
from warnings import warn

import hyperspy.api as hs
import numpy as np
from orix.crystal_map import Phase
from orix.projections import StereographicProjection
from orix.vector import Vector3d
from scipy.interpolate import interpn
from tqdm import tqdm

from kikuchipy.signals._kikuchipy_signal import KikuchipySignal2D
from kikuchipy.signals.util._master_pattern import _lambert2vector


_logger = logging.getLogger(__name__)


class KikuchiMasterPattern(KikuchipySignal2D, hs.signals.Signal2D):
    """General class for Kikuchi master patterns.

    Not meant to be used directly, see derived classes like
    :class:`~kikuchipy.signals.EBSDMasterPattern`.

    Parameters
    ----------
    *args
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    hemisphere : str
        Which hemisphere the data contains, either ``"upper"``,
        ``"lower"``, or ``"both"``.
    phase : ~orix.crystal_map.Phase
        The phase describing the crystal structure used in the master
        pattern simulation.
    projection : str
        Which projection the pattern is in, ``"stereographic"`` or
        ``"lambert"``.
    **kwargs
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    """

    _custom_properties = ["hemisphere", "phase", "projection"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hemisphere = kwargs.get("hemisphere")
        self._phase = kwargs.get("phase", Phase())
        self._projection = kwargs.get("projection")

    @property
    def _has_multiple_energies(self) -> bool:
        return "energy" in [i.name for i in self.axes_manager.navigation_axes]

    @property
    def hemisphere(self) -> str:
        """Return or set which hemisphere(s) the signal contains.

        Options are ``"upper"``, ``"lower"`` or ``"both"``.

        Parameters
        ----------
        value : str
            Which hemisphere(s) the signal contains.
        """
        return self._hemisphere

    @hemisphere.setter
    def hemisphere(self, value: str):
        self._hemisphere = value

    @property
    def phase(self) -> Phase:
        """Return or set the phase describing the crystal structure used
        in the master pattern simulation.

        Parameters
        ----------
        value : ~orix.crystal_map.Phase
            The phase used in the master pattern simulation.
        """
        return self._phase

    @phase.setter
    def phase(self, value: Phase):
        self._phase = value

    @property
    def projection(self) -> str:
        """Return or set which projection the pattern is in.

        Parameters
        ----------
        value : str
            Which projection the pattern is in, either
            ``"stereographic"`` or ``"lambert"``.
        """
        return self._projection

    @projection.setter
    def projection(self, value: str):
        self._projection = value

    def as_lambert(self, show_progressbar: Optional[bool] = None) -> Any:
        """Return a new master pattern in the Lambert projection
        :cite:`callahan2013dynamical`.

        Only implemented for non-lazy signals.

        Returns
        -------
        lambert_master_pattern
            Master pattern in the Lambert projection with the same data
            shape but in 32-bit floating point data dtype.

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> import kikuchipy as kp
        >>> mp_sp = kp.data.nickel_ebsd_master_pattern_small()
        >>> mp_sp.projection
        'stereographic'
        >>> mp_lp = mp_sp.as_lambert()
        >>> mp_lp.projection
        'lambert'
        >>> _ = hs.plot.plot_images([mp_sp, mp_lp], per_row=2)
        """
        if self.projection == "lambert":
            warn("Already in the Lambert projection, returning a deepcopy", UserWarning)
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

        iterable = np.ndindex(nav_shape[::-1])
        if show_progressbar or (
            show_progressbar is None and hs.preferences.General.show_progressbar
        ):
            iterable = tqdm(iterable, total=n_iterations)

        for idx in iterable:
            data_i = interpn(values=self.data[idx], **kwargs)
            data_out[idx] = data_i.reshape(sig_shape)

        return self.__class__(
            data_out,
            axes=list(self.axes_manager.as_dictionary().values()),
            phase=self.phase.deepcopy(),
            projection="lambert",
            hemisphere=deepcopy(self.hemisphere),
        )

    def plot_spherical(
        self,
        energy: Union[int, float, None] = None,
        return_figure: bool = False,
        style: str = "surface",
        plotter_kwargs: Union[dict] = None,
        show_kwargs: Union[dict] = None,
    ) -> "pyvista.Plotter":
        """Plot the master pattern sphere.

        This requires the master pattern to be in the stereographic
        projection and both hemispheres to be present.

        Parameters
        ----------
        energy
            Acceleration voltage in kV used to simulate the master
            pattern to plot. If not given, the highest energy is used.
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
            :meth:`pyvista.Plotter.show` if ``return_figure=False``.

        Returns
        -------
        pl
            Only returned if ``return_figure=True``.

        Notes
        -----
        Requires :mod:`pyvista` (see :ref:`the installation guide
        <optional-dependencies>`).

        Examples
        --------
        >>> import kikuchipy as kp
        >>> mp = kp.data.nickel_ebsd_master_pattern_small(projection="stereographic")
        >>> mp.plot_spherical()
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
        master_upper
            Upper hemisphere of the master pattern.
        master_lower
            Lower hemisphere of master pattern.
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

    # --- Inherited methods from KikuchipySignal2D overwritten

    def deepcopy(self) -> Any:
        """Return a deep copy using :func:`copy.deepcopy`.

        Returns
        -------
        s_new
            Identical signal without shared memory.

        Examples
        --------
        >>> import numpy as np
        >>> import kikuchipy as kp
        >>> mp = kp.data.nickel_ebsd_master_pattern_small()
        >>> mp2 = mp.deepcopy()
        >>> np.may_share_memory(mp.data, mp2.data)
        False
        """
        return super().deepcopy()

    # --- Inherited methods from Signal2D overwritten

    def set_signal_type(self, signal_type: str = "") -> None:
        if "master" in signal_type.lower():
            properties = self._get_custom_properties()
            super().set_signal_type(signal_type)
            self._set_custom_properties(properties)
        else:
            properties = self._custom_properties
            super().set_signal_type(signal_type)
            _logger.info("Delete custom properties when setting signal type")
            for name in properties:
                try:
                    self.__delattr__("_" + name)
                except AttributeError:  # pragma: no cover
                    pass
