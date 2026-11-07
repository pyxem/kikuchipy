#
# Copyright 2019-2026 the kikuchipy developers
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
#

"""Base signal class for Kikuchi master patterns of specific diffraction
modailities to subclass.

This class has functionality applicable to all subclasses.
"""

import logging
from typing import TYPE_CHECKING
from warnings import warn

import hyperspy.api as hs
import numpy as np
import orix.crystal_map as ocm
import orix.projections as opr
import orix.vector as ove
from tqdm import tqdm
from typing_extensions import Self

from kikuchipy._constants import verify_dependency_or_raise
from kikuchipy._kikuchi_sphere._lambert_projection import (
    _lambert_to_stereo_coords,
    _project_lambert_to_stereo,
    _project_stereo_to_lambert,
    _stereo_to_lambert_coords,
)
from kikuchipy._utils.vector import (
    ValidHemispheres,
    ValidProjections,
    parse_hemisphere,
    parse_projection,
)
from kikuchipy.signals._kikuchipy_signal import KikuchipySignal2D
from kikuchipy.signals.util._overwrite_hyperspy_methods import insert_doc_disclaimer

if TYPE_CHECKING:  # pragma: no cover
    from pyvista import Plotter


_logger = logging.getLogger(__name__)


class KikuchiMasterPattern(KikuchipySignal2D, hs.signals.Signal2D):
    """Base class for Kikuchi master patterns.

    Not meant to be used directly, see derived classes like
    :class:`~kikuchipy.signals.EBSDMasterPattern`.

    Parameters
    ----------
    *args
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    hemisphere : str
        Which hemisphere the data contains, either "upper", "lower", or
        "both".
    phase : ~orix.crystal_map.Phase
        The phase describing the crystal structure used in the master
        pattern simulation.
    projection : str
        Which projection the pattern is in, "stereographic" or
        "lambert".
    **kwargs
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    """

    _custom_attributes = ["hemisphere", "phase", "projection"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hemisphere: ValidHemispheres | None = kwargs.get("hemisphere")
        self._phase = kwargs.get("phase", ocm.Phase())
        self._projection: ValidProjections | None = kwargs.get("projection")

    # -------------------------- Properties -------------------------- #

    @property
    def hemisphere(self) -> ValidHemispheres | None:
        """Return or set which hemisphere(s) the signal contains.

        Options are "upper", "lower", or "both".

        Parameters
        ----------
        value : str
            Which hemisphere(s) the signal contains.
        """
        return self._hemisphere

    @hemisphere.setter
    def hemisphere(self, value: ValidHemispheres) -> None:
        self._hemisphere = parse_hemisphere(value)

    @property
    def phase(self) -> ocm.Phase:
        """Return or set the phase describing the crystal structure used
        in the master pattern simulation.

        Parameters
        ----------
        value : ~orix.crystal_map.Phase
            The phase used in the master pattern simulation.
        """
        return self._phase

    @phase.setter
    def phase(self, value: ocm.Phase) -> None:
        self._phase = value

    @property
    def projection(self) -> ValidProjections | None:
        """Return or set which projection the pattern is in.

        Parameters
        ----------
        value : str
            Which projection the pattern is in, either "stereographic"
            or "lambert".
        """
        return self._projection

    @projection.setter
    def projection(self, value: ValidProjections) -> None:
        self._projection = parse_projection(value)

    @property
    def _has_multiple_energies(self) -> bool:
        return "energy" in [i.name for i in self.axes_manager.navigation_axes]

    # --------------------------- Methods ---------------------------- #

    def as_lambert(self, show_progressbar: bool | None = None) -> Self:
        """Return a new master pattern in the Lambert projection
        :cite:`callahan2013dynamical`.

        Only implemented for non-lazy signals.

        Parameters
        ----------
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.

        Returns
        -------
        master_pattern
            Master pattern in the Lambert projection with the same data
            shape, but in 32-bit floating point data dtype.

        Raises
        ------
        ValueError
            If :attr:`projection` is "lambert".
        """
        if self.projection == "lambert":
            raise ValueError("Master pattern already in the Lambert projection")

        if self._lazy is True:
            raise NotImplementedError("Only implemented for non-lazy signals")

        sig_shape = self._signal_shape_rc
        n = sig_shape[0]

        # Precompute once: for each Lambert output pixel, the fractional
        # row/column in the stereographic input image
        row_frac, col_frac = _lambert_to_stereo_coords(n)

        nav_shape = self.axes_manager.navigation_shape
        data_out = np.zeros(self.data.shape, dtype=np.float32)

        iterable = np.ndindex(nav_shape[::-1])
        if show_progressbar or (
            show_progressbar is None and hs.preferences.General.show_progressbar
        ):
            n_iterations = max(self.axes_manager.navigation_size, 1)
            iterable = tqdm(iterable, total=n_iterations)

        for idx in iterable:
            lam = _project_stereo_to_lambert(self.data[idx], row_frac, col_frac)
            lam = lam.reshape(sig_shape)
            data_out[idx] = lam

        master_pattern = self.__class__(
            data_out,
            axes=list(self.axes_manager.as_dictionary().values()),
            phase=self.phase.deepcopy(),
            projection="lambert",
            hemisphere=self.hemisphere,
        )

        return master_pattern

    def as_stereo(
        self,
        *,
        exclude_outside_equator: bool = True,
        show_progressbar: bool | None = None,
    ) -> Self:
        """Return a new master pattern in the stereographic projection
        :cite:`callahan2013dynamical`.

        Only implemented for non-lazy signals.

        Parameters
        ----------
        exclude_outside_equator
            Whether to set intensities outside the equator to zero.
            Default is True. If False, these pixels get intensities from
            the opposite hemisphere. If only one hemisphere is present,
            it is used as the opposite hemisphere as well, which
            requires the phase's point group to have inversion symmetry.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.

        Returns
        -------
        master_pattern
            Master pattern in the stereographic projection with the same
            data shape but in 32-bit floating point data dtype.

        Raises
        ------
        ValueError
            If *exclude_outside_equator* is False, only one hemisphere
            is present, and the phase's point group does not have
            inversion symmetry. Or if :attr:`projection` is
            "stereographic".
        """
        if self.projection == "stereographic":
            raise ValueError("Master pattern already in the stereographic projection")

        if self._lazy is True:
            raise NotImplementedError("Not implemented for lazy signals")

        if not exclude_outside_equator and self.hemisphere != "both":
            pg = self.phase.point_group
            if pg is None or not pg.contains_inversion:
                raise ValueError(
                    "Both hemispheres must be present in the master pattern to include "
                    "intensities outside the equator if the phase's point group does "
                    "not have inversion symmetry"
                )

        if show_progressbar is None:
            show_progressbar = hs.preferences.General.show_progressbar

        sig_shape = self._signal_shape_rc
        side_length = sig_shape[0]

        # Precompute once: for each stereographic output pixel, the
        # fractional row/column in the Lambert input image
        row_frac, col_frac, outside_equator = _stereo_to_lambert_coords(
            side_length, exclude_outside_equator=exclude_outside_equator
        )

        # Index of the hemisphere axis in the data array, if present
        hemisphere_axis = None
        if self.hemisphere == "both":
            try:
                hemisphere_axis = self.axes_manager["hemisphere"].index_in_array
            except ValueError:
                # Readers and simulators put the hemispheres first (last
                # navigation axis in HyperSpy)
                warn("No navigation axis named 'hemisphere', assuming last axis")
                hemisphere_axis = 0

        nav_shape = self._navigation_shape_rc
        data_out = np.zeros(self.data.shape, dtype=np.float32)

        iterable = np.ndindex(nav_shape)
        if show_progressbar:
            n_iterations = max(self.axes_manager.navigation_size, 1)
            iterable = tqdm(iterable, total=n_iterations)

        for idx in iterable:
            if exclude_outside_equator:
                pattern_opposite = None
            elif hemisphere_axis is None:
                pattern_opposite = self.data[idx][::-1, ::-1]
            else:
                idx_opposite = list(idx)
                idx_opposite[hemisphere_axis] = 1 - idx[hemisphere_axis]
                pattern_opposite = self.data[tuple(idx_opposite)]

            sp = _project_lambert_to_stereo(
                self.data[idx],
                row_frac,
                col_frac,
                outside_equator=outside_equator,
                pattern_lambert_opposite=pattern_opposite,
            )
            sp = sp.reshape(sig_shape)
            data_out[idx] = sp

        master_pattern = self.__class__(
            data_out,
            axes=list(self.axes_manager.as_dictionary().values()),
            phase=self.phase.deepcopy(),
            projection="stereographic",
            hemisphere=self.hemisphere,
        )

        return master_pattern

    def plot_spherical(
        self,
        energy: int | float | None = None,
        return_figure: bool = False,
        style: str = "surface",
        plotter_kwargs: dict | None = None,
        show_kwargs: dict | None = None,
    ) -> "Plotter | None":
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
            further modification and then plotting. Default is False. If
            True, the figure is not plotted.
        style
            Visualization style of the mesh, either "surface" (default),
            "wireframe" or "points". In general, "surface" is
            recommended when zoomed out, while "points" is recommended
            when zoomed in. See :meth:`pyvista.Plotter.add_mesh` for
            details.
        plotter_kwargs
            Dictionary of keyword arguments passed to
            :class:`pyvista.Plotter`.
        show_kwargs
            Dictionary of keyword arguments passed to
            :meth:`pyvista.Plotter.show` if *return_figure* is False.

        Returns
        -------
        pl
            Only returned if *return_figure* is True.

        Notes
        -----
        Requires :mod:`pyvista` (see :ref:`the installation guide
        <dependencies>`).

        Examples
        --------
        >>> import kikuchipy as kp
        >>> mp = kp.data.nickel_ebsd_master_pattern_small(projection="stereographic")
        >>> mp.plot_spherical()
        """
        verify_dependency_or_raise("pyvista", "3D plotting on the sphere")

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
        stereo2sphere = opr.InverseStereographicProjection()
        v1 = stereo2sphere.xy2vector(x.ravel(), y.ravel())
        stereo2sphere.pole = 1
        v2 = stereo2sphere.xy2vector(x.ravel(), y.ravel())
        v3 = ove.Vector3d.stack((v1, v2)).flatten()

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

    # ----- Inherited methods from KikuchipySignal2D overwritten ----- #

    def deepcopy(self) -> Self:
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

    # --------- Inherited methods from Signal2D overwritten ---------- #

    @insert_doc_disclaimer(
        cls=hs.signals.Signal2D, meth=hs.signals.Signal2D.set_signal_type
    )
    def set_signal_type(self, signal_type: str = "") -> None:
        if "master" in signal_type.lower():
            attrs = self._get_custom_attributes()
            super().set_signal_type(signal_type)
            self._set_custom_attributes(attrs)
        else:
            attrs = self._custom_attributes
            super().set_signal_type(signal_type)
            _logger.debug("Delete custom attributes when setting signal type")
            for name in attrs:
                try:
                    self.__delattr__("_" + name)
                except AttributeError:  # pragma: no cover
                    pass

    # ------------------------ Private methods ----------------------- #

    def _get_master_pattern_arrays_from_energy(
        self, energy: int | float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
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
            idx = float(energy)
            master_patterns = self.inav[idx].data
        else:  # Assume a single energy
            master_patterns = self.data
        if self.hemisphere == "both":
            master_upper, master_lower = master_patterns
        else:
            # TODO: The missing hemisphere of a centrosymmetric phase is
            # the given one rotated 180 degrees, since I(x, y, -z) =
            # I(-x, -y, z), not a copy. A copy is only correct if the
            # point group also has a 2-fold axis along z, and is wrong
            # for e.g. -1, -3, and -3m. Use
            # np.ascontiguousarray(master_patterns[..., ::-1, ::-1]) for
            # the missing hemisphere (contiguous to avoid recompiling
            # Numba functions). This also fixes a "lower" master pattern
            # being returned as the upper one. Affects get_patterns(),
            # refinement, plot_spherical(), and the detector plotter.
            master_upper = master_lower = master_patterns
        return master_upper, master_lower
