# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
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

"""Kikuchi bands and zone axes used in geometrical EBSD simulations."""

from typing import Union

from diffsims.crystallography import ReciprocalLatticePoint
import numpy as np
from orix.crystal_map import Phase
from orix.vector import Vector3d


class KikuchiBand(ReciprocalLatticePoint):
    """Kikuchi bands used in geometrical EBSD simulations."""

    def __init__(
        self,
        phase: Phase,
        hkl: Union[Vector3d, np.ndarray, tuple],
        hkl_detector: Union[Vector3d, np.ndarray, list, tuple],
        in_pattern: Union[np.ndarray, list, tuple],
        gnomonic_radius: Union[float, np.ndarray] = 10,
    ):
        """Center positions of Kikuchi bands on the detector for n
        simulated patterns.

        This class extends the
        :class:`~diffsims.crystallography.ReciprocalLatticePoint` class
        with EBSD detector pixel and gnomonic coordinates for each band
        (or point).

        Parameters
        ----------
        phase
            A phase container with a crystal structure and a space and
            point group describing the allowed symmetry operations.
        hkl
            All Miller indices present in any of the n patterns.
        hkl_detector
            Detector coordinates for all Miller indices per pattern, in
            the shape navigation_shape + (n_hkl, 3).
        in_pattern
            Boolean array of shape navigation_shape + (n_hkl,)
            indicating whether an hkl is visible in a pattern.
        gnomonic_radius
            Only plane trace coordinates of bands with Hesse normal
            form distances below this radius is returned when called
            for.

        Examples
        --------
        This class is ment to be part of a GeometricalEBSDSimulation
        generated from an EBSDSimulationGenerator object. However, a
        KikuchiBand object with no navigation shape and two bands can be
        created in the following way:

        >>> import numpy as np
        >>> from orix.crystal_map import Phase
        >>> from kikuchipy.simulations.features import KikuchiBand
        >>> p = Phase(name="ni", space_group=225)
        >>> p.structure.lattice.setLatPar(3.52, 3.52, 3.52, 90, 90, 90)
        >>> bands = KikuchiBand(
        ...     phase=p,
        ...     hkl=np.array([[-1, 1, 1], [-2, 0, 0]]),
        ...     hkl_detector=np.array(
        ...         [[0.26, 0.32, 0.26], [-0.21, 0.45, 0.27]]
        ...     ),
        ...     in_pattern=np.ones(2, dtype=bool),
        ...     gnomonic_radius=10,
        ... )
        >>> bands
        KikuchiBand (|2)
        Phase: ni (m-3m)
        [[-1  1  1]
         [ 0 -2  0]]
        """
        super().__init__(phase=phase, hkl=hkl)
        self._hkl_detector = Vector3d(hkl_detector)
        self._in_pattern = np.asarray(in_pattern)
        self.gnomonic_radius = gnomonic_radius

    @property
    def hkl_detector(self) -> Vector3d:
        """Detector coordinates for all bands per pattern."""
        return self._hkl_detector

    @property
    def gnomonic_radius(self) -> np.ndarray:
        """Only plane trace coordinates of bands with Hesse normal form
        distances below this radius are returned when called for. Per
        navigation point.
        """
        return self._gnomonic_radius

    @gnomonic_radius.setter
    def gnomonic_radius(self, value: Union[np.ndarray, list, float]):
        """Only plane trace coordinates of bands with Hesse normal form
        distances below this radius are returned when called for. Per
        navigation point.
        """
        r = np.atleast_1d(value)
        if r.size == 1:
            r = r * np.ones(self.navigation_shape)
        self._gnomonic_radius = np.atleast_1d(r.reshape(self.navigation_shape))

    @property
    def navigation_shape(self) -> tuple:
        """Navigation shape."""
        return self.hkl_detector.shape[:-1]

    @property
    def navigation_dimension(self) -> int:
        """Number of navigation dimensions (a maximum of 2)."""
        return len(self.navigation_shape)

    @property
    def _data_shape(self) -> tuple:
        """Navigation shape + number of bands."""
        return self.navigation_shape + (self.size,)

    @property
    def in_pattern(self) -> np.ndarray:
        """Which bands are visible in which patterns."""
        return self._in_pattern

    @property
    def x_detector(self) -> np.ndarray:
        """X detector coordinate for all bands per pattern."""
        return self.hkl_detector.data[..., 0]

    @property
    def y_detector(self) -> np.ndarray:
        """Y detector coordinate for all bands per pattern."""
        return self.hkl_detector.data[..., 1]

    @property
    def z_detector(self) -> np.ndarray:
        """Z detector coordinate for all bands per pattern."""
        return self.hkl_detector.data[..., 2]

    @property
    def x_gnomonic(self) -> np.ndarray:
        """X coordinate in the gnomonic projection plane on the detector
        for all bands per pattern.
        """
        return self.x_detector / self.z_detector

    @property
    def y_gnomonic(self) -> np.ndarray:
        """Y coordinate in the gnomonic projection plane on the detector
        for all bands per pattern.
        """
        return self.y_detector / self.z_detector

    @property
    def hesse_distance(self) -> np.ndarray:
        """Distance from the PC (origin) per band, i.e. the right-angle
        component of the distance to the pole.
        """
        return np.tan(0.5 * np.pi - self.hkl_detector.theta.data)

    @property
    def within_gnomonic_radius(self) -> np.ndarray:
        """Return whether a plane trace is within the `gnomonic_radius`
        as a boolean array.
        """
        # TODO: Should be part of GeometricalEBSDSimulation, not here
        is_full_upper = self.z_detector > -1e-5
        gnomonic_radius = self._get_reshaped_gnomonic_radius(
            self.hesse_distance.ndim
        )
        in_circle = np.abs(self.hesse_distance) < gnomonic_radius
        return np.logical_and(in_circle, is_full_upper)

    @property
    def hesse_alpha(self) -> np.ndarray:
        """Hesse angle alpha. Only angles for the planes within the
        `gnomonic_radius` are returned.
        """
        hesse_distance = self.hesse_distance
        hesse_distance[~self.within_gnomonic_radius] = np.nan
        gnomonic_radius = self._get_reshaped_gnomonic_radius(
            hesse_distance.ndim
        )
        return np.arccos(hesse_distance / gnomonic_radius)

    @property
    def plane_trace_coordinates(self) -> np.ndarray:
        """Plane trace coordinates P1, P2 on the form [y0, x0, y1, x1]
        per band in the plane of the detector in gnomonic coordinates.

        Coordinates for the planes outside the `gnomonic_radius` are set
        to NaN.
        """
        # Get alpha1 and alpha2 angles (NaN for bands outside gnomonic radius)
        phi = self.hkl_detector.phi.data
        hesse_alpha = self.hesse_alpha
        plane_trace = np.zeros(self.navigation_shape + (self.size, 4))
        alpha1 = phi - np.pi + hesse_alpha
        alpha2 = phi - np.pi - hesse_alpha

        # Calculate start and end points for the plane traces
        plane_trace[..., 0] = np.cos(alpha1)
        plane_trace[..., 1] = np.cos(alpha2)
        plane_trace[..., 2] = np.sin(alpha1)
        plane_trace[..., 3] = np.sin(alpha2)

        # And remember to multiply by the gnomonic radius
        gnomonic_radius = self._get_reshaped_gnomonic_radius(plane_trace.ndim)
        return gnomonic_radius * plane_trace

    @property
    def hesse_line_x(self) -> np.ndarray:
        return -self.hesse_distance * np.cos(self.hkl_detector.phi.data)

    @property
    def hesse_line_y(self) -> np.ndarray:
        return -self.hesse_distance * np.sin(self.hkl_detector.phi.data)

    def __getitem__(self, key):
        """Get a deepcopy subset of the KikuchiBand object.

        Properties have different shapes, so care must be taken when
        slicing. As an example, consider a 2 x 3 map with 4 bands. Three
        data shapes are considered:
        * navigation shape (2, 3) (gnomonic_radius)
        * band shape (4,) (hkl, structure_factor, theta)
        * full shape (2, 3, 4) (hkl_detector, in_pattern)
        """
        # These are overwritten as the input key length is investigated
        nav_slice, band_slice = key, key  # full_slice = key
        nav_ndim = self.navigation_dimension
        n_keys = len(key) if hasattr(key, "__iter__") else 1
        if n_keys == 0:  # The case with key = ()/slice(None). Return everything
            band_slice = slice(None)
        elif n_keys == 1:
            if nav_ndim != 0:
                band_slice = slice(None)
        elif n_keys == 2:
            if nav_ndim == 0:
                raise IndexError("Not enough axes to slice")
            elif nav_ndim == 1:
                nav_slice = key[0]
                band_slice = key[1]
            else:  # nav_slice = key
                band_slice = slice(None)
        elif n_keys == 3:  # Maximum number of slices
            if nav_ndim < 2:
                raise IndexError("Not enough axes to slice")
            else:
                nav_slice = key[:2]
                band_slice = key[2]
        new_bands = KikuchiBand(
            phase=self.phase,
            hkl=self.hkl[band_slice],
            hkl_detector=self.hkl_detector[key],
            in_pattern=self.in_pattern[key],
            gnomonic_radius=self.gnomonic_radius[nav_slice],
        )
        new_bands._structure_factor = self.structure_factor[band_slice]
        new_bands._theta = self.theta[band_slice]
        return new_bands

    def __repr__(self):
        shape_str = _get_dimension_str(
            nav_shape=self.navigation_shape, sig_shape=(self.size,)
        )
        first_line = f"{self.__class__.__name__} {shape_str}"
        return "\n".join([first_line] + super().__repr__().split("\n")[1:])

    def _get_reshaped_gnomonic_radius(self, ndim: int) -> np.ndarray:
        add_ndim = ndim - self.gnomonic_radius.ndim
        return self.gnomonic_radius.reshape(
            self.gnomonic_radius.shape + (1,) * add_ndim
        )

    def unique(self, **kwargs):
        # TODO: Fix transfer of properties in this class and other inheriting
        #  classes in diffsims when creating a new class object
        raise NotImplementedError

    def symmetrise(self, **kwargs):
        # TODO: Fix transfer of properties in this class and other inheriting
        #  classes in diffsims when creating a new class object
        raise NotImplementedError

    @classmethod
    def from_min_dspacing(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_highest_hkl(cls, **kwargs):
        raise NotImplementedError


class ZoneAxis(ReciprocalLatticePoint):
    """Zone axes used in geometrical EBSD simulations."""

    def __init__(
        self,
        phase: Phase,
        uvw: Union[Vector3d, np.ndarray, list, tuple],
        uvw_detector: Union[Vector3d, np.ndarray, list, tuple],
        in_pattern: Union[np.ndarray, list, tuple],
        gnomonic_radius: Union[float, np.ndarray] = 10,
    ):
        """Positions of zone axes on the detector.

        Parameters
        ----------
        phase
            A phase container with a crystal structure and a space and
            point group describing the allowed symmetry operations.
        uvw
            Miller indices.
        uvw_detector
            Zone axes coordinates on the detector.
        in_pattern
            Boolean array of shape (n, n_hkl) indicating whether an hkl
            is visible in a pattern.
        gnomonic_radius
            Only plane trace coordinates of bands with Hesse normal
            form distances below this radius is returned when called
            for.
        """
        super().__init__(phase=phase, hkl=uvw)
        self._uvw_detector = Vector3d(uvw_detector)
        self._in_pattern = np.asarray(in_pattern)
        self.gnomonic_radius = gnomonic_radius

    @property
    def uvw_detector(self) -> Vector3d:
        """Detector coordinates for all zone axes per pattern."""
        return self._uvw_detector

    @property
    def gnomonic_radius(self) -> np.ndarray:
        """Only zone axes within this distance from the PC are returned
        when called for. Per navigation point.
        """
        return self._gnomonic_radius

    @gnomonic_radius.setter
    def gnomonic_radius(self, value: Union[np.ndarray, list, float]):
        """Only plane trace coordinates of bands with Hesse normal form
        distances below this radius are returned when called for. Per
        navigation point.
        """
        r = np.atleast_1d(value)
        if r.size == 1:
            r = r * np.ones(self.navigation_shape)
        self._gnomonic_radius = np.atleast_1d(r.reshape(self.navigation_shape))

    @property
    def navigation_shape(self) -> tuple:
        """Navigation shape."""
        return self.uvw_detector.shape[:-1]

    @property
    def navigation_dimension(self) -> int:
        """Number of navigation dimensions (a maximum of 2)."""
        return len(self.navigation_shape)

    @property
    def _data_shape(self) -> tuple:
        """Navigation shape + number of bands."""
        return self.navigation_shape + (self.size,)

    @property
    def in_pattern(self) -> np.ndarray:
        """Which bands are visible in which patterns."""
        return self._in_pattern

    @property
    def x_detector(self) -> np.ndarray:
        """X detector coordinate for all zone axes per pattern."""
        return self.uvw_detector.data[..., 0]

    @property
    def y_detector(self) -> np.ndarray:
        """Y detector coordinate for all zone axes per pattern."""
        return self.uvw_detector.data[..., 1]

    @property
    def z_detector(self) -> np.ndarray:
        """Z detector coordinate for all zone axes per pattern."""
        return self.uvw_detector.data[..., 2]

    @property
    def x_gnomonic(self) -> np.ndarray:
        """X coordinate in the gnomonic projection plane on the detector
        for all zone axes per pattern.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.x_detector / self.z_detector

    @property
    def y_gnomonic(self) -> np.ndarray:
        """X coordinate in the gnomonic projection plane on the detector
        for all zone axes per pattern.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.y_detector / self.z_detector

    @property
    def r_gnomonic(self) -> np.ndarray:
        """Gnomonic radius for all zone axes per pattern."""
        return np.sqrt(self.x_gnomonic ** 2 + self.y_gnomonic ** 2)

    @property
    def within_gnomonic_radius(self) -> np.ndarray:
        """Return whether a zone axis is within the `gnomonic_radius`
        as a boolean array.
        """
        # TODO: Should be part of GeometricalEBSDSimulation, not here
        is_full_upper = self.z_detector > -1e-5
        gnomonic_radius = self._get_reshaped_gnomonic_radius(
            self.navigation_dimension + 1
        )
        in_circle = self.r_gnomonic < gnomonic_radius
        return np.logical_and(in_circle, is_full_upper)

    @property
    def _xy_within_gnomonic_radius(self) -> np.ndarray:
        xy = np.ones(self._data_shape + (2,)) * np.nan
        within = self.within_gnomonic_radius
        xy[within, 0] = self.x_gnomonic[within]
        xy[within, 1] = self.y_gnomonic[within]
        return xy

    def __repr__(self):
        shape_str = _get_dimension_str(
            nav_shape=self.navigation_shape, sig_shape=(self.size,)
        )
        first_line = f"{self.__class__.__name__} {shape_str}"
        return "\n".join([first_line] + super().__repr__().split("\n")[1:])

    def _get_reshaped_gnomonic_radius(self, ndim: int) -> np.ndarray:
        add_ndim = ndim - self.gnomonic_radius.ndim
        return self.gnomonic_radius.reshape(
            self.gnomonic_radius.shape + (1,) * add_ndim
        )

    def __getitem__(self, key):
        """Get a deepcopy subset of the ZoneAxis object.

        Properties have different shapes, so care must be taken when
        slicing. As an example, consider a 2 x 3 map with 4 zone axes.
        Three data shapes are considered:
        * navigation shape (2, 3) (gnomonic_radius)
        * zone axes shape (4,) (hkl, structure_factor, theta)
        * full shape (2, 3, 4) (uvw_detector, in_pattern)
        """
        # These are overwritten as the input key length is investigated
        nav_slice, za_slice = key, key  # full_slice = key
        nav_ndim = self.navigation_dimension
        n_keys = len(key) if hasattr(key, "__iter__") else 1
        if n_keys == 0:  # The case with key = ()/slice(None). Return everything
            za_slice = slice(None)
        elif n_keys == 1:
            if nav_ndim != 0:
                za_slice = slice(None)
        elif n_keys == 2:
            if nav_ndim == 0:
                raise IndexError("Not enough axes to slice")
            elif nav_ndim == 1:
                nav_slice = key[0]
                za_slice = key[1]
            else:  # nav_slice = key
                za_slice = slice(None)
        elif n_keys == 3:  # Maximum number of slices
            if nav_ndim < 2:
                raise IndexError("Not enough axes to slice")
            else:
                nav_slice = key[:2]
                za_slice = key[2]
        new_za = ZoneAxis(
            phase=self.phase,
            uvw=self.hkl[za_slice],
            uvw_detector=self.uvw_detector[key],
            in_pattern=self.in_pattern[key],
            gnomonic_radius=self.gnomonic_radius[nav_slice],
        )
        new_za._structure_factor = self.structure_factor[za_slice]
        new_za._theta = self.theta[za_slice]
        return new_za

    def unique(self, **kwargs):
        # TODO: Fix transfer of properties in this class and other inheriting
        #  classes in diffsims when creating a new class object
        raise NotImplementedError

    def symmetrise(self, **kwargs):
        # TODO: Fix transfer of properties in this class and other inheriting
        #  classes in diffsims when creating a new class object
        raise NotImplementedError

    @classmethod
    def from_min_dspacing(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_highest_hkl(cls, **kwargs):
        raise NotImplementedError


def _get_dimension_str(nav_shape: tuple, sig_shape: tuple):
    """Adapted from HyperSpy's AxesManager._get_dimension_str."""
    dim_str = "("
    if len(nav_shape) > 0:
        for axis in nav_shape:
            dim_str += f"{axis}, "
    dim_str = dim_str.rstrip(", ")
    dim_str += "|"
    if len(sig_shape) > 0:
        for axis in sig_shape:
            dim_str += f"{axis}, "
    dim_str = dim_str.rstrip(", ")
    dim_str += ")"
    return dim_str
