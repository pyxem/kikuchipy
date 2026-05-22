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

"""Implementation of the EBSD detector class.

The class is implemented with the following goals in mind:

- Enable it's use as many places in the code as possible.
  This means that in this file, to avoid circular imports, we must
  import from other modules inside the using detector methods, rather
  than at the top of the file.
- To ease understanding and extending the class, we should put
  implementations of fitting, plotting, IO, and such in separate
  modules, rather than here.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload
import warnings

from diffsims.crystallography import ReciprocalLatticeVector
import numpy as np
from orix.crystal_map import PhaseList
from orix.quaternion import Rotation
from typing_extensions import Self, get_args

from kikuchipy._constants import dependency_version
from kikuchipy._utils._detector_coordinates import (
    convert_coordinates,
    get_coordinate_conversions,
)
from kikuchipy._utils.deprecated import VisibleDeprecationWarning
from kikuchipy.indexing._hough_indexing import _get_indexer_from_detector

# Repeated in plotting module
DETECTOR_PLOT_FORMATS = Literal["detector", "gnomonic"]
PROJECTION_CENTER_PLOT_MODES = Literal["map", "scatter", "3d"]

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.axes as maxes
    import matplotlib.figure as mfigure

    if dependency_version["pyebsdindex"] is not None:
        from pyebsdindex.ebsd_index import EBSDIndexer
    if dependency_version["ipywidgets"] is not None:
        import ipywidgets


PC_CONVENTIONS = Literal[
    "bruker",
    # tsl
    "tsl",
    "edax",
    "amatek",
    # oxford
    "oxford",
    "aztec",
    # emsoft
    "emsoft",
    "emsoft4",
    "emsoft5",
]
PC_CONVENTIONS_SINGLE = Literal["bruker", "tsl", "oxford", "emsoft"]
PC_CONVENTIONS_ALIASES: dict[PC_CONVENTIONS_SINGLE, list[PC_CONVENTIONS]] = {
    "bruker": ["bruker"],
    "tsl": ["tsl", "edax", "amatek"],
    "oxford": ["oxford", "aztec"],
    "emsoft": ["emsoft", "emsoft4", "emsoft5"],
}


class EBSDDetector:
    r"""An EBSD detector defining the detector's view of the sample.

    The detector stores information of the detector shape, pixel size,
    binning factor, tilt, azimuthal, and twist, the sample tilt, and
    the projection/pattern center (PC) per beam position. Given one or
    multiple PCs, the detector's gnomonic coordinates are calculated.

    A detector defines the transformations needed to project patterns
    generated in the sample to the detector.

    Parameters
    ----------
    shape
        Number of detector rows :math:`N_y` and columns :math:`N_x`, in
        pixels. Default is (1, 1).
    px_size
        Size of the square unbinned detector pixel :math:`\delta`, in
        microns. Default is 1.0.
    binning
        Detector binning factor :math:`b`, i.e. how many pixels are
        binned into one. Default is 1, meaning no binning.
    tilt
        Detector tilt :math:`\theta` about the detector horizontal
        :math:`X_d`, in degrees. Default is 0. A positive angle means
        features on the detector appear to move upward (assuming all
        other defaults).
    azimuthal
        Detector tilt :math:`\omega` about the detector vertical
        :math:`Y_d`, pointing downwards, in degrees. Default is 0. A
        positive angle means features on the detector appear to move
        toward the right (assuming all other defaults).
    twist
        Detector tilt :math:`\gamma` about the detector normal
        :math:`Z_d`, pointing towards the sample, in degrees. Default is
        0.0. A positive angle means features on the detector appear to
        move counter-clockwise about the detector center (assuming all
        other defaults).
    sample_tilt
        Sample tilt :math:`\sigma` about the sample horizontal,
        :math:`Y_d`, in degrees. Default is 70. Note that the sample
        horizontal :math:`Y_s` is parallel to the detector horizontal,
        :math:`X_d` (assuming all other defaults).
    pc
        X, Y, and Z coordinates of the projection centers (PCs) in the
        given *convention*. Default is [0.5, 0.5, 0.5]. The PC describes
        the location of the beam on the sample surface measured relative
        to the detector screen. See *Notes* for the definitions of and
        conversions between conventions. If multiple PCs are given, they
        are assumed to be on the form [[x0, y0, z0], [x1, y1, z1], ...].
    convention
        Convention of the given *pc*. Default is Bruker's convention.
        This determines how to convert *pc* to the internal (Bruker's)
        definition. Options are "edax"/"tsl"/"amatek", "oxford"/"aztec",
        "bruker", "emsoft", "emsoft4", and "emsoft5". "emsoft" and
        "emsoft5" is the same convention. See *Notes* for conversions
        between conventions.

    Notes
    -----
    The pattern on the detector is always viewed from the detector
    towards the sample. Pattern width and height in pixels are here
    given as :math:`N_x` and :math:`N_y`, respectively (possibly
    binned). PCs are stored and used internally in Bruker's convention.

    The Bruker PC coordinates :math:`(x_B^*, y_B^*, z_B^*)` are defined
    in fractions of :math:`N_x`, :math:`N_y`, and :math:`N_y`,
    respectively. :math:`x_B^*` and :math:`y_B^*` are defined with
    respect to the upper left corner of the detector. The Bruker PC
    coordinates are used internally, called :math:`(PC_x, PC_y, PC_z)`
    in the rest of the documentation when there is no reference to
    Bruker specifically.

    The EDAX TSL PC coordinates :math:`(x_T^*, y_T^*, z_T^*)` are
    defined in fractions of :math:`N_x`, :math:`N_y`, and
    :math:`\min(N_x, N_y)` with respect to the lower left corner of the
    detector, respectively.

    Given these definitions, the following is the conversion from a PC
    in EDAX TSL's convention to a PC in Bruker's convention

    .. math::

        x_B^* &= x_T^*,\\
        y_B^* &= 1 - y_T^*,\\
        z_B^* &= \frac{\min(N_x, N_y)}{N_y} z_T^*.

    The Oxford Instruments PC coordinates :math:`(x_O^*, y_O^*, z_O^*)`
    are defined in fractions of :math:`N_x` with respect to the lower
    left corner of the detector. The conversion from Oxford Instruments
    to Bruker is therefore given as

    .. math::

        x_B^* &= x_O^*,\\
        y_B^* &= 1 - y_O^* \frac{N_x}{N_y},\\
        z_B^* &= \frac{N_x}{N_y} z_O^*.

    The EMsoft PC coordinates :math:`(x_{pc}, y_{pc})` are defined as
    the number of pixels (with subpixel accuracy) with respect to the
    center of the detector. :math:`x_{pc}` points towards the right and
    :math:`y_{pc}` points upwards. The final PC coordinate, :math:`L`,
    is the detector distance, in microns. Note that prior to EMsoft
    v5.0, :math:`x_{pc}` was defined towards the left. The conversion
    from EMsoft to Bruker is then, finally, given as

    .. math::

        x_B^* &= \frac{1}{2} - \frac{x_{pc}}{N_x b},\\
        y_B^* &= \frac{1}{2} - \frac{y_{pc}}{N_y b},\\
        z_B^* &= \frac{L}{N_y b \delta},

    where :math:`\delta` is the unbinned detector pixel size in microns
    and :math:`b` is the binning factor.

    The calculation of gnomonic coordinates is based on the
    supplementary material to :cite:`britton2016tutorial`.
    """

    if dependency_version["psygnal"] is not None:
        from psygnal import Signal

        _sample_tilt_changed = Signal(float)
        _tilt_changed = Signal(float)
        _azimuthal_changed = Signal(float)
        _twist_changed = Signal(float)
        _pc_changed = Signal(np.ndarray)

    def __init__(
        self,
        shape: tuple[int, int] = (1, 1),
        px_size: float = 1.0,
        binning: int = 1,
        tilt: float = 0.0,
        azimuthal: float = 0.0,
        twist: float = 0.0,
        sample_tilt: float = 70.0,
        pc: np.ndarray | list | tuple = (0.5, 0.5, 0.5),
        convention: PC_CONVENTIONS | None = "bruker",
    ) -> None:
        self._has_signals = hasattr(self, "_sample_tilt_changed")

        self.shape = shape
        self.px_size = px_size
        # .binning returns an integer, while ._binning should always be
        # a float. Use the latter for any computation.
        self.binning = binning
        self.tilt = tilt
        self.azimuthal = azimuthal
        self.twist = twist
        self.sample_tilt = sample_tilt

        self.pc = pc
        if convention is None:
            # TODO: Remove option to pass None and this warning after
            # 0.13 has been released.
            warnings.warn(
                message=(
                    "Passing 'None' is deprecated and will give an error in version "
                    "0.14. Pass the default value 'bruker' to avoid this warning."
                ),
                category=VisibleDeprecationWarning,
            )
            convention = "bruker"
        self._set_pc_in_bruker_convention(convention)

    # -------------------------- Properties -------------------------- #
    # Grouped by topic

    # ----------------------- Angle properties ----------------------- #

    @property
    def sample_tilt(self) -> float:
        r"""Return or set the sample tilt in degrees.

        The sample tilt :math:`\sigma` is about the sample horizontal,
        :math:`Y_d`. Note that the sample horizontal :math:`Y_d` is
        parallel to the detector horizontal, :math:`X_d`.

        Parameters
        ----------
        value : float
            Sample tilt in degrees.
        """
        return self._sample_tilt

    @sample_tilt.setter
    def sample_tilt(self, value: float) -> None:
        if not isinstance(value, Number):
            raise ValueError(f"Invalid sample tilt {value}. Must be a number.")
        self._sample_tilt = float(value)

        if self._has_signals:
            self._sample_tilt_changed.emit(self._sample_tilt)

    @property
    def tilt(self) -> float:
        r"""Return or set the detector tilt :math:`\theta` about the
        detector horizontal :math:`X_d`, in degrees.

        Parameters
        ----------
        value : float
            Detector tilt in degrees.
        """
        return self._tilt

    @tilt.setter
    def tilt(self, value: float) -> None:
        if not isinstance(value, Number):
            raise ValueError(f"Invalid detector tilt {value}. Must be a number.")
        self._tilt = float(value)

        if self._has_signals:
            self._tilt_changed.emit(self._tilt)

    @property
    def azimuthal(self) -> float:
        r"""Return or set the detector tilt :math:`\omega` about the
        detector vertical :math:`Y_d`, pointing downwards, in degrees.

        A positive angle means features on the detector appear to move
        toward the right (assuming all other defaults).

        Parameters
        ----------
        value : float
            Azimuthal detector tilt in degrees.
        """
        return self._azimuthal

    @azimuthal.setter
    def azimuthal(self, value: float) -> None:
        if not isinstance(value, Number):
            raise ValueError(f"Invalid azimuthal {value}. Must be a number.")
        self._azimuthal = float(value)

        if self._has_signals:
            self._azimuthal_changed.emit(self._azimuthal)

    @property
    def twist(self) -> float:
        r"""Return or set the detector twist :math:`\gamma` about the
        detector normal :math:`Z_d`, pointing towards the sample, in
        degrees.

        A positive angle means features on the detector appear to move
        counter-clockwise about the detector center (assuming all other
        defaults).

        Parameters
        ----------
        value : float
            Detector twist in degrees.
        """
        return self._twist

    @twist.setter
    def twist(self, value: float) -> None:
        if not isinstance(value, Number):
            raise ValueError(f"Invalid twist {value}. Must be a number.")
        self._twist = float(value)

        if self._has_signals:
            self._twist_changed.emit(self._twist)

    @property
    def euler(self) -> np.ndarray:
        r"""Return the detector Euler angles in the Bunge convention
        (ZXZ) in degrees.

        The Euler angles are given by the :attr:`azimuthal`,
        :attr:`tilt`, and the :attr:`twist`,
        :math:`(-\omega, 90^{\circ} + \theta, -\gamma)`.
        """
        return np.array([-self.azimuthal, 90.0 + self.tilt, -self.twist], dtype=float)

    # ----------------- Projection center properties ----------------- #

    @property
    def pc(self) -> np.ndarray:
        """Return or set all projection center (PC) coordinates.

        Parameters
        ----------
        value : numpy.ndarray, list, or tuple
            PC coordinates. If multiple PCs are given, they are assumed
            to be on the form [[x0, y0, z0], [x1, y1, z1], ...]. Default
            is [[0.5, 0.5, 0.5]].

        Notes
        -----
        The PC coordinates are stored in Bruker's convention. See the
        *Notes* in :class:`EBSDDetector` for more information.
        """
        return self._pc

    @pc.setter
    def pc(self, value: np.ndarray | list | tuple) -> None:
        self._pc = np.atleast_2d(value).astype(float)

        if self._has_signals:
            self._pc_changed.emit(self._pc)

    @property
    def pc_average(self) -> np.ndarray:
        """Return the average projection center.

        Notes
        -----
        The PC coordinates are stored in Bruker's convention. See the
        *Notes* in :class:`EBSDDetector` for more information.
        """
        ndim = self.pc.ndim
        axis = ()
        if ndim == 2:
            axis += (0,)
        elif ndim == 3:
            axis += (0, 1)
        return np.nanmean(self.pc, axis=axis)

    @property
    def pc_flattened(self) -> np.ndarray:
        """Return a flattened array of projection center coordinates of
        shape (:attr:`navigation_size`, 3).

        Notes
        -----
        The PC coordinates are stored in Bruker's convention. See the
        *Notes* in :class:`EBSDDetector` for more information.
        """
        return self.pc.reshape((-1, 3))

    @property
    def pcx(self) -> np.ndarray:
        """Return or set the projection center (PC) x coordinates.

        Parameters
        ----------
        value : numpy.ndarray, list, tuple or float
            PC x coordinates in Bruker's convention. If multiple x
            coordinates are passed, they are assumed to be on the form
            [x0, x1,...].

        Notes
        -----
        The PC coordinates are stored in Bruker's convention. See the
        *Notes* in :class:`EBSDDetector` for more information.
        """
        return self.pc[..., 0]

    @pcx.setter
    def pcx(self, value: np.ndarray | list | tuple | float):
        self._pc[..., 0] = np.atleast_2d(value).astype(float)

        if self._has_signals:
            self._pc_changed.emit(self._pc)

    @property
    def pcy(self) -> np.ndarray:
        """Return or set the projection center (PC) y coordinates.

        Parameters
        ----------
        value : numpy.ndarray, list, tuple or float
            PC y coordinates in Bruker's convention. If multiple y
            coordinates are passed, they are assumed to be on the form
            [y0, y1,...].

        Notes
        -----
        The PC coordinates are stored in Bruker's convention. See the
        *Notes* in :class:`EBSDDetector` for more information.
        """
        return self.pc[..., 1]

    @pcy.setter
    def pcy(self, value: np.ndarray | list | tuple | float):
        self._pc[..., 1] = np.atleast_2d(value).astype(float)

        if self._has_signals:
            self._pc_changed.emit(self._pc)

    @property
    def pcz(self) -> np.ndarray:
        """Return or set the projection center (PC) z coordinates.

        Parameters
        ----------
        value : numpy.ndarray, list, tuple or float
            PC z coordinates in Bruker's convention. If multiple z
            coordinates are passed, they are assumed to be on the form
            [z0, z1,...].

        Notes
        -----
        The PC coordinates are stored in Bruker's convention. See the
        *Notes* in :class:`EBSDDetector` for more information.
        """
        return self.pc[..., 2]

    @pcz.setter
    def pcz(self, value: np.ndarray | list | tuple | float):
        self._pc[..., 2] = np.atleast_2d(value).astype(float)

        if self._has_signals:
            self._pc_changed.emit(self._pc)

    @property
    def px_size(self) -> float:
        r"""Return the pixel size.

        This is the size of the square unbinned detector pixel
        :math:`\delta`, in microns.

        Parameters
        ----------
        value : float
            Pixel size.
        """
        return self._px_size

    @px_size.setter
    def px_size(self, value: float) -> None:
        if not isinstance(value, Number):
            raise ValueError(f"Invalid pixel size {value}. Must be a number.")
        self._px_size = float(value)

    # ----------------------- Shape properties ----------------------- #

    @property
    def shape(self) -> tuple[int, int]:
        """Return or set the number of detector rows :math:`N_y` and
        columns :math:`N_x` in pixels.

        Parameters
        ----------
        value : tuple[int, int]
            Two integers :math:`(N_y, N_x)`.
        """
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int, int]) -> None:
        self._shape = self._parse_valid_shape_or_raise(value)

    @property
    def navigation_dimension(self) -> int:
        """Return the number of navigation dimensions of the projection
        center array.
        """
        return len(self.navigation_shape)

    @property
    def navigation_shape(self) -> tuple[int] | tuple[int, int]:
        """Return or set the navigation shape of the projection center
        array.

        Parameters
        ----------
        value : tuple[int] or tuple[int, int]
            Navigation shape, with a maximum dimension of 2.
        """
        return self.pc.shape[: self.pc.ndim - 1]

    @navigation_shape.setter
    def navigation_shape(self, value: tuple[int] | tuple[int, int]) -> None:
        ndim = len(value)
        if ndim > 2:
            raise ValueError(f"A maximum dimension of 2 is allowed, 2 < {ndim}")
        else:
            self.pc = self.pc.reshape(value + (3,))

            if self._has_signals:
                self._pc_changed.emit(self._pc)

    @property
    def navigation_size(self) -> int:
        """Return the number of projection centers."""
        return int(np.prod(self.navigation_shape))

    @property
    def ncols(self) -> int:
        """Return the number of detector pixel columns :math:`N_x`."""
        return self.shape[1]

    @property
    def nrows(self) -> int:
        """Return the number of detector pixel rows :math:`N_y`."""
        return self.shape[0]

    @property
    def aspect_ratio(self) -> float:
        """Return the number of detector columns :math:`N_x` divided by
        the number of detector rows :math:`N_y`.
        """
        return self.ncols / self.nrows

    @property
    def bounds(self) -> np.ndarray:
        """Return the detector bounds :math:`(0, N_x - 1, 0, N_y - 1)`
        in pixel coordinates.
        """
        return np.array([0, self.ncols - 1, 0, self.nrows - 1])

    @property
    def size(self) -> int:
        """Return the number of detector pixels :math:`N_x N_y`."""
        return self.nrows * self.ncols

    @property
    def unbinned_shape(self) -> tuple[int, int]:
        """Return the unbinned detector shape in pixels.

        This is given by :math:`(N_y b, N_x b)`.
        """
        return tuple(np.array(self.shape, dtype=int) * self._binning)

    # ----------------------- Size properties ------------------------ #

    @property
    def binning(self) -> int:
        r"""Return or set the integer detector binning factor, :math:`b`.

        Parameters
        ----------
        value : int
            Detector binning factor.
        """
        return int(self._binning)

    @binning.setter
    def binning(self, value: int) -> None:
        if not isinstance(value, Number):
            raise ValueError(f"Invalid binning {value}. Must be an integer.")
        self._binning = float(value)

    @property
    def height(self) -> float:
        r"""Return the detector height in microns.

        This is given by :math:`N_y \delta b`.
        """
        return self.nrows * self.px_size * self._binning

    @property
    def width(self) -> float:
        r"""Return the detector width in microns.

        This is given by :math:`N_x \delta b`.
        """
        return self.ncols * self.px_size * self._binning

    @property
    def px_size_binned(self) -> float:
        r"""Return the binned pixel size in microns.

        This is given by :math:`\delta b`.
        """
        return self.px_size * self._binning

    @property
    def specimen_scintillator_distance(self) -> np.ndarray:
        """Return the specimen-to-scintillator distance, known in EMsoft
        as :math:`L`, given in microns.
        """
        return self.pcz * self.height

    # ---------------- Gnomonic projection properties ---------------- #

    @property
    def x_min(self) -> np.ndarray | float:
        """Return the left bound in gnomonic coordinates.

        The calculation of gnomonic coordinates is based on the
        supplementary material to :cite:`britton2016tutorial`.
        """
        return -self.aspect_ratio * (self.pcx / self.pcz)

    @property
    def x_max(self) -> np.ndarray | float:
        """Return the right bound in gnomonic coordinates.

        The calculation of gnomonic coordinates is based on the
        supplementary material to :cite:`britton2016tutorial`.
        """
        return self.aspect_ratio * (1 - self.pcx) / self.pcz

    @property
    def x_range(self) -> np.ndarray:
        """Return the horizontal detector range in gnomonic coordinates.

        The calculation of gnomonic coordinates is based on the
        supplementary material to :cite:`britton2016tutorial`.
        """
        return np.dstack((self.x_min, self.x_max)).reshape(self.navigation_shape + (2,))

    @property
    def y_min(self) -> np.ndarray | float:
        """Return the upper bound in gnomonic coordinates.

        The calculation of gnomonic coordinates is based on the
        supplementary material to :cite:`britton2016tutorial`.
        """
        return -(1 - self.pcy) / self.pcz

    @property
    def y_max(self) -> np.ndarray | float:
        """Return the lower bound in gnomonic coordinates.

        The calculation of gnomonic coordinates is based on the
        supplementary material to :cite:`britton2016tutorial`.
        """
        return self.pcy / self.pcz

    @property
    def y_range(self) -> np.ndarray:
        """Return the vertical detector range in gnomonic coordinates.

        The calculation of gnomonic coordinates is based on the
        supplementary material to :cite:`britton2016tutorial`.
        """
        return np.dstack((self.y_min, self.y_max)).reshape(self.navigation_shape + (2,))

    @property
    def x_scale(self) -> np.ndarray:
        """Return the width of a pixel in gnomonic coordinates.

        The calculation of gnomonic coordinates is based on the
        supplementary material to :cite:`britton2016tutorial`.
        """
        if self.ncols == 1:
            x_scale = np.diff(self.x_range)
        else:
            x_scale = np.diff(self.x_range) / (self.ncols - 1)
        return x_scale.reshape(self.navigation_shape)

    @property
    def y_scale(self) -> np.ndarray:
        """Return the height of a pixel in gnomonic coordinates.

        The calculation of gnomonic coordinates is based on the
        supplementary material to :cite:`britton2016tutorial`.
        """
        if self.nrows == 1:
            y_scale = np.diff(self.y_range)
        else:
            y_scale = np.diff(self.y_range) / (self.nrows - 1)
        return y_scale.reshape(self.navigation_shape)

    @property
    def gnomonic_bounds(self) -> np.ndarray:
        """Return the detector bounds [x0, x1, y0, y1] in gnomonic
        coordinates.

        The calculation of gnomonic coordinates is based on the
        supplementary material to :cite:`britton2016tutorial`.
        """
        return np.concatenate((self.x_range, self.y_range), axis=-1)

    @property
    def r_max(self) -> np.ndarray:
        """Return the maximum distance from :attr:`pc` to the detector
        edge in gnomonic coordinates.

        The calculation of gnomonic coordinates is based on the
        supplementary material to :cite:`britton2016tutorial`.
        """
        corners = np.zeros(self.navigation_shape + (4,))
        corners[..., 0] = self.x_min**2 + self.y_min**2  # Upper left
        corners[..., 1] = self.x_max**2 + self.y_min**2  # Upper right
        corners[..., 2] = self.x_max**2 + self.y_max**2  # Lower right
        corners[..., 3] = self.x_min**2 + self.y_min**2  # Lower left
        return np.atleast_2d(np.sqrt(np.max(corners, axis=-1)))

    @property
    def coordinate_conversion_factors(self) -> dict:
        """Return factors for converting coordinates on the detector
        from pixel units to gnomonic units or vice versa.

        The dict returned contains the keys "pix_to_gn",
        containing factors for converting pixel to gnomonic
        coordinates, and "gn_to_pix", containing factors for
        converting gnomonic to pixel coordinates.
        Under each of these keys is a further dict with the
        keys: "m_x", "c_x", "m_y" and "c_y". These are the
        slope (m) and y-intercept (c) corresponding to
        y = mx + c, which describes the linear conversion
        of the coordinates. A (different) linear relationship
        is required for x (column) and y(row) coordinates,
        hence the two sets of m and c parameters.
        The shape of each array of conversion factors
        typically corresponds to the navigation shape
        of an EBSDDetector.
        """
        return get_coordinate_conversions(self.gnomonic_bounds, self.bounds)

    @property
    def sample_to_detector(self) -> Rotation:
        """Return the orientation matrix which transforms
        vectors in the sample reference frame, CSs, to the
        detector reference frame, CSd.

        Notes
        -----

        This is the matrix U_s as defined in the paper by
        Britton et al. :cite:`britton2016tutorial`.

        The return value has type orix.quaternion.Rotation.
        To obtain a np.ndarray from this, call
        u_s_rot.to_matrix(), or u_s_rot.to_matrix().squeeze(),
        to obtain a 2D, 3x3 array.

        The matrix describing the reverse transformation, i.e.
        from the detector reference frame, CSd, to the sample
        reference frame, CSs, can be obtained like this:
        ~EBSDDetector.sample_to_detector
        """
        u_sample = Rotation.from_euler([0, self.sample_tilt, 0], degrees=True)
        u_d = Rotation.from_euler(self.euler, degrees=True)
        u_d_g = u_d.to_matrix().squeeze()
        u_detector = Rotation.from_matrix(u_d_g.T)
        u_s_bruker = u_sample * u_detector
        sample_to_detector = (
            Rotation.from_axes_angles((0, 0, -1), -np.pi / 2) * u_s_bruker
        )
        return sample_to_detector

    @property
    def _average_gnomonic_bounds(self) -> np.ndarray:
        return np.nanmean(
            self.gnomonic_bounds, axis=(0, 1, 2)[: self.navigation_dimension]
        )

    # ------------------------ Dunder methods ------------------------ #

    def __repr__(self) -> str:
        decimals = 3
        pc_average = tuple(map(float, self.pc_average.round(decimals)))
        shape = tuple(map(int, self.shape))
        sample_tilt = str(np.round(self.sample_tilt, decimals))
        tilt = str(np.round(self.tilt, decimals))
        azimuthal = str(np.round(self.azimuthal, decimals))
        twist = str(np.round(self.twist, decimals))
        px_size = str(np.round(self.px_size, decimals))
        binning = str(self.binning)

        s = f"{self.__class__.__name__}\n"
        s += f"  shape (Ny, Nx):     {shape}\n"
        s += f"  pc (PCx, PCy, PCz): {pc_average}\n"
        s += f"  sample_tilt:        {sample_tilt} deg\n"
        s += f"  tilt:               {tilt} deg\n"
        s += f"  azimuthal:          {azimuthal} deg\n"
        s += f"  twist:              {twist} deg\n"
        s += f"  binning:            {binning}\n"
        s += f"  px_size:            {px_size} um"

        return s

    # ------------------------ Public methods ------------------------ #

    @classmethod
    def load(cls, fname: Path | str) -> Self:
        """Return an EBSD detector loaded from a text file saved with
        :meth:`save`.

        Parameters
        ----------
        fname
            Full path to file.

        Returns
        -------
        detector
            Loaded EBSD detector.
        """
        from kikuchipy.io._detectors import read_ebsd_detector_from_file

        detector_kw = read_ebsd_detector_from_file(fname)
        detector = cls(**detector_kw)

        return detector

    def convert_detector_coordinates(
        self,
        coords: np.ndarray,
        direction: str,
        detector_index: int | tuple | None = None,
    ) -> np.ndarray:
        """Convert between gnomonic and pixel coordinates on the detector screen.


        Parameters
        ----------
        coords
            A 2D array of coordinates of any shape whereby the
            x and y coordinates to be converted are stored in
            the last axis.
        direction
            Either "pix_to_gn" or "gn_to_pix", depending on the
            direction of conversion needed.
        detector_index
            Index showing which conversion factors in *conversions[direction]*
            should be applied to *coords*.
            If None, **all** conversion factors in *conversions[direction]*
            are applied to *coords*.
            If an int is supplied, this refers to an index in a 1D dataset.
            A 1D tuple *e.g.* (3,) can also be passed for a 1D dataset.
            A 2D index can be specified by supplying a tuple *e.g.* (2, 3).
            The default value is None

        Returns
        -------
        coords_out
            Array of coords but with values converted as specified
            by direction. The shape is either the same as the input
            or is the navigation shape then the shape of the input.

        Examples
        --------

        Convert a single point on the detector in pixel coordinates into
        gnomonic coordinates for all patterns in the dataset.

        >>> import numpy as np
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> det = s.detector
        >>> det.navigation_shape
        (3, 3)
        >>> coords = np.array([[36.2, 12.7]])
        >>> coords.shape
        (1, 2)
        >>> coords_out = det.convert_detector_coordinates(coords, "pix_to_gn", None)
        >>> coords_out.shape
        (3, 3, 1, 2)
        >>> coords_out.squeeze()
        array([[[ 0.36223464,  0.00664684],
                [ 0.35762801, -0.00304659],
                [ 0.35361398, -0.00042112]],
        <BLANKLINE>
               [[ 0.36432453,  0.00973461],
                [ 0.35219231,  0.00567801],
                [ 0.34417285,  0.00404584]],
        <BLANKLINE>
               [[ 0.36296371,  0.00072557],
                [ 0.34447751,  0.00538137],
                [ 0.36136688,  0.00180754]]])

        Convert three points on the detector in pixel coordinates into
        gnomonic coordinates for the pattern at navigation index (1, 2)
        in the dataset.

        >>> import numpy as np
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> det = s.detector
        >>> det.navigation_shape
        (3, 3)
        >>> coords = np.array([[36.2, 12.7], [2.5, 43.7], [8.2, 27.7]])
        >>> coords.shape
        (3, 2)
        >>> coords_out = det.convert_detector_coordinates(coords, "pix_to_gn", (1, 2))
        >>> coords_out.shape
        (3, 2)
        >>> coords_out
        array([[ 0.34417285,  0.00404584],
               [-0.77639565, -1.02674418],
               [-0.58686329, -0.49472353]])
        """
        coords_out = convert_coordinates(
            coords, direction, self.coordinate_conversion_factors, detector_index
        )
        return coords_out

    def crop(self, extent: tuple[int, int, int, int] | list[int]) -> Self:
        """Return a new detector with its :attr:`shape` cropped and
        :attr:`pc` values updated accordingly.

        Parameters
        ----------
        extent
            The (top, bottom, left, right) pixels giving the extent of
            the cropped detector.

        Returns
        -------
        new_detector
            A new detector with a new shape and PC values.
        """
        ny, nx = self.shape

        # Unpack extent, making sure it does not go outside the original
        # shape
        top, bottom, left, right = extent
        top = max(top, 0)
        bottom = min(bottom, ny)
        left = max(left, 0)
        right = min(right, nx)

        ny_new, nx_new = bottom - top, right - left
        if any([ny_new <= 0, nx_new <= 0]) or any(
            [not isinstance(e, int) for e in extent]
        ):
            raise ValueError(
                "Extent (top, bottom, left, right) must be integers and given so that"
                " bottom > top and right > left"
            )

        pcx_new = (self.pcx * nx - left) / nx_new
        pcy_new = (self.pcy * ny - top) / ny_new
        pcz_new = self.pcz * ny / ny_new

        return self.__class__(
            shape=(ny_new, nx_new),
            pc=np.dstack((pcx_new, pcy_new, pcz_new)),
            tilt=self.tilt,
            sample_tilt=self.sample_tilt,
            binning=int(self._binning),
            px_size=self.px_size,
            azimuthal=self.azimuthal,
        )

    def deepcopy(self) -> Self:
        """Return a deep copy using :func:`copy.deepcopy`.

        Returns
        -------
        detector
            Identical detector without shared memory.
        """
        return deepcopy(self)

    @overload
    def estimate_xtilt(
        self,
        detect_outliers: bool = False,
        plot: bool = True,
        degrees: bool = False,
        return_figure: Literal[False] = False,
        return_outliers: Literal[False] = False,
        figure_kwargs: dict | None = None,
    ) -> float: ...

    @overload
    def estimate_xtilt(
        self,
        detect_outliers: bool = False,
        plot: bool = True,
        degrees: bool = False,
        return_figure: Literal[False] = False,
        return_outliers: Literal[True] = ...,
        figure_kwargs: dict | None = None,
    ) -> tuple[float, np.ndarray]: ...

    @overload
    def estimate_xtilt(
        self,
        detect_outliers: bool = False,
        plot: Literal[True] = True,
        degrees: bool = False,
        return_figure: Literal[True] = ...,
        return_outliers: Literal[False] = False,
        figure_kwargs: dict | None = None,
    ) -> "tuple[float, mfigure.Figure]": ...

    @overload
    def estimate_xtilt(
        self,
        detect_outliers: bool = False,
        plot: Literal[True] = True,
        degrees: bool = False,
        return_figure: Literal[True] = ...,
        return_outliers: Literal[True] = ...,
        figure_kwargs: dict | None = None,
    ) -> "tuple[float, np.ndarray, mfigure.Figure]": ...

    @overload
    def estimate_xtilt(
        self,
        detect_outliers: bool = False,
        plot: Literal[False] = ...,
        degrees: bool = False,
        return_figure: Literal[True] = ...,
        return_outliers: Literal[False] = False,
        figure_kwargs: dict | None = None,
    ) -> float: ...

    @overload
    def estimate_xtilt(
        self,
        detect_outliers: bool = False,
        plot: Literal[False] = ...,
        degrees: bool = False,
        return_figure: Literal[True] = ...,
        return_outliers: Literal[True] = ...,
        figure_kwargs: dict | None = None,
    ) -> tuple[float, np.ndarray]: ...

    def estimate_xtilt(
        self,
        detect_outliers: bool = False,
        plot: bool = True,
        degrees: bool = False,
        return_figure: bool = False,
        return_outliers: bool = False,
        figure_kwargs: dict | None = None,
    ) -> (
        float
        | tuple[float, np.ndarray]
        | "tuple[float, mfigure.Figure]"
        | "tuple[float, np.ndarray, mfigure.Figure]"
    ):
        r"""Return an estimate of the tilt about the detector
        :math:`X_d` axis.

        The tilt is assumed to bring the sample plane normal into
        coincidence with the detector plane normal (but in the opposite
        direction). See :cite:`winkelmann2020refined` for further
        details.

        See the :ref:`reference frame tutorial
        </tutorials/reference_frames.ipynb>` for details on the
        detector-sample geometry.

        An estimate is found by linear regression of :attr:`pcz` vs.
        :attr:`pcy`.

        Parameters
        ----------
        detect_outliers
            Whether to attempt to detect outliers. If False (default), a
            linear fit to all points is performed. If True, a robust fit
            using the RANSAC algorithm is performed instead, which also
            detects outliers.
        plot
            Whether to plot data points and the estimated line. Default
            is True.
        degrees
            Whether to return the estimated tilt in degrees, radians
            otherwise.
        return_figure
            Whether to return the plotted figure. Default is False.
        return_outliers
            Whether to return a mask with True for PC values considered
            outliers. Default is False. If True, *detect_outliers* is
            set to True.
        figure_kwargs
            Keyword arguments passed to
            :func:`matplotlib.pyplot.Figure` if *plot* is True.

        Returns
        -------
        x_tilt
            Estimated tilt about detector :math:`X_d` in radians, unless
            *degrees* is True.
        outliers
            Returned if *return_outliers* is True, in the shape of
            :attr:`navigation_shape`.
        fig
            Returned if *plot* is True and *return_figure* is True.

        Notes
        -----
        This method is adapted from Aimo Winkelmann's function
        ``fit_xtilt()`` in the *xcdskd* Python package. See
        :cite:`winkelmann2020refined` for their use of related
        functions.

        See Also
        --------
        sklearn.linear_model.LinearRegression,
        sklearn.linear_model.RANSACRegressor,
        :meth:`~kikuchipy.detectors.EBSDDetector.estimate_xtilt_ztilt`,
        :meth:`~kikuchipy.detectors.EBSDDetector.fit_pc`
        """
        from kikuchipy.detectors._fit_projection_center import (
            estimate_xtilt_linear,
            estimate_xtilt_linear_robust,
        )
        from kikuchipy.draw._projection_center_plot import plot_xtilt_estimate

        if self.navigation_size == 1:
            raise ValueError("Estimation requires more than one projection center")

        if return_outliers:
            detect_outliers = True

        if detect_outliers:
            x_tilt, regressor, is_outlier = estimate_xtilt_linear_robust(detector=self)
        else:
            x_tilt, regressor = estimate_xtilt_linear(detector=self)
            is_outlier = None

        if degrees:
            out = (float(np.rad2deg(x_tilt)),)
        else:
            out = (x_tilt,)

        if is_outlier is not None:
            is_outlier_2d = is_outlier.reshape(self.navigation_shape)
            out += (is_outlier_2d,)

        if plot:
            pcy = self.pcy.reshape((-1, 1))
            pcz = self.pcz.reshape((-1, 1))
            pcz_fit = np.linspace(np.min(pcz), np.max(pcz), 2)
            pcy_fit = regressor.predict(pcz_fit[:, np.newaxis])
            if figure_kwargs is None:
                figure_kwargs = {}
            fig = plot_xtilt_estimate(
                pcy=pcy,
                pcz=pcz,
                pcy_fit=pcy_fit,
                pcz_fit=pcz_fit,
                x_tilt=x_tilt,
                is_outlier=is_outlier,
                **figure_kwargs,
            )
            if return_figure:
                out += (fig,)

        if len(out) == 1:
            out = out[0]

        return out

    def estimate_xtilt_ztilt(
        self, degrees: bool = False, is_outlier: list | tuple | np.ndarray | None = None
    ) -> tuple[float, float]:
        r"""Return estimated tilts about the detector :math:`X_d` and
        :math:`Z_d` axes.

        These tilts bring the sample plane normal into coincidence with
        the detector plane normal (but in the opposite direction). See
        :cite:`winkelmann2020refined` for further details.

        See the :ref:`reference frame tutorial
        </tutorials/reference_frames.ipynb>` for details on the
        detector-sample geometry.

        Estimates are found by fitting a hyperplane to :attr:`pc` using
        singular value decomposition.

        Parameters
        ----------
        degrees
            Whether to return the estimated tilt in degrees, radians
            otherwise.
        is_outlier
            Boolean array with True for PCs to not include in the fit.
            If not given, all PCs are used. Must be of
            :attr:`navigation_shape`.

        Returns
        -------
        x_tilt
            Estimated tilt about detector :math:`X_d` in radians, unless
            *degrees* is True.
        z_tilt
            Estimated tilt about detector :math:`Z_d` in radians, unless
            *degrees* is True.

        Notes
        -----
        This method is adapted from Aimo Winkelmann's function
        ``fit_plane()`` in the *xcdskd* Python package. Its use is
        described in :cite:`winkelmann2020refined`.

        Winkelmann et al. refers to Gander & Hrebicek, "Solving Problems
        in Scientific Computing", 3rd Ed., Chapter 6, p. 97 for the
        implementation of the hyperplane fitting.

        See Also
        --------
        estimate_xtilt, fit_pc
        """
        from kikuchipy.detectors._fit_projection_center import fit_hyperplane

        if self.navigation_size == 1:
            raise ValueError("Estimation requires more than one projection center")

        pc = self.pc
        if isinstance(is_outlier, np.ndarray):
            pc = pc[~is_outlier]
        pc = pc.reshape((-1, 3))

        pc_centered = pc - pc.mean(axis=0)
        x_tilt, z_tilt, *_ = fit_hyperplane(pc_centered)

        if degrees:
            x_tilt = np.rad2deg(x_tilt)
            z_tilt = np.rad2deg(z_tilt)

        return x_tilt, z_tilt

    def extrapolate_pc(
        self,
        pc_indices: tuple | list | np.ndarray,
        navigation_shape: tuple,
        step_sizes: tuple,
        shape: tuple | None = None,
        px_size: float | None = None,
        binning: int | float | None = None,
        is_outlier: tuple | list | np.ndarray | None = None,
    ) -> Self:
        r"""Return a new detector with projection centers (PCs) in a 2D
        map extrapolated from an average PC.

        Parameters
        ----------
        pc_indices
            2D map pixel coordinates (row, column) of each :attr:`pc`,
            possibly outside *navigation_shape*. Must be a flattened
            array of shape (2,) + :attr:`navigation_size`.
        navigation_shape
            Shape of the output PC array (n rows, n columns).
        step_sizes
            Vertical and horizontal step sizes (dy, dx).
        shape
            Detector (signal) shape (n rows, n columns). If not given,
            (:attr:`nrows`, :attr:`ncols`) is used.
        px_size
            Unbinned detector pixel size. If not given, :attr:`px_size`
            is used.
        binning
            Detector binning factor. If not given, :attr:`binning` is
            used.
        is_outlier
            Boolean array with *True* for PCs to not include in the fit.
            If not given, all PCs are used. Must be of
            :attr:`navigation_shape`.

        Returns
        -------
        new_detector
            Detector with :attr:`navigation_shape` given by input
            *navigation_shape*.

        Notes
        -----
        The average PC :math:`\bar{PC}` is calculated from :attr:`pc`,
        possibly excluding some PCs based on the *is_outlier* mask. The
        sample position having this PC, :math:`(\bar{x}, \bar{y})`, is
        assumed to be the one obtained by averaging *pc_indices*. All
        other PCs :math:`(PC_x, PC_y, PC_z)` in positions :math:`(x, y)`
        are then extrapolated based on the following equations given in
        appendix A in :cite:`singh2017application`:

        .. math::
            PC_x &= \bar{PC_x} + (\bar{x} - x) \cdot \Delta x / (\delta \cdot N_x \cdot b),\\
            PC_y &= \bar{PC_y} + (\bar{y} - y) \cdot \Delta y \cdot \cos{\alpha} / (\delta \cdot N_y \cdot b),\\
            PC_z &= \bar{PC_z} - (\bar{y} - y) \cdot \Delta y \cdot \sin{\alpha} / (\delta \cdot N_y \cdot b),\\

        where :math:`(\Delta y, \Delta x)` are the vertical and
        horizontal step sizes, respectively, :math:`(N_y, N_x)` are the
        number of binned detector rows and columns, respectively, the
        angle :math:`\alpha = 90^{\circ} - \sigma + \theta`, where
        :math:`\sigma` is the sample tilt and :math:`\theta` is the
        detector tilt, :math:`\delta` is the unbinned detector pixel
        size and :math:`b` is the binning factor.
        """
        dy, dx = step_sizes
        pc_indices = np.atleast_2d(pc_indices).T
        ny, nx = navigation_shape
        if shape is None:
            shape = self.shape
        nrows, ncols = shape
        if px_size is None:
            px_size = self.px_size
        if binning is None:
            binning = float(self.binning)

        pc = self.pc_flattened

        if is_outlier is not None:
            is_inlier = ~np.asarray(is_outlier)
            pc = pc[is_inlier]
            pc_indices = pc_indices[is_inlier]

        # Calculate mean PC and position
        pc_mean = pc.mean(axis=0)
        pc_indices_mean = pc_indices.mean(axis=1)
        pc_indices_mean = np.round(pc_indices_mean).astype(int)

        # Make PC plane
        alpha = np.deg2rad(90 - self.sample_tilt + self.tilt)
        y, x = np.indices((ny, nx), dtype=float)
        factor = px_size * binning
        d_pcx = -(pc_indices_mean[1] - x) * dx / (factor * ncols)
        d_pcy = -(pc_indices_mean[0] - y) * dy * np.cos(alpha) / (factor * nrows)
        d_pcz = +(pc_indices_mean[0] - y) * dy * np.sin(alpha) / (factor * nrows)
        pcx = pc_mean[0] - d_pcx
        pcy = pc_mean[1] - d_pcy
        pcz = pc_mean[2] - d_pcz

        new_detector = self.deepcopy()
        new_detector._shape = shape
        new_detector.pc = np.stack((pcx, pcy, pcz), axis=2)
        new_detector.px_size = px_size
        new_detector.binning = int(binning)

        return new_detector

    @overload
    def fit_pc(
        self,
        pc_indices: list | tuple | np.ndarray,
        map_indices: list | tuple | np.ndarray,
        transformation: Literal["projective", "affine"] = ...,
        is_outlier: np.ndarray | None = None,
        plot: bool = True,
        return_figure: Literal[False] = False,
        figure_kwargs: dict[str, Any] | None = None,
    ) -> Self: ...

    @overload
    def fit_pc(
        self,
        pc_indices: list | tuple | np.ndarray,
        map_indices: list | tuple | np.ndarray,
        transformation: Literal["projective", "affine"] = ...,
        is_outlier: np.ndarray | None = None,
        plot: bool = True,
        return_figure: Literal[True] = True,
        figure_kwargs: dict[str, Any] | None = None,
    ) -> "tuple[Self, mfigure.Figure]": ...

    def fit_pc(
        self,
        pc_indices: list | tuple | np.ndarray,
        map_indices: list | tuple | np.ndarray,
        transformation: Literal["projective", "affine"] = "projective",
        is_outlier: np.ndarray | None = None,
        plot: bool = True,
        return_figure: bool = False,
        figure_kwargs: dict[str, Any] | None = None,
    ) -> "Self | tuple[Self, mfigure.Figure]":
        """Return a new detector with interpolated projection centers
        (PCs) for all points in a map by fitting a plane to :attr:`pc`.

        See :cite:`winkelmann2020refined` for further details.

        Parameters
        ----------
        pc_indices
            2D coordinates (row, column) of each :attr:`pc` in
            *map_indices*. Must be a flattened array of shape (2,) +
            :attr:`navigation_shape`.
        map_indices
            2D coordinates (row, column) of all map points in a regular
            grid to interpolate PCs for. Must be a flattened array of
            shape (2,) + :attr:`navigation_shape`.
        transformation
            Which transformation function to use when fitting PCs,
            either "projective" (default) or "affine". Both
            transformations preserve co-planarity of map points. The
            projective transformation allows parallel lines in the map
            point grid to become non-parallel within the sample plane.
        is_outlier
            Boolean array with True for PCs to not include in the fit.
            If not given, all PCs are used. Must be of
            :attr:`navigation_shape`.
        plot
            Whether to plot the experimental and estimated PCs (default
            is True).
        return_figure
            Whether to return the figure if *plot* is True (default is
            False).
        figure_kwargs
            Keyword arguments passed to
            :func:`matplotlib.pyplot.Figure` if *plot* is True.

        Returns
        -------
        new_detector
            New detector with as many interpolated PCs as indices given
            in *map_indices* and an estimated sample tilt. The detector
            tilt is assumed to be constant.
        fig
            Figure of experimental and estimated PCs, returned if
            *plot* is True and *return_figure* is True.

        Raises
        ------
        ValueError
            If :attr:`navigation_size` is 1 or if the *pc_indices* or
            *map_indices* arrays have the incorrect shape.

        See Also
        --------
        :meth:`~kikuchipy.detectors.EBSDDetector.estimate_xtilt`,
        :meth:`~kikuchipy.detectors.EBSDDetector.estimate_xtilt_ztilt`,
        :meth:`~kikuchipy.detectors.EBSDDetector.extrapolate_pc`

        Notes
        -----
        This method is adapted from Aimo Winkelmann's functions
        ``fit_affine()`` and ``fit_projective()`` in the *xcdskd* Python
        package. Their uses are described in
        :cite:`winkelmann2020refined`. Winkelmann et al. refers to a
        code example from StackOverflow
        (https://stackoverflow.com/a/20555267/3228100) for the affine
        transformation.
        """
        from kikuchipy.detectors._fit_projection_center import fit_plane_to_pc
        from kikuchipy.draw._projection_center_plot import plot_projection_center_fit

        n_pc = self.navigation_size
        if n_pc == 1:
            raise ValueError("Fitting requires multiple projection centers (PCs)")

        pc_indices = np.asarray(pc_indices)
        map_indices = np.asarray(map_indices)

        nav_shape = self.navigation_shape
        if pc_indices.shape != (2,) + nav_shape:
            raise ValueError(
                f"`pc_indices` array shape {pc_indices.shape} must be equal to "
                f"{(2,) + nav_shape}"
            )

        if map_indices.ndim not in [2, 3] or map_indices.shape[0] != 2:
            raise ValueError(
                f"`map_indices` array shape {map_indices.shape} must be (2, m columns) "
                "or (2, n rows, m columns)"
            )

        if is_outlier is not None and (
            not isinstance(is_outlier, np.ndarray)
            or not np.issubdtype(is_outlier.dtype, np.bool_)
            or is_outlier.size != n_pc
        ):
            raise ValueError(
                "`is_outlier` must be a boolean array of a size equal to the number of "
                "PCs"
            )

        pc_fit, pc_fit_map, pc_flat, x_tilt, intercept, slope = fit_plane_to_pc(
            detector=self,
            pc_indices=pc_indices,
            map_indices=map_indices,
            is_outlier=is_outlier,
            transformation=transformation,
        )

        x_tilt_deg = np.rad2deg(x_tilt)
        new_detector = self.deepcopy()
        new_detector.pc = pc_fit_map
        new_detector.sample_tilt = 90 - x_tilt_deg - new_detector.tilt

        if plot:
            if figure_kwargs is None:
                figure_kwargs = {}
            if isinstance(is_outlier, np.ndarray):
                nav_shape = (int(np.sum(~is_outlier)),)
            pc_fit_2d = pc_fit.reshape(nav_shape + (3,))
            fig = plot_projection_center_fit(
                pc=pc_flat,
                pc_fit=pc_fit_2d,
                fit_intercept=intercept,
                fit_slope=slope,
                **figure_kwargs,
            )
        else:
            fig = None

        if return_figure and fig is not None:
            return new_detector, fig
        else:
            return new_detector

    def get_indexer(
        self,
        phase_list: PhaseList,
        reflectors: (
            list[ReciprocalLatticeVector | np.ndarray | list | tuple | None] | None
        ) = None,
        **kwargs,
    ) -> "EBSDIndexer":
        r"""Return a PyEBSDIndex EBSD indexer.

        Parameters
        ----------
        phase_list
            List of phases. :class:`~pyebsdindex.ebsd_index.EBSDIndexer`
            only supports a list containing one face-centered cubic
            (FCC) phase, one body-centered cubic (BCC) phase or both.
        reflectors
            List of reflectors or pole families :math:`\{hkl\}` to use
            in indexing for each phase. If not passed, the default in
            :func:`pyebsdindex.tripletvote.addphase` is used. For each
            phase, the reflectors can either be a NumPy array, a list,
            a tuple, a
            :class:`~diffsis.crystallography.ReciprocalLatticeVector`,
            or None.
        **kwargs
            Keyword arguments passed to
            :class:`~pyebsdindex.ebsd_index.EBSDIndexer`, except for the
            following arguments which cannot be passed since they are
            determined from the detector or *phase_list*:
            ``phaselist`` (not to be confused with *phase_list*),
            ``vendor``, ``PC``, ``sampleTilt``, ``camElev`` and
            ``patDim``.

        Returns
        -------
        pyebsdindex.ebsd_index.EBSDIndexer
            Indexer instance for use with PyEBSDIndex or in
            :meth:`~kikuchipy.signals.EBSD.hough_indexing`.
            ``indexer.PC`` is set equal to :attr:`pc_flattened`.

        Notes
        -----
        Requires that PyEBSDIndex is installed, which is an optional
        dependency of kikuchipy. See :ref:`dependencies` for details.

        See Also
        --------
        pyebsdindex.tripletvote.addphase
        """
        return _get_indexer_from_detector(
            phase_list=phase_list,
            shape=self.shape,
            pc=self.pc_flattened.squeeze(),
            sample_tilt=self.sample_tilt,
            tilt=self.tilt,
            reflectors=reflectors,
            **kwargs,
        )

    def pc_emsoft(self, version: int = 5) -> np.ndarray:
        r"""Return the projection center(s) :attr:`pc` in the EMsoft
        convention.

        Parameters
        ----------
        version
            Which EMsoft PC convention to use. The direction of the x PC
            coordinate, :math:`x_{pc}`, flipped in version 5.

        Returns
        -------
        new_pc
            PC in the EMsoft convention.

        Notes
        -----
        The PC coordinate conventions of Bruker, EDAX TSL, Oxford
        Instruments, and EMsoft are given in :class:`EBSDDetector`. The
        PC is stored in the Bruker convention internally, so the
        conversion is

        .. math::

            x_{pc} &= N_x b \left(\frac{1}{2} - x_B^*\right),\\
            y_{pc} &= N_y b \left(\frac{1}{2} - y_B^*\right),\\
            L &= N_y b \delta z_B^*,

        where :math:`N_x` and :math:`N_y` are number of detector columns
        and rows, :math:`b` is binning, :math:`\delta` is the unbinned
        pixel size, :math:`(x_B^*, y_B^*, z_B^*)` are the Bruker PC
        coordinates, and :math:`(x_{pc}, y_{pc}, L)` are the returned
        EMsoft PC coordinates.

        Examples
        --------
        >>> import kikuchipy as kp
        >>> det = kp.detectors.EBSDDetector(
        ...     shape=(60, 80),
        ...     pc=(0.4, 0.2, 0.6),
        ...     convention="bruker",
        ...     px_size=59.2,
        ...     binning=8,
        ... )
        >>> det.pc_emsoft()
        array([[   64. ,   144. , 17049.6]])
        >>> det.pc_emsoft(4)
        array([[  -64. ,   144. , 17049.6]])
        """
        return self._pc_bruker2emsoft(version=version)

    def pc_bruker(self) -> np.ndarray:
        """Return PC in the Bruker convention, given in the class
        description.

        Returns
        -------
        new_pc
            PC in the Bruker convention.
        """
        return self.pc

    def pc_tsl(self) -> np.ndarray:
        r"""Return PC in the EDAX TSL convention.

        Returns
        -------
        new_pc
            PC in the EDAX TSL convention.

        Notes
        -----
        The PC coordinate conventions of Bruker, EDAX TSL, Oxford
        Instruments and EMsoft are given in the class description. The
        PC is stored in the Bruker convention internally, so the
        conversion is

        .. math::

            x_T^* &= x_B^*,\\
            y_T^* &= \frac{N_y}{N_x} (1 - y_B^*),\\
            z_T^* &= \frac{N_y}{N_x} z_B^*,

        where :math:`N_x` and :math:`N_y` are number of detector columns
        and rows, :math:`(x_B^*, y_B^*, z_B^*)` are the Bruker PC
        coordinates, and :math:`(x_T^*, y_T^*, z_T^*)` are the returned
        EDAX TSL PC coordinates.

        Examples
        --------
        >>> import kikuchipy as kp
        >>> det = kp.detectors.EBSDDetector(
        ...     shape=(60, 80),
        ...     pc=(0.4, 0.2, 0.6),
        ...     convention="bruker",
        ... )
        >>> det.pc_tsl()
        array([[0.4, 0.8, 0.6]])
        """
        return self._pc_bruker2tsl()

    def pc_oxford(self) -> np.ndarray:
        """Return PC in the Oxford convention.

        Returns
        -------
        new_pc
            PC in the Oxford convention.

        Notes
        -----
        The PC coordinates are stored in Bruker's convention. See the
        *Notes* in :class:`EBSDDetector` for more information.
        """
        return self._pc_bruker2oxford()

    @overload
    def plot(
        self,
        coordinates: DETECTOR_PLOT_FORMATS = "detector",
        show_pc: bool = ...,
        pc_kwargs: dict | None = None,
        pattern: np.ndarray | None = None,
        pattern_kwargs: dict | None = None,
        draw_gnomonic_circles: bool = False,
        gnomonic_angles: np.ndarray | list | None = None,
        gnomonic_circles_kwargs: dict | None = None,
        zoom: float = ...,
        return_figure: Literal[False] = ...,
    ) -> None: ...

    @overload
    def plot(
        self,
        coordinates: DETECTOR_PLOT_FORMATS = "detector",
        show_pc: bool = ...,
        pc_kwargs: dict | None = None,
        pattern: np.ndarray | None = None,
        pattern_kwargs: dict | None = None,
        draw_gnomonic_circles: bool = False,
        gnomonic_angles: np.ndarray | list | None = None,
        gnomonic_circles_kwargs: dict | None = None,
        zoom: float = ...,
        return_figure: Literal[True] = ...,
    ) -> "None | mfigure.Figure | mfigure.SubFigure": ...

    def plot(
        self,
        coordinates: DETECTOR_PLOT_FORMATS = "detector",
        show_pc: bool = True,
        pc_kwargs: dict | None = None,
        pattern: np.ndarray | None = None,
        pattern_kwargs: dict | None = None,
        draw_gnomonic_circles: bool = False,
        gnomonic_angles: np.ndarray | list[float] | None = None,
        gnomonic_circles_kwargs: dict | None = None,
        zoom: float = 1.0,
        return_figure: bool = False,
    ) -> "None | mfigure.Figure | mfigure.SubFigure":
        """Plot the detector screen viewed from the detector towards the
        sample.

        The plotting of gnomonic circles and general style is adapted
        from the supplementary material to :cite:`britton2016tutorial`
        by Aimo Winkelmann.

        Parameters
        ----------
        coordinates
            Which coordinates to use, "detector" (default) or
            "gnomonic".
        show_pc
            Show the average projection center in the Bruker convention.
            Default is True.
        pc_kwargs
            The following arguments are used:

            - "s": PC size. Default is 100.
            - "zorder": The z-order. Default is 10.
        pattern
            A pattern to put on the detector. If not given, no pattern
            is displayed. The pattern array must have the same shape as
            the detector.
        pattern_kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.imshow`.
        draw_gnomonic_circles
            Draw circles for angular distances from pattern. Default is
            False. Circle positions are only correct when *coordinates*
            are gnomonic.
        gnomonic_angles
            Which angular distances to plot if
            *draw_gnomonic_circles* is True, in degrees. Default is from
            10 to 80 degrees in steps of 10 degrees.
        gnomonic_circles_kwargs
            Keyword arguments passed to
            :meth:`matplotlib.patches.Circle`.
        zoom
            Whether to zoom in/out from the detector, e.g. to show the
            extent of the gnomonic projection circles. Note that a zoom
            > 1 zooms out. Default is 1 (no zoom).
        return_figure
            Whether to return the figure. Default is False.

        Returns
        -------
        fig
            Matplotlib figure instance, if *return_figure* is True.

        See Also
        --------
        :class:`~kikuchipy.draw.EBSDDetectorPlotter`,
        :meth:`~kikuchipy.detectors.EBSDDetector.plot_side_view`,
        :meth:`~kikuchipy.detectors.EBSDDetector.plot_top_view`,
        :meth:`~kikuchipy.detectors.EBSDDetector.plot_interactive`
        """
        from kikuchipy.draw._ebsd_detector_plot import plot_ebsd_detector

        if pattern_kwargs is None:
            pattern_kwargs = {}
        if pc_kwargs is None:
            pc_kwargs = {}
        if gnomonic_circles_kwargs is None:
            gnomonic_circles_kwargs = {}
        fig = plot_ebsd_detector(
            detector=self,
            coords_fmt=coordinates,
            zoom=zoom,
            show_pc=show_pc,
            draw_gnomonic_circles=draw_gnomonic_circles,
            pattern=pattern,
            pattern_kwargs=pattern_kwargs,
            pc_kwargs=pc_kwargs,
            gnomonic_circles_kwargs=gnomonic_circles_kwargs,
            gnomonic_angles=gnomonic_angles,
        )

        if return_figure:
            return fig

    @overload
    def plot_side_view(
        self,
        ax: "maxes.Axes | None" = None,
        legend: bool = False,
        dimensionless: bool = True,
        interactive: Literal[False] = False,
        return_figure: Literal[False] = False,
        **kwargs,
    ) -> None: ...

    @overload
    def plot_side_view(
        self,
        ax: "maxes.Axes | None" = None,
        legend: bool = False,
        dimensionless: bool = True,
        interactive: Literal[True] = True,
        return_figure: Literal[False] = False,
        **kwargs,
    ) -> "ipywidgets.VBox": ...

    @overload
    def plot_side_view(
        self,
        ax: "maxes.Axes | None" = None,
        legend: bool = False,
        dimensionless: bool = True,
        interactive: Literal[False] = False,
        return_figure: Literal[True] = True,
        **kwargs,
    ) -> "mfigure.Figure | mfigure.SubFigure": ...

    @overload
    def plot_side_view(
        self,
        ax: "maxes.Axes | None" = None,
        legend: bool = False,
        dimensionless: bool = True,
        interactive: Literal[True] = True,
        return_figure: Literal[True] = True,
        **kwargs,
    ) -> "tuple[ipywidgets.VBox, mfigure.Figure | mfigure.SubFigure]": ...

    def plot_side_view(
        self,
        ax: "maxes.Axes | None" = None,
        legend: bool = False,
        dimensionless: bool = True,
        interactive: bool = False,
        return_figure: bool = False,
        **kwargs,
    ) -> "None | ipywidgets.VBox | tuple[ipywidgets.VBox, mfigure.Figure | mfigure.SubFigure] | mfigure.Figure | mfigure.SubFigure":
        r"""Plot the EBSD detector-sample geometry in a 2D side-view.

        The view is looking down the microscope Z axis and shows the
        X-Y plane.

        Parameters
        ----------
        ax
            The Matplotlib axis to plot in. If not given, a new figure
            and axis are created.
        legend
            Whether to show a legend in the upper right corner.
            Default is False.
        dimensionless
            Whether to ignore the
            :attr:`~kikuchipy.detectors.EBSDDetector.px_size` when
            drawing the plot axes. Default is True.
        interactive
            Whether to return slider controls to interactively vary the
            relevant detector-sample geometry parameters in the
            side-view. Default is False. If True, the detector is varied
            inplace. Requires that :mod:`ipywidgets` is installed. If
            :mod:`psygnal` is installed, the plot is driven by signals
            emitted from the detector property setters.

            The following attributes can be varied:

            - :attr:`sample_tilt`
            - :attr:`tilt`
            - :attr:`pcy` and :attr:`pcz`

            The remaining attributes are ignored.
        return_figure
            Whether to return the figure. Default is False.
        **kwargs
            Keyword arguments passed to
            :func:`~matplotlib.pyplot.figure` if *ax* is not given.

        Returns
        -------
        controls
            The slider controls. Must be displayed using IPython.
        fig
            Figure showing the detector-sample geometry.

        See Also
        --------
        :meth:`~kikuchipy.detectors.EBSDDetector.plot_top_view`,
        :meth:`~kikuchipy.detectors.EBSDDetector.plot`,
        :meth:`~kikuchipy.detectors.EBSDDetector.plot_interactive`
        """
        from kikuchipy.draw._ebsd_detector_plot import (
            plot_detector_sample_geometry_side_view,
        )
        from kikuchipy.draw._ebsd_detector_plot_widgets import (
            plot_detector_sample_geometry_side_view_interactive,
        )

        if interactive:
            controls, fig = plot_detector_sample_geometry_side_view_interactive(
                detector=self,
                ax=ax,
                legend=legend,
                dimensionless=dimensionless,
                **kwargs,
            )
            if return_figure:
                return controls, fig
            else:
                return controls
        else:
            fig = plot_detector_sample_geometry_side_view(
                detector=self,
                ax=ax,
                legend=legend,
                dimensionless=dimensionless,
                **kwargs,
            )
            if return_figure:
                return fig

    @overload
    def plot_top_view(
        self,
        ax: "maxes.Axes | None" = None,
        legend: bool = False,
        dimensionless: bool = True,
        interactive: bool = False,
        return_figure: Literal[False] = False,
        **kwargs,
    ) -> None: ...

    @overload
    def plot_top_view(
        self,
        ax: "maxes.Axes | None" = None,
        legend: bool = False,
        dimensionless: bool = True,
        interactive: bool = False,
        return_figure: Literal[True] = True,
        **kwargs,
    ) -> "mfigure.Figure | mfigure.SubFigure": ...

    @overload
    def plot_top_view(
        self,
        ax: "maxes.Axes | None" = None,
        legend: bool = False,
        dimensionless: bool = True,
        interactive: Literal[True] = True,
        return_figure: bool = False,
        **kwargs,
    ) -> "ipywidgets.VBox": ...

    @overload
    def plot_top_view(
        self,
        ax: "maxes.Axes | None" = None,
        legend: bool = False,
        dimensionless: bool = True,
        interactive: Literal[True] = True,
        return_figure: Literal[True] = True,
        **kwargs,
    ) -> "tuple[ipywidgets.VBox, mfigure.Figure | mfigure.SubFigure]": ...

    def plot_top_view(
        self,
        ax: "maxes.Axes | None" = None,
        legend: bool = False,
        dimensionless: bool = True,
        interactive: bool = False,
        return_figure: bool = False,
        **kwargs,
    ) -> "None | ipywidgets.VBox | tuple[ipywidgets.VBox, mfigure.Figure | mfigure.SubFigure] | mfigure.Figure | mfigure.SubFigure":
        r"""Plot the EBSD detector-sample geometry in a 2D top-view.

        The view is looking down the microscope Z axis and shows the
        X-Y plane. This is the view in which the detector azimuthal
        angle :math:`\omega` is visible. The effect of

        Parameters
        ----------
        ax
            The Matplotlib axis to plot in. If not given, a new figure
            and axis are created.
        legend
            Whether to show a legend in the upper right corner.
            Default is False.
        dimensionless
            Whether to ignore the
            :attr:`~kikuchipy.detectors.EBSDDetector.px_size` when
            drawing the plot axes. Default is True.
        interactive
            Whether to return slider controls to interactively vary the
            relevant detector-sample geometry parameters in the
            top-view. Default is False. If True, the detector is varied
            inplace. Requires that :mod:`ipywidgets` is installed. If
            :mod:`psygnal` is installed, the plot is driven by signals
            emitted from the detector property setters.

            The following attributes can be varied:

            - :attr:`azimuthal`
            - :attr:`pcx` and :attr:`pcz`

            The remaining attributes are ignored.
        return_figure
            Whether to return the figure. Default is False.
        **kwargs
            Keyword arguments passed to
            :func:`~matplotlib.pyplot.figure` if *ax* is not given.

        Returns
        -------
        fig
            Figure showing the detector-sample geometry from the top.

        See Also
        --------
        :meth:`~kikuchipy.detectors.EBSDDetector.plot_side_view`,
        :meth:`~kikuchipy.detectors.EBSDDetector.plot`,
        :meth:`~kikuchipy.detectors.EBSDDetector.plot_interactive`
        """
        from kikuchipy.draw._ebsd_detector_plot import (
            plot_detector_sample_geometry_top_view,
        )
        from kikuchipy.draw._ebsd_detector_plot_widgets import (
            plot_detector_sample_geometry_top_view_interactive,
        )

        if interactive:
            controls, fig = plot_detector_sample_geometry_top_view_interactive(
                detector=self,
                ax=ax,
                legend=legend,
                dimensionless=dimensionless,
                **kwargs,
            )
            if return_figure:
                return controls, fig
            else:
                return controls
        else:
            fig = plot_detector_sample_geometry_top_view(
                detector=self,
                ax=ax,
                legend=legend,
                dimensionless=dimensionless,
                **kwargs,
            )
            if return_figure:
                return fig

    @overload
    def plot_interactive(
        self,
        inplace: bool = True,
        legend: bool = False,
        dimensionless: bool = True,
        coordinates: DETECTOR_PLOT_FORMATS = "gnomonic",
        return_figure: Literal[False] = False,
        **kwargs,
    ) -> "ipywidgets.Widget": ...

    @overload
    def plot_interactive(
        self,
        inplace: bool = True,
        legend: bool = False,
        dimensionless: bool = True,
        coordinates: DETECTOR_PLOT_FORMATS = "gnomonic",
        return_figure: Literal[True] = True,
        **kwargs,
    ) -> "tuple[ipywidgets.Widget, mfigure.Figure]": ...

    def plot_interactive(
        self,
        inplace: bool = True,
        legend: bool = False,
        dimensionless: bool = True,
        coordinates: DETECTOR_PLOT_FORMATS = "gnomonic",
        return_figure: bool = False,
        **kwargs,
    ) -> "ipywidgets.Widget | tuple[ipywidgets.Widget, mfigure.Figure]":
        """Plot the side view, top view, and detector plane side by
        side with interactive slider controls.

        Parameters
        ----------
        inplace
            Whether the interactive changes affect the detector inplace.
            Default is True.
        legend
            Whether to show a legend in the upper right corner of the
            side and top view plots. Default is False.
        dimensionless
            Whether to ignore the
            :attr:`~kikuchipy.detectors.EBSDDetector.px_size` when
            drawing the side-view plot axes. Default is True.
        coordinates
            Detector plane coordinate format: "gnomonic" (default) or
            "detector".
        return_figure
            Whether to return the figure. Default is False. If False,
            :meth:`matplotlib.figure.Figure.show` is called before
            returning.
        **kwargs
            Keyword arguments passed to
            :func:`~matplotlib.pyplot.figure`.

        Returns
        -------
        widgets
            The widget containing the sliders. Required to display the
            interactive controls.

        See Also
        --------
        :class:`~kikuchipy.draw.EBSDDetectorPlotter`,
        :meth:`~kikuchipy.detectors.EBSDDetector.plot`,
        :meth:`~kikuchipy.detectors.EBSDDetector.plot_side_view`,
        :meth:`~kikuchipy.detectors.EBSDDetector.plot_top_view`

        Notes
        -----
        Requires that :mod:`ipywidgets` is installed.

        Parameters to vary:

        - :attr:`sample_tilt`
        - Detector :attr:`tilt`
        - Detector :attr:`azimuthal`
        - Average :attr:`pc`, (PCx, PCy, PCz), individually

        If :mod:`psygnal` is installed and *inplace* is True, the plot
        can be updated by any change to the above parameters, not just
        via the slider controls. However, any change not done via the
        sliders will not affect the sliders.
        """
        from kikuchipy.draw._ebsd_detector_plot_widgets import EBSDDetectorPlotter

        plotter = EBSDDetectorPlotter(
            self,
            inplace=inplace,
            legend=legend,
            dimensionless=dimensionless,
            coords_fmt=coordinates,
        )
        fig, widgets = plotter.show(**kwargs)

        if return_figure:
            return widgets, fig
        else:
            return widgets

    @overload
    def plot_pc(
        self,
        mode: PROJECTION_CENTER_PLOT_MODES = "map",
        return_figure: Literal[False] = False,
        orientation: Literal["horizontal", "vertical"] = "horizontal",
        annotate: bool = False,
        figure_kwargs: dict | None = None,
        **kwargs,
    ) -> None: ...

    @overload
    def plot_pc(
        self,
        mode: PROJECTION_CENTER_PLOT_MODES = ...,
        return_figure: Literal[True] = True,
        orientation: Literal["horizontal", "vertical"] = ...,
        annotate: bool = ...,
        figure_kwargs: dict | None = ...,
        **kwargs,
    ) -> "mfigure.Figure": ...

    def plot_pc(
        self,
        mode: PROJECTION_CENTER_PLOT_MODES = "map",
        return_figure: bool = False,
        orientation: Literal["horizontal", "vertical"] = "horizontal",
        annotate: bool = False,
        figure_kwargs: dict | None = None,
        **kwargs,
    ) -> "None | mfigure.Figure":
        """Plot all projection centers (PCs).

        Parameters
        ----------
        mode
            String describing how to plot PCs. Options are "map"
            (default), "scatter" and "3d". If map mode,
            :attr:`navigation_dimension` must be 2.
        return_figure
            Whether to return the figure (default is False).
        orientation
            Whether to align the plots in a "horizontal" (default) or
            "vertical" orientation.
        annotate
            Whether to label each pattern with its 1D index into
            :attr:`pc_flattened` when *mode* is "scatter". Default is
            False.
        figure_kwargs
            Keyword arguments to pass to
            :func:`matplotlib.pyplot.figure` upon figure creation. Note
            that ``layout="tight"`` is used by default unless another
            layout is given.
        **kwargs
            Keyword arguments passed to the plotting function, which is
            :meth:`~matplotlib.axes.Axes.imshow` if *mode* is "map",
            :meth:`~matplotlib.axes.Axes.scatter` if *mode* is
            "scatter", and
            :meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.scatter` if
            *mode* is "3d".

        Returns
        -------
        fig
            Figure is returned if *return_figure* is True.
        """
        from kikuchipy.draw._projection_center_plot import plot_all_projection_centers

        # Ensure there are PCs to plot
        if self.navigation_size == 1:
            raise ValueError(
                "Detector must have more than one projection center value to plot"
            )
        if mode == "map" and self.navigation_dimension != 2:
            raise ValueError(
                "Detector's navigation dimension must be 2D when plotting PCs in a map"
            )

        # Ensure mode is OK
        modes = ["map", "scatter", "3d"]
        if not isinstance(mode, str) or mode not in modes:
            raise ValueError(
                f"Plot mode {mode!r} must be one of the following strings {modes}"
            )

        if figure_kwargs is None:
            figure_kwargs = {}

        fig = plot_all_projection_centers(
            pc=self.pc,
            navigation_size=self.navigation_size,
            mode=mode,
            orientation=orientation,
            annotate=annotate,
            figure_kwargs=figure_kwargs,
            **kwargs,
        )

        if return_figure:
            return fig

    def save(
        self, filename: str | Path, convention: PC_CONVENTIONS = "bruker", **kwargs
    ) -> None:
        """Save the detector in a text file with projection centers
        (PCs) in the given convention.

        Parameters
        ----------
        filename
            Name of text file to write to. See :func:`~numpy.savetxt`
            for supported file formats.
        convention
            PC convention. Default is Bruker's convention. Options are
            "tsl"/"edax"/"amatek", "oxford"/"aztec", "bruker", and
            "emsoft"/"emsoft4"/"emsoft5". "emsoft" and "emsoft5" is the
            same convention. See *Notes* in :class:`EBSDDetector` for
            conversions between conventions.
        **kwargs
            Keyword arguments passed to :func:`~numpy.savetxt`, e.g.
            ``fmt="%.4f"`` to reduce the number of PC decimals from the
            default 7 to 4.
        """
        from kikuchipy.io._detectors import write_ebsd_detector_to_file

        write_ebsd_detector_to_file(
            detector=self, filename=filename, convention=convention, **kwargs
        )

    # ------------------------ Private methods ----------------------- #

    @staticmethod
    def _get_pc_convention_or_raise(conv: PC_CONVENTIONS) -> PC_CONVENTIONS_SINGLE:
        conv_lower = conv.lower()
        for k, v in PC_CONVENTIONS_ALIASES.items():
            if conv_lower in v:
                return k
        else:
            options = get_args(PC_CONVENTIONS)
            options_str = ", ".join(options)
            raise ValueError(
                f"Invalid projection/pattern center convention {conv!r}. Options are "
                f"{options_str}."
            )

    def _get_pc_in_bruker_convention(
        self, convention: PC_CONVENTIONS = "bruker"
    ) -> np.ndarray:
        """Convert current :attr:`pc` to Bruker's convention from
        another convention.

        Parameters
        ----------
        convention
            Convention of the current PCs. Default is "bruker".

        Returns
        -------
        pc
            PC array in Bruker's convention.
        """
        conv = self._get_pc_convention_or_raise(convention)
        if conv == "tsl":
            return self._pc_tsl2bruker()
        elif conv == "oxford":
            return self._pc_oxford2bruker()
        elif conv == "emsoft":
            try:
                version = int(convention[-1])
            except ValueError:
                version = 5
            return self._pc_emsoft2bruker(version)
        else:
            return self.pc

    def _get_pc_in_convention(
        self, convention: PC_CONVENTIONS = "bruker"
    ) -> np.ndarray:
        """Convert current :attr:`pc` from Bruker's convention to
        another convention.

        Parameters
        ----------
        convention
            Convention of the output PCs. Default is "bruker", which
            means the PCs are returned without conversion.

        Returns
        -------
        pc
            PC array in the given *convention*.
        """
        conv = self._get_pc_convention_or_raise(convention)
        if conv == "tsl":
            return self._pc_bruker2tsl()
        elif conv == "oxford":
            return self._pc_bruker2oxford()
        elif conv == "emsoft":
            try:
                version = int(convention[-1])
            except ValueError:
                version = 5
            return self._pc_bruker2emsoft(version)
        else:
            return self.pc

    @staticmethod
    def _parse_valid_shape_or_raise(value: tuple[int, int]) -> tuple[int, int]:
        if (
            not isinstance(value, Iterable)
            or len(value) != 2
            or not all(isinstance(ni, Number) for ni in value)
        ):
            raise ValueError(
                f"Invalid shape {value}. Must be an iterable of two integers."
            )
        ny, nx = value
        shape = (int(ny), int(nx))
        return shape

    def _pc_emsoft2bruker(self, version: int = 5) -> np.ndarray:
        new_pc = np.zeros_like(self.pc, dtype=float)
        pcx = self.pcx
        if version < 5:
            pcx = -pcx
        new_pc[..., 0] = 0.5 - (pcx / (self.ncols * self._binning))
        new_pc[..., 1] = 0.5 - (self.pcy / (self.nrows * self._binning))
        new_pc[..., 2] = self.pcz / (self.nrows * self._binning * self.px_size)
        return new_pc

    def _pc_tsl2bruker(self) -> np.ndarray:
        new_pc = deepcopy(self.pc)
        new_pc[..., 1] = 1 - self.pcy
        new_pc[..., 2] *= min([self.nrows, self.ncols]) / self.nrows
        return new_pc

    def _pc_oxford2bruker(self) -> np.ndarray:
        new_pc = deepcopy(self.pc)
        new_pc[..., 1] = 1 - self.pcy * self.aspect_ratio
        new_pc[..., 2] *= self.aspect_ratio
        return new_pc

    def _pc_bruker2emsoft(self, version: int = 5) -> np.ndarray:
        new_pc = np.zeros_like(self.pc, dtype=float)
        new_pc[..., 0] = (0.5 - self.pcx) * self.ncols * self._binning
        if version < 5:
            new_pc[..., 0] = -new_pc[..., 0]
        new_pc[..., 1] = (0.5 - self.pcy) * self.nrows * self._binning
        new_pc[..., 2] = self.pcz * self.nrows * self._binning * self.px_size
        return new_pc

    def _pc_bruker2tsl(self) -> np.ndarray:
        new_pc = deepcopy(self.pc)
        new_pc[..., 1] = 1 - self.pcy
        new_pc[..., 2] /= min([self.nrows, self.ncols]) / self.nrows
        return new_pc

    def _pc_bruker2oxford(self) -> np.ndarray:
        new_pc = deepcopy(self.pc)
        new_pc[..., 1] = (1 - self.pcy) / self.aspect_ratio
        new_pc[..., 2] /= self.aspect_ratio
        return new_pc

    def _set_pc_in_bruker_convention(
        self, convention: PC_CONVENTIONS = "bruker"
    ) -> None:
        self.pc = self._get_pc_in_bruker_convention(convention)
