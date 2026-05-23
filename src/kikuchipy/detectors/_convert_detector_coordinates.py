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

"""Functions for converting between pixel and gnomonic detector
coordinates.
"""

from typing import Literal
import warnings

import numba as nb
import numpy as np

from kikuchipy._constants import VisibleDeprecationWarning

# TODO: Remove deprecated "detector" format and this then unnecessary
# handling once 0.12 is released
ALL_DETECTOR_PLOT_FORMATS = Literal["pixel", "gnomonic", "detector"]
DETECTOR_PLOT_FORMATS = Literal["pixel", "gnomonic"]
COORDINATE_CONVERSION_DIRECTIONS = Literal["pix_to_gn", "gn_to_pix"]


def parse_coordinate_format(fmt: ALL_DETECTOR_PLOT_FORMATS) -> DETECTOR_PLOT_FORMATS:
    if fmt == "detector":
        warnings.warn(
            "Pass 'pixel' instead. Passing 'detector' is deprecated and will throw an "
            "error in 0.13.0",
            VisibleDeprecationWarning,
        )
        fmt = "pixel"
    return fmt


@nb.njit(cache=True, fastmath=True, nogil=True)
def get_pixel_to_gnomonic_coords_conversion_numba(
    gnomonic_bounds: np.ndarray, bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xg_min = gnomonic_bounds[..., 0]
    xg_max = gnomonic_bounds[..., 1]
    yg_min = gnomonic_bounds[..., 2]
    yg_max = gnomonic_bounds[..., 3]
    m_x = (xg_max - xg_min) / (bounds[1] + 1)
    c_x = xg_min
    m_y = (yg_min - yg_max) / (bounds[3] + 1)
    c_y = yg_max
    return m_x, c_x, m_y, c_y


@nb.njit(cache=True, fastmath=True, nogil=True)
def get_gnomonic_to_pixel_coords_conversion_numba(
    gnomonic_bounds: np.ndarray, bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xg_min = gnomonic_bounds[..., 0]
    xg_max = gnomonic_bounds[..., 1]
    yg_min = gnomonic_bounds[..., 2]
    yg_max = gnomonic_bounds[..., 3]
    m_x = (bounds[1] + 1) / (xg_max - xg_min)
    c_x = -xg_min * m_x
    m_y = (bounds[3] + 1) / (yg_min - yg_max)
    c_y = -yg_max * m_y
    return m_x, c_x, m_y, c_y


def get_pixel_to_gnomonic_coords_conversion(
    gnomonic_bounds: np.ndarray, bounds: np.ndarray
) -> dict[str, np.ndarray]:
    """Get conversion factors for pixel -> gnomonic coordinates."""
    m_x, c_x, m_y, c_y = get_pixel_to_gnomonic_coords_conversion_numba(
        gnomonic_bounds=np.atleast_2d(gnomonic_bounds),
        bounds=np.asarray(bounds, dtype=np.float64),
    )
    return {"m_x": m_x, "c_x": c_x, "m_y": m_y, "c_y": c_y}


def get_gnomonic_to_pixel_coords_conversion(
    gnomonic_bounds: np.ndarray, bounds: np.ndarray
) -> dict[str, np.ndarray]:
    """Get conversion factors for gnomonic -> pixel coordinates."""
    m_x, c_x, m_y, c_y = get_gnomonic_to_pixel_coords_conversion_numba(
        gnomonic_bounds=np.atleast_2d(gnomonic_bounds),
        bounds=np.asarray(bounds, dtype=np.float64),
    )
    return {"m_x": m_x, "c_x": c_x, "m_y": m_y, "c_y": c_y}


def get_coordinate_conversion(
    gnomonic_bounds: np.ndarray,
    bounds: np.ndarray,
    direction: COORDINATE_CONVERSION_DIRECTIONS,
) -> dict[str, np.ndarray]:
    """Get conversion factors for a requested conversion direction."""
    if direction == "pix_to_gn":
        return get_pixel_to_gnomonic_coords_conversion(gnomonic_bounds, bounds)
    else:
        return get_gnomonic_to_pixel_coords_conversion(gnomonic_bounds, bounds)


def convert_coordinates(
    coords: np.ndarray,
    conversion: dict[str, np.ndarray],
    detector_index: int | tuple | None = None,
) -> np.ndarray:
    """Convert coordinates using one conversion-factor dictionary."""
    coords = np.atleast_2d(coords)

    nav_shape = conversion["m_x"].shape
    nav_ndim = len(nav_shape)

    if detector_index is None:
        if coords.ndim >= nav_ndim + 2 and coords.shape[:nav_ndim] == nav_shape:
            # one or more sets of coords, different for each image
            out_shape = coords.shape
        else:
            # one or more sets of coords, the same for each image
            out_shape = nav_shape + coords.shape

        extra_axes = list(range(nav_ndim, len(out_shape) - 1))

        m_x = np.expand_dims(conversion["m_x"], extra_axes)
        c_x = np.expand_dims(conversion["c_x"], extra_axes)
        m_y = np.expand_dims(conversion["m_y"], extra_axes)
        c_y = np.expand_dims(conversion["c_y"], extra_axes)

        coords_out = _convert_coordinates(
            coords,
            out_shape,
            m_x,
            c_x,
            m_y,
            c_y,
        )

    else:
        m_x = np.asarray(conversion["m_x"][detector_index])
        c_x = np.asarray(conversion["c_x"][detector_index])
        m_y = np.asarray(conversion["m_y"][detector_index])
        c_y = np.asarray(conversion["c_y"][detector_index])

        index_shape = m_x.shape
        if len(index_shape) == 0:
            out_shape = coords.shape
        else:
            out_shape = index_shape + coords.shape
            extra_axes = list(range(len(index_shape), len(out_shape) - 1))
            m_x = np.expand_dims(m_x, extra_axes)
            c_x = np.expand_dims(c_x, extra_axes)
            m_y = np.expand_dims(m_y, extra_axes)
            c_y = np.expand_dims(c_y, extra_axes)

        coords_out = _convert_coordinates(
            coords,
            out_shape,
            m_x,
            c_x,
            m_y,
            c_y,
        )

    return coords_out


@nb.njit(cache=True, fastmath=True, nogil=True)
def _convert_coordinates(
    coords: np.ndarray,
    out_shape: tuple,
    m_x: np.ndarray,
    c_x: np.ndarray,
    m_y: np.ndarray,
    c_y: np.ndarray,
) -> np.ndarray:
    """Return converted coordinates from linear conversion factors."""
    coords_out = np.zeros(out_shape, dtype=float)

    # Coordinates are ordered as (y, x) or (gy, gx).
    coords_out[..., 0] = m_y * coords[..., 0] + c_y
    coords_out[..., 1] = m_x * coords[..., 1] + c_x

    return coords_out


def convert_pixel_to_gnomonic_coords(
    gnomonic_bounds: np.ndarray,
    bounds: np.ndarray,
    coords: np.ndarray,
    pos: int | tuple | None = None,
) -> np.ndarray:
    """Convert pixel coordinates to gnomonic coordinates."""
    conversion = get_pixel_to_gnomonic_coords_conversion(gnomonic_bounds, bounds)
    return convert_coordinates(coords, conversion, pos)


def convert_gnomonic_to_pixel_coords(
    gnomonic_bounds: np.ndarray,
    bounds: np.ndarray,
    coords: np.ndarray,
    pos: int | tuple | None = None,
) -> np.ndarray:
    """Convert gnomonic coordinates to pixel coordinates."""
    conversion = get_gnomonic_to_pixel_coords_conversion(gnomonic_bounds, bounds)
    return convert_coordinates(coords, conversion, pos)
