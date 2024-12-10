# Copyright 2019-2024 The kikuchipy developers
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

"""Functions for converting between pixel and gnomonic detector coordinates."""

from typing import Union

import numpy as np


def get_coordinate_conversions(gnomonic_bounds: np.ndarray, bounds: np.ndarray) -> dict:
    """
    Get factors for converting between pixel and gnomonic coordinates.

    Return a dict 'conversions' containing the keys
    "pix_to_gn", containing factors for converting
    pixel to gnomonic coordinates, and "gn_to_pix",
    containing factors for converting gnomonic to pixel
    coordinates.
    Under each of these keys is a further dict with the
    keys: "m_x", "c_x", "m_y" and "c_y". These are the
    slope (m) and y-intercept (c) corresponding to
    y = mx + c, which describes the linear conversion
    of the coordinates. A (different) linear relationship
    is required for x (column) and y(row) coordinates,
    hence the two sets of m and c parameters.

    Parameters
    ----------
    gnomonic_bounds
        Array of shape at least (4,) containing the
        gnomonic bounds of the EBSD detector screen.
        Typically obtained as the "gnomonic_bounds"
        property of an EBSDDetector.
    bounds
        Array of four ints giving the detector bounds
        [x0, x1, y0, y1] in pixel coordinates. Typically
        obtained from the "bounds" property of an
        EBSDDetector.

    Returns
    -------
    conversions
        Contains the keys "pix_to_gn", containing factors
        for converting pixel to gnomonic coordinates, and
        "gn_to_pix", containing factors for converting
        gnomonic to pixel coordinates.
        Under each of these keys is a further dict with the
        keys: "m_x", "c_x", "m_y" and "c_y". These are the
        slope (m) and y-intercept (c) corresponding to
        y = mx + c, which describes the linear conversion
        of the coordinates. A (different) linear relationship
        is required for x (column) and y(row) coordinates,
        hence the two sets of m and c parameters.

    Examples
    --------
    Create an EBSD detector and get the coordinate conversion factors.

    >>> import numpy as np
    >>> import kikuchipy as kp
    >>> from kp._utils._detector_coordinates import get_coordinate_conversions
    >>> det = kp.detectors.EBSDDetector(
    ...     shape=(60, 60),
    ...     pc=np.ones((10, 20, 3)) * (0.421, 0.779, 0.505),
    ...     convention="edax",
    ...     px_size=70,
    ...     binning=8,
    ...     tilt=5,
    ...     sample_tilt=70,
    ... )
    >>> det
    EBSDDetector(shape=(60, 60), pc=(0.421, 0.221, 0.505), sample_tilt=70.0, tilt=5.0, azimuthal=0.0, twist=0.0, binning=8.0, px_size=70.0 um)
    >>> det.navigation_shape
    (10, 20)
    >>> det.bounds
    array([ 0, 59,  0, 59])
    >>> det.gnomonic_bounds[0, 0]
    array([-0.83366337,  1.14653465, -1.54257426,  0.43762376])
    >>> conversions = get_coordinate_conversions(det.gnomonic_bounds, det.bounds)
    >>> conversions["pix_to_gn"]["m_x"].shape
    (10, 20)
    """
    gnomonic_bounds = np.atleast_2d(gnomonic_bounds)

    m_pix_to_gn_x = (gnomonic_bounds[..., 1] - gnomonic_bounds[..., 0]) / (
        bounds[1] + 1
    )
    c_pix_to_gn_x = gnomonic_bounds[..., 0]

    m_pix_to_gn_y = (gnomonic_bounds[..., 2] - gnomonic_bounds[..., 3]) / (
        bounds[3] + 1
    )
    c_pix_to_gn_y = gnomonic_bounds[..., 3]

    m_gn_to_pix_x = 1 / m_pix_to_gn_x
    c_gn_to_pix_x = -c_pix_to_gn_x / m_pix_to_gn_x

    m_gn_to_pix_y = 1 / m_pix_to_gn_y
    c_gn_to_pix_y = -c_pix_to_gn_y / m_pix_to_gn_y

    conversions = {
        "pix_to_gn": {
            "m_x": m_pix_to_gn_x,
            "c_x": c_pix_to_gn_x,
            "m_y": m_pix_to_gn_y,
            "c_y": c_pix_to_gn_y,
        },
        "gn_to_pix": {
            "m_x": m_gn_to_pix_x,
            "c_x": c_gn_to_pix_x,
            "m_y": m_gn_to_pix_y,
            "c_y": c_gn_to_pix_y,
        },
    }

    return conversions


def convert_coordinates(
    coords: np.ndarray,
    direction: str,
    conversions: dict,
    detector_index: Union[None, tuple, int] = None,
) -> np.ndarray:
    """
    Convert between gnomonic and pixel coordinates.

    Parameters
    ----------
    coords
        An array of coordinates of any shape whereby the
        x and y coordinates to be converted are stored in
        the last axis.
    direction
        Either "pix_to_gn" or "gn_to_pix", depending on the
        direction of conversion needed.
    conversions
        Dict containing the conversion parameters. Usually
        the output of get_coordinate_conversions().
        Contains the keys "pix_to_gn", containing factors
        for converting pixel to gnomonic coordinates, and
        "gn_to_pix", containing factors for converting
        gnomonic to pixel coordinates.
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
    detector_index
        Index showing which conversion factors in *conversions[direction]*
        should be applied to *coords*.
        If None, **all** conversion factors in *conversions[direction]*
        are applied to *coords*.
        If an int is supplied, this refers to an index in a 1D dataset.
        A 1D tuple *e.g.* (3,) can also be passed for a 1D dataset.
        A 2D index can be specified by supplying a tuple *e.g.* (2, 3).
        The default value is None.

    Returns
    -------
    coords_out
        Array of coords but with values converted as specified
        by direction. The shape is either the same as the input
        or is the navigation shape then the shape of the input.

    Examples
    --------

    Convert 300 xy coordinates for all patterns in a dataset.

    >>> import numpy as np
    >>> import kikuchipy as kp
    >>> from kp._utils._detector_coordinates import (get_coordinate_conversions, convert_coordinates)
    >>> s = kp.data.nickel_ebsd_small()
    >>> det = s.detector
    >>> det.navigation_shape
    (3, 3)
    >>> coords_2d = np.random.randint(0, 60, (300, 2))
    >>> coords_2d.shape
    (300, 2)
    >>> conv = get_coordinate_conversions(det.gnomonic_bounds, det.bounds)
    >>> coords_out = convert_coordinates(coords_2d, "pix_to_gn", conv, None)
    >>> coords_out.shape
    (3, 3, 300, 2)

    Convert 300 xy coordinates for the pattern at index (1, 2) in a dataset.

    >>> import numpy as np
    >>> import kikuchipy as kp
    >>> from kp._utils._detector_coordinates import (get_coordinate_conversions, convert_coordinates)
    >>> s = kp.data.nickel_ebsd_small()
    >>> det = s.detector
    >>> det.navigation_shape
    (3, 3)
    >>> coords_2d = np.random.randint(0, 60, (300, 2))
    >>> coords_2d.shape
    (300, 2)
    >>> conv = get_coordinate_conversions(det.gnomonic_bounds, det.bounds)
    >>> coords_out = convert_coordinates(coords_2d, "pix_to_gn", conv, (1, 2))
    >>> coords_out.shape
    (300, 2)

    Convert 17 sets of 300 xy coordinates, different for each pattern in a dataset.

    >>> import numpy as np
    >>> import kikuchipy as kp
    >>> from kp._utils._detector_coordinates import (get_coordinate_conversions, convert_coordinates)
    >>> s = kp.data.nickel_ebsd_small()
    >>> det = s.detector
    >>> det.navigation_shape
    (3, 3)
    >>> coords_2d = np.random.randint(0, 60, (3, 3, 17, 300, 2))
    >>> coords_2d.shape
    (3, 3, 17, 300, 2)
    >>> conv = get_coordinate_conversions(det.gnomonic_bounds, det.bounds)
    >>> coords_out = convert_coordinates(coords_2d, "pix_to_gn", conv, None)
    >>> coords_out.shape
    (3, 3, 17, 300, 2)

    Convert 17 sets of 300 xy coordinates, the same for all pattern in a dataset.

    >>> import numpy as np
    >>> import kikuchipy as kp
    >>> from kp._utils._detector_coordinates import (get_coordinate_conversions, convert_coordinates)
    >>> s = kp.data.nickel_ebsd_small()
    >>> det = s.detector
    >>> det.navigation_shape
    (3, 3)
    >>> coords_2d = np.random.randint(0, 60, (17, 300, 2))
    >>> coords_2d.shape
    (17, 300, 2)
    >>> conv = get_coordinate_conversions(det.gnomonic_bounds, det.bounds)
    >>> coords_out = convert_coordinates(coords_2d, "pix_to_gn", conv, None)
    >>> coords_out.shape
    (3, 3, 17, 300, 2)
    """
    coords = np.atleast_2d(coords)

    nav_shape = conversions[direction]["m_x"].shape
    nav_ndim = len(nav_shape)

    if isinstance(detector_index, type(None)):
        detector_index = ()
        if coords.ndim >= nav_ndim + 2 and coords.shape[:nav_ndim] == nav_shape:
            # one or more sets of coords, different for each image
            out_shape = coords.shape
        else:
            # one or more sets of coords, the same for each image
            out_shape = nav_shape + coords.shape

        extra_axes = list(range(nav_ndim, len(out_shape) - 1))

        coords_out = _convert_coordinates(
            coords,
            out_shape,
            detector_index,
            np.expand_dims(conversions[direction]["m_x"], extra_axes),
            np.expand_dims(conversions[direction]["c_x"], extra_axes),
            np.expand_dims(conversions[direction]["m_y"], extra_axes),
            np.expand_dims(conversions[direction]["c_y"], extra_axes),
        )

    else:
        if isinstance(detector_index, int):
            detector_index = tuple([detector_index])

        out_shape = coords.shape

        coords_out = _convert_coordinates(
            coords,
            out_shape,
            detector_index,
            conversions[direction]["m_x"],
            conversions[direction]["c_x"],
            conversions[direction]["m_y"],
            conversions[direction]["c_y"],
        )

    return coords_out


def _convert_coordinates(
    coords: np.ndarray,
    out_shape: tuple,
    detector_index: tuple,
    m_x: Union[np.ndarray, float],
    c_x: Union[np.ndarray, float],
    m_y: Union[np.ndarray, float],
    c_y: Union[np.ndarray, float],
) -> np.ndarray:
    """
    Return converted coordinate depending on arguments.

    This function is usually called by convert_coordinates().

    Parameters
    ----------
    coords
        An array of coordinates whereby the x and y coordinates
        to be converted are stored in the last axis
    direction
        Either "pix_to_gn" or "gn_to_pix", depending on the
        direction of conversion needed.
    conversions
        dict


    Parameters
    ----------
    coords
        An array of coordinates whereby the x and y coordinates
        to be converted are stored in the last axis.
    out_shape
        Tuple of ints giving the output shape.
    detector_index
        Tuple giving the detector index..
    m_x
        Conversion factor m for x coordinate.
    c_x
        Conversion factor c for x coordinate.
    m_y
        Conversion factor m for y coordinate.
    c_y
        Conversion factor c for y coordinate.

    Returns
    -------
    coords_out
        Array of coords the same shape as the input but
        with converted values.
    """
    coords_out = np.zeros(out_shape, dtype=float)

    coords_out[..., 0] = m_x[detector_index] * coords[..., 0] + c_x[detector_index]
    coords_out[..., 1] = m_y[detector_index] * coords[..., 1] + c_y[detector_index]

    return coords_out
