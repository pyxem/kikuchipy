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

"""Private tools for projecting parts of dynamically simulated master
patterns into a detector.
"""

from typing import Optional, Union

import numba as nb
import numpy as np

from kikuchipy.pattern._pattern import _rescale
from kikuchipy.projections.lambert_projection import _vector2xy


# Reusable constants
SQRT_PI_HALF = np.sqrt(np.pi / 2)


def _get_direction_cosines_from_detector(detector) -> np.ndarray:
    return _get_direction_cosines(
        pcx=detector.pcx,
        pcy=detector.pcy,
        pcz=detector.pcz,
        nrows=detector.nrows,
        ncols=detector.ncols,
        tilt=detector.tilt,
        azimuthal=detector.azimuthal,
        sample_tilt=detector.sample_tilt,
    )


@nb.jit(nogil=True, nopython=True)
def _get_direction_cosines(
    pcx: np.ndarray,
    pcy: np.ndarray,
    pcz: np.ndarray,
    nrows: int,
    ncols: int,
    tilt: float,
    azimuthal: float,
    sample_tilt: float,
) -> np.ndarray:
    """Get the direction cosines between the detector and sample as done
    in EMsoft and :cite:`callahan2013dynamical`.

    Parameters
    ----------
    pcx
        Projection center (PC) x coordinates.
    pcy
        PC y coordinates.
    pcz
        PC z coordinates.
    nrows
        Number of detector rows.
    ncols
        Number of detector columns.
    tilt
        Detector tilt from horizontal in radians.
    azimuthal
        Sample tilt about the sample RD axis in radians.
    sample_tilt
        Sample tilt from horizontal in radians.

    Returns
    -------
    r_g_array
        Direction cosines for each detector pixel.

    See Also
    --------
    kikuchipy.detectors.EBSDDetector
    """
    nrows_array = np.arange(nrows)
    ncols_array = np.arange(ncols)

    # Bruker to EMsoft's v5 PC convention
    xpc = ncols * (0.5 - pcx)
    ypc = nrows * (0.5 - pcy)
    zpc = nrows * pcz

    det_x = -((-xpc - (1 - ncols) * 0.5) - ncols_array)
    det_y = (ypc - (1 - nrows) * 0.5) - nrows_array

    alpha = (np.pi / 2) - np.deg2rad(sample_tilt) + np.deg2rad(tilt)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    azimuthal = np.deg2rad(azimuthal)
    cw = np.cos(azimuthal)
    sw = np.sin(azimuthal)

    Ls = -sw * det_x + zpc * cw
    Lc = cw * det_x + zpc * sw

    r_g_array = np.zeros((nrows, ncols, 3))

    for i in nb.prange(nrows):
        ii = nrows - i - 1
        for j in nb.prange(ncols):
            r_g_array[i, j, 0] = det_y[ii] * ca + sa * Ls[j]
            r_g_array[i, j, 1] = Lc[j]
            r_g_array[i, j, 2] = -sa * det_y[ii] + ca * Ls[j]

    # Normalize
    norm = np.sqrt(np.sum(np.square(r_g_array), axis=-1))
    norm = np.expand_dims(norm, axis=-1)
    r_g_array = np.divide(r_g_array, norm)

    return r_g_array


@nb.jit(cache=True, nogil=True, nopython=True)
def _project_patterns_from_master_pattern(
    rotations: np.ndarray,
    direction_cosines: np.ndarray,
    master_north: np.ndarray,
    master_south: np.ndarray,
    npx: int,
    npy: int,
    scale: float,
    rescale: bool,
    out_min: Union[int, float],
    out_max: Union[int, float],
    dtype_out: Optional[type] = np.float32,
) -> np.ndarray:
    """Project one simulated EBSD pattern onto a detector per rotation,
    given one direction cosine per detector pixel, describing the
    detector's view of the sample.

    Parameters
    ----------
    rotations
        Array of rotations of shape (..., 4) for a given chunk as
        quaternions.
    direction_cosines
        Direction cosines (unit vectors) between detector and sample of
        shape (nrows, ncols, 3).
    master_north
        Northern hemisphere of the master pattern.
    master_south
        Southern hemisphere of the master pattern.
    npx
        Number of pixels in the x-direction on the master pattern.
    npy
        Number of pixels in the y-direction on the master pattern.
    scale
        Factor to scale up from square Lambert projection to the master
        pattern.
    rescale
        Whether to rescale pattern intensities.
    dtype_out
        NumPy data type of the returned patterns, by default 32-bit
        float.

    Returns
    -------
    numpy.ndarray
        3D or 4D array with simulated patterns.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    nav_shape = rotations.shape[:-1]
    sig_shape = direction_cosines.shape[:-1]
    n_pixels = sig_shape[0] * sig_shape[1]
    simulated = np.zeros(nav_shape + (n_pixels,), dtype=dtype_out)

    direction_cosines_flat = direction_cosines.reshape((-1, 3))

    for i in np.ndindex(nav_shape):
        simulated[i] = _project_single_pattern_from_master_pattern(
            rotation=rotations[i],
            direction_cosines=direction_cosines_flat,
            master_north=master_north,
            master_south=master_south,
            npx=npx,
            npy=npy,
            scale=scale,
            n_pixels=n_pixels,
            rescale=rescale,
            out_min=out_min,
            out_max=out_max,
            dtype_out=dtype_out,
        )

    return simulated.reshape(nav_shape + sig_shape)


@nb.jit(cache=True, nogil=True, nopython=True)
def _project_single_pattern_from_master_pattern(
    rotation: np.ndarray,
    direction_cosines: np.ndarray,
    master_north: np.ndarray,
    master_south: np.ndarray,
    npx: int,
    npy: int,
    scale: float,
    n_pixels: int,
    rescale: bool,
    out_min: Union[int, float],
    out_max: Union[int, float],
    dtype_out: type,
) -> np.ndarray:
    """Project a single EBSD pattern onto a detector given one rotation
    and one direction cosine per detector pixel, describing the
    detector's view of the sample.

    Parameters
    ----------
    rotation
        Array of one rotation of shape (4,).
    direction_cosines
        Direction cosines (unit vectors) between detector and sample of
        shape (n_pixels, 3).
    master_north
        Northern hemisphere of the master pattern.
    master_south
        Southern hemisphere of the master pattern.
    npx
        Number of pixels in the x-direction on the master pattern.
    npy
        Number of pixels in the y-direction on the master pattern.
    scale
        Factor to scale up from square Lambert projection to the master
        pattern.
    n_pixels
        Number of detector pixels.
    rescale
        Whether to rescale pattern intensities.
    dtype_out
        NumPy data type of the returned patterns.

    Returns
    -------
    numpy.ndarray
        Simulated EBSD pattern.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    # Rotate the detector's view of the crystal
    rotated_direction_cosines = _rotate_vector(rotation, direction_cosines)

    (nii, nij, niip, nijp, di, dj, dim, djm,) = _get_lambert_interpolation_parameters(
        v=rotated_direction_cosines, npx=npx, npy=npy, scale=scale
    )

    # Loop over the detector pixels and fill in intensities one by one
    # from the correct hemisphere of the master pattern
    pattern = np.zeros((n_pixels,))
    for i in nb.prange(n_pixels):
        if rotated_direction_cosines[i, 2] >= 0:
            mp = master_north
        else:
            mp = master_south
        pattern[i] = _get_pixel_from_master_pattern(
            mp, nii[i], nij[i], niip[i], nijp[i], di[i], dj[i], dim[i], djm[i]
        )

    # Potentially rescale pattern intensities to desired data type
    if rescale:
        pattern = _rescale(pattern, np.min(pattern), np.max(pattern), out_min, out_max)

    return pattern.astype(dtype_out)


@nb.jit(
    "float64[:, :](float64[:], float64[:, :])", nogil=True, cache=True, nopython=True
)
def _rotate_vector(rotation: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotation of vector(s) by a quaternion.

    Parameters
    ----------
    rotation
        Rotation as an array of shape (4,) and data type 64-bit floats.
    vector
        Vector(s) as an array of shape (n, 3) and data type 64-bit
        floats.

    Returns
    -------
    rotated_vectors

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    # TODO: Implement this and similar functions in orix
    a, b, c, d = rotation
    x = vector[:, 0]
    y = vector[:, 1]
    z = vector[:, 2]
    rotated_vector = np.zeros(vector.shape)
    aa = a ** 2
    bb = b ** 2
    cc = c ** 2
    dd = d ** 2
    ac = a * c
    ab = a * b
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d
    rotated_vector[:, 0] = (aa + bb - cc - dd) * x + 2 * ((ac + bd) * z + (bc - ad) * y)
    rotated_vector[:, 1] = (aa - bb + cc - dd) * y + 2 * ((ad + bc) * x + (cd - ab) * z)
    rotated_vector[:, 2] = (aa - bb - cc + dd) * z + 2 * ((ab + cd) * y + (bd - ac) * x)
    return rotated_vector


@nb.jit(
    (
        "Tuple((int32[:], int32[:], int32[:], int32[:], float64[:], float64[:], "
        "float64[:], float64[:]))(float64[:, :], int64, int64, float64)"
    ),
    nogil=True,
    nopython=True,
)
def _get_lambert_interpolation_parameters(
    v: np.ndarray,
    npx: int,
    npy: int,
    scale: float,
) -> tuple:
    """Return interpolation parameters in the square Lambert projection
    from vectors (direction cosines), as implemented in EMsoft.

    Parameters
    ----------
    v
        Vectors (direction cosines) of shape (n, 3) and as 64-bit
        floats.
    npx
        Number of horizontal master pattern pixels as a 64-bit integer.
    npy
        Number of vertical master pattern pixels as a 64-bit integer.
    scale
        Factor to scale up from the square Lambert projection to the
        master pattern, as a 64-bit float.

    Returns
    -------
    nii : numpy.ndarray
        1D array of each vector's row coordinate as 32-bit integers.
    nij : numpy.ndarray
        1D array of each vector's column coordinate as 32-bit integers.
    niip : numpy.ndarray
        1D array of each vector's neighbouring row coordinate as 32-bit
        integers.
    nijp : numpy.ndarray
        1D array of each vector's neighbouring column coordinate as
        32-bit integers.
    di : numpy.ndarray
        1D array of each vector's row interpolation weight factor as
        64-bit floats.
    dj : numpy.ndarray
        1D array of each vector's column interpolation weight factor as
        64-bit floats.
    dim : numpy.ndarray
        1D array of each vector's neighbouring row interpolation weight
        factor as 64-bit floats.
    djm : numpy.ndarray
        1D array of each vector's neighbouring column interpolation
        weight factor as 64-bit floats.
    """
    xy = scale * _vector2xy(v) / SQRT_PI_HALF

    i = xy[:, 1]
    j = xy[:, 0]

    dtype = np.int32
    n = i.size
    nii = np.zeros(n, dtype=dtype)
    nij = np.zeros(n, dtype=dtype)
    niip = np.zeros(n, dtype=dtype)
    nijp = np.zeros(n, dtype=dtype)
    di = np.zeros(n)
    dj = np.zeros(n)
    dim = np.zeros(n)
    djm = np.zeros(n)
    for ii in nb.prange(n):
        i_this = i[ii]
        j_this = j[ii]

        nii_i = dtype(i_this + scale)
        nij_i = dtype(j_this + scale)
        niip_i = nii_i + 1
        nijp_i = nij_i + 1
        if niip_i > npx:
            niip_i = nii_i  # pragma: no cover
        if nijp_i > npy:
            nijp_i = nij_i  # pragma: no cover
        if nii_i < 0:
            nii_i = niip_i  # pragma: no cover
        if nij_i < 0:
            nij_i = nijp_i  # pragma: no cover

        nii[ii] = nii_i
        nij[ii] = nij_i
        niip[ii] = niip_i
        nijp[ii] = nijp_i
        di[ii] = i_this - nii_i + scale
        dj[ii] = j_this - nij_i + scale
        dim[ii] = 1 - di[ii]
        djm[ii] = 1 - dj[ii]

    return nii, nij, niip, nijp, di, dj, dim, djm


@nb.jit(cache=True, nogil=True, nopython=True)
def _get_pixel_from_master_pattern(mp, nii, nij, niip, nijp, di, dj, dim, djm):
    """Return an intensity from a master pattern in the square Lambert
    projection using bi-linear interpolation.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    return (
        mp[nii, nij] * dim * djm
        + mp[niip, nij] * di * djm
        + mp[nii, nijp] * dim * dj
        + mp[niip, nijp] * di * dj
    )
