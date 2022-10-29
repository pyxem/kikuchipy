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

# The following copyright notice is included because the following
# functionality in this file is derived and adapted from EMsoft:
# - Determination of the direction cosines on the unit sphere for a
#   given projection centre (PC) and crystal orientation
# - Interpolation from the unit sphere to the square Lambert projection

# #####################################################################
# Copyright (c) 2013-2022, Marc De Graef Research Group/Carnegie Mellon
# University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#  - Neither the names of Marc De Graef, Carnegie Mellon University nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ######################################################################

"""Private tools for projecting parts of dynamically simulated master
patterns into a detector.
"""

from typing import Optional, Tuple, Union

from numba import njit
import numba as nb
import numpy as np

from kikuchipy.pattern._pattern import _rescale_with_min_max
from kikuchipy.projections.lambert_projection import _vector2xy
from kikuchipy._rotation import _rotate_vector


# Reusable constants
SQRT_PI_HALF = np.sqrt(np.pi / 2)


def _get_direction_cosines_from_detector(
    detector: "EBSDDetector", mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Return direction cosines for one or more projection centers
    (PCs).

    Parameters
    ----------
    detector
        EBSD detector with one or more PCs.
    mask
        1D signal mask with ``True`` values for pixels to get direction
        cosines for.

    Returns
    -------
    dc
        Flattened direction cosines.
    """
    if detector.navigation_shape == (1,):
        pcx, pcy, pcz = detector.pc.squeeze().astype(np.float64)
        func = _get_direction_cosines_for_fixed_pc
    else:
        pcx, pcy, pcz = detector.pc.reshape((-1, 3)).T.astype(np.float64)
        func = _get_direction_cosines_for_varying_pc
    if mask is None:
        mask = np.ones(detector.shape, dtype=bool).ravel()
    return func(
        pcx=pcx,
        pcy=pcy,
        pcz=pcz,
        nrows=detector.nrows,
        ncols=detector.ncols,
        tilt=detector.tilt,
        azimuthal=detector.azimuthal,
        sample_tilt=detector.sample_tilt,
        mask=mask,
    )


@njit(
    "Tuple((float64, float64, float64, float64))(float64, float64, float64)",
    cache=True,
    nogil=True,
    fastmath=True,
)
def _get_cosine_sine_of_alpha_and_azimuthal(
    sample_tilt: float, tilt: float, azimuthal: float
) -> Tuple[float, float, float, float]:
    alpha = (np.pi / 2) - np.deg2rad(sample_tilt) + np.deg2rad(tilt)
    azimuthal = np.deg2rad(azimuthal)
    return np.cos(alpha), np.sin(alpha), np.cos(azimuthal), np.sin(azimuthal)


@njit(
    (
        "float64[:, :]"
        "(float64, float64, float64, int64, int64, float64, float64, float64, bool_[:])"
    ),
    cache=True,
    nogil=True,
    fastmath=True,
)
def _get_direction_cosines_for_fixed_pc(
    pcx: float,
    pcy: float,
    pcz: float,
    nrows: int,
    ncols: int,
    tilt: float,
    azimuthal: float,
    sample_tilt: float,
    mask: np.ndarray,
) -> np.ndarray:
    """Return direction cosines for a single projection center (PC).

    Algorithm adapted from EMsoft, see :cite:`callahan2013dynamical`.

    Parameters
    ----------
    pcx
        PC x coordinate.
    pcy
        PC y coordinate.
    pcz
        PC z coordinate.
    nrows
        Number of detector rows.
    ncols
        Number of detector columns.
    tilt
        Detector tilt from horizontal in degrees.
    azimuthal
        Sample tilt about the sample RD axis in degrees.
    sample_tilt
        Sample tilt from horizontal in degrees.
    mask
        1D signal mask with ``True`` values for pixels to get direction
        cosines for.

    Returns
    -------
    r_g_array
        Direction cosines for detector pixels in the mask of shape
        (n_pixels, 3) and data type of 64-bit floats.

    See Also
    --------
    kikuchipy.detectors.EBSDDetector

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    nrows_array = np.arange(nrows - 1, -1, -1)
    ncols_array = np.arange(ncols)

    # Bruker to EMsoft's v5 PC convention
    xpc = ncols * (0.5 - pcx)
    ypc = nrows * (0.5 - pcy)
    zpc = nrows * pcz

    det_x = xpc + (1 - ncols) * 0.5 + ncols_array
    det_y = ypc - (1 - nrows) * 0.5 - nrows_array

    ca, sa, cw, sw = _get_cosine_sine_of_alpha_and_azimuthal(
        sample_tilt=sample_tilt,
        tilt=tilt,
        azimuthal=azimuthal,
    )
    Ls = -sw * det_x + zpc * cw
    Lc = cw * det_x + zpc * sw

    idx_1d = np.arange(nrows * ncols)[mask]
    rows = idx_1d // ncols
    cols = np.mod(idx_1d, ncols)
    n_pixels = idx_1d.size
    r_g_array = np.zeros((n_pixels, 3), dtype=np.float64)
    for i in nb.prange(n_pixels):
        r_g_array[i, 0] = det_y[rows[i]] * ca + sa * Ls[cols[i]]
        r_g_array[i, 1] = Lc[cols[i]]
        r_g_array[i, 2] = -sa * det_y[rows[i]] + ca * Ls[cols[i]]

    # Normalize
    norm = np.sqrt(np.sum(np.square(r_g_array), axis=-1))
    norm = np.expand_dims(norm, axis=-1)
    r_g_array = np.divide(r_g_array, norm)

    return r_g_array


@njit(
    (
        "float64[:, :, :]"
        "(float64[:], float64[:], float64[:], int64, int64, float64, float64, float64, bool_[:])"
    ),
    cache=True,
    nogil=True,
    fastmath=True,
)
def _get_direction_cosines_for_varying_pc(
    pcx: np.ndarray,
    pcy: np.ndarray,
    pcz: np.ndarray,
    nrows: int,
    ncols: int,
    tilt: float,
    azimuthal: float,
    sample_tilt: float,
    mask: np.ndarray,
) -> np.ndarray:
    """Return sets of direction cosines for varying projection centers
    (PCs).

    Algorithm adapted from EMsoft, see :cite:`callahan2013dynamical`.

    Parameters
    ----------
    pcx
        PC x coordinates. Must be a 1D array.
    pcy
        PC y coordinates. Must be a 1D array.
    pcz
        PC z coordinates. Must be a 1D array.
    nrows
        Number of detector rows.
    ncols
        Number of detector columns.
    tilt
        Detector tilt from horizontal in degrees.
    azimuthal
        Sample tilt about the sample RD axis in degrees.
    sample_tilt
        Sample tilt from horizontal in degrees.
    mask
        1D signal mask with ``True`` values for pixels to get direction
        cosines for.

    Returns
    -------
    r_g_array
        Direction cosines for each detector pixel for each PC, of shape
        (n PCs, n_pixels, 3) and data type of 64-bit floats.

    See Also
    --------
    kikuchipy.detectors.EBSDDetector

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    nrows_array = np.arange(nrows - 1, -1, -1)
    ncols_array = np.arange(ncols)

    ca, sa, cw, sw = _get_cosine_sine_of_alpha_and_azimuthal(
        sample_tilt=sample_tilt,
        tilt=tilt,
        azimuthal=azimuthal,
    )

    det_x_factor = (1 - ncols) * 0.5
    det_y_factor = (1 - nrows) * 0.5

    idx_1d = np.arange(nrows * ncols)[mask]
    rows = idx_1d // ncols
    cols = np.mod(idx_1d, ncols)

    n_pcs = pcx.size
    n_pixels = idx_1d.size
    r_g_array = np.zeros((n_pcs, n_pixels, 3), dtype=np.float64)

    for i in nb.prange(n_pcs):
        # Bruker to EMsoft's v5 PC convention
        xpc = ncols * (0.5 - pcx[i])
        ypc = nrows * (0.5 - pcy[i])
        zpc = nrows * pcz[i]

        det_x = xpc + det_x_factor + ncols_array
        det_y = ypc - det_y_factor - nrows_array

        Ls = -sw * det_x + zpc * cw
        Lc = cw * det_x + zpc * sw

        for j in nb.prange(n_pixels):
            r_g_array[i, j, 0] = det_y[rows[j]] * ca + sa * Ls[cols[j]]
            r_g_array[i, j, 1] = Lc[cols[j]]
            r_g_array[i, j, 2] = -sa * det_y[rows[j]] + ca * Ls[cols[j]]

    # Normalize
    norm = np.sqrt(np.sum(np.square(r_g_array), axis=-1))
    norm = np.expand_dims(norm, axis=-1)
    r_g_array = np.divide(r_g_array, norm)

    return r_g_array


@njit(cache=True, nogil=True, fastmath=True)
def _project_patterns_from_master_pattern_with_fixed_pc(
    rotations: np.ndarray,
    direction_cosines: np.ndarray,
    master_upper: np.ndarray,
    master_lower: np.ndarray,
    npx: int,
    npy: int,
    scale: float,
    rescale: bool,
    out_min: Union[int, float],
    out_max: Union[int, float],
    dtype_out: Optional[type] = np.float32,
) -> np.ndarray:
    """Return one or more simulated EBSD patterns projected from a
    master pattern with a fixed projection center (PC).

    Parameters
    ----------
    rotations
        2D array of quaternions of shape (n, 4) for a given chunk.
    direction_cosines
        Single set of direction cosines (unit vectors) between detector
        and sample of shape (m pixels, 3) (the PC).
    master_upper
        Upper hemisphere of the master pattern.
    master_lower
        Lower hemisphere of the master pattern.
    npx
        Number of pixels in the x-direction on the master pattern.
    npy
        Number of pixels in the y-direction on the master pattern.
    scale
        Factor to scale up from square Lambert projection to the master
        pattern.
    rescale
        Whether to rescale pattern intensities.
    out_min
        Minimum intensity of output patterns.
    out_max
        Maximum intensity of output patterns.
    dtype_out
        NumPy data type of the returned patterns, by default 32-bit
        float.

    Returns
    -------
    simulated
        2D array of simulated patterns with flattened navigation and
        signal dimensions.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n = rotations.shape[0]
    simulated = np.zeros((n, direction_cosines.shape[0]), dtype=dtype_out)
    for i in range(n):
        simulated[i] = _project_single_pattern_from_master_pattern(
            rotation=rotations[i],
            direction_cosines=direction_cosines,
            master_upper=master_upper,
            master_lower=master_lower,
            npx=npx,
            npy=npy,
            scale=scale,
            rescale=rescale,
            out_min=out_min,
            out_max=out_max,
            dtype_out=dtype_out,
        )
    return simulated


@njit(cache=True, nogil=True, fastmath=True)
def _project_patterns_from_master_pattern_with_varying_pc(
    rotations: np.ndarray,
    direction_cosines: np.ndarray,
    master_upper: np.ndarray,
    master_lower: np.ndarray,
    npx: int,
    npy: int,
    scale: float,
    rescale: bool,
    out_min: Union[int, float],
    out_max: Union[int, float],
    dtype_out: Optional[type] = np.float32,
) -> np.ndarray:
    """Return simulated EBSD patterns projected from a master pattern
    with varying projection centers (PCs).

    Parameters
    ----------
    rotations
        2D array of quaternions of shape (n, 4) for a given chunk.
    direction_cosines
        Sets of direction cosines (unit vectors) between detector and
        sample of shape (n, m pixels, 3) (the PC), one set per rotation.
    master_upper
        Upper hemisphere of the master pattern.
    master_lower
        Lower hemisphere of the master pattern.
    npx
        Number of pixels in the x-direction on the master pattern.
    npy
        Number of pixels in the y-direction on the master pattern.
    scale
        Factor to scale up from square Lambert projection to the master
        pattern.
    rescale
        Whether to rescale pattern intensities.
    out_min
        Minimum intensity of output patterns.
    out_max
        Maximum intensity of output patterns.
    dtype_out
        NumPy data type of the returned patterns, by default 32-bit
        float.

    Returns
    -------
    simulated
        2D array of simulated patterns with flattened navigation and
        signal dimensions.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n = rotations.shape[0]
    simulated = np.empty((n, direction_cosines.shape[1]), dtype=dtype_out)
    for i in range(n):
        simulated[i] = _project_single_pattern_from_master_pattern(
            rotation=rotations[i],
            direction_cosines=direction_cosines[i],
            master_upper=master_upper,
            master_lower=master_lower,
            npx=npx,
            npy=npy,
            scale=scale,
            rescale=rescale,
            out_min=out_min,
            out_max=out_max,
            dtype_out=dtype_out,
        )
    return simulated


@njit(cache=True, nogil=True, fastmath=True)
def _project_single_pattern_from_master_pattern(
    rotation: np.ndarray,
    direction_cosines: np.ndarray,
    master_upper: np.ndarray,
    master_lower: np.ndarray,
    npx: int,
    npy: int,
    scale: float,
    rescale: bool,
    out_min: Union[int, float],
    out_max: Union[int, float],
    dtype_out: type,
) -> np.ndarray:
    """Return a single 1D EBSD pattern projected from a master pattern.

    Parameters
    ----------
    rotation
        Array of one quaternion of shape (4,).
    direction_cosines
        Set of direction cosines (unit vectors) between detector and
        sample of shape (m pixels, 3).
    master_upper
        Upper hemisphere of the master pattern.
    master_lower
        Lower hemisphere of the master pattern.
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
        NumPy data type of the returned patterns.

    Returns
    -------
    patterns
        1D array of a simulated EBSD pattern of data type ``dtype_out``.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    # Rotate the detector's view of the crystal
    dc_rotated = _rotate_vector(rotation, direction_cosines)

    (nii, nij, niip, nijp, di, dj, dim, djm) = _get_lambert_interpolation_parameters(
        v=dc_rotated, npx=npx, npy=npy, scale=scale
    )

    # Loop over the detector pixels and fill in intensities one by one
    # from the correct hemisphere of the master pattern
    pattern = np.zeros(direction_cosines.shape[0])
    for i in nb.prange(pattern.size):
        if dc_rotated[i, 2] >= 0:
            mp = master_upper
        else:
            mp = master_lower
        pattern[i] = _get_pixel_from_master_pattern(
            mp, nii[i], nij[i], niip[i], nijp[i], di[i], dj[i], dim[i], djm[i]
        )

    # Potentially rescale pattern intensities to desired data type
    if rescale:
        pattern = _rescale_with_min_max(
            pattern, np.min(pattern), np.max(pattern), out_min, out_max
        )

    return pattern.astype(dtype_out)


@njit(
    (
        "Tuple((int32[:], int32[:], int32[:], int32[:], float64[:], float64[:], "
        "float64[:], float64[:]))(float64[:, :], int64, int64, float64)"
    ),
    cache=True,
    nogil=True,
    fastmath=True,
)
def _get_lambert_interpolation_parameters(
    v: np.ndarray,
    npx: int,
    npy: int,
    scale: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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
    nii
        1D array of each vector's row coordinate as 32-bit integers.
    nij
        1D array of each vector's column coordinate as 32-bit integers.
    niip
        1D array of each vector's neighbouring row coordinate as 32-bit
        integers.
    nijp
        1D array of each vector's neighbouring column coordinate as
        32-bit integers.
    di
        1D array of each vector's row interpolation weight factor as
        64-bit floats.
    dj
        1D array of each vector's column interpolation weight factor as
        64-bit floats.
    dim
        1D array of each vector's neighbouring row interpolation weight
        factor as 64-bit floats.
    djm
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
    # Use loop to avoid numpy.where() for the conditionals
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


@njit(cache=True, nogil=True, fastmath=True)
def _get_pixel_from_master_pattern(
    mp: np.ndarray,
    nii: int,
    nij: int,
    niip: int,
    nijp: int,
    di: float,
    dj: float,
    dim: float,
    djm: float,
) -> np.ndarray:
    """Return an intensity from a master pattern in the square Lambert
    projection using bi-linear interpolation.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    # Cannot specify output data type because master pattern array can
    # be either integer or float
    return (
        mp[nii, nij] * dim * djm
        + mp[niip, nij] * di * djm
        + mp[nii, nijp] * dim * dj
        + mp[niip, nijp] * di * dj
    )


@njit("float64[:, :](float64[:], float64[:])", cache=True, nogil=True, fastmath=True)
def _lambert2vector(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lambert (X, Y) to vector (x, y, z) projection
    :cite:`callahan2013dynamical`.

    Parameters
    ----------
    x, y
        1D arrays of square grid x and y coordinates with 64- bit
        floating point data type.

    Returns
    -------
    cart
        2D array (n, 3) of vectors. The vectors are not normalized, so
        they might not be on the unit sphere.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    n = x.size
    cart = np.zeros((n, 3), dtype=np.float64)

    for i in nb.prange(n):
        xi = x[i] * np.sqrt(np.pi / 2)
        yi = y[i] * np.sqrt(np.pi / 2)

        xi_abs = abs(xi)
        yi_abs = abs(yi)

        if max([xi_abs, yi_abs]) == 0:
            cart[i] = [0, 0, 1]
        else:
            if xi_abs <= yi_abs:
                q = 2 * yi * np.sqrt(np.pi - yi**2) / np.pi
                qq = xi * np.pi * 0.25 / yi
                cart[i] = [q * np.sin(qq), q * np.cos(qq), 1 - 2 * yi**2 / np.pi]
            else:
                q = 2 * xi * np.sqrt(np.pi - xi**2) / np.pi
                qq = yi * np.pi * 0.25 / xi
                cart[i] = [q * np.cos(qq), q * np.sin(qq), 1 - 2 * xi**2 / np.pi]

    return cart
