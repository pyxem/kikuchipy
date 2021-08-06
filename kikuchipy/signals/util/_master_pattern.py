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

"""Tools for projecting parts of dynamically simulated master patterns
into a detector.
"""

from typing import Optional, Union

import numba as nb
import numpy as np
from orix.quaternion import Rotation
from orix.vector import Vector3d

from kikuchipy.pattern import rescale_intensity
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


@nb.njit(nogil=True)
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
            r_g_array[i][j][0] = det_y[ii] * ca + sa * Ls[j]
            r_g_array[i][j][1] = Lc[j]
            r_g_array[i][j][2] = -sa * det_y[ii] + ca * Ls[j]

    # Normalize
    norm = np.sqrt(np.sum(np.square(r_g_array), axis=-1))
    norm = np.expand_dims(norm, axis=-1)
    r_g_array = np.divide(r_g_array, norm)

    return r_g_array


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
            niip_i = nii_i
        if nijp_i > npy:
            nijp_i = nij_i
        if nii_i < 0:
            nii_i = niip_i
        if nij_i < 0:
            nij_i = nijp_i

        nii[ii] = nii_i
        nij[ii] = nij_i
        niip[ii] = niip_i
        nijp[ii] = nijp_i
        di[ii] = i_this - nii_i + scale
        dj[ii] = j_this - nij_i + scale
        dim[ii] = 1 - di[ii]
        djm[ii] = 1 - dj[ii]

    return nii, nij, niip, nijp, di, dj, dim, djm


def _get_patterns_chunk(
    rotations_array: np.ndarray,
    dc: Vector3d,
    master_north: np.ndarray,
    master_south: np.ndarray,
    npx: int,
    npy: int,
    scale: Union[int, float],
    rescale: bool,
    dtype_out: Optional[type] = np.float32,
) -> np.ndarray:
    """Get the EBSD patterns on the detector for each rotation in the
    chunk. Each pattern is found by a bi-quadratic interpolation of the
    master pattern as described in EMsoft.

    Parameters
    ----------
    rotations_array
        Array of rotations of shape (..., 4) for a given chunk in
        quaternions.
    dc
        Direction cosines unit vector between detector and sample.
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
    """
    rot = Rotation(rotations_array)
    sig_shape = dc.shape
    dc_flat = dc.reshape(dc.size)
    nav_shape = rot.shape
    #    simulated = np.empty(shape=nav_shape + sig_shape, dtype=dtype_out)
    n_pixels = dc_flat.shape
    simulated = np.zeros(shape=nav_shape + n_pixels, dtype=dtype_out)

    interpolation_kwargs = dict(npx=int(npx), npy=int(npy), scale=float(scale))
    for i in np.ndindex(nav_shape):
        dc_rot = rot[i] * dc_flat
        (
            nii,
            nij,
            niip,
            nijp,
            di,
            dj,
            dim,
            djm,
        ) = _get_lambert_interpolation_parameters(v=dc_rot.data, **interpolation_kwargs)
        pattern = np.where(
            dc_rot.z >= 0,
            (
                master_north[nii, nij] * dim * djm
                + master_north[niip, nij] * di * djm
                + master_north[nii, nijp] * dim * dj
                + master_north[niip, nijp] * di * dj
            ),
            (
                master_south[nii, nij] * dim * djm
                + master_south[niip, nij] * di * djm
                + master_south[nii, nijp] * dim * dj
                + master_south[niip, nijp] * di * dj
            ),
        )
        if rescale:
            pattern = rescale_intensity(pattern, dtype_out=dtype_out)
        #        simulated[i] = pattern.reshape(sig_shape)
        simulated[i] = pattern

    return simulated.reshape(nav_shape + sig_shape)
