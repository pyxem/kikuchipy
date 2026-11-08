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

"""Functions for interpolation."""

import numba as nb
import numpy as np


@nb.njit(
    "float64(float64[:, :], float64, float64, int64, int64)",
    cache=True,
    nogil=True,
)
def _bilinear_lookup_nan_skip(
    image: np.ndarray, row_frac: float, col_frac: float, nrows: int, ncols: int
) -> float:
    """Bilinear interpolation at a single fractional coordinate,
    skipping NaN-valued neighbors and renormalizing over the valid ones.

    Returns NaN only if all four neighbors are NaN.
    """
    r0 = int(row_frac)
    c0 = int(col_frac)
    if r0 + 1 < nrows:
        r1 = r0 + 1
    else:
        r1 = r0
    if c0 + 1 < ncols:
        c1 = c0 + 1
    else:
        c1 = c0
    dr = row_frac - r0
    dc = col_frac - c0

    i00 = image[r0, c0]
    i01 = image[r0, c1]
    i10 = image[r1, c0]
    i11 = image[r1, c1]

    total_w = 0.0
    result = 0.0

    if not np.isnan(i00):
        w00 = (1.0 - dr) * (1.0 - dc)
        total_w += w00
        result += w00 * i00

    if not np.isnan(i01):
        w01 = (1.0 - dr) * dc
        total_w += w01
        result += w01 * i01

    if not np.isnan(i10):
        w10 = dr * (1.0 - dc)
        total_w += w10
        result += w10 * i10

    if not np.isnan(i11):
        w11 = dr * dc
        total_w += w11
        result += w11 * i11

    if total_w > 0.0:
        return result / total_w

    return np.nan


@nb.njit(
    "float64(float64[:, :], float64, float64, int64, int64)",
    cache=True,
    fastmath=True,
)
def _bilinear_lookup(
    image: np.ndarray, row_frac: float, col_frac: float, nrows: int, ncols: int
) -> float:
    """Bilinear interpolation at a single fractional square grid
    coordinate.

    Parameters
    ----------
    image
        2D array to sample from.
    row_frac
        Fractional row index. Caller guarantees
        0 <= row_frac <= nrows-1.
    col_frac
        Fractional column index. Caller guarantees
        0 <= col_frac <= ncols-1.
    nrows
        Image height.
    ncols
        Image width.

    Returns
    -------
    out
        Bilinearly interpolated float64 value.
    """
    r0 = int(row_frac)
    c0 = int(col_frac)
    if r0 + 1 < nrows:
        r1 = r0 + 1
    else:
        r1 = r0
    if c0 + 1 < ncols:
        c1 = c0 + 1
    else:
        c1 = c0
    dr = row_frac - r0
    dc = col_frac - c0
    # fmt: off
    out = (
          (1.0 - dr) * (1.0 - dc) * image[r0, c0]
        + (1.0 - dr) * dc         * image[r0, c1]
        + dr         * (1.0 - dc) * image[r1, c0]
        + dr         * dc         * image[r1, c1]
    )
    # fmt: on
    return out


@nb.njit(
    "float64[:](float64[:, :], float64[:], float64[:])",
    cache=True,
    nogil=True,
    parallel=True,
)
def _bilinear_interpolate_nan_skip(
    image: np.ndarray, row_frac: np.ndarray, col_frac: np.ndarray
) -> np.ndarray:
    """Bilinear interpolation of a 2D image, skipping NaN-valued pixels.

    Uses :func:`_bilinear_lookup_nan_skip` so that NaN pixels in *image*
    are excluded from the weighted average and the remaining weights are
    renormalized.  This prevents NaN (or zero-filled) out-of-mask pixels
    from darkening the result at mask boundaries.

    Parameters
    ----------
    image
        2D float64 array to sample from.  NaN marks pixels to exclude.
    row_frac
        1D array of fractional row indices.
    col_frac
        1D array of fractional column indices.

    Returns
    -------
    out
        1D float64 array of interpolated values. NaN where the
        coordinate falls outside the image bounds or all four neighbors
        are NaN.
    """
    nrows = image.shape[0]
    ncols = image.shape[1]
    n = row_frac.shape[0]
    out = np.full(n, np.nan)
    for i in nb.prange(n):
        rf = row_frac[i]
        cf = col_frac[i]
        if np.isnan(rf) or np.isnan(cf):
            continue
        if rf < 0.0 or rf > nrows - 1 or cf < 0.0 or cf > ncols - 1:
            continue
        out[i] = _bilinear_lookup_nan_skip(image, rf, cf, nrows, ncols)
    return out
