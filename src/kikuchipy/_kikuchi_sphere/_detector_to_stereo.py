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

"""Functions for projecting from the detector to the unit sphere."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numba as nb
import numpy as np
import orix.quaternion as oqu
from tqdm import tqdm

from kikuchipy._kikuchi_sphere._interpolation import _bilinear_lookup

if TYPE_CHECKING:  # pragma: no cover
    from kikuchipy.detectors._ebsd_detector import EBSDDetector


def _get_unit_sphere_vectors(
    size: int, exclude_outside_equator: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inverse-stereographic unit vectors for both hemispheres.

    Parameters
    ----------
    size
        Side length of the square stereographic grid.
    exclude_outside_equator
        Whether to exclude pixels outside the equator in the returned
        mask. If False, all pixels are included. Pixels outside the disk
        then map to vectors on the opposite hemisphere. Default is True.

    Returns
    -------
    v_upper
        Unit vectors on the upper hemisphere (z >= 0), shape (size*size,
        3), C-contiguous float64.
    v_lower
        Unit vectors on the lower hemisphere (z <= 0), shape (size*size,
        3), C-contiguous float64.
    inside_equator
        Boolean mask of shape (size*size,) marking pixels inside the
        unit disk. All True if *exclude_outside_equator* is False.
    """
    arr = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    X, Y = np.meshgrid(arr, arr[::-1])
    flat_X = X.ravel()
    flat_Y = Y.ravel()
    r2 = flat_X**2 + flat_Y**2
    if exclude_outside_equator:
        inside_equator = r2 <= 1.0
    else:
        inside_equator = np.ones(r2.size, dtype=bool)

    denom = 1.0 + r2
    xy = np.stack([2.0 * flat_X / denom, 2.0 * flat_Y / denom], axis=1)
    z_pole = (1.0 - r2) / denom  # +z for upper, -z for lower

    v_upper = np.ascontiguousarray(np.column_stack([xy, z_pole]))
    v_lower = np.ascontiguousarray(np.column_stack([xy, -z_pole]))

    return v_upper, v_lower, inside_equator


def _get_crystal_to_detector_matrices(
    mat_s2c: oqu.Orientation | oqu.Rotation,
    detector: EBSDDetector,
    symmetrize: bool = False,
) -> np.ndarray:
    """Return the matrices mapping crystal unit vectors to the detector
    frame, v_det = mat @ v_crystal, for each rotation and, optionally,
    each of its proper symmetrically equivalent rotations.

    Parameters
    ----------
    mat_s2c
        Rotations transforming vectors from the sample to the crystal.
        Must be an orientation with a valid symmetry if *symmetrize* is
        True.
    detector
        EBSD detector providing the sample-to-detector rotation.
    symmetrize
        Whether to include all proper symmetrically equivalent
        rotations. Default is False.

    Returns
    -------
    mat_c2d
        C-contiguous float64 array of shape (n_rotations, n_operators,
        3, 3). The number of operators is 1 if *symmetrize* is False.
    """
    rot = oqu.Rotation(mat_s2c).flatten()
    if symmetrize:
        sym_ops = oqu.Rotation(mat_s2c.symmetry.proper_subgroup)
    else:
        sym_ops = oqu.Rotation.identity()

    # Symmetrically equivalent rotations (applied in the crystal) of
    # shape (n_operators, n_rotations), inverted to crystal-to-sample
    mat_c2s = (~sym_ops.outer(rot)).to_matrix()

    # Detector <- sample <- crystal
    mat_s2d = detector.sample_to_detector.to_matrix().squeeze()
    mat_c2d = mat_s2d @ mat_c2s

    mat_c2d = mat_c2d.transpose(1, 0, 2, 3)
    mat_c2d = np.ascontiguousarray(mat_c2d, dtype=np.float64)

    return mat_c2d


@nb.njit(
    (
        "float64[:]"
        "(float64[:, :], boolean[:], float64[:, :], float64[:, :], float64, float64, "
        "float64, float64, int64, int64)"
    ),
    cache=True,
    fastmath=True,
    nogil=True,
    parallel=True,
)
def _project_detector_to_sphere(
    v_c: np.ndarray,
    inside_equator: np.ndarray,
    mat_c2d: np.ndarray,
    pattern: np.ndarray,
    gn_x0: float,
    gn_y1: float,
    x_scale: float,
    y_scale: float,
    nrows: int,
    ncols: int,
) -> np.ndarray:
    """Project unit-sphere vectors onto the detector and sample the
    pattern.

    Parameters
    ----------
    v_c
        (n, 3) unit vectors in the crystal frame, C-contiguous float64.
    inside_equator
        (n,) boolean mask. Only pixels within the stereographic equator
        are considered.
    mat_c2d
        Rotation matrix (3, 3) transforming vectors from crystal to
        detector.
    pattern
        Pattern of shape (nrows, ncols), float64.
    gn_x0
        Left gnomonic bound.
    gn_y1
        Top gnomonic bound.
    x_scale
        Gnomonic-to-pixel horizontal scale factor (gnomonic units per
        pixel).
    y_scale
        Gnomonic-to-pixel vertical scale factor (gnomonic units per
        pixel).
    nrows
        Number of detector rows.
    ncols
        Number of detector columns.

    Returns
    -------
    out
        Interpolated intensities (n,), float64. NaN where invalid.

    Notes
    -----
    For each pixel inside the equator the vector is rotated into the
    detector frame using the pre-built orientation matrix *mat_c2d*,
    gnomonic-projected, and the detector intensity is recovered via
    bilinear interpolation. Pixels that fall outside the detector face
    or behind it receive NaN.
    """
    n = v_c.shape[0]
    out = np.full(n, np.nan)

    ncols_1 = ncols - 1
    nrows_1 = nrows - 1

    for i in nb.prange(n):
        if not inside_equator[i]:
            continue

        # Rotate crystal-frame vector into detector frame
        # fmt: off
        vz = (
              mat_c2d[2, 0] * v_c[i, 0]
            + mat_c2d[2, 1] * v_c[i, 1]
            + mat_c2d[2, 2] * v_c[i, 2]
        )
        # fmt: on
        # Gnomonic projection only valid toward detector, vz > 0
        if vz <= 0.0:
            continue

        # fmt: off
        vx = (
              mat_c2d[0, 0] * v_c[i, 0]
            + mat_c2d[0, 1] * v_c[i, 1]
            + mat_c2d[0, 2] * v_c[i, 2]
        )
        vy = (
              mat_c2d[1, 0] * v_c[i, 0]
            + mat_c2d[1, 1] * v_c[i, 1]
            + mat_c2d[1, 2] * v_c[i, 2]
        )
        # fmt: on

        # Gnomonic projection
        col_frac = (vx / vz - gn_x0) / x_scale - 0.5
        row_frac = (gn_y1 - vy / vz) / y_scale - 0.5
        if col_frac < 0.0 or col_frac > ncols_1 or row_frac < 0.0 or row_frac > nrows_1:
            continue

        # Bilinear interpolation on the regular detector grid
        out[i] = _bilinear_lookup(
            image=pattern,
            row_frac=row_frac,
            col_frac=col_frac,
            nrows=nrows,
            ncols=ncols,
        )

    return out


@nb.njit(
    "(float64[:, :], int32[:, :], float64[:, :])",
    cache=True,
    nogil=True,
    parallel=True,
)
def _accumulate_intensities(
    accumulator: np.ndarray, counts: np.ndarray, partial_sphere: np.ndarray
) -> None:
    """Add one projection's contributions into the running totals
    in-place.

    Parameters
    ----------
    accumulator
        (2, n) float64 running sum of valid intensities. Updated
        in-place.
    counts
        (2, n) int32 number of valid contributions per pixel. Updated
        in-place.
    partial_sphere
        (2, n) float64 stereographic projection from a single symmetry
        operator; NaN marks pixels with no contribution.
    """
    n = partial_sphere.shape[1]
    for i in nb.prange(n):
        for hemi in range(2):
            value = partial_sphere[hemi, i]
            if not np.isnan(value):
                accumulator[hemi, i] += value
                counts[hemi, i] += 1


@nb.njit(cache=True, nogil=True, parallel=True)
def _normalize_accumulator(accumulator: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Return the sphere from the given *accumulator*, divided by the
    number of *counts*, i.e. the number of times a rotated pattern
    contributed to that pixel on the sphere.

    Parameters
    ----------
    accumulator
        (2, n) float64 running sum of valid intensities.
    counts
        (2, n) int32 number of valid contributions per pixel.

    Returns
    -------
    sphere
        Normalized *accumulator*; NaN marks pixels with no contribution.
    """
    sphere = np.full(accumulator.shape, np.nan)
    n = accumulator.shape[1]
    for j in nb.prange(n):
        for i in range(2):
            if counts[i, j] > 0:
                sphere[i, j] = accumulator[i, j] / counts[i, j]
    return sphere


def _detector_to_stereo(
    detector: EBSDDetector,
    pattern: np.ndarray | da.Array,
    mat_s2c: oqu.Orientation | oqu.Rotation,
    size: int = 1_001,
    symmetrize: bool = False,
    exclude_outside_equator: bool = True,
    show_progressbar: bool = False,
) -> np.ndarray:
    """Return the mean of one or more patterns on a flat EBSD detector
    projected onto the stereographic equatorial plane, for both
    hemispheres.

    Parameters
    ----------
    detector
        EBSD detector with either one PC or *n_patterns* PCs.
    pattern
        Detector images of shape (n_patterns, nrows, ncols). If a Dask
        array, chunks along the first axis are computed one at a time.
    mat_s2c
        Rotations transforming vectors from the sample to the crystal,
        of size *n_patterns*. Must be an orientation with a valid
        symmetry if *symmetrize* is True.
    n
        Side length of the square output grid. Default is 1001.
    symmetrize
        Whether to also project each pattern for all proper
        symmetrically equivalent orientations. Default is False.
    exclude_outside_equator
        Whether to leave pixels outside the equator as NaN. If False,
        pixels outside the equator get intensities from the opposite
        hemisphere. Default is True.
    show_progressbar
        Whether to show a progressbar over the patterns. Default is
        False.

    Returns
    -------
    sphere
        Array of shape (2, n, n) with the per-pixel mean intensity over
        all contributing patterns and operators. Index 0 is the upper
        hemisphere, index 1 is the lower hemisphere. Pixels that
        received no contribution are NaN.

    Notes
    -----
    Inverse mapping is used: for each stereographic output coordinate
    (X, Y) the corresponding fractional detector pixel is found and
    sampled via bilinear interpolation.
    """
    n_patterns, nrows, ncols = pattern.shape

    v_upper, v_lower, inside_equator = _get_unit_sphere_vectors(
        size, exclude_outside_equator=exclude_outside_equator
    )
    mat_c2d = _get_crystal_to_detector_matrices(
        mat_s2c=mat_s2c, detector=detector, symmetrize=symmetrize
    )
    gn_params = detector._get_gnomonic_parameters()
    n_pcs = gn_params.shape[0]

    accumulator = np.zeros((2, size * size), dtype=np.float64)
    counts = np.zeros((2, size * size), dtype=np.int32)

    # Overwritten in every iteration
    partial_sphere = np.empty((2, size * size), dtype=np.float64)

    if isinstance(pattern, da.Array):
        chunk_ends = np.cumsum(pattern.chunks[0])
        chunk_starts = chunk_ends - np.array(pattern.chunks[0])
    else:
        chunk_starts = np.array([0])
        chunk_ends = np.array([n_patterns])

    pbar = tqdm(total=n_patterns, disable=not show_progressbar)
    for start, end in zip(chunk_starts, chunk_ends):
        patterns_chunk = np.asarray(pattern[start:end])
        for i in range(start, end):
            pattern_f64 = np.ascontiguousarray(
                patterns_chunk[i - start], dtype=np.float64
            )
            if n_pcs == 1:
                gn_x0, gn_y1, x_scale, y_scale = gn_params[0]
            else:
                gn_x0, gn_y1, x_scale, y_scale = gn_params[i]

            for mat_c2d_i in mat_c2d[i]:
                for hemi, v in enumerate((v_upper, v_lower)):
                    partial_sphere[hemi] = _project_detector_to_sphere(
                        v_c=v,
                        inside_equator=inside_equator,
                        mat_c2d=mat_c2d_i,
                        pattern=pattern_f64,
                        gn_x0=gn_x0,
                        gn_y1=gn_y1,
                        x_scale=x_scale,
                        y_scale=y_scale,
                        nrows=nrows,
                        ncols=ncols,
                    )
                _accumulate_intensities(accumulator, counts, partial_sphere)
            pbar.update(1)
    pbar.close()

    sphere = _normalize_accumulator(accumulator, counts)
    sphere = sphere.reshape(2, size, size)

    return sphere
