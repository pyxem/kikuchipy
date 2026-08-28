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

import numba as nb
import numpy as np
import orix.quaternion as oqu

from kikuchipy._kikuchi_sphere._interpolation import _bilinear_lookup
from kikuchipy.detectors._ebsd_detector import EBSDDetector


def _get_unit_sphere_vectors(
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inverse-stereographic unit vectors for both hemispheres.

    Parameters
    ----------
    n
        Side length of the square stereographic grid.

    Returns
    -------
    v_upper
        Unit vectors on the upper hemisphere (z >= 0), shape (n*n, 3),
        C-contiguous float64.
    v_lower
        Unit vectors on the lower hemisphere (z <= 0), shape (n*n, 3),
        C-contiguous float64.
    inside_disk
        Boolean mask of shape (n*n,) marking pixels inside the unit
        disk.
    """
    arr = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    X, Y = np.meshgrid(arr, arr[::-1])
    flat_X = X.ravel()
    flat_Y = Y.ravel()
    r2 = flat_X**2 + flat_Y**2
    inside_disk = r2 <= 1.0

    denom = 1.0 + r2
    xy = np.stack([2.0 * flat_X / denom, 2.0 * flat_Y / denom], axis=1)
    z_pole = (1.0 - r2) / denom  # +z for upper, -z for lower

    v_upper = np.ascontiguousarray(np.column_stack([xy, z_pole]))
    v_lower = np.ascontiguousarray(np.column_stack([xy, -z_pole]))

    return v_upper, v_lower, inside_disk


def _combine_transformations(
    rot_s2c: oqu.Orientation | oqu.Rotation | oqu.Quaternion, detector: EBSDDetector
) -> np.ndarray:
    """Return the (3, 3) matrix mapping crystal unit vectors to the
    detector frame: v_det = rotm @ v_crystal.

    Combines the inverse crystal orientation and the sample-to-detector
    rotation into a single C-contiguous float64 matrix.
    """
    # Detector <- sample <- crystal
    R_c2d = detector.sample_to_detector * (~rot_s2c)
    rotm_c2d = np.ascontiguousarray(R_c2d.to_matrix().squeeze(), dtype=np.float64)
    return rotm_c2d


@nb.njit(
    (
        "float64[:]"
        "(float64[:, :], bool[:], float64[:, :], float64[:, :], float64, float64, "
        "float64, float64, int64, int64)"
    ),
    cache=True,
    fastmath=True,
    nogil=True,
    parallel=True,
)
def _project_detector_to_sphere(
    v_c: np.ndarray,
    inside_disk: np.ndarray,
    rotm_c2d: np.ndarray,
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
    inside_disk
        (n,) boolean mask. Only pixels within the stereographic disk are
        considered.
    rotm_c2d
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
    For each pixel inside the unit disk the vector is rotated into the
    detector frame using the pre-built orientation matrix *om_c2d*,
    gnomonic-projected, and the detector intensity is recovered via
    bilinear interpolation. Pixels that fall outside the detector face
    or behind it receive NaN.
    """
    n = v_c.shape[0]
    out = np.full(n, np.nan)

    for i in nb.prange(n):
        if not inside_disk[i]:
            continue

        # Rotate crystal-frame vector into detector frame
        # fmt: off
        vz = (
              rotm_c2d[2, 0] * v_c[i, 0]
            + rotm_c2d[2, 1] * v_c[i, 1]
            + rotm_c2d[2, 2] * v_c[i, 2]
        )
        # fmt: on
        # Gnomonic projection only valid toward detector, vz > 0
        if vz <= 0.0:
            continue

        # fmt: off
        vx = (
              rotm_c2d[0, 0] * v_c[i, 0]
            + rotm_c2d[0, 1] * v_c[i, 1]
            + rotm_c2d[0, 2] * v_c[i, 2]
        )
        vy = (
              rotm_c2d[1, 0] * v_c[i, 0]
            + rotm_c2d[1, 1] * v_c[i, 1]
            + rotm_c2d[1, 2] * v_c[i, 2]
        )
        # fmt: on

        # Gnomonic projection
        col_frac = (vx / vz - gn_x0) / x_scale - 0.5
        row_frac = (gn_y1 - vy / vz) / y_scale - 0.5
        if (
            col_frac < 0.0
            or col_frac > ncols - 1
            or row_frac < 0.0
            or row_frac > nrows - 1
        ):
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
            if counts[i, j] >= 0:
                sphere[i, j] = accumulator[i, j] / counts[i, j]
    return sphere


def _detector_to_stereo(
    pattern: np.ndarray,
    detector: EBSDDetector,
    rot_s2c: oqu.Orientation | oqu.Rotation | oqu.Quaternion,
    n: int = 1001,
) -> np.ndarray:
    """Return a pattern on a flat EBSD detector projected onto the
    stereographic equatorial plane, for both hemispheres.

    Parameters
    ----------
    pattern
        2D detector image of shape (nrows, ncols).
    detector
        EBSDDetector with a single PC.
    rot_s2c
        Rotation transforming vectors from the sample to the crystal.
    n
        Side length of the square output grid.

    Returns
    -------
    out
        Array of shape (2, *n*, *n*): index 0 is the upper hemisphere,
        index 1 is the lower hemisphere.

    Notes
    -----
    Inverse mapping is used: for each stereographic output coordinate
    (X, Y) the corresponding fractional detector pixel is found and
    sampled via bilinear interpolation.  Pixels outside the detector
    are NaN.
    """
    v_upper, v_lower, inside_disk = _get_unit_sphere_vectors(n)
    rotm_c2d = _combine_transformations(rot_s2c=rot_s2c, detector=detector)
    pattern_f64 = np.ascontiguousarray(pattern, dtype=np.float64)

    gn = detector.gnomonic_bounds.squeeze()  # [x0, x1, y0, y1]
    gn_x0 = float(gn[0])
    gn_y1 = float(gn[3])
    x_scale = float((gn[1] - gn[0]) / detector.ncols)
    y_scale = float((gn[3] - gn[2]) / detector.nrows)
    nrows, ncols = int(detector.nrows), int(detector.ncols)

    out = np.empty((2, n * n), dtype=np.float64)
    out[0] = _project_detector_to_sphere(
        v_upper,
        inside_disk,
        rotm_c2d,
        pattern_f64,
        gn_x0,
        gn_y1,
        x_scale,
        y_scale,
        nrows,
        ncols,
    )
    out[1] = _project_detector_to_sphere(
        v_lower,
        inside_disk,
        rotm_c2d,
        pattern_f64,
        gn_x0,
        gn_y1,
        x_scale,
        y_scale,
        nrows,
        ncols,
    )

    return out.reshape(2, n, n)


def _detector_to_stereo_symmetrized(
    pattern: np.ndarray,
    detector: EBSDDetector,
    ori_s2c: oqu.Orientation,
    n: int = 1001,
) -> np.ndarray:
    """Return the symmetry-averaged stereographic projection over all
    proper symmetrically equivalent orientations.

    Parameters
    ----------
    pattern
        2D detector image of shape (nrows, ncols).
    detector
        EBSD detector with a single PC.
    ori_s2c
        Orientation transforming vectors from the sample to the crystal,
        with a valid symmetry set.
    n
        Side length of the square output grid. Default is 1001.

    Returns
    -------
    sphere
        Array of shape (2, n, n) with the per-pixel mean intensity over
        all contributing operators.  Index 0 is the upper hemisphere,
        index 1 is the lower hemisphere. Pixels that received no
        contribution are NaN.
    """
    v_upper, v_lower, inside_disk = _get_unit_sphere_vectors(n)
    pattern_f64 = np.ascontiguousarray(pattern, dtype=np.float64)

    gn = detector.gnomonic_bounds.squeeze()  # [x0, x1, y0, y1]
    gn_x0 = float(gn[0])
    gn_y1 = float(gn[3])
    x_scale = float((gn[1] - gn[0]) / detector.ncols)
    y_scale = float((gn[3] - gn[2]) / detector.nrows)
    nrows, ncols = int(detector.nrows), int(detector.ncols)

    # Extract sample-to-detector rotation matrix; only the crystal
    # orientation changes per operator so we avoid recomputing this each
    # iteration.
    rotm_s2d = detector.sample_to_detector.to_matrix().squeeze()

    accumulator = np.zeros((2, n * n), dtype=np.float64)
    counts = np.zeros((2, n * n), dtype=np.int32)

    # Overwritten in every iteration
    partial_sphere = np.empty((2, n * n), dtype=np.float64)

    sym_ops = ori_s2c.symmetry.proper_subgroup
    for sym_op in sym_ops:
        # Symmetrically equivalent rotation (applied in the crystal).
        # Inverted and rotation matrix extracted.
        rotm_c2s_eq = (~(sym_op * ori_s2c)).to_matrix().squeeze()
        # Detector <- sample <- crystal
        rotm_c2d = np.ascontiguousarray(rotm_s2d @ rotm_c2s_eq)

        partial_sphere[0] = _project_detector_to_sphere(
            v_upper,
            inside_disk,
            rotm_c2d,
            pattern_f64,
            gn_x0,
            gn_y1,
            x_scale,
            y_scale,
            nrows,
            ncols,
        )
        partial_sphere[1] = _project_detector_to_sphere(
            v_lower,
            inside_disk,
            rotm_c2d,
            pattern_f64,
            gn_x0,
            gn_y1,
            x_scale,
            y_scale,
            nrows,
            ncols,
        )
        _accumulate_intensities(accumulator, counts, partial_sphere)

    sphere = _normalize_accumulator(accumulator, counts)
    sphere = sphere.reshape(2, n, n)

    return sphere
