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

"""Geometrical Kikuchi pattern simulation."""

from typing import Literal

import diffpy.structure as dst
from diffsims.crystallography import ReciprocalLatticeVector
import numba as nb
import numpy as np
from orix.quaternion import Rotation

from kikuchipy.detectors._ebsd_detector import EBSDDetector


@nb.njit(
    "Tuple((float64[:, :], float64[:, :]))"
    "(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64)",
    cache=True,
    fastmath=True,
    nogil=True,
)
def _bands_and_zone_axes_kernel(
    hkl: np.ndarray,
    uvw: np.ndarray,
    u_kstar: np.ndarray,
    u_k: np.ndarray,
    max_r: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gnomonic coordinates of projected reciprocal *hkl* and
    direct *uvw* crystal vectors.

    This function shouldn't use parallelization; the idea is that this
    function itself should be run in parallel.
    """
    n_hkl = hkl.shape[0]
    n_uvw = uvw.shape[0]

    band_buf = np.empty((n_hkl, 4), dtype=np.float64)
    za_buf = np.empty((n_uvw, 2), dtype=np.float64)
    nb_bands = 0
    nb_za = 0

    # Bands
    for i in range(n_hkl):
        v_h = hkl[i, 0]
        v_k = hkl[i, 1]
        v_l = hkl[i, 2]
        dx = v_h * u_kstar[0, 0] + v_k * u_kstar[1, 0] + v_l * u_kstar[2, 0]
        dy = v_h * u_kstar[0, 1] + v_k * u_kstar[1, 1] + v_l * u_kstar[2, 1]
        dz = v_h * u_kstar[0, 2] + v_k * u_kstar[1, 2] + v_l * u_kstar[2, 2]
        if dz < 0.0:
            continue
        xy_norm = np.sqrt(dx * dx + dy * dy)
        if xy_norm == 0.0:
            continue
        hesse = dz / xy_norm
        if np.abs(hesse) >= max_r:
            continue
        alpha = np.arccos(hesse / max_r)
        az = np.atan2(dy, dx) - np.pi
        a1 = az + alpha
        a2 = az - alpha
        band_buf[nb_bands, 0] = max_r * np.cos(a1)
        band_buf[nb_bands, 1] = max_r * np.sin(a1)
        band_buf[nb_bands, 2] = max_r * np.cos(a2)
        band_buf[nb_bands, 3] = max_r * np.sin(a2)
        nb_bands += 1

    # Zone axes
    for i in range(n_uvw):
        u0 = uvw[i, 0]
        u1 = uvw[i, 1]
        u2 = uvw[i, 2]
        dx = u0 * u_k[0, 0] + u1 * u_k[1, 0] + u2 * u_k[2, 0]
        dy = u0 * u_k[0, 1] + u1 * u_k[1, 1] + u2 * u_k[2, 1]
        dz = u0 * u_k[0, 2] + u1 * u_k[1, 2] + u2 * u_k[2, 2]
        if dz <= 0.0:
            continue
        xg = dx / dz
        yg = dy / dz
        if np.sqrt(xg * xg + yg * yg) >= max_r:
            continue
        za_buf[nb_za, 0] = xg
        za_buf[nb_za, 1] = yg
        nb_za += 1

    return band_buf[:nb_bands], za_buf[:nb_za]


def _get_geometrical_bands_and_zone_axes_gnomonic(
    hkl: np.ndarray,
    uvw: np.ndarray,
    base: np.ndarray,
    recbase: np.ndarray,
    rotation_matrix: np.ndarray,
    u_s: np.ndarray,
    max_r_gnomonic: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return gnomonic coordinates of visible Kikuchi bands
    :math:`{hkl}` and zone axes :math:`\left<uvw\right>` for a single
    crystal orientation.

    Parameters
    ----------
    hkl
        Miller indices of the reflectors, shape (n, 3).
    uvw
        Zone axis indices, shape (m, 3). These should be obtained from
        the cross products of *hkl*.
    base
        Direct lattice basis (3, 3), from
        ``phase.structure.lattice.base``.
    recbase
        Reciprocal lattice basis (3, 3), from
        ``phase.structure.lattice.recbase``.
    rotation_matrix
        Crystal orientation rotation matrix (3, 3), from
        ``rotation.to_matrix().squeeze()``.
    u_s
        Sample-to-detector rotation matrix (3, 3), from
        ``detector.sample_to_detector.to_matrix().squeeze()``.
    max_r_gnomonic
        Maximum gnomonic radius. Default is 10.0.

    Returns
    -------
    band_coords
        Gnomonic plane-trace endpoints (x0, y0, x1, y1) of visible
        Kikuchi bands, shape (k, 4).
    zone_axes_coords
        Gnomonic (x, y) positions of visible zone axes, shape (l, 2).
    """
    u_os = rotation_matrix @ u_s

    # Reciprocal lattice -> crystal -> sample -> detector
    u_kstar = recbase.T @ u_os

    # Direct lattice -> crystal -> sample -> detector
    u_k = base @ u_os

    u_kstar = np.ascontiguousarray(u_kstar, dtype=np.float64)
    u_k = np.ascontiguousarray(u_k, dtype=np.float64)
    hkl = np.ascontiguousarray(hkl, dtype=np.float64)
    uvw = np.ascontiguousarray(uvw, dtype=np.float64)

    band_coords, zone_axes_coords = _bands_and_zone_axes_kernel(
        hkl, uvw, u_kstar, u_k, max_r_gnomonic
    )

    return band_coords, zone_axes_coords


def _get_zone_axes_from_hkl(hkl: np.ndarray) -> np.ndarray:
    """Return unique zone axis directions *uvw* as pairwise cross
    products of *hkl*.

    Parameters
    ----------
    hkl
        Miller indices, shape (n, 3).

    Returns
    -------
    uvw
        Zone axis directions, shape (m, 3). Zero vectors and duplicates
        are removed.
    """
    uvw = np.cross(hkl[:, None, :], hkl[None, :, :]).reshape(-1, 3)
    uvw = uvw[~np.all(np.isclose(uvw, 0), axis=1)]
    return np.unique(np.round(uvw, 6), axis=0)


def _get_geometrical_simulation(
    reflectors: ReciprocalLatticeVector,
    detector: EBSDDetector,
    rotation: Rotation,
    uvw: np.ndarray | None = None,
    coords_fmt: Literal["detector", "gnomonic"] = "detector",
) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates of visible Kikuchi bands and zone axes.

    Parameters
    ----------
    reflectors
        Reciprocal lattice vectors (reflectors).
    detector
        EBSD detector with navigation shape (1,).
    rotation
        Single crystal orientation.
    uvw
        Pre-computed zone axis directions, shape (m, 3). If not given
        (default), they are derived as pairwise cross products of the
        *reflectors* hkl indices.

        Pass a pre-computed array when calling this function repeatedly
        with the same *reflectors* to avoid redundant work.
    coords_fmt
        Coordinate format for the returned arrays:

        - "detector" (default) returns uncalibrated detector-pixel
          coordinates
        - "gnomonic" returns gnomonic coordinates with the origin at the
          projection center

    Returns
    -------
    band_coords
        Plane-trace endpoints (x0, y0, x1, y1) of visible Kikuchi bands
        in the requested coordinate system, shape (k, 4).
    zone_axes_coords
        (x, y) positions of visible zone axes in the requested
        coordinate system, shape (l, 2).
    """
    # Setup
    lattice: dst.Lattice = reflectors.phase.structure.lattice
    hkl = reflectors.hkl.astype(float).reshape(-1, 3)
    base = np.asarray(lattice.base).reshape(3, 3)
    recbase = np.asarray(lattice.recbase).reshape(3, 3)
    rotation_matrix = rotation.to_matrix().squeeze()
    u_s = detector.sample_to_detector.to_matrix().squeeze()
    max_r_gnomonic = np.float64(np.max(detector.r_max))

    # PC components and detector scales (needed for both coordinate
    # formats to filter zone axes by rectangular detector bounds)
    pcx, pcy, pcz = detector.pc_average
    aspect_ratio = detector.aspect_ratio
    x_scale = float(detector.x_scale.item())
    y_scale = float(detector.y_scale.item())

    x_range = detector.x_range.squeeze().copy()
    y_range = detector.y_range.squeeze().copy()
    x_range[0] -= x_scale
    x_range[1] += x_scale
    y_range[0] -= y_scale
    y_range[1] += y_scale

    if uvw is None:
        uvw = _get_zone_axes_from_hkl(hkl)

    band_gnomonic, za_gnomonic = _get_geometrical_bands_and_zone_axes_gnomonic(
        hkl=hkl,
        uvw=uvw,
        base=base,
        recbase=recbase,
        rotation_matrix=rotation_matrix,
        u_s=u_s,
        max_r_gnomonic=max_r_gnomonic,
    )

    if coords_fmt == "gnomonic":
        if za_gnomonic.shape[0] == 0:
            return band_gnomonic, np.empty((0, 2))

        # Filter zone axes by the rectangular detector bounds
        xg = za_gnomonic[:, 0]
        yg = za_gnomonic[:, 1]
        within = (
            (xg >= x_range[0])
            & (xg <= x_range[1])
            & (yg >= y_range[0])
            & (yg <= y_range[1])
        )
        if not within.any():
            return band_gnomonic, np.empty((0, 2))

        return band_gnomonic, za_gnomonic[within]

    # Bands
    band_det = band_gnomonic.copy()
    band_det[:, [0, 2]] = (
        band_gnomonic[:, [0, 2]] + (pcx / pcz) * aspect_ratio
    ) / x_scale
    band_det[:, [1, 3]] = (-band_gnomonic[:, [1, 3]] + (pcy / pcz)) / y_scale

    # Zone axes
    if za_gnomonic.shape[0] == 0:
        return band_det, np.empty((0, 2))

    xg = za_gnomonic[:, 0]
    yg = za_gnomonic[:, 1]
    za_det_x = (xg + (pcx / pcz) * aspect_ratio) / x_scale
    za_det_y = (-yg + (pcy / pcz)) / y_scale

    # Filter zone axes by rectangular gnomonic bounds, extended by one
    # pixel in each direction to include zone axes on the border
    within = (
        (xg >= x_range[0])
        & (xg <= x_range[1])
        & (yg >= y_range[0])
        & (yg <= y_range[1])
    )
    if not within.any():
        return band_det, np.empty((0, 2))

    za_det = np.column_stack([za_det_x[within], za_det_y[within]])

    return band_det, za_det
