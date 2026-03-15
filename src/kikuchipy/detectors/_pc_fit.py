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

import logging

import numpy as np
from orix.quaternion import Rotation
from orix.vector import Vector3d
import scipy.stats as scs
from skimage.transform import ProjectiveTransform
from sklearn.linear_model import LinearRegression, RANSACRegressor

_logger = logging.getLogger(__name__)


def fit_hyperplane(
    pc_centered: np.ndarray,
) -> tuple[float, float, Rotation, Rotation, np.ndarray]:
    # Hyperplane fit
    pc_trim_mean = scs.trim_mean(pc_centered, proportiontocut=0.1)
    pc_trim_centered = pc_centered - pc_trim_mean[np.newaxis, :]
    # u @ np.diag(s) @ vh = (u * s) @ vh
    u, s, vh = np.linalg.svd(pc_trim_centered, full_matrices=False)

    # Check handedness of the coordinate system spanned by the plane
    # normals. The determinant should be 1.
    determinant = np.linalg.det(vh)
    _logger.debug(f"Determinant of SVD: {determinant:.7f}")

    # Extract estimated sample plane unit normal vector
    sample_normal = Vector3d(vh[2])
    sample_normal = sample_normal.unit
    if sample_normal.z < 0:
        # Make normal point towards detector screen
        sample_normal = -sample_normal
    _logger.debug(f"Sample plane normal from SVD: {sample_normal.data.squeeze()}")

    # Tilt about the detector x and z axes
    vx, vy, vz = sample_normal.data.squeeze()
    x_tilt = np.arccos(vz)
    z_tilt = np.pi / 2 - np.arctan2(vy, vx)

    # Check that rotation of [001] gives the surface normal in the
    # detector system
    rot_xtilt = Rotation.from_axes_angles([1, 0, 0], -x_tilt)
    rot_ztilt = Rotation.from_axes_angles([0, 0, 1], -z_tilt)
    sample_normal_tilts = rot_ztilt * rot_xtilt * Vector3d.zvector()
    _logger.debug(
        f"Sample plane normal from tilts: {sample_normal_tilts.data.squeeze()}"
    )

    return x_tilt, z_tilt, rot_xtilt, rot_ztilt, pc_trim_mean


def fit_pc_projective(
    pc_centered_flat: np.ndarray,
    pc_indices_flat: np.ndarray,
    map_indices_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    *_, rot_xtilt, rot_ztilt, pc_trim_mean = fit_hyperplane(pc_centered_flat)

    v_pc_centered = Vector3d(pc_centered_flat)
    v_pc_trim_mean = Vector3d(pc_trim_mean)
    v_pc_plane = ~(rot_ztilt * rot_xtilt) * (v_pc_centered - v_pc_trim_mean)

    # Get transformation matrix
    tform = ProjectiveTransform()
    status = tform.estimate(pc_indices_flat[:, :2], v_pc_plane.data[:, :2])
    _logger.debug(f"Status of projective transformation: {status}")
    matrix = tform.params.T

    # PC coordinates projected from beam indices and tilt parameters
    pc_fit = np.dot(pc_indices_flat, matrix)
    pc_fit /= pc_fit[:, 2, None]
    pc_fit[:, 2] = 0
    v_pc_fit = rot_ztilt * rot_xtilt * Vector3d(pc_fit)
    v_pc_fit += v_pc_trim_mean

    # Do the same for interpolated PC coordinates
    pc_fit_map = np.dot(map_indices_flat, matrix)
    pc_fit_map /= pc_fit_map[:, 2, None]
    pc_fit_map[:, 2] = 0
    v_pc_fit_map = rot_ztilt * rot_xtilt * Vector3d(pc_fit_map)
    v_pc_fit_map += v_pc_trim_mean

    return v_pc_fit.data, v_pc_fit_map.data


def fit_pc_affine(
    pc_flat: np.ndarray, pc_indices_flat: np.ndarray, map_indices_flat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Solve the least squares problem X * A = Y
    # Source: https://stackoverflow.com/a/20555267/3228100
    matrix, res, *_ = np.linalg.lstsq(pc_indices_flat, pc_flat, rcond=None)
    _logger.debug(f"Residuals of least squares fit: {res}")

    pc_fit = np.dot(pc_indices_flat, matrix)
    pc_fit_map = np.dot(map_indices_flat, matrix)

    return pc_fit, pc_fit_map


def estimate_xtilt_linear(
    pcy: np.ndarray, pcz: np.ndarray
) -> tuple[float, LinearRegression]:
    """Return an estimated X tilt from (PCy, PCz) using a linear model."""
    regressor = LinearRegression()
    regressor.fit(pcz, pcy)
    slope = regressor.coef_
    slope = slope.squeeze()
    x_tilt = float(np.pi / 2 + np.arctan(slope))
    return x_tilt, regressor


def estimate_xtilt_linear_robust(
    pcy: np.ndarray, pcz: np.ndarray
) -> tuple[float, RANSACRegressor, np.ndarray]:
    """Return an estimated X tilt from (PCy, PCz) using a robust linear
    model with detection of outliers.
    """
    regressor = RANSACRegressor()
    regressor.fit(pcz, pcy)
    slope = regressor.estimator_.coef_
    slope = slope.squeeze()
    x_tilt = float(np.pi / 2 + np.arctan(slope))
    is_outlier = ~regressor.inlier_mask_
    return x_tilt, regressor, is_outlier
