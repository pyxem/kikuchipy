# -*- coding: utf-8 -*-
# Copyright 2019-2020 The kikuchipy developers
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

from typing import Union

import numpy as np
from orix.quaternion import rotation as rot
from orix.vector import Vector3d
from orix import sampling, quaternion
import matplotlib.pyplot as plt


from kikuchipy.detectors.ebsd_detector import EBSDDetector
from kikuchipy.projections.lambert_projection import LambertProjection
from kikuchipy import load

master_pattern = load(
    r"C:\Users\laler\Desktop\Project - Kikuchipy\Ni-MP.h5",
    projection="lambert",
    hemisphere="both",
)

euler_angles = r"C:\Users\laler\EMsoftData\testfile.txt"

xpc = 0
ypc = 0
L = 15000
theta_c = 10
sigma = 70
delta = 50

detector = EBSDDetector(
    shape=(480, 640),  # row, columns - y, x
    px_size=delta,
    pc=(xpc, ypc, L),
    # convention="emsoft",
    tilt=theta_c,
    sample_tilt=sigma,
)

r = sampling.sample_generators.get_sample_fundamental(
    resolution=59, space_group=225
)


class EBSDDetectorPattern:
    @classmethod
    def get_patterns(cls, master_pattern, rotations, detector: EBSDDetector):

        number_of_rotations = rotations.shape[0]

        pattern_catalogue = np.zeros(
            (detector.nrows, detector.ncols, number_of_rotations)
        )

        num_rotations = np.arange(0, number_of_rotations)
        direction_cosines = _get_direction_cosines(
            detector, number_of_rotations
        )

        master_north = master_pattern.data[0]
        master_south = master_pattern.data[1]

        # npx assuming (row, col) these should be equal
        npx = master_pattern.data.shape[3]
        # npy
        npy = master_pattern.data.shape[2]

        # Sum the energy axis
        master_north = np.sum(master_north, 0)
        master_south = np.sum(master_south, 0)

        scale_factor = (npx - 1) / 2

        ii, jj = np.meshgrid(
            np.arange(0, detector.nrows),
            np.arange(0, detector.ncols),
            indexing="ij",
        )

        # Current direction cosines output (column, row, rotation, xyz)
        rotated_dc = rotations * direction_cosines[jj, ii]

        (
            nix,
            niy,
            nixp,
            niyp,
            dx,
            dy,
            dxm,
            dym,
        ) = _get_lambert_interpolation_parameters(
            rotated_dc, scale_factor, npx, npy
        )

        pattern_catalogue[..., num_rotations] = np.where(
            rotated_dc.z >= 0,
            (
                +master_north[niy, nix] * dxm * dym
                + master_north[niyp, nix] * dx * dym
                + master_north[niy, nixp] * dxm * dy
                + master_north[niyp, nixp] * dx * dy
            ),
            (
                +master_south[niy, nix] * dxm * dym
                + master_south[niyp, nix] * dx * dym
                + master_south[niy, nixp] * dxm * dy
                + master_south[niyp, nixp] * dx * dy
            ),
        )
        pattern_catalogue = np.rot90(pattern_catalogue, 2)
        return pattern_catalogue


def _get_direction_cosines(detector: EBSDDetector, num_rot):
    xpc = detector.pc[..., 0]
    ypc = detector.pc[..., 1]
    L = detector.pc[..., 2]  # This will be wrong in the future
    # Scintillator coordinates in microns
    scin_x = (
        -((-xpc - (1.0 - detector.ncols) * 0.5) - np.arange(0, detector.ncols))
        * detector.px_size
    )
    scin_y = (
        (ypc - (1.0 - detector.nrows) * 0.5) - np.arange(0, detector.nrows)
    ) * detector.px_size

    # Auxilliary angle to rotate between reference frames
    theta_c = np.radians(detector.tilt)
    sigma = np.radians(detector.sample_tilt)

    alpha = (np.pi / 2) - sigma + theta_c
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    omega = np.radians(0)  # angle between normal of sample and detector
    cw = np.cos(omega)
    sw = np.sin(omega)

    r_g_array = np.zeros((detector.ncols, detector.nrows, 3))

    ii, jj = np.meshgrid(
        np.arange(0, detector.nrows),
        np.arange(0, detector.ncols),
        indexing="ij",
    )

    Ls = -sw * scin_x + L * cw
    Lc = cw * scin_x + L * sw

    r_g_array[jj, ii, 0] = scin_y[ii] * ca + sa * Ls[jj]
    r_g_array[jj, ii, 1] = Lc[jj]
    r_g_array[jj, ii, 2] = -sa * scin_y[ii] + ca * Ls[jj]

    r_g_array = np.repeat(r_g_array[:, :, np.newaxis, :], num_rot, axis=2)
    r_g_array = np.flipud(r_g_array)
    r_g = Vector3d(r_g_array)

    return r_g.unit


def _get_lambert_interpolation_parameters(
    rotated_direction_cosines, scale, npx, npy
):
    # Normalized direction cosines to Rosca-Lambert projection
    xy = (
        scale
        * LambertProjection.project(rotated_direction_cosines)
        / (np.sqrt(np.pi / 2))
    )

    x = xy[..., 0]
    y = xy[..., 1]
    nix = x.astype(int) + scale
    niy = y.astype(int) + scale
    nixp = nix + 1
    niyp = niy + 1
    nixp = np.where(nixp < npx - 1, nixp, nix)
    niyp = np.where(niyp < npy - 1, niyp, niy)
    nix = np.where(nix < 0, nixp, nix)
    niy = np.where(niy < 0, niyp, niy)
    dx = x - nix + scale
    dy = y - niy + scale
    dxm = 1.0 - dx
    dym = 1.0 - dy

    return (
        nix.astype(int),
        niy.astype(int),
        nixp.astype(int),
        niyp.astype(int),
        dx,
        dy,
        dxm,
        dym,
    )


patterns = EBSDDetectorPattern.get_patterns(master_pattern, r, detector)

plt.imshow(patterns[..., 0], cmap="gray")
plt.show()
