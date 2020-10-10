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
import matplotlib.pyplot as plt


from kikuchipy.detectors.ebsd_detector import EBSDDetector
from kikuchipy.projections.lambert_projection import LambertProjection
from kikuchipy import load

master_pattern = load(
    r"C:\Users\laler\Desktop\Project - Kikuchipy\Ni-MP.h5",
    projection="lambert",
    hemisphere="both",
)
xpc = 0.00001
ypc = 0.00001
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


class EBSDDetectorPattern:
    @classmethod
    def get_pattern_eu(
        cls,
        master_pattern,
        detector: EBSDDetector,
        phi1,
        Phi,
        phi2,
        direction_cosines,
    ):
        rotation = rot.Rotation.from_euler(np.radians((phi1, Phi, phi2)))
        ebsd_detector_pattern = np.zeros((detector.nrows, detector.ncols))

        master_north = master_pattern.data[0]
        master_south = master_pattern.data[1]

        energy_bins = int(
            master_pattern.axes_manager["energy"].axis[-1]
            - master_pattern.axes_manager["energy"].axis[0]
        )

        # npx
        npx = master_pattern.data.shape[2]
        # npy
        npy = master_pattern.data.shape[3]

        # 500
        scale_factor = (npx - 1) / 2

        ii, jj = np.meshgrid(
            np.arange(0, detector.nrows),
            np.arange(0, detector.ncols),
            indexing="ij",
        )

        # Current direction cosines output (row, column, xyz)
        rotated_dc = rotation * direction_cosines[ii, jj]

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
        # it should look at all the energy thingies probably
        print(nix[0, 0])
        print(niy[0, 0])
        print(ii)
        ebsd_detector_pattern[ii, jj] = np.where(
            rotated_dc.z >= 0,
            (
                +master_north.data[
                    15,
                    nix[ii.astype(int), jj.astype(int)],
                    niy[ii.astype(int), jj.astype(int)],
                ]
                * dxm
                * dym
                + master_north.data[15, nixp, niy] * dx * dym
                + master_north.data[15, nix, niyp] * dxm * dy
                + master_north.data[15, nixp, niyp] * dx * dy
            ),
            (
                +master_south.data[15, nix, niy] * dxm * dym
                + master_south.data[15, nixp, niy] * dx * dym
                + master_south.data[15, nix, niyp] * dxm * dy
                + master_south.data[15, nixp, niyp] * dx * dy
            ),
        )
        return ebsd_detector_pattern


# This should probably be its own method in the detector module
# detector.direction_cosines or something
def _get_direction_cosines(detector: EBSDDetector):
    xpc = detector.pc[..., 0]
    ypc = detector.pc[..., 1]
    L = detector.pc[..., 2]  # This will be wrong in the future

    # Scintillator coordinates in microns
    scin_x = (
        -(-xpc - (1.0 - detector.ncols) * 0.5 - np.arange(0, detector.ncols))
        * detector.px_size
    )
    scin_y = (
        ypc - (1.0 - detector.nrows) * 0.5 - np.arange(0, detector.nrows)
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

    r_g_array = np.zeros((detector.nrows, detector.ncols, 3))

    ii, jj = np.meshgrid(
        np.arange(0, detector.nrows),
        np.arange(0, detector.ncols),
        indexing="ij",
    )

    Ls = -sw * scin_x + L * cw
    Lc = cw * scin_x + L * sw

    # NOTE EMSOFT HAS CHANGED THE INDICES THIS IS NOT DONE HERE YET
    # SEE WHAT YOU NEED TO DO OR IF YOU NEED TO DO SOMETHING AT ALL :)
    r_g_array[ii, jj, 0] = scin_y[ii] * ca + sa * Ls[jj]
    r_g_array[ii, jj, 1] = Lc[jj]
    r_g_array[ii, jj, 2] = -sa * scin_y[ii] + ca * Ls[jj]

    r_g = Vector3d(r_g_array)
    # Current output shape (row, column, xyz) vs EMsoft (column, row, xyz) I think
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
    dx = x - nix
    dy = y - niy
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


a = EBSDDetectorPattern.get_pattern_eu(
    master_pattern, detector, 0, 0, 0, _get_direction_cosines(detector)
)

plt.imshow(a)
plt.show()
