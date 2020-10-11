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

euler_angles = r"C:\Users\laler\EMsoftData\testfile.txt"

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
    def get_patterns(cls, master_pattern, anglefile, detector: EBSDDetector):

        # I am not sure this is the fastest way to do things, limited experience with io
        with open(anglefile, "r") as f:
            angle_type = f.readline()
            number_of_rotations = int(f.readline())

            if angle_type.strip() == "eu":
                angle_array = np.zeros((number_of_rotations, 3))
                i = -1
                for line in f:
                    i += 1
                    li = line.split()
                    angle_array[i, 0] = float(li[0])
                    angle_array[i, 1] = float(li[1])
                    angle_array[i, 2] = float(li[2])
            # TODO: Implement quaternion
            else:
                raise ValueError

        pattern_catalogue = np.zeros(
            (detector.nrows, detector.ncols, number_of_rotations)
        )

        num_rotations = np.arange(0, number_of_rotations)
        direction_cosines = _get_direction_cosines(
            detector, number_of_rotations
        )

        #       THis part can be optimized, but let it as is for now to get a MVP
        for i in range(number_of_rotations):
            pattern_catalogue[..., i] = EBSDDetectorPattern.get_pattern(
                master_pattern,
                detector,
                direction_cosines,
                angle_array[i, 0],
                angle_array[i, 0],
                angle_array[i, 0],
            )

        # pattern_catalogue[..., num_rotations] = EBSDDetectorPattern.get_pattern(master_pattern, detector, direction_cosines, angle_array[num_rotations, 0], angle_array[num_rotations, 1], angle_array[num_rotations, 2])
        # Rotate 180 degrees for now and flip
        # pattern_catalogue = np.rot90(pattern_catalogue, 2)

        return pattern_catalogue

    @classmethod
    def get_pattern(
        cls, master_pattern, detector: EBSDDetector, direction_cosines, *args
    ):
        # This can be determined from anglefile, but not sure if it is faster?
        if len(args) == 3:
            euler = np.column_stack(
                np.radians(args)
            )  # Not needed in current implementation but should be down the road
            rotation = rot.Rotation.from_euler(euler)
        elif len(args) == 4:
            rotation = rot.Rotation(args)
        else:  # This can probably be removed if get_pattern becomes a private method
            raise ValueError(
                "Rotation angles need to be in Bunge-Euler or Quaternion!"
            )
        ebsd_detector_pattern = np.zeros((detector.nrows, detector.ncols))

        master_north = master_pattern.data[0]
        master_south = master_pattern.data[1]

        # NYI
        energy_bins = int(
            master_pattern.axes_manager["energy"].axis[-1]
            - master_pattern.axes_manager["energy"].axis[0]
        )

        # npx assuming (row, col) these should be equal
        npx = master_pattern.data.shape[3]
        # npy
        npy = master_pattern.data.shape[2]

        # 500
        scale_factor = (npx - 1) / 2

        ii, jj = np.meshgrid(
            np.arange(0, detector.nrows),
            np.arange(0, detector.ncols),
            indexing="ij",
        )

        # Current direction cosines output (column, row, rotation, xyz)
        rotated_dc = rotation * direction_cosines[jj, ii]

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
        # This could probably be a sum of MPNH between emin emax
        # and MPSH between emin emax
        ebsd_detector_pattern[ii, jj] = np.where(
            rotated_dc.z >= 0,
            (
                +master_north[
                    15,
                    nix[ii, jj],
                    niy[ii, jj],
                ]
                * dxm
                * dym
                + master_north[15, nixp, niy] * dx * dym
                + master_north[15, nix, niyp] * dxm * dy
                + master_north[15, nixp, niyp] * dx * dy
            ),
            (
                +master_south[15, nix, niy] * dxm * dym
                + master_south[15, nixp, niy] * dx * dym
                + master_south[15, nix, niyp] * dxm * dy
                + master_south[15, nixp, niyp] * dx * dy
            ),
        )
        return ebsd_detector_pattern


# This should probably be its own method in the detector module
# detector.direction_cosines or something
def _get_direction_cosines(detector: EBSDDetector, num_rot):
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

    # r_g_array = np.repeat(r_g_array[:, :,np.newaxis, :], num_rot, axis=2)

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


# TODO: I believe the patterns need to be rotated 180 degrees and then inverted
# a = EBSDDetectorPattern.get_pattern(
#      master_pattern, detector, _get_direction_cosines(detector, 1), 120, 45, 60)
patterns = EBSDDetectorPattern.get_patterns(
    master_pattern, euler_angles, detector
)
#
# # These are currently upside down compared (and inverted?) to EMsoft!
# # first rotation
plt.imshow(patterns[..., 1], cmap="gray")
plt.axis("off")
# # 2nd rotation
# plt.imshow(patterns[..., 1], cmap="gray") # Looks a bit more rotated than the key
# #...
# # nth rotation
# #plt.imshow(patterns[..., n-1], cmap="gray")
#


# plt.imshow(a, cmap="gray")

plt.show()
