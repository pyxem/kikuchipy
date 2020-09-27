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
from scipy import interpolate


from kikuchipy.detectors import ebsd_detector
from kikuchipy.projections import lambert_projection


class SimulateEBSDPattern:
    @classmethod
    # def my_awesome_method(cls,xpc, ypc, L, phi1, Phi, phi2):
    # def my_awesome_method(cls, xpc, ypx, L, a, b, c, d):
    def get_pattern(cls, xpc, ypc, L, *args):
        # Create Orix Rotation object with input values either:
        # A) Bunge-Euler triplets, or
        # B) Quaternions
        if len(args) == 3:
            rotation = rot.Rotation.from_euler(args)
        elif len(args) == 4:
            rotation = rot.Rotation(args)
        else:
            raise ValueError("Rotation need to be Bunge-Euler or Quaternion!")

        # Detector model
        detector = ebsd_detector.EBSDDetector()

        """
        This stuff is from the Callahan paper
        
        """

        # Detector tilt
        theta_c = detector.tilt
        # Sample tilt
        sigma = detector.sample_tilt

        alpha = np.pi / 2 - sigma + theta_c

        # NYI
        ys = 0
        xs = 0

        # Screen coordinates in sample reference frame
        ypc_ys = ypc - ys
        xpc_xs = xpc - xs
        r_g_x = ypc_ys * np.cos(alpha) + L * np.sin(alpha)
        r_g_y = xpc_xs
        r_g_z = -ypc_ys * np.sin(alpha) + L * np.cos(alpha)

        r_g = Vector3d(np.column_stack((r_g_x, r_g_y, r_g_z)))

        # r_g = np.column_stack(r_g_x, r_g_y, r_g_z)
        # Length of above
        # rho_s = np.sqrt(L**2 + ypc_ys**2 + xpc_xs**2)
        # Direction cosines of a screen pixel in the (RD, TD, ND) reference frame
        # r_g_hat = r_g / rho_s
        r_g_hat = r_g.unit

        northern = True
        # The following is VERY WIP Lines 77 - 81
        if r_g_hat.x.data < 0:  # If True we are in southern hemisphere
            northern = False
            r_g_hat = -r_g_hat

        # Rotate our vector according to our Rotation object
        rotated_vector = rotation * r_g_hat

        # Line 400 in EMdymod.f90 very relevant

        detnumsx = detector.nrows
        detnumsy = detector.ncols
        numquats = 1

        # npx: Number of pixels along square semi-edge
        # npy: Should be same as npx
        # nix: Coordinates of point
        # niy:
        # nixp: and neighboring point
        # niyp:
        # dx interpolation weight factors
        # dy
        # dxm
        # dym

        npx = detnumsx
        npy = detnumsy

        # Sum of the Northern Hemisphere Master Pattern (along a dimension?)
        mLPNHsum = np.array((1, 1, 1))

        (
            nix,
            niy,
            nixp,
            niyp,
            dx,
            dy,
            dxm,
            dym,
        ) = _get_lambert_interpolation_parameters(rotated_vector, 1, npx, npy)

        accum_e_detector = 1  # Should be an array with interpolated intensities
        ebsd_pattern = np.zeros((detnumsx, detnumsy, numquats))

        for k in range(numquats):
            for i in range(detnumsx):
                for j in range(detnumsy):
                    ebsd_pattern[i, j, k] = (
                        accum_e_detector * mLPNHsum[nix, niy, k] * dxm * dym
                        + mLPNHsum[nixp, niy, k] * dx * dym
                        + mLPNHsum[nix, niyp, k] * dxm * dy
                        + mLPNHsum[nixp, niyp, k] * dx * dy
                    )


def _get_lambert_interpolation_parameters(
    rotated_direction_cosines, scale, npx, npy
):
    # Direction cosines to Rosca-Lambert projection
    xy = scale * lambert_projection.LambertProjection.project(
        rotated_direction_cosines
    )
    x = xy[..., 0]
    y = xy[..., 1]
    nix = int(npx + x) - npx
    niy = int(npy + y) - npy
    nixp = nix + 1
    niyp = niy + 1
    if nixp > npx:
        nixp = nix
    if niyp > npy:
        niyp = niy
    if nix < -npx:
        nix = nixp
    if niy < -npy:
        niy = niyp
    dx = x - nix
    dy = y - niy
    dxm = 1.0 - dx
    dym = 1.0 - dy

    return nix, niy, nixp, niyp, dx, dy, dxm, dym
