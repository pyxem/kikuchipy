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


from kikuchipy.detectors import ebsd_detector
from kikuchipy.projections import lambert_projection
from kikuchipy import load


class SimulateEBSDPattern:
    @classmethod
    # def get_pattern(cls,xpc, ypc, L, phi1, Phi, phi2):
    # def get_pattern(cls, xpc, ypx, L, a, b, c, d):
    def get_pattern(cls, xpc, ypc, L, *args):
        print(len(args))
        if len(args) == 3:
            rotation = rot.Rotation.from_euler(np.radians(args))
        elif len(args) == 4:
            rotation = rot.Rotation(args)
        else:
            raise ValueError("Rotation need to be Bunge-Euler or Quaternion!")

        # Detector model
        detector = ebsd_detector.EBSDDetector(
            shape=(480, 640),  # row, columns - y, x
            px_size=50.0,
            pc=(xpc, ypc, L),
            # convention="emsoft",
            tilt=10,
            sample_tilt=70,
        )

        """
        This stuff is from the Callahan paper
        
        """

        # detnumsx = detector.ncols
        # detnumsy = detector.nrows
        detnumsx = detector.ncols  # 640
        detnumsy = detector.nrows  # 480

        npx = 1001
        npy = 1001

        master_pattern = load(
            r"C:\Users\laler\Desktop\Project - Kikuchipy\Ni-MP.h5",
            projection="lambert",
            hemisphere="both",
        )

        MPNH = master_pattern.data[0]
        MPSH = master_pattern.data[1]

        ebsd_pattern = np.zeros((detnumsx, detnumsy))
        energy_bins = int(
            master_pattern.axes_manager["energy"].axis[-1]
            - master_pattern.axes_manager["energy"].axis[0]
        )

        # For-loop nightmare will be changed obviously :))
        scale_factor = 500
        direction_cos = _get_direction_cosines(detector)
        for i in range(0, detector.ncols):
            for j in range(0, detector.nrows):
                rotated_dc = rotation * direction_cos[i, j]
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
                for k in range(
                    energy_bins, energy_bins + 1
                ):  # No background intensity
                    if rotated_dc.z >= 0:  # Northern
                        ebsd_pattern[i, j] += (
                            +MPNH.data[k, nix, niy] * dxm * dym
                            + MPNH.data[k, nixp, niy] * dx * dym
                            + MPNH.data[k, nix, niyp] * dxm * dy
                            + MPNH.data[k, nixp, niyp] * dx * dy
                        )
                    else:  # Southern
                        ebsd_pattern[i, j] += (
                            +MPSH.data[k, nix, niy] * dxm * dym
                            + MPSH.data[k, nixp, niy] * dx * dym
                            + MPSH.data[k, nix, niyp] * dxm * dy
                            + MPSH.data[k, nixp, niyp] * dx * dy
                        )
        return ebsd_pattern


def _get_direction_cosines(detector: ebsd_detector.EBSDDetector):
    xpc = detector.pc[..., 0]
    ypc = detector.pc[..., 1]
    L = detector.pc[..., 2]  # This will be wrong in the future
    # For some reason we are also dealing with L in microns??
    # These are in microns for some reason
    scin_x = (
        -(-xpc - (1 - detector.ncols) * 0.5 - np.arange(0, detector.ncols))
        * detector.px_size
    )
    scin_y = (
        ypc - (1 - detector.nrows) * 0.5 - np.arange(0, detector.nrows)
    ) * detector.px_size

    sigma = np.radians(detector.sample_tilt)
    theta_c = np.radians(detector.tilt)

    alpha = (np.pi / 2) - sigma + theta_c

    ca = np.cos(alpha)
    sa = np.cos(alpha)

    omega = 0  # angle between normal of sample and detector

    cw = np.cos(np.radians(omega))
    sw = np.sin(np.radians(omega))

    epl = 479

    r_g_arr = np.zeros((640, 480, 3))
    for j in range(0, 640):
        Ls = -sw * scin_x[j] + L * cw
        Lc = cw * scin_x[j] + L * sw
        for i in range(0, 480):
            r_g_arr[j, epl - i, 0] = scin_y[i] * ca + sa * Ls
            r_g_arr[j, epl - i, 1] = Lc
            r_g_arr[j, epl - i, 2] = -sa * scin_y[i] + ca * Ls

    r_g = Vector3d(r_g_arr)
    r_g_hat = r_g.unit
    return r_g_hat


def _get_lambert_interpolation_parameters(
    rotated_direction_cosines, scale, npx, npy
):
    # Normalized direction cosines to Rosca-Lambert projection
    xy = (
        scale
        * lambert_projection.LambertProjection.project(
            rotated_direction_cosines
        )
        / (np.sqrt(np.pi / 2))
    )

    x = xy[..., 0]
    y = xy[..., 1]
    nix = int(npx + x) - npx + scale
    niy = int(npy + y) - npy + scale
    nixp = nix + 1
    niyp = niy + 1
    if nixp > 1000:
        nixp = nix
    if niyp > 1000:
        niyp = niy
    if nix < 0:
        nix = nixp
    if niy < 0:
        niy = niyp
    dx = x - nix
    dy = y - niy
    dxm = 1.0 - dx
    dym = 1.0 - dy
    return nix, niy, nixp, niyp, dx, dy, dxm, dym


a = SimulateEBSDPattern.get_pattern(0.00001, 0.00001, 15000, 0, 0, 0)
from matplotlib.colors import NoNorm

norm = plt.Normalize(a.min(), a.max())
plt.imshow(a, norm=norm, cmap="gray", interpolation="none", filternorm=False)
plt.show()
