# Copyright 2019-2024 The kikuchipy developers
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

"""Creation of dummy Oxford Instrument's H5OINA file for testing and IO
documentation.
"""

from pathlib import Path

import h5py
import numpy as np
from orix.quaternion import Rotation

from kikuchipy.data import nickel_ebsd_small


def create_dummy_oxford_h5ebsd_file(path: Path) -> None:
    """Create a dummy H5OINA file with the given path.

    Both processed and unprocessed patterns (processed + 1) are written
    to file.
    """
    # Quaternions determined from indexing
    # fmt: off
    qu_grain1 = (0.9542, -0.0183, -0.2806,  0.1018)
    qu_grain2 = (0.9542,  0.0608, -0.2295, -0.1818)
    rot = Rotation(
        [
            qu_grain1, qu_grain2, qu_grain2,
            qu_grain1, qu_grain2, qu_grain2,
            qu_grain1, qu_grain2, qu_grain2,
        ]
    )
    # fmt: on
    euler = rot.to_euler()

    s = nickel_ebsd_small()
    ny, nx = s._navigation_shape_rc
    n = ny * nx
    sy, sx = s._signal_shape_rc
    dx = s.axes_manager["x"].scale

    f = h5py.File(path, mode="w")

    # Top group
    f.create_dataset("Format Version", data=b"5.0")
    f.create_dataset("Index", data=b"1")
    f.create_dataset("Manufacturer", data=b"Oxford Instruments")
    f.create_dataset("Software Version", data=b"6.0.8014.1")
    scan = f.create_group("1")

    # EBSD
    ebsd = scan.create_group("EBSD")
    ones = np.ones(n)

    # Data
    data = ebsd.create_group("Data")
    data.create_dataset("Band Contrast", dtype="uint8", data=ones)
    data.create_dataset("Band Slope", dtype="uint8", data=ones)
    data.create_dataset("Bands", dtype="uint8", data=ones)
    data.create_dataset("Beam Position X", dtype="float32", data=ones)
    data["Beam Position X"].attrs["Unit"] = "um"
    data.create_dataset("Beam Position Y", dtype="float32", data=ones)
    data["Beam Position Y"].attrs["Unit"] = "um"
    data.create_dataset("Detector Distance", dtype="float32", data=ones)
    data.create_dataset("Error", dtype="uint8", data=ones)
    data["Error"].attrs["HighMAD"] = [np.int32(5)]
    data["Error"].attrs["LowBandContrast"] = [np.int32(3)]
    data["Error"].attrs["LowBandSlope"] = [np.int32(4)]
    data["Error"].attrs["NoSolution"] = [np.int32(2)]
    data["Error"].attrs["NotAnalyzed"] = [np.int32(0)]
    data["Error"].attrs["Replaced"] = [np.int32(7)]
    data["Error"].attrs["Success"] = [np.int32(1)]
    data["Error"].attrs["UnexpectedError"] = [np.int32(6)]
    data.create_dataset("Euler", dtype="float32", data=euler)
    data["Euler"].attrs["Unit"] = "rad"
    data.create_dataset("Mean Angular Deviation", dtype="float32", data=ones)
    data["Euler"].attrs["Mean Angular Deviation"] = "rad"
    data.create_dataset("Pattern Center X", dtype="float32", data=ones)
    data.create_dataset("Pattern Center Y", dtype="float32", data=ones)
    data.create_dataset("Pattern Quality", dtype="float32", data=ones)
    data.create_dataset("Phase", dtype="uint8", data=ones)
    s.remove_static_background()
    data.create_dataset(
        "Processed Patterns", dtype="uint8", data=s.data.reshape((n, sy, sx))
    )
    data.create_dataset(
        "Unprocessed Patterns", dtype="uint8", data=s.data.reshape((n, sy, sx)) + 1
    )
    x = np.array([0, 1, 2] * 3) * dx
    data.create_dataset("X", dtype="float32", data=x)
    data["X"].attrs["Unit"] = "um"
    data.create_dataset("Y", dtype="float32", data=np.sort(x))
    data["Y"].attrs["Unit"] = "um"

    # Header
    header = ebsd.create_group("Header")
    header.create_dataset("Beam Voltage", dtype="float32", data=20)
    header["Beam Voltage"].attrs["Unit"] = "kV"
    header.create_dataset("Magnification", dtype="float32", data=2000)
    header.create_dataset("Working Distance", dtype="float32", data=23.5)
    header["Working Distance"].attrs["Unit"] = "mm"
    header.create_dataset("X Cells", dtype="int32", data=3)
    header["X Cells"].attrs["Unit"] = "px"
    header.create_dataset("Y Cells", dtype="int32", data=3)
    header["Y Cells"].attrs["Unit"] = "px"
    header.create_dataset("X Step", dtype="float32", data=1.5)
    header["X Step"].attrs["Unit"] = "um"
    header.create_dataset("Y Step", dtype="float32", data=1.5)
    header["Y Step"].attrs["Unit"] = "um"
    header.create_dataset("Pattern Height", dtype="int32", data=60)
    header["Pattern Height"].attrs["Unit"] = "px"
    header.create_dataset("Pattern Width", dtype="int32", data=60)
    header["Pattern Width"].attrs["Unit"] = "px"
    header.create_dataset(
        "Processed Static Background", dtype="uint8", data=s.static_background
    )
    header.create_dataset("Tilt Angle", dtype="float32", data=np.deg2rad(69.9))
    header.create_dataset(
        "Detector Orientation Euler", dtype="float32", data=np.deg2rad([0, 91.5, 0])
    )
    header.create_dataset("Camera Binning Mode", data=b"8x8 (60x60 px)")

    f.close()
