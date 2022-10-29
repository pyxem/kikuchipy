# Copyright 2019-2022 The kikuchipy developers
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

"""Creation of a dummy Oxford Instruments h5ebsd file (H5OINA) used when
testing the Oxford h5ebsd reader.

Only the groups and datasets read in the reader are included.
"""

import os

from h5py import File
import numpy as np
from orix.quaternion import Rotation

import kikuchipy as kp


# Orientations determined from indexing
grain1 = (0.9542, -0.0183, -0.2806, 0.1018)
grain2 = (0.9542, 0.0608, -0.2295, -0.1818)
rot = Rotation([grain1, grain2, grain2, grain1, grain2, grain2, grain1, grain2, grain2])
euler = rot.to_euler()

s = kp.data.nickel_ebsd_small()
ny, nx = s.axes_manager.navigation_shape[::-1]
n = ny * nx
sy, sx = s.axes_manager.signal_shape[::-1]
dx = s.axes_manager["x"].scale

dir_data = os.path.abspath(os.path.dirname(__file__))


f = File(os.path.join(dir_data, "patterns.h5oina"), mode="w")

# Top group
f.create_dataset("Format Version", data=b"5.0")
f.create_dataset("Index", data=b"1")
f.create_dataset("Manufacturer", data=b"Oxford Instruments")
f.create_dataset("Software Version", data=b"6.0.8014.1")
scan = f.create_group("1")

# EBSD
ebsd = scan.create_group("EBSD")
ones = np.ones(n)
zeros = np.zeros(n)

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

f.close()
