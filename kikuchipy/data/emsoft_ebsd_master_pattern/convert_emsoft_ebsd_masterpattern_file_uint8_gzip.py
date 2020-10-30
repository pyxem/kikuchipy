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

"""Procedure to reduce the file size of a EMsoft EBSD master pattern
file, returned from the EMMCOpenCL and EMEBSDmaster programs, by
reducing the data type from float32 to uint8 and using gzip compression.
"""

import os

from h5py import File
import numpy as np

import kikuchipy as kp


# Read data
datadir = "/home/hakon/kode/emsoft/emdata/crystal_data"
phase = "ni"
fname = f"{phase}_mc_mp_20kv2.h5"
hemisphere = "both"
mp_lp = kp.load(
    filename=os.path.join(datadir, phase, fname),
    hemisphere=hemisphere,
    projection="lambert",
)
mp_sp = kp.load(
    filename=os.path.join(datadir, phase, fname),
    hemisphere=hemisphere,
    projection="spherical",
)

# Rescale to uint8
dtype_out = np.uint8
mp_lp.rescale_intensity(dtype_out=dtype_out)
mp_sp.rescale_intensity(dtype_out=dtype_out)

# Get HDF5 groups and datasets in original EMsoft file
f = File(os.path.join(datadir, phase, fname), mode="r")
orig_data = kp.io.plugins.h5ebsd.hdf5group2dict(group=f["/"], recursive=True)

# Overwrite master patterns in original data
lp_shape = (1, 1) + mp_lp.axes_manager.signal_shape[::-1]
orig_data["EMData"]["EBSDmaster"]["mLPNH"] = mp_lp.inav[0].data.reshape(
    lp_shape
)
orig_data["EMData"]["EBSDmaster"]["mLPSH"] = mp_lp.inav[1].data.reshape(
    lp_shape
)
sp_shape = (1,) + mp_sp.axes_manager.signal_shape[::-1]
orig_data["EMData"]["EBSDmaster"]["masterSPNH"] = mp_sp.inav[0].data.reshape(
    sp_shape
)
orig_data["EMData"]["EBSDmaster"]["masterSPSH"] = mp_sp.inav[1].data.reshape(
    sp_shape
)

# Create new HDF5 file (included in the kikuchipy.data module)
f2 = File(
    os.path.join(datadir, phase, "ni_mc_mp_20kv_uint8_gzip_opts9.h5"), mode="w"
)
kp.io.plugins.h5ebsd.dict2h5ebsdgroup(
    dictionary=orig_data, group=f2, compression="gzip", compression_opts=9
)

# Finally, close both files
f2.close()
f.close()
