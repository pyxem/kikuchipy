# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
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

"""Read support for EBSD patterns in Oxford Instrument's binary .ebsp
format.
"""

import os
from typing import Dict, List, Tuple

import dask.array as da
import numpy as np


# Plugin characteristics
# ----------------------
format_name = "Oxford"
description = "Read support for Oxford Instrument's .ebsp file."
full_support = False
# Recognised file extension
file_extensions = ["ebsp"]
default_extension = 0
# Writing capabilities (signal dimensions, navigation dimensions)
writes = False


def file_reader(
    filename: str, navigation_shape: Tuple[int, int], lazy: bool = False,
) -> List[Dict]:
    file = OxfordBinaryFile(filename)
    scan = file.read(navigation_shape=navigation_shape, lazy=lazy)
    return [scan]


def file_writer(filename: str):
    file = OxfordBinaryFile(filename)


class OxfordBinaryFile:
    def __init__(self, filename: str):
        self.filename = filename

    def read(self, navigation_shape: Tuple[int, int], lazy: bool = False):
        nr, nc = navigation_shape
        n = nr * nc

        dtype = np.uint8
        bits = np.iinfo(dtype).bits
        between = 34

        # Open file and set pointer to start of file
        f = open(self.filename, mode="rb")

        # Get signal shape
        f.seek(0)
        pattern_starts = np.fromfile(f, dtype=int, count=n, offset=8)
        f.seek(0)
        sig_shape = np.fromfile(
            f, dtype=np.uint16, count=5, offset=pattern_starts[0]
        )
        sr, sc = sig_shape[[2, 4]].astype(int)

        # Create a memory map from data on disk
        data_size = n * sr * sc + (n - 1) * between
        offset2 = bits * (n + 1) + (2 * bits) * 1
        f.seek(0)
        # Could use numpy.fromfile() when lazy=False directly here, but
        # this reading route has a memory peak greater than the data
        # in memory. Use dask instead.
        data0 = np.memmap(
            f, shape=(data_size,), dtype=dtype, mode="r", offset=offset2
        )

        # Reshape data for easy removal of header info from the pattern
        # intensities
        data0 = da.pad(data0, pad_width=(0, between))
        data0 = data0.reshape((n, sr * sc + between))
        data0 = data0[..., :-between]
        data_shape = (nr, nc, sr, sc)
        data0 = data0.reshape(data_shape)

        if lazy:
            data = data0
        else:
            # Create array in memory and overwrite inplace
            data = np.zeros(data_shape, dtype=dtype)
            da.store(sources=data0, targets=data, compute=True)

        units = ["um"] * 4
        names = ["y", "x", "dy", "dx"]
        scales = np.ones(4)
        axes = [
            {
                "size": data.shape[i],
                "index_in_array": i,
                "name": names[i],
                "scale": scales[i],
                "offset": 0.0,
                "units": units[i],
            }
            for i in range(data.ndim)
        ]

        metadata = dict(
            General=dict(
                original_filename=self.filename,
                title=os.path.splitext(os.path.split(self.filename)[1])[0],
            ),
            Signal=dict(signal_type="EBSD", record_by="image"),
        )
        original_metadata = dict()

        scan = dict(
            axes=axes,
            data=data,
            metadata=metadata,
            original_metadata=original_metadata,
        )

        f.close()

        return scan

    def write(self):
        return
