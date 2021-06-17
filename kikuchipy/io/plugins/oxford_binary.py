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
    filename: str,
    navigation_shape: Tuple[int, int],
    lazy: bool = False,
) -> List[Dict]:
    file = OxfordBinaryFile(filename)
    scan = file.read(navigation_shape=navigation_shape, lazy=lazy)
    return [scan]


def file_writer(filename: str):
    file = OxfordBinaryFile(filename)


class OxfordBinaryFile:
    def __init__(self, filename: str):
        """Set up an Oxford binary .ebsp file for reading or writing.

        Parameters
        ----------
        filename
            Full path to file on disk.
        """
        self.filename = filename
        self.file = None

    def open(self):
        """Open the file in readable mode."""
        self.file = open(self.filename, mode="rb")

    def close(self):
        """Close file."""
        self.file.close()

    def get_signal_shape(self, offset: int) -> Tuple[int, int]:
        """Return signal shape (n rows, n columns).

        Parameters
        ----------
        offset
            Byte offset to the file position with the information.
        """
        file = self.file
        file.seek(0)
        sig_shape = np.fromfile(file, dtype=np.uint16, count=5, offset=offset)
        sr, sc = sig_shape[[2, 4]].astype(int)
        return sr, sc

    def read(self, navigation_shape: Tuple[int, int], lazy: bool = False):
        nr, nc = navigation_shape
        n = nr * nc

        self.open()
        file = self.file

        # Get byte positions for the start of each pattern
        file.seek(0)
        pattern_starts = np.fromfile(file, dtype=int, count=n, offset=8)
        pattern_positions = np.argsort(pattern_starts)
        first_pattern_position = pattern_starts[0]

        sr, sc = self.get_signal_shape(offset=first_pattern_position)

        dtype = np.uint8
        bits = np.iinfo(dtype).bits
        n_bytes_header = 34

        # Create a memory map from data on disk
        data_size = n * sr * sc + (n - 1) * n_bytes_header
        offset2 = bits * (n + 1) + (2 * bits) * 1
        file.seek(0)
        # Could use numpy.fromfile() when lazy=False directly here, but
        # this reading route has a memory peak greater than the data
        # in memory. Use dask instead.
        data0 = np.memmap(
            file, shape=(data_size,), dtype=dtype, mode="r", offset=offset2
        )

        # Reshape data for easy removal of header info from the pattern
        # intensities
        data0 = da.pad(data0, pad_width=(0, n_bytes_header))
        data0 = data0.reshape((n, sr * sc + n_bytes_header))

        # Sort if necessary
        if not np.allclose(np.diff(pattern_positions), 1):
            data0 = data0[pattern_positions]

        data0 = data0[..., :-n_bytes_header]
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

        scan = dict(
            axes=axes,
            data=data,
            metadata=metadata,
            original_metadata=dict(),
        )

        self.close()

        return scan

    def write(self):
        return


def _guess_number_of_patterns(
    file, offset: int, assumed_n_pixels: int = 3600
) -> Tuple[int, np.ndarray]:
    """Guess the number of patterns in the file based upon an assumed
    lower bound for the number of pattern pixels and the file size.

    Parameters
    ----------
    file : io.FileIO
        Oxford Instrument's binary file (.ebsp) with EBSD patterns.
    offset : int
        File byte offset for the pattern starts, either 0 or 8 depending
        on the .ebsp file format version.
    assumed_n_pixels : int
        Assumed lower bound for the number of pattern pixels.

    Returns
    -------
    n_patterns
        Number of EBSD patterns in the file.
    pattern_starts
        Byte positions of the pattern starts in the file.
    """
    file.seek(0)
    file_byte_size = os.path.getsize(file.name)
    header_size = 34

    max_assumed_n_patterns = file_byte_size // (assumed_n_pixels + header_size)
    assumed_pattern_starts = np.fromfile(
        file, dtype="q", count=max_assumed_n_patterns, offset=offset
    )

    diff_pattern_starts_bytes = np.diff(assumed_pattern_starts)
    n_pixels = diff_pattern_starts_bytes[0] - header_size
    is_actual_pattern_starts = diff_pattern_starts_bytes == (
        n_pixels + header_size
    )
    n_patterns = np.sum(is_actual_pattern_starts) + 1

    pattern_starts = assumed_pattern_starts[:n_patterns]

    return n_patterns, pattern_starts
