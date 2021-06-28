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

"""Read support for EBSD patterns in Oxford Instruments' binary .ebsp
format.

The reader assumes that the file is uncompressed and that patterns are
stored as 8-bit unsigned integers.
"""

import os
from typing import Dict, List, Tuple

import dask.array as da
import numpy as np


# Plugin characteristics
# ----------------------
format_name = "Oxford binary"
description = "Read support for Oxford Instruments' binary .ebsp file."
full_support = False
# Recognised file extension
file_extensions = ["ebsp"]
default_extension = 0
# Writing capabilities (signal dimensions, navigation dimensions)
writes = False


def file_reader(
    filename: str,
    navigation_shape: Tuple[int, int] = None,
    lazy: bool = False,
) -> List[Dict]:
    """Return a list with one dictionary containing the EBSD
    patterns from the file in a 'data' key along with an
    'axes' key and 'metadata' and 'original_metadata' keys.

    Parameters
    ----------
    navigation_shape
        Number of map rows and columns, in that order. If not given,
        the number of patterns in the file is guessed from the file
        header, and the navigation shape of the returned pattern
        array will be one-dimensional.
    lazy
        Whether to load the patterns lazily. Default is False.

    Returns
    -------
    list of dict
        Data, axes, metadata, and original metadata in a dictionary
        within a list. Data is returned as :class:`numpy.ndarray` if
        `lazy` is False, otherwise as :class:`dask.array.Array`.
    """
    file = OxfordBinaryFile(filename)
    scan = file.read(navigation_shape=navigation_shape, lazy=lazy)
    return [scan]


class OxfordBinaryFile:
    """Binary file with EBSD patterns stored in Oxford Instruments'
    .ebsp format.

    It is assumed that the file is uncompressed and that patterns are
    stored as 8-bit unsigned integers.
    """

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
        """Open file in readable mode."""
        self.file = open(self.filename, mode="rb")

    def close(self):
        """Close file."""
        self.file.close()

    def get_signal_shape(self, offset: int) -> Tuple[int, int]:
        """Return signal shape as (number of map rows, number of map
        columns).

        Parameters
        ----------
        offset
            Byte offset to the file position with the information.
        """
        file = self.file
        file.seek(0)
        sig_shape = np.fromfile(file, dtype=np.uint16, count=5, offset=offset)
        sr, sc = sig_shape[[2, 4]].astype(np.int64)
        return sr, sc

    def guess_number_of_patterns(self, assumed_n_pixels: int = 1600) -> int:
        """Guess the number of patterns in the file based upon an
        assumed lower bound for the number of pattern pixels and the
        file size.

        Parameters
        ----------
        assumed_n_pixels
            Assumed lower bound for the number of pattern pixels.
            Default is 1600 pixels.

        Returns
        -------
        n_patterns
            Guess of the number of EBSD patterns in the file.
        """
        file = self.file
        file.seek(0)
        file_byte_size = os.path.getsize(file.name)
        metadata_size = 34

        max_assumed_n_patterns = file_byte_size // (assumed_n_pixels + metadata_size)
        assumed_pattern_starts = np.fromfile(
            file, dtype=np.int64, count=max_assumed_n_patterns, offset=8
        )
        diff_pattern_starts_bytes = np.diff(assumed_pattern_starts)

        # Determine outliers by a distance to the mean greater than a
        # number of standard deviations
        mean = np.mean(diff_pattern_starts_bytes)
        std = np.std(diff_pattern_starts_bytes)
        distance = abs(diff_pattern_starts_bytes - mean)
        max_std = 2
        not_outlier = distance < max_std * std
        outliers_start_idx = np.argmax(not_outlier < 1) - 1
        not_outlier = np.ones(max_assumed_n_patterns, dtype=bool)
        not_outlier[outliers_start_idx:] = False
        n_patterns = np.sum(not_outlier)

        return n_patterns

    def read(
        self, navigation_shape: Tuple[int, int] = None, lazy: bool = False
    ) -> Dict:
        """Return a dictionary containing the EBSD patterns from the
        file in a 'data' key along with an 'axes' key and 'metadata'
        and 'original_metadata' keys.

        Parameters
        ----------
        navigation_shape
            Number of map rows and columns, in that order. If not given,
            the number of patterns in the file is guessed from the file
            header, and the navigation shape of the returned pattern
            array will be one-dimensional.
        lazy
            Whether to load the patterns lazily. Default is False.

        Returns
        -------
        scan
            Data, axes, metadata, and original metadata. Data is
            returned as :class:`numpy.ndarray` if `lazy` is False,
            otherwise as :class:`dask.array.Array`.
        """
        self.open()
        file = self.file

        pattern_starts = None
        if navigation_shape is None:
            n_patterns = self.guess_number_of_patterns()
            data_shape = (n_patterns,)
        else:
            n_patterns = np.prod(navigation_shape)
            data_shape = navigation_shape

        # Get byte positions for the start of each pattern
        file.seek(0)
        pattern_starts = np.fromfile(file, dtype=np.int64, count=n_patterns, offset=8)
        first_pattern_position = 8 + n_patterns * 8

        sr, sc = self.get_signal_shape(offset=first_pattern_position)
        data_shape += (sr, sc)

        # Create a memory map from data on disk
        dtype = np.uint8
        header_size = 16
        footer_size = 18
        metadata_size = header_size + footer_size
        data_size = n_patterns * (sr * sc + metadata_size)

        # Raise explanatory error if byte position of first pattern
        # seems obviously wrong
        file_size = os.path.getsize(file.name)
        if data_size > file_size:
            raise ValueError(
                f"Assumed number of {n_patterns} patterns with {sr * sc} pixels leads "
                f"to a data size {data_size} greater than the file size {file_size}"
            )

        # Could use numpy.fromfile() when lazy=False directly here, but
        # this reading route has a memory peak greater than the data
        # in memory. Use dask instead.
        file.seek(0)
        data0 = np.memmap(
            file,
            shape=(data_size,),
            dtype=dtype,
            mode="r",
            offset=first_pattern_position,
        )
        data0 = da.from_array(data0)

        # Reshape data for easy removal of header and footer info from
        # the pattern intensities
        data0 = data0.reshape((n_patterns, -1))

        # Sort if necessary
        if not np.allclose(np.diff(pattern_starts), 1):
            pattern_positions = (
                (pattern_starts - first_pattern_position) / (data_size / n_patterns)
            ).astype(np.int64)
            data0 = data0[pattern_positions]

        data0 = data0[:, header_size : header_size + sr * sc]
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
