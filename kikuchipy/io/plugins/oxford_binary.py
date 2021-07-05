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

"""Read support for uncompressed EBSD patterns in Oxford Instruments'
binary .ebsp format.
"""

import os
import struct
from typing import Dict, List, Optional, Tuple

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


def file_reader(filename, navigation_shape=None, lazy=False):
    pass


class OxfordBinaryFile:
    pattern_header_size = 16
    pattern_footer_size = 18
    # fmt: off
    pattern_header_dtype = [
        ("is_compressed",   np.int32, (1,)),
        ("nrows",           np.int32, (1,)),
        ("ncols",           np.int32, (1,)),
        ("n_bytes",         np.int32, (1,)),
    ]
    # fmt: on

    def __init__(self, filename: str, navigation_shape: Optional[Tuple[int]] = None):
        self.file = open(filename, mode="rb")

        self.version = self.get_version()

        if navigation_shape is None:
            self.n_patterns = self.guess_number_of_patterns()
        else:
            self.n_patterns = np.prod(navigation_shape)
        self.pattern_starts = self.get_pattern_starts()

        first_pattern_position = self.first_pattern_position

        is_compressed, nrows, ncols, n_bytes = self.get_single_pattern_header(
            first_pattern_position
        )
        self.is_compressed = is_compressed
        self.signal_shape = (nrows, ncols)
        self.n_bytes = n_bytes
        if n_bytes == self.n_pixels:
            self.dtype = np.uint8
        else:
            self.dtype = np.uint16

        self.pattern_footer_dtype = self.get_pattern_footer_dtype(
            first_pattern_position
        )

        self.mmap = self.get_memmap()

    @property
    def all_indexed(self) -> np.ndarray:
        return self.is_indexed.all()

    @property
    def pattern_dtype(self) -> np.dtype:
        if self.n_bytes == self.n_pixels:
            return np.uint8
        else:
            return np.uint16

    @property
    def first_pattern_position(self) -> int:
        return self.pattern_order_byte_position + self.n_patterns * 8

    @property
    def is_indexed(self):
        return self.pattern_starts != 0

    @property
    def metadata_size(self):
        return self.pattern_header_size + self.pattern_footer_size

    @property
    def n_pixels(self):
        return np.prod(self.signal_shape)

    @property
    def pattern_order(self) -> np.ndarray:
        bytes_per_pattern = self.n_bytes + self.metadata_size
        pp = (self.pattern_starts - self.first_pattern_position) / bytes_per_pattern
        return pp.astype(int)

    @property
    def pattern_order_byte_position(self) -> int:
        if self.version != 0:
            return 8
        else:
            return 0

    def get_memmap(self):
        file_dtype = self.pattern_header_dtype
        file_dtype += [("pattern", self.pattern_dtype, self.signal_shape)]
        footer_dtype = self.pattern_footer_dtype
        if len(footer_dtype) != 0:
            file_dtype += footer_dtype

        return np.memmap(
            self.file,
            dtype=file_dtype,
            shape=self.n_patterns,
            mode="r",
            offset=self.first_pattern_position,
        )

    def get_pattern_starts(self) -> np.ndarray:
        self.file.seek(self.pattern_order_byte_position)
        return np.fromfile(self.file, dtype=np.int64, count=self.n_patterns)

    def get_single_pattern_header(self, offset: int) -> Tuple[bool, int, int, int]:
        self.file.seek(offset)
        header = np.fromfile(self.file, dtype=self.pattern_header_dtype, count=1)
        return (
            bool(header["is_compressed"][0]),
            int(header["nrows"]),
            int(header["ncols"]),
            int(header["n_bytes"]),
        )

    def get_pattern_footer_dtype(self, offset: int) -> List[tuple]:
        self.file.seek(offset + self.pattern_header_size + self.n_bytes)
        footer_dtype = ()
        map_col_dtype = ("map_col", np.float64, (1,))
        map_row_dtype = ("map_row", np.float64, (1,))
        if self.version == 1:
            footer_dtype += (map_col_dtype, map_row_dtype)
            self.pattern_footer_size = 16
        elif self.version > 1:
            pattern_footer_size = 2
            if struct.unpack("?", self.file.read(1))[0]:  # bool
                footer_dtype += (("has_map_col", bool, (1,)), map_col_dtype)
                pattern_footer_size += 8
            if struct.unpack("?", self.file.read(1))[0]:  # bool
                footer_dtype += (("has_map_row", bool, (1,)), map_row_dtype)
                pattern_footer_size += 8
            self.pattern_footer_size = pattern_footer_size
        return list(footer_dtype)

    def get_version(self) -> int:
        self.file.seek(0)
        version = struct.unpack("q", self.file.read(8))[0]  # int64
        if version < 0:
            return -version
        else:
            return 0

    def get_scan(
        self, navigation_shape: Optional[Tuple[int]] = None, lazy: bool = False
    ) -> list:

        is_non_indexed = self.pattern_starts == 0
        n_non_indexed = np.sum(is_non_indexed)
        # TODO: Fix this
        if n_non_indexed > 0:
            raise ValueError(
                "Cannot read EBSD patterns from a file with only non-indexed patterns"
            )

        header = self.get_single_pattern_header(offset=self.first_pattern_position)
        self.signal_shape = (header["sr"], header["sc"])
        self.dtype = header["dtype"]
        # TODO: Fix this
        if header["is_compressed"]:
            raise ValueError("Cannot read EBSD patterns from a compressed file")

        data = self.get_patterns(lazy=lazy)

        self.file.close()

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

        return scan

    def get_patterns(self, lazy: bool) -> np.ndarray:
        first_pattern_position = self.first_pattern_position
        self.file.seek(first_pattern_position)

        bytes_per_pattern = self.n_pixels * self.n_bytes + self.metadata_size
        data_shape = self.navigation_shape + (bytes_per_pattern,)
        patterns = da.from_array(
            np.memmap(self.file, shape=data_shape, dtype=self.dtype, mode="r")
        )

        # Sort if necessary
        pattern_starts = self.pattern_starts
        if not np.allclose(np.diff(pattern_starts), 1):
            pattern_order = (
                (pattern_starts - first_pattern_position) / bytes_per_pattern
            ).astype(np.int64)
            patterns = patterns[pattern_order]

        # Remove header and footer
        pattern_header_size = self.pattern_header_size
        patterns = patterns[..., pattern_header_size : pattern_header_size + n_pixels]

        if lazy:
            data = patterns
        else:
            # Create array in memory and overwrite inplace
            data = np.zeros(data_shape, dtype=self.dtype)
            da.store(sources=patterns, targets=data, compute=True)

        return data

    def guess_number_of_patterns(self, min_assumed_n_pixels: int = 1600) -> int:
        """Guess the number of patterns in the file based upon an
        assumed lower bound for the number of pattern pixels and the
        file size.

        Parameters
        ----------
        min_assumed_n_pixels
            Assumed lower bound for the number of pattern pixels.
            Default is 1600 pixels.

        Returns
        -------
        n_patterns
            Guess of the number of EBSD patterns in the file.
        """
        self.file.seek(self.pattern_order_byte_position)
        file_byte_size = os.path.getsize(self.file.name)

        # Maximum bytes to read based on minimum assumed pixel size
        max_assumed_n_patterns = file_byte_size // (
            min_assumed_n_pixels + self.pattern_header_size
        )

        # Read assumed pattern starts
        assumed_pattern_starts = np.fromfile(
            self.file, dtype=np.int64, count=max_assumed_n_patterns
        )

        # It is assumed that a jump in bytes from one pattern position
        # to the next does not exceed a number of maximum bytes one
        # pattern can take up in the file. The array index where
        # this happens (plus 2) is assumed to be the number of patterns
        # in the file.
        diff_pattern_starts = np.diff(assumed_pattern_starts)
        max_assumed_n_pixels = 1024 * 1244
        # n pixels x 2 (can be uint16)
        max_assumed_pattern_size = max_assumed_n_pixels * 2 + self.pattern_header_size
        # 20x is chosen as a sufficiently high jump in bytes
        pattern_start = abs(diff_pattern_starts) > 20 * max_assumed_pattern_size
        n_patterns = np.nonzero(pattern_start)[0][0] + 1

        return n_patterns
