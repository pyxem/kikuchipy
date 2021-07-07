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
from typing import List, Tuple, Union

import dask.array as da
import numpy as np

from kikuchipy.signals.util._dask import get_chunking

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


def file_reader(filename: str, lazy: bool = False) -> list:
    with open(filename, mode="rb") as f:
        obf = OxfordBinaryFile(f)
        scan = obf.get_scan(lazy=lazy)
    return [scan]


class OxfordBinaryFile:
    pattern_header_size = 16
    pattern_header_dtype = [
        ("is_compressed", np.int32, (1,)),
        ("nrows", np.int32, (1,)),
        ("ncols", np.int32, (1,)),
        ("n_bytes", np.int32, (1,)),
    ]

    def __init__(self, file):
        self.file = file

        self.version = self.get_version()

        self.n_patterns = self.guess_number_of_patterns()
        self.pattern_starts = self.get_pattern_starts()

        is_compressed, nrows, ncols, n_bytes = self.get_single_pattern_header(
            self.first_pattern_position
        )
        self.is_compressed = is_compressed
        if self.is_compressed:
            raise NotImplementedError(
                f"Cannot read compressed EBSD patterns from {self.file.name}"
            )
        self.signal_shape = (nrows, ncols)
        self.n_bytes = n_bytes
        if n_bytes == self.n_pixels:
            self.dtype = np.uint8
        else:
            self.dtype = np.uint16

        self.pattern_footer_dtype = self.get_pattern_footer_dtype(
            self.first_pattern_position
        )

        if not self.all_patterns_present:
            self.navigation_shape = (self.n_indexed_patterns,)
            self.step_sizes = [
                1,
            ]
            self.axes_names = ["x", "dy", "dx"]
            self.scan_unit = "px"
        else:
            nrows, ncols, step_size = self.get_navigation_shape_and_step_size()
            self.navigation_shape = (nrows, ncols)
            self.step_sizes = [step_size, step_size]
            self.axes_names = ["y", "x", "dy", "dx"]
            self.scan_unit = "um"

        self.memmap = self.get_memmap()

    @property
    def all_patterns_present(self) -> np.ndarray:
        return self.pattern_is_present.all()

    @property
    def data_shape(self) -> tuple:
        return self.navigation_shape + self.signal_shape

    @property
    def first_pattern_position(self) -> int:
        return self.pattern_order_byte_position + self.n_patterns * 8

    @property
    def indexed_pattern_starts(self) -> np.ndarray:
        return self.pattern_starts[self.pattern_is_present]

    @property
    def pattern_is_present(self) -> np.ndarray:
        return self.pattern_starts != 0

    @property
    def metadata_size(self) -> int:
        return self.pattern_header_size + self.pattern_footer_size

    @property
    def n_indexed_patterns(self) -> int:
        return np.sum(self.pattern_is_present)

    @property
    def n_pixels(self) -> int:
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
        file_dtype = self.pattern_header_dtype + [
            ("pattern", self.dtype, self.signal_shape)
        ]
        footer_dtype = self.pattern_footer_dtype
        if len(footer_dtype) != 0:
            file_dtype += footer_dtype

        return np.memmap(
            self.file,
            dtype=file_dtype,
            shape=self.n_indexed_patterns,
            mode="r",
            offset=self.first_pattern_position,
        )

    def get_navigation_shape_and_step_size(self) -> Tuple[int, int, int]:
        pattern_starts = self.indexed_pattern_starts
        first_footer = self.get_single_pattern_footer(pattern_starts[0])
        second_footer = self.get_single_pattern_footer(pattern_starts[1])
        last_footer = self.get_single_pattern_footer(pattern_starts[-1])

        first_row = first_footer["map_row"][0][0]
        last_row = last_footer["map_row"][0][0]

        first_col = first_footer["map_col"][0][0]
        second_col = second_footer["map_col"][0][0]
        last_col = last_footer["map_col"][0][0]

        step_size = second_col - first_col

        min_row_i = int(abs(np.around(first_row / step_size)))
        max_row_i = int(abs(np.around(last_row / step_size)) + 1)
        nrows = max_row_i + min_row_i

        min_col_i = int(abs(np.around(first_col / step_size)))
        max_col_i = int(abs(np.around(last_col / step_size) + 1))
        ncols = max_col_i + min_col_i

        return nrows, ncols, step_size

    def get_pattern_starts(self) -> np.ndarray:
        self.file.seek(self.pattern_order_byte_position)
        return np.fromfile(self.file, dtype=np.int64, count=self.n_patterns)

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
        else:
            self.pattern_footer_size = 0
        return list(footer_dtype)

    def get_patterns(self, lazy: bool) -> Union[np.ndarray, da.Array]:
        data = self.memmap["pattern"]
        if lazy:
            data = da.from_array(data)

        is_sorted = np.allclose(np.diff(self.pattern_order), 1)
        if not is_sorted and self.all_patterns_present:
            data = data[self.pattern_order]

        if data.shape != self.data_shape:
            data = data.reshape(self.data_shape)

        if lazy:
            chunks = get_chunking(
                data_shape=self.data_shape, nav_dim=2, sig_dim=2, dtype=self.dtype
            )
            data = data.rechunk(chunks)

        return data

    def get_single_pattern_footer(self, offset: int) -> tuple:
        self.file.seek(offset + self.pattern_header_size + self.n_bytes)
        return np.fromfile(self.file, dtype=self.pattern_footer_dtype, count=1)

    def get_single_pattern_header(self, offset: int) -> Tuple[bool, int, int, int]:
        self.file.seek(offset)
        header = np.fromfile(self.file, dtype=self.pattern_header_dtype, count=1)
        return (
            bool(header["is_compressed"][0]),
            int(header["nrows"]),
            int(header["ncols"]),
            int(header["n_bytes"]),
        )

    def get_version(self) -> int:
        self.file.seek(0)
        version = struct.unpack("q", self.file.read(8))[0]  # int64
        if version < 0:
            return -version
        else:
            return 0

    def get_scan(self, lazy: bool) -> dict:
        data = self.get_patterns(lazy=lazy)

        units = [self.scan_unit] * 4
        scales = self.step_sizes + [1, 1]
        axes = [
            {
                "size": data.shape[i],
                "index_in_array": i,
                "name": self.axes_names[i],
                "scale": scales[i],
                "offset": 0.0,
                "units": units[i],
            }
            for i in range(data.ndim)
        ]
        fname = self.file.name
        metadata = dict(
            General=dict(
                original_filename=fname,
                title=os.path.splitext(os.path.split(fname)[1])[0],
            ),
            Signal=dict(signal_type="EBSD", record_by="image"),
        )

        order = self.pattern_order[self.pattern_is_present]
        om = dict(
            y_beam=self.memmap["map_row"][..., 0][order],
            x_beam=self.memmap["map_col"][..., 0][order],
            map1d_id=np.arange(self.n_patterns)[self.pattern_is_present],
            file_order=self.pattern_order[self.pattern_is_present],
        )

        scan = dict(
            axes=axes,
            data=data,
            metadata=metadata,
            original_metadata=om,
        )

        return scan

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
