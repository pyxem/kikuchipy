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
binary .ebsp file format.

Information about the file format was provided by Oxford Instruments.
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


def file_reader(filename: str, lazy: bool = False) -> List[dict]:
    """Read EBSD patterns from an Oxford Instruments' binary .ebsp file.

    Only uncompressed patterns can be read. If only non-indexed patterns
    are stored in the file, the navigation shape will be 1D.

    Parameters
    ----------
    filename
        File path to .ebsp file.
    lazy
        Read the data lazily without actually reading the data from disk
        until required. Default is False.

    Returns
    -------
    scan
        Data, axes, metadata and original metadata.

    Notes
    -----
    Information about the .ebsp file format was provided by Oxford
    Instruments.
    """
    with open(filename, mode="rb") as f:
        obf = OxfordBinaryFileReader(f)
        scan = obf.get_scan(lazy=lazy)
    return [scan]


class OxfordBinaryFileReader:
    """Oxford Instruments' binary .ebsp file reader."""

    # Header for each pattern in the file
    pattern_header_size = 16
    pattern_header_dtype = [
        ("is_compressed", np.int32, (1,)),
        ("nrows", np.int32, (1,)),
        ("ncols", np.int32, (1,)),
        ("n_bytes", np.int32, (1,)),
    ]

    def __init__(self, file: object):
        """Prepare to read EBSD patterns from an open Oxford
        Instruments' binary .ebsp file.

        File header, byte positions of patterns, and pattern headers and
        footers are read upon initialization to determine the
        navigation (map) shape, signal (detector) shape, signal data
        type (uint8 or uint16), and whether all or only non-indexed
        patterns are stored in the file.

        A memory map (:func:`numpy.memmap`) is created at the end,
        pointing to, but not reading, the patterns on disk.

        Parameters
        ----------
        file
            Open Oxford Instruments' binary .ebsp file with uncompressed
            patterns.
        """
        self.file = file  # Already open file

        self.version = self.get_version()

        # Number of patterns in the file is not known, so this is
        # guessed from the file header where the file byte positions of
        # the patterns are stored
        self.n_patterns = self.guess_number_of_patterns()
        self.pattern_starts = self.get_pattern_starts()

        # Determine whether we can read the file, signal shape, and data
        # type
        is_compressed, nrows, ncols, n_bytes = self.get_single_pattern_header(
            self.first_pattern_position
        )
        if is_compressed:
            raise NotImplementedError(
                f"Cannot read compressed EBSD patterns from {self.file.name}"
            )
        self.signal_shape = (nrows, ncols)
        self.n_bytes = n_bytes
        if n_bytes == np.prod(self.signal_shape):
            self.dtype = np.uint8
        else:
            self.dtype = np.uint16

        # While the pattern header is always in the same format across
        # .ebsp file versions, this is not the case for the pattern
        # footer. Here we determine it's format.
        self.pattern_footer_dtype = self.get_pattern_footer_dtype(
            self.first_pattern_position
        )

        # Allow for reading of files where only non-indexed patterns are
        # stored in the file
        if not self.all_patterns_present or len(self.pattern_footer_dtype) == 0:
            self.navigation_shape = (self.n_patterns_present,)
            self.step_sizes = [1]
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
    def all_patterns_present(self) -> bool:
        """Whether all or only non-indexed patterns are stored in the
        file.
        """
        return self.pattern_is_present.all()

    @property
    def data_shape(self) -> tuple:
        """Full data shape."""
        return self.navigation_shape + self.signal_shape

    @property
    def first_pattern_position(self) -> int:
        """File byte position of first pattern after the file header."""
        return self.pattern_starts_byte_position + self.n_patterns * 8

    @property
    def pattern_is_present(self) -> np.ndarray:
        """Boolean array indicating whether a pattern listed in the file
        header is present in the file or not. If not, its
        `pattern_starts` entry is zero.
        """
        return self.pattern_starts != 0

    @property
    def n_patterns_present(self) -> int:
        """Number of patterns actually stored in the file."""
        return np.sum(self.pattern_is_present)

    @property
    def pattern_order(self) -> np.ndarray:
        """Flattened index of each consecutive pattern in the file into
        the 2D navigation (map) shape. This usually varies within rows,
        but not across rows.
        """
        metadata_size = self.pattern_header_size + self.pattern_footer_size
        bytes_per_pattern = self.n_bytes + metadata_size
        pp = (self.pattern_starts - self.first_pattern_position) / bytes_per_pattern
        return pp.astype(int)

    @property
    def pattern_starts_byte_position(self) -> int:
        """File byte position of file byte positions of patterns. For
        .ebsp file version 0, this is at the first byte, while for later
        versions, this is at the ninth byte, after the file version.
        """
        if self.version != 0:
            return 8
        else:
            return 0

    def get_memmap(self) -> np.memmap:
        """Return a memory map of the pattern header, actual patterns,
        and a potential pattern footer.

        The memory map has the shape of (n indexed patterns,), and the
        patterns have the correct signal shape (n rows, n columns).

        If the pattern footer is available, the memory map has these
        fields:

        ============= =============== ===================
        Name          Data type       Shape
        ============= =============== ===================
        is_compressed int32           (1,)
        nrows         int32           (1,)
        ncols         int32           (1,)
        n_bytes       int32           (1,)
        pattern       uint8 or uint16 (n rows, n columns)
        has_beam_x    bool            (1,)
        beam_x        float64         (1,)
        has_beam_y    bool            (1,)
        beam_y        float64         (1,)
        ============= =============== ===================

        Returns
        -------
        numpy.memmap
        """
        pattern_dtype = ("pattern", self.dtype, self.signal_shape)
        footer_dtype = self.pattern_footer_dtype
        file_dtype = self.pattern_header_dtype + [pattern_dtype]
        if len(footer_dtype) != 0:
            file_dtype += footer_dtype

        return np.memmap(
            self.file,
            dtype=file_dtype,
            shape=self.n_patterns_present,
            mode="r",
            offset=self.first_pattern_position,
        )

    def get_navigation_shape_and_step_size(self) -> Tuple[int, int, int]:
        """Return the navigation shape and step size.

        An equal step size between rows and columns is assumed.

        The navigation shape is determined by evaluating the beam
        row and column position of the upper left and lower right
        patterns. The step size is determined from the difference in
        column position of the upper left pattern and the pattern in the
        next column.

        Returns
        -------
        nrows
            Number of navigation (map) rows.
        ncols
            Number of navigation (map) columns.
        step_size
            Step size between rows and columns.
        """
        pattern_starts = self.pattern_starts[self.pattern_is_present]
        first_footer = self.get_single_pattern_footer(pattern_starts[0])
        second_footer = self.get_single_pattern_footer(pattern_starts[1])
        last_footer = self.get_single_pattern_footer(pattern_starts[-1])

        first_y = first_footer["beam_y"][0][0]
        last_y = last_footer["beam_y"][0][0]
        first_x = first_footer["beam_x"][0][0]
        second_x = second_footer["beam_x"][0][0]
        last_x = last_footer["beam_x"][0][0]

        step_size = second_x - first_x  # um

        first_yi = np.around(first_y / step_size)
        last_yi = np.around(last_y / step_size)
        nrows = int(abs(last_yi - first_yi)) + 1
        first_xi = np.around(first_x / step_size)
        last_xi = np.around(last_x / step_size)
        ncols = int(abs(last_xi - first_xi)) + 1

        return nrows, ncols, step_size

    def get_pattern_starts(self) -> np.ndarray:
        """Return the file byte positions of each pattern.

        Parameters
        ----------
        pattern_starts
            Integer array of file byte positions.
        """
        self.file.seek(self.pattern_starts_byte_position)
        return np.fromfile(self.file, dtype=np.int64, count=self.n_patterns)

    def get_pattern_footer_dtype(self, offset: int) -> List[tuple]:
        """Return the pattern footer data types to be used when memory
        mapping.

        Parameters
        ----------
        footer_dtype
            Format of each pattern footer as a list of tuples with a
            field name, data type and size. The format depends on the
            :attr:`~self.version`.
        """
        self.file.seek(offset + self.pattern_header_size + self.n_bytes)
        footer_dtype = ()
        beam_x_dtype = ("beam_x", np.float64, (1,))
        beam_y_dtype = ("beam_y", np.float64, (1,))
        if self.version == 1:
            footer_dtype += (beam_x_dtype, beam_y_dtype)
            self.pattern_footer_size = 16
        elif self.version > 1:
            pattern_footer_size = 2
            if struct.unpack("?", self.file.read(1))[0]:  # bool
                footer_dtype += (("has_beam_x", bool, (1,)), beam_x_dtype)
                pattern_footer_size += 8
                self.file.seek(8, 1)  # Move 8 bytes
            if struct.unpack("?", self.file.read(1))[0]:  # bool
                footer_dtype += (("has_beam_y", bool, (1,)), beam_y_dtype)
                pattern_footer_size += 8
            self.pattern_footer_size = pattern_footer_size
        else:
            self.pattern_footer_size = 0
        return list(footer_dtype)

    def get_patterns(self, lazy: bool) -> Union[np.ndarray, da.Array]:
        """Return the EBSD patterns in the file.

        The patterns are read from the memory map. They are sorted into
        their correct navigation (map) position if necessary.

        Parameters
        ----------
        lazy
            Whether to return a :class:`numpy.ndarray` or
            :class:`dask.array.Array`.

        Returns
        -------
        data
            EBSD patterns of shape (n navigation rows, n navigation
            columns, n signal rows, n signal columns).
        """
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
                data_shape=self.data_shape,
                nav_dim=len(self.navigation_shape),
                sig_dim=2,
                dtype=self.dtype,
            )
            data = data.rechunk(chunks)

        return data

    def get_single_pattern_footer(self, offset: int) -> tuple:
        """Return a single pattern footer with pattern beam positions.

        Parameters
        ----------
        offset
            File byte pattern start of the pattern of interest.

        Returns
        -------
        footer
            The format of this depends on the file
            :attr:`~self.version`.
        """
        self.file.seek(offset + self.pattern_header_size + self.n_bytes)
        return np.fromfile(self.file, dtype=self.pattern_footer_dtype, count=1)

    def get_single_pattern_header(self, offset: int) -> Tuple[bool, int, int, int]:
        """Return a single pattern header.

        Parameters
        ----------
        offset
            File byte pattern start of the pattern of interest.

        Returns
        -------
        is_compressed
            Whether the pattern is compressed.
        nrows
            Number of signal (detector) rows.
        ncols
            Number of signal (detector) columns.
        n_bytes
            Number of pattern bytes.
        """
        self.file.seek(offset)
        header = np.fromfile(self.file, dtype=self.pattern_header_dtype, count=1)
        return (
            bool(header["is_compressed"][0]),
            int(header["nrows"]),
            int(header["ncols"]),
            int(header["n_bytes"]),
        )

    def get_version(self) -> int:
        """Return the .ebsp file version.

        The first version of the .ebsp format did not store a version
        number. Subsequent versions store the version number as a
        negative number.

        Returns
        -------
        version
        """
        self.file.seek(0)
        version = struct.unpack("q", self.file.read(8))[0]  # int64
        if version < 0:
            return -version
        else:
            # Didn't actually read the version, just a pattern start
            # byte position, which can either be 0 (not in the file) or
            # a positive number (byte position)
            return 0

    def get_scan(self, lazy: bool) -> dict:
        """Return a dictionary with the necessary information to
        initialize an :class:`~kikuchipy.signals.EBSD` instance.

        Parameters
        ----------
        lazy
            Whether to return the EBSD patterns as a
            :class:`numpy.ndarray` or :class:`dask.array.Array`.

        Returns
        -------
        scan
            Dictionary of axes, data, metadata and original metadata.
        """
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
            map1d_id=np.arange(self.n_patterns)[self.pattern_is_present],
            file_order=order,
        )
        if "beam_y" in self.memmap.dtype.names:
            om["beam_y"] = beam_y = self.memmap["beam_y"][..., 0][order]
        if "beam_x" in self.memmap.dtype.names:
            om["beam_x"] = beam_x = self.memmap["beam_x"][..., 0][order]

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
        self.file.seek(self.pattern_starts_byte_position)
        file_byte_size = os.path.getsize(self.file.name)

        # Read assumed pattern starts
        max_assumed_n_patterns = file_byte_size // (
            min_assumed_n_pixels + self.pattern_header_size
        )
        assumed_pattern_starts = np.fromfile(
            self.file, dtype=np.int64, count=max_assumed_n_patterns
        )

        # It is assumed that a jump in bytes from one pattern position
        # to the next does not exceed a number of maximum bytes one
        # pattern can take up in the file. The array index where
        # this happens (plus 2) is assumed to be the number of patterns
        # in the file.
        diff_pattern_starts = np.diff(assumed_pattern_starts)
        max_assumed_n_pixels = 1024 * 1344
        # n pixels x 2 (can be uint16)
        max_assumed_pattern_size = max_assumed_n_pixels * 2 + self.pattern_header_size
        # 20x is chosen as a sufficiently high jump in bytes
        pattern_start = abs(diff_pattern_starts) > 20 * max_assumed_pattern_size
        n_patterns = np.nonzero(pattern_start)[0][0] + 1

        return n_patterns
