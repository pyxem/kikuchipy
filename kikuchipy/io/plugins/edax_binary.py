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

"""Reader of EBSD data from EDAX TSL UP1/2 files.

The reader is adapted from the EDAX UP1/2 reader in PyEBSDIndex.
"""

import os
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union
import warnings

import dask.array as da
import numpy as np

from kikuchipy.signals.util import get_chunking


__all__ = ["file_reader"]


# Plugin characteristics
# ----------------------
format_name = "EDAX binary"
description = (
    "Read support for electron backscatter diffraction patterns stored "
    "in a binary file formatted in EDAX TSL's UP1/UP2 format with file "
    "extension '.up1' or '.up2'. The reader is adapted from the EDAX "
    "UP1/2 reader in PyEBSDIndex."
)
full_support = False
# Recognised file extension
file_extensions = ["up1", "up2"]
default_extension = 0
# Writing capabilities (signal dimensions, navigation dimensions)
writes = False


def file_reader(
    filename: Union[str, Path],
    nav_shape: Optional[Tuple[int, int]] = None,
    lazy: bool = False,
) -> List[dict]:
    """Read EBSD patterns from an EDAX binary UP1/2 file.

    Not meant to be used directly; use :func:`~kikuchipy.load`.

    Parameters
    ----------
    filename
        File path to UP1/2 file with ``"up1"`` or ``"up2"`` extension.
    nav_shape
        Navigation shape, as (n map rows, n map columns), of the
        returned :class:`~kikuchipy.signals.EBSD` signal, matching the
        number of patterns in the file. If not given, this shape will
        be attempted to be determined from the file. If it could not be,
        the returned signal will have only one navigation dimension. If
        patterns were acquired in an hexagonal grid, the returned signal
        will have only one navigation dimension irrespective of this
        parameter's value.
    lazy
        Read the data lazily without actually reading the data from disk
        until required. Default is ``False``.

    Returns
    -------
    scan
        Data, axes, metadata and original metadata.

    Raises
    ------
    ValueError
        If file version is 2, since only version 1 or >= 3 is supported.
    ValueError
        If ``nav_shape`` does not match the number of patterns in
        the file.

    Warns
    -----
    UserWarning
        If patterns were acquired in an hexagonal grid, since then the
        returned signal will have only one navigation dimension, even
        though ``nav_shape`` is given.

    Notes
    -----
    Reader adapted from the EDAX UP1/2 reader in PyEBSDIndex.
    """
    with open(filename, mode="rb") as f:
        reader = EDAXBinaryFileReader(f)
        scan = reader.read_scan(nav_shape=nav_shape, lazy=lazy)
    return [scan]


class EDAXBinaryFileReader:
    """EDAX TSL's binary UP1/2 file reader.

    Parameters
    ----------
    file
        Open EDAX binary UP1/2 file with uncompressed patterns.
    """

    def __init__(self, file: BinaryIO):
        """Prepare to read EBSD patterns from an open EDAX UP1/2 file."""
        self.file = file  # Already open file

        ext = os.path.splitext(file.name)[1][1:].lower()
        self.dtype = {"up1": np.uint8, "up2": np.uint16}[ext]

        file.seek(0)
        self.version = np.fromfile(self.file, "uint32", 1)[0]
        if self.version == 2:
            raise ValueError("Only files with version 1 or >= 3, not 2, can be read")

    def read_header(self) -> Dict[str, int]:
        """Read and return header information.

        Returns
        -------
        dictionary
        """
        self.file.seek(4)
        sx, sy, pattern_offset = np.fromfile(self.file, "uint32", count=3)
        file_size = Path(self.file.name).stat().st_size

        if self.version == 1:
            n_patterns = int(
                (file_size - pattern_offset) / (sx * sy * self.dtype(0).nbytes)
            )
            nx, ny = n_patterns, 1
            dx, dy = 1, 1
            is_hex = False
        else:  # Version >= 3
            nx, ny = np.fromfile(self.file, "uint32", 2, offset=1)

            is_hex = bool(np.fromfile(self.file, "uint8", 1)[0])
            if is_hex:
                warnings.warn(
                    "Returned signal has one navigation dimension since an hexagonal "
                    "grid is not supported"
                )
                n_patterns = int(
                    (file_size - pattern_offset) / (sx * sy * self.dtype(0).nbytes)
                )
                nx, ny = n_patterns, 1
            else:
                n_patterns = int(nx * ny)

            dx, dy = np.fromfile(self.file, "float64", 2)

        return {
            "sx": sx,
            "sy": sy,
            "pattern_offset": pattern_offset,
            "nx": nx,
            "ny": ny,
            "n_patterns": n_patterns,
            "dx": dx,
            "dy": dy,
            "is_hex": is_hex,
        }

    def read_scan(
        self, nav_shape: Optional[Tuple[int, int]] = None, lazy: bool = False
    ) -> dict:
        """Return a dictionary with scan information and patterns.

        Parameters
        ----------
        nav_shape
            Navigation shape, as (n map rows, n map columns). Default is
            ``None``.
        lazy
            Whether to reader patterns lazily. Default is ``False``.

        Returns
        -------
        dictionary
            Dictionary of scan information.
        """
        header = self.read_header()

        if nav_shape is not None and not header["is_hex"]:
            ny, nx = nav_shape
            n_patterns = int(ny * nx)
            if n_patterns != header["n_patterns"]:
                raise ValueError(
                    f"Given `nav_shape` {nav_shape} does not match the number of "
                    f"patterns in the file, {header['n_patterns']}."
                )
        else:
            ny, nx = header["ny"], header["nx"]
            n_patterns = header["n_patterns"]

        sy, sx = header["sy"], header["sx"]
        data_shape = (ny, nx, sy, sx)
        if ny == 1:
            data_shape = data_shape[1:]
        ndim = len(data_shape)
        nav_dim = ndim - 2

        if lazy:
            data = np.memmap(
                self.file,
                dtype=self.dtype,
                shape=data_shape,
                mode="r",
                offset=header["pattern_offset"],
            )
            data = da.from_array(data)
            chunks = get_chunking(
                data_shape=data_shape, nav_dim=nav_dim, sig_dim=2, dtype=self.dtype
            )
            data = data.rechunk(chunks)
        else:
            data = np.fromfile(self.file, self.dtype, n_patterns * sy * sx)
            data = data.reshape(data_shape)

        units = ["um"] * ndim
        scales = [header["dy"], header["dx"]] + [1, 1][:nav_dim]
        axes_names = ["y", "x"][-nav_dim:] + ["dy", "dx"]
        axes = [
            {
                "size": data.shape[i],
                "index_in_array": i,
                "name": axes_names[i],
                "scale": scales[i],
                "offset": 0.0,
                "units": units[i],
            }
            for i in range(ndim)
        ]
        fname = self.file.name
        metadata = dict(
            General=dict(
                original_filename=fname,
                title=os.path.splitext(os.path.split(fname)[1])[0],
            ),
            Signal=dict(signal_type="EBSD", record_by="image"),
        )

        scan = dict(
            axes=axes,
            data=data,
            metadata=metadata,
        )

        return scan
