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

from typing import List, Tuple, Union

import dask.array as da
from h5py import File
import numpy as np

from kikuchipy.io.plugins.h5ebsd import hdf5group2dict
from kikuchipy.io.plugins.emsoft_ebsd_master_pattern import (
    _crystal_data_2_metadata,
)

# Plugin characteristics
# ----------------------
format_name = "emsoft_ebsd"
description = (
    "Read support for dynamically simulated electron backscatter diffraction "
    "patterns stored in EMsoft's h5ebsd file format."
)
full_support = False
# Recognised file extension
file_extensions = ["h5", "hdf5"]
default_extension = 0
# Writing capabilities
writes = False

# Unique HDF5 footprint
footprint = ["emdata/ebsd/ebsdpatterns"]


def file_reader(
    filename: str,
    scan_size: Union[None, int, Tuple[int, ...]] = None,
    lazy: bool = False,
    **kwargs,
) -> List[dict]:
    """Read dynamically simulated electron backscatter diffraction
    patterns from EMsoft's h5ebsd file format [Jackson2014]_.

    Parameters
    ----------
    filename
        Full file path of the HDF file.
    scan_size
        Scan size in number of patterns in width and height.
    lazy
        Open the data lazily without actually reading the data from disk
        until requested. Allows opening datasets larger than available
        memory. Default is False.
    kwargs :
        Keyword arguments passed to h5py.File.

    Returns
    -------
    signal_dict_list: list of dicts
        Data, axes, metadata and original metadata.
    """
    mode = kwargs.pop("mode", "r")
    f = File(filename, mode=mode, **kwargs)

    _check_file_format(f)

    # Set metadata and original metadata dictionaries
    md = {
        "Signal": {"signal_type": "EBSD", "record_by": "image"},
        "General": {
            "title": f.filename.split("/")[-1].split(".")[0],
            "original_filename": f.filename.split("/")[-1],
        },
        "Sample": {
            "Phases": {
                "1": _crystal_data_2_metadata(hdf5group2dict(f["CrystalData"]))
            },
        },
    }
    # Read all data except patterns
    contents = hdf5group2dict(
        f["/"], data_dset_names=["EBSDPatterns"], recursive=True
    )
    scan = {"metadata": md, "original_metadata": contents}

    # Read patterns
    dataset = f["EMData/EBSD/EBSDPatterns"]
    if lazy:
        chunks = "auto" if dataset.chunks is None else dataset.chunks
        patterns = da.from_array(dataset, chunks=chunks)
    else:
        patterns = np.asanyarray(dataset)

    # Reshape data if desired
    sy = contents["NMLparameters"]["EBSDNameList"]["numsy"]
    sx = contents["NMLparameters"]["EBSDNameList"]["numsx"]
    if scan_size is not None:
        if isinstance(scan_size, int):
            new_shape = (scan_size, sy, sx)
        else:
            new_shape = scan_size + (sy, sx)
        patterns = patterns.reshape(new_shape)

    scan["data"] = patterns

    # Set navigation and signal axes
    pixel_size = contents["NMLparameters"]["EBSDNameList"]["delta"]
    ndim = patterns.ndim
    units = ["px", "um", "um"]
    names = ["x", "dy", "dx"]
    scales = np.array([1, pixel_size, pixel_size])
    if ndim == 4:
        units = ["px"] + units
        names = ["y"] + names
        scales = np.append([1], scales)
    scan["axes"] = [
        {
            "size": patterns.shape[i],
            "index_in_array": i,
            "name": names[i],
            "scale": scales[i],
            "offset": 0,
            "units": units[i],
        }
        for i in range(patterns.ndim)
    ]

    if not lazy:
        f.close()

    return [scan]


def _check_file_format(file: File):
    """Return whether the HDF file is in EMsoft's h5ebsd file format.

    Parameters
    ----------
    file: h5py:File
    """
    try:
        program_name = file["EMheader/EBSD/ProgramName"][:][0].decode()
        if program_name != "EMEBSD.f90":
            raise KeyError
    except KeyError:
        raise IOError(f"'{file.filename}' is not in EMsoft's h5ebsd format.")
