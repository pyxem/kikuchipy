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
from kikuchipy.signals.util._metadata import (
    ebsd_metadata,
    metadata_nodes,
    _set_metadata_from_mapping,
)


# Plugin characteristics
# ----------------------
format_name = "emsoft_ebsd"
description = (
    "Read support for dynamically simulated electron backscatter diffraction "
    "patterns stored in EMsoft's HDF5 file format produced by their EMEBSD.f90 "
    "program."
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
    patterns from EMsoft's format produced by their EMEBSD.f90 program.

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

    # Read original metadata
    omd = hdf5group2dict(
        f["/"], data_dset_names=["EBSDPatterns"], recursive=True
    )

    # Set metadata and original metadata dictionaries
    md = _get_metadata(omd)
    md.update(
        {
            "Signal": {"signal_type": "EBSD", "record_by": "image"},
            "General": {
                "title": f.filename.split("/")[-1].split(".")[0],
                "original_filename": f.filename.split("/")[-1],
            },
            "Sample": {
                "Phases": {
                    "1": _crystal_data_2_metadata(
                        hdf5group2dict(f["CrystalData"])
                    )
                }
            },
        }
    )
    scan = {"metadata": md, "original_metadata": omd}

    # Read patterns
    dataset = f["EMData/EBSD/EBSDPatterns"]
    if lazy:
        chunks = "auto" if dataset.chunks is None else dataset.chunks
        patterns = da.from_array(dataset, chunks=chunks)
    else:
        patterns = np.asanyarray(dataset)

    # Reshape data if desired
    sy = omd["NMLparameters"]["EBSDNameList"]["numsy"]
    sx = omd["NMLparameters"]["EBSDNameList"]["numsx"]
    if scan_size is not None:
        if isinstance(scan_size, int):
            new_shape = (scan_size, sy, sx)
        else:
            new_shape = scan_size + (sy, sx)
        patterns = patterns.reshape(new_shape)

    scan["data"] = patterns

    # Set navigation and signal axes
    pixel_size = omd["NMLparameters"]["EBSDNameList"]["delta"]
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
    """Return whether the HDF file is in EMsoft's format.

    Parameters
    ----------
    file: h5py:File
    """
    try:
        program_name = file["EMheader/EBSD/ProgramName"][:][0].decode()
        if program_name != "EMEBSD.f90":
            raise KeyError
    except KeyError:
        raise IOError(
            f"'{file.filename}' is not in EMsoft's format returned by their "
            "EMEBSD.f90 program."
        )


def _get_metadata(omd: dict) -> dict:
    """Return metadata dictionary from original metadata dictionary.

    Parameters
    ----------
    omd
        Dictionary with original metadata.

    Returns
    -------
    md
        Dictionary with metadata.
    """
    md = ebsd_metadata()
    sem_node, ebsd_node = metadata_nodes(["sem", "ebsd"])
    md.set_item(f"{ebsd_node}.manufacturer", "EMsoft")
    mapping = {
        f"{ebsd_node}.version": ["EMheader", "EBSD", "Version"],
        f"{ebsd_node}.binning": ["NMLparameters", "EBSDNameList", "binning"],
        f"{ebsd_node}.elevation_angle": [
            "NMLparameters",
            "EBSDNameList",
            "thetac",
        ],
        f"{ebsd_node}.exposure_time": [
            "NMLparameters",
            "EBSDNameList",
            "dwelltime",
        ],
        f"{ebsd_node}.xpc": ["NMLparameters", "EBSDNameList", "xpc"],
        f"{ebsd_node}.ypc": ["NMLparameters", "EBSDNameList", "ypc"],
        f"{ebsd_node}.zpc": ["NMLparameters", "EBSDNameList", "L"],
        f"{sem_node}.beam_energy": [
            "NMLparameters",
            "EBSDNameList",
            "energymax",
        ],
    }
    _set_metadata_from_mapping(omd, md, mapping)
    return md.as_dictionary()
