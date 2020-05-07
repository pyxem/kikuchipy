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

from typing import List, Optional, Tuple

import dask.array as da
from h5py import File, Group, Dataset
import numpy as np

from kikuchipy.io.plugins.h5ebsd import hdf5group2dict

# Plugin characteristics
# ----------------------
format_name = "emsoft_masterpattern"
description = (
    "Read support for electron backscatter diffraction master patterns "
    "stored in EMsoft's HDF5 file format."
)
full_support = False
# Recognised file extension
file_extensions = ["h5", "hdf5"]
default_extension = 0
# Writing capabilities
writes = False

# Unique HDF5 footprint
footprint = "emdata/ebsdmaster"


def file_reader(
    filename: str,
    energy_range: Optional[range] = None,
    projection: str = "spherical",
    hemisphere: str = "north",
    lazy: bool = False,
    **kwargs,
) -> List[dict]:
    """Read electron backscatter diffraction master patterns from
    EMsoft's HDF5 file format.

    Parameters
    ----------
    filename
        Full file path of the HDF file.
    energy_range
        Range of beam energies for patterns to read. If None is passed
        (default), all available energies are read.
    projection
        Projection(s) to read. Options are "spherical" (default) or
        "lambert".
    hemisphere
        Projection hemisphere(s) to read. Options are "north" (default),
        "south" or "both". If "both", these will be stacked in the
        vertical navigation axis.
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

    # Check if the file is valid
    check_file_format(f)

    # Get data shape and slices
    data_group = f["EMData/EBSDmaster"]
    energies = data_group["EkeVs"][()]
    data_shape, data_slices = get_data_shape_slices(
        npx=f["NMLparameters/EBSDMasterNameList/npx"][()],
        energies=energies,
        energy_range=energy_range,
    )
    i_min = data_slices[0].start
    i_min = 0 if i_min is None else i_min
    min_energy = energies[i_min]

    # Account for the Lambert projections being stored as having a 1-dimension
    # before the energy dimension
    if projection == "lambert":
        data_slices = (slice(None, None),) + data_slices

    # Get HDF5 data sets
    datasets = get_datasets(
        data_group=data_group, projection=projection, hemisphere=hemisphere,
    )
    data_shape = (len(datasets),) + data_shape

    # Set up data reading
    data_kwargs = {}
    if lazy:
        if datasets[0].chunks is None or datasets[0].shape != data_shape:
            data_kwargs["chunks"] = "auto"
        else:
            data_kwargs["chunks"] = datasets[0].chunks
        data_read_func = da.from_array
        data_stack_func = da.stack
    else:
        data_read_func = np.asanyarray
        data_stack_func = np.stack

    # Read data
    data = data_read_func(datasets[0][data_slices], **data_kwargs)
    if data_shape[0] == 2:
        data = data_stack_func(
            [data, data_read_func(datasets[1][data_slices], **data_kwargs)],
            axis=0,
        )

    # Remove 1-dimensions
    data = data.squeeze()

    ny, nx, sy, sx = data_shape
    units = ["hemisphere", "keV", "px", "px"]
    names = ["y", "x", "dy", "dx"]
    scales = np.ones(4)
    offsets = [0, min_energy, -sy // 2, -sx // 2]
    dim_idx = []
    if ny != 1:
        dim_idx.append(0)
    if nx != 1:
        dim_idx.append(1)
    dim_idx += [2, 3]

    # Create axis object
    axes = [
        {
            "size": data.shape[i],
            "index_in_array": i,
            "name": names[j],
            "scale": scales[j],
            "offset": offsets[j],
            "units": units[j],
        }
        for i, j in zip(range(data.ndim), dim_idx)
    ]

    md = {
        "Signal": {"signal_type": "EBSDMasterPattern", "record_by": "image",},
    }

    output = {
        "axes": axes,
        "data": data,
        "metadata": md,
        "original_metadata": {},
    }

    if not lazy:
        f.close()

    return [
        output,
    ]


def check_file_format(file: File):
    """Return whether the HDF file is in EMsoft's master pattern file
    format.

    Parameters
    ----------
    file: h5py:File

    """

    try:
        program_name = file["EMheader/EBSDmaster/ProgramName"][:][0].decode()
        if program_name != "EMEBSDmaster.f90":
            raise KeyError
    except KeyError:
        raise IOError(
            f"'{file.filename}' is not in EMsoft's master pattern format."
        )


def get_data_shape_slices(
    npx: int, energies: np.ndarray, energy_range: Optional[tuple] = None,
) -> Tuple[Tuple, Tuple[slice, ...]]:
    """Determine data shape from number of pixels in a master pattern
    quadrant and an energy array.

    Parameters
    ----------
    npx
        Number of pixels along x-direction of the square master pattern.
    energies
        Beam energies.
    energy_range
        Range of sought energies.

    Returns
    -------
    data_shape
        Shape of data.
    data_slices
        Data to get, determined from `energy_range`.

    """

    data_shape = (npx * 2 + 1,) * 2
    data_slices = (slice(None, None),) * 2
    if energy_range is None:
        data_slices = (slice(None, None),) + data_slices
        data_shape = (len(energies),) + data_shape
    else:
        i_min, i_max = [
            np.argwhere(energies == i)[0][0] + 1 for i in energy_range
        ]
        data_slices = (slice(i_min, i_max),) + data_slices
        data_shape = (i_max - i_min,) + data_shape

    return data_shape, data_slices


def get_datasets(
    data_group: Group, projection: str, hemisphere: str,
) -> List[Dataset]:
    """Get datasets from projection and hemisphere.

    Parameters
    ----------
    data_group
        HDF5 data group with data sets.
    projection
        "spherical" or "lambert" projection.
    hemisphere
        "north" hemisphere, "south" hemisphere, or "both".

    Returns
    -------
    datasets
        List of HDF5 data sets.

    """

    mapping_p = {"spherical": "masterSP", "lambert": "mLP"}
    mapping_h = {"north": "NH", "south": "SH"}
    hemispheres = ["north", "south"]
    hemisphere = hemisphere.lower()

    if hemisphere == "both":
        dset_names = [mapping_p[projection] + mapping_h[h] for h in hemispheres]
        datasets = [data_group[n] for n in dset_names]
    elif hemisphere in hemispheres:
        dset_name = mapping_p[projection] + mapping_h[hemisphere]
        datasets = [
            data_group[dset_name],
        ]
    else:
        raise ValueError(
            f"'hemisphere' argument {hemisphere} must be one of {hemispheres}."
        )

    return datasets
