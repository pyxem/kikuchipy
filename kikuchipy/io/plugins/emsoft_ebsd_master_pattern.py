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

"""Read support for simulated EBSD master patterns in EMsoft's HDF5
format.
"""

import os
from typing import List, Optional, Tuple

import dask.array as da
from h5py import File, Group, Dataset
import numpy as np

from kikuchipy.io.plugins.h5ebsd import hdf5group2dict
from kikuchipy.io.plugins.emsoft_ebsd import _crystaldata2phase


# Plugin characteristics
# ----------------------
format_name = "emsoft_ebsd_master_pattern"
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
footprint = ["emdata/ebsdmaster"]


def file_reader(
    filename: str,
    energy: Optional[range] = None,
    projection: str = "stereographic",
    hemisphere: str = "north",
    lazy: bool = False,
    **kwargs,
) -> List[dict]:
    """Read electron backscatter diffraction master patterns from
    EMsoft's HDF5 file format :cite:`callahan2013dynamical`.

    Parameters
    ----------
    filename
        Full file path of the HDF file.
    energy
        Desired beam energy or energy range. If None is passed
        (default), all available energies are read.
    projection
        Projection(s) to read. Options are "stereographic" (default) or
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
    _check_file_format(f)

    # Set metadata and original metadata dictionary
    md = {
        "Signal": {"signal_type": "EBSDMasterPattern", "record_by": "image"},
        "General": {
            "title": f.filename.split("/")[-1].split(".")[0],
            "original_filename": f.filename.split("/")[-1],
        },
    }
    nml_params = hdf5group2dict(f["NMLparameters"], recursive=True)

    # Get phase information and add it to both the original metadata and
    # a Phase object
    crystal_data = hdf5group2dict(f["CrystalData"])
    nml_params["CrystalData"] = crystal_data
    phase = _crystaldata2phase(crystal_data)

    # Get the phase name
    try:
        xtal_name = os.path.split(nml_params["MCCLNameList"]["xtalname"])[0]
        phase_name = os.path.splitext(xtal_name)[0]
    except KeyError:
        phase_name = None
    phase.name = phase_name

    # Get data shape and slices
    data_group = f["EMData/EBSDmaster"]
    energies = data_group["EkeVs"][()]
    data_shape, data_slices = _get_data_shape_slices(
        npx=nml_params["EBSDMasterNameList"]["npx"],
        energies=energies,
        energy=energy,
    )
    i_min = data_slices[0].start
    i_min = 0 if i_min is None else i_min
    min_energy = energies[i_min]

    # Get HDF5 data sets
    datasets = _get_datasets(
        data_group=data_group, projection=projection, hemisphere=hemisphere,
    )

    # TODO: Data shape and slices are easier to handle if the reader
    #  was a class (in addition to file_reader()) instead of a series of
    #  function
    dataset_shape = data_shape
    if projection.lower() == "lambert":
        data_slices = (slice(None, None),) + data_slices
        data_shape = (data_group["numset"][:][0],) + data_shape

    data_shape = (len(datasets),) + data_shape

    # Set up data reading
    data_kwargs = {}
    if lazy:
        if datasets[0].chunks is None or datasets[0].shape != dataset_shape:
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

    if projection.lower() == "lambert":
        if hemisphere.lower() == "both":
            sum_axis = 1
            data_shape = (data_shape[0],) + data_shape[2:]
        else:
            sum_axis = 0
            data_shape = data_shape[1:]
        data = data.sum(axis=sum_axis).astype(data.dtype)

    # Remove 1-dimensions
    data = data.squeeze()

    # Axes scales
    energy_scale = nml_params["MCCLNameList"]["Ebinsize"]
    scales = np.array([1, energy_scale, 1, 1])

    ny, nx, sy, sx = data_shape
    names = ["hemisphere", "energy", "height", "width"]
    units = ["", "keV", "px", "px"]
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

    output = {
        "axes": axes,
        "data": data,
        "metadata": md,
        "original_metadata": nml_params,
        "phase": phase,
        "projection": projection,
        "hemisphere": hemisphere,
    }

    if not lazy:
        f.close()

    return [output]


def _check_file_format(file: File):
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


def _get_data_shape_slices(
    npx: int, energies: np.ndarray, energy: Optional[tuple] = None,
) -> Tuple[Tuple, Tuple[slice, ...]]:
    """Determine the data shape from half the master pattern side
    length, number of asymmetric positions if the square Lambert
    projection is to be read, and an energy or energy range.

    Parameters
    ----------
    npx
        Half the number of pixels along x-direction of the square master
        pattern. Half is used because that is what EMsoft uses.
    energies
        Beam energies.
    energy
        Desired beam energy or energy range.

    Returns
    -------
    data_shape
        Shape of data.
    data_slices
        Data to get, determined from `energy`.
    """
    data_shape = (npx * 2 + 1,) * 2
    data_slices = (slice(None, None),) * 2
    if energy is None:
        data_slices = (slice(None, None),) + data_slices
        data_shape = (len(energies),) + data_shape
    elif hasattr(energy, "__iter__"):
        i_min = np.argwhere(energies >= energy[0])[0][0]
        i_max = np.argwhere(energies <= energy[1])[-1][0] + 1
        data_slices = (slice(i_min, i_max),) + data_slices
        data_shape = (i_max - i_min,) + data_shape
    else:  # Assume integer
        # Always returns one integer
        index = np.abs(energies - energy).argmin()
        data_slices = (slice(index, index + 1),) + data_slices
        data_shape = (1,) + data_shape
    return data_shape, data_slices


def _get_datasets(
    data_group: Group, projection: str, hemisphere: str,
) -> List[Dataset]:
    """Get datasets from projection and hemisphere.

    Parameters
    ----------
    data_group
        HDF5 data group with data sets.
    projection
        "stereographic" or "lambert" projection.
    hemisphere
        "north" hemisphere, "south" hemisphere, or "both".

    Returns
    -------
    datasets
        List of HDF5 data sets.
    """
    hemisphere = hemisphere.lower()
    projection = projection.lower()

    projections = {"stereographic": "masterSP", "lambert": "mLP"}
    hemispheres = {"north": "NH", "south": "SH"}

    if projection not in projections.keys():
        raise ValueError(
            f"'projection' value {projection} must be one of "
            f"{projections.keys()}"
        )

    if hemisphere == "both":
        dset_names = [
            projections[projection] + hemispheres[h] for h in hemispheres.keys()
        ]
        datasets = [data_group[n] for n in dset_names]
    elif hemisphere in hemispheres:
        dset_name = projections[projection] + hemispheres[hemisphere]
        datasets = [data_group[dset_name]]
    else:
        raise ValueError(
            f"'hemisphere' value {hemisphere} must be one of "
            f"{hemispheres.keys()}."
        )

    return datasets
