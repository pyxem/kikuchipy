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

import abc
import os
from typing import List, Optional, Tuple

import dask.array as da
from h5py import Dataset, File, Group
import numpy as np

from kikuchipy.io.plugins.emsoft_ebsd import _crystaldata2phase
from kikuchipy.io.plugins._h5ebsd import _hdf5group2dict


class EMsoftMasterPatternReader(abc.ABC):
    """Abstract class for readers of kikuchi diffraction master patterns
    from EMsoft's HDF5 file format.
    """

    def __init__(
        self,
        filename: str,
        energy: Optional[range] = None,
        projection: str = "stereographic",
        hemisphere: "str" = "upper",
        lazy: bool = False,
    ):
        self.filename = filename
        self.energy = energy
        self.projection = projection.lower()
        self.hemisphere = hemisphere.lower()
        self.lazy = lazy

    @property
    @abc.abstractmethod
    def diffraction_type(self):
        return NotImplemented  # pragma: no cover

    @property
    @abc.abstractmethod
    def cl_parameters_group_name(self):
        return NotImplemented  # pragma: no cover

    @property
    @abc.abstractmethod
    def energy_string(self):
        return NotImplemented  # pragma: no cover

    def read(self, **kwargs) -> List[dict]:
        """Read kikuchi diffraction master patterns.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to h5py.File.

        Returns
        -------
        signal_dict_list
            Data, axes, metadata and original metadata.
        """
        mode = kwargs.pop("mode", "r")
        f = File(self.filename, mode=mode, **kwargs)

        _check_file_format(f, self.diffraction_type)

        # Set data group names
        diff_type = self.diffraction_type
        signal_type = f"{diff_type}MasterPattern"
        if diff_type == "TKD":
            signal_type = "EBSDMasterPattern"
        data_group_path = f"EMData/{diff_type}master"
        name_list_name = f"{diff_type}MasterNameList"

        # Set metadata and original metadata dictionary
        md = {
            "Signal": {"signal_type": signal_type, "record_by": "image"},
            "General": {
                "title": f.filename.split("/")[-1].split(".")[0],
                "original_filename": f.filename.split("/")[-1],
            },
        }
        nml_params = _hdf5group2dict(f["NMLparameters"], recursive=True)

        # Get phase information and add it to both the original metadata
        # and a Phase object
        crystal_data = _hdf5group2dict(f["CrystalData"])
        nml_params["CrystalData"] = crystal_data
        phase = _crystaldata2phase(crystal_data)

        # Get phase name
        if phase.name == "":
            try:
                xtal_name = os.path.split(nml_params["MCCLNameList"]["xtalname"])[0]
                phase.name = os.path.splitext(xtal_name)[0]
            except KeyError:
                phase.name = None

        # Get data shape and slices
        data_group = f[data_group_path]
        energies = data_group[self.energy_string][()]
        data_shape, data_slices = _get_data_shape_slices(
            npx=nml_params[name_list_name]["npx"], energies=energies, energy=self.energy
        )
        i_min = data_slices[0].start
        i_min = 0 if i_min is None else i_min
        min_energy = energies[i_min]

        projection = self.projection
        hemisphere = self.hemisphere

        # Get HDF5 data sets
        datasets = _get_datasets(
            data_group=data_group, projection=projection, hemisphere=hemisphere
        )

        # TODO: Take EMsoft NML file parameter combinesites into account
        dataset_shape = data_shape
        if projection == "lambert" and diff_type != "ECP":
            data_slices = (slice(None, None),) + data_slices
            data_shape = (data_group["numset"][:][0],) + data_shape

        data_shape = (len(datasets),) + data_shape

        # Set up data reading
        data_kwargs = {}
        if self.lazy:
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
                [data, data_read_func(datasets[1][data_slices], **data_kwargs)]
            )

        if projection == "lambert" and diff_type != "ECP":
            if self.hemisphere == "both":
                sum_axis = 1
                data_shape = (data_shape[0],) + data_shape[2:]
            else:
                sum_axis = 0
                data_shape = data_shape[1:]
            data = data.sum(axis=sum_axis).astype(data.dtype)

        # Remove 1-dimensions
        data = data.squeeze()

        if projection == "stereographic":
            # Mirror about horizontal (flip up-down)
            data = data[..., ::-1, :]

        # Axes scales
        group_name = f"{self.cl_parameters_group_name}NameList"
        energy_scale = nml_params[group_name]["Ebinsize"]
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

        if not self.lazy:
            f.close()

        return [output]


def _check_file_format(file: File, diffraction_type: str):
    """Raise an error if the HDF5 file is not in EMsoft's master
    pattern file format.

    Parameters
    ----------
    file
        HDF5 file.
    diffraction_type
        ``"EBSD"``, ``"TKD"`` or ``"ECP"``.

    Raises
    ------
    KeyError
        If the program that created the file is not named
        ``"EM<diffraction_type>master.f90"``.
    IOError
        If the file is not in the EMsoft master pattern file format.
    """
    try:
        program_name_path = f"EMheader/{diffraction_type}master/ProgramName"
        program_name = file[program_name_path][:][0].decode()
        if program_name != f"EM{diffraction_type}master.f90":
            raise KeyError
    except KeyError:
        raise IOError(f"'{file.filename}' is not in EMsoft's master pattern format")


def _get_data_shape_slices(
    npx: int,
    energies: np.ndarray,
    energy: Optional[tuple] = None,
) -> Tuple[Tuple, Tuple[slice, ...]]:
    """Return data shape from half the master pattern side length,
    number of asymmetric positions if the square Lambert projection is
    to be read, and an energy or energy range.

    Parameters
    ----------
    npx
        Half the number of pixels along x-direction of the square master
        pattern. Half is used because that is what EMsoft uses.
    energies
        Beam energies.
    energy
        Desired beam energy or energy range. If not given, it is assumed
        that all master patterns will be read.

    Returns
    -------
    data_shape
        Shape of data.
    data_slices
        Data to get, determined from ``energy``.
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


def _get_datasets(data_group: Group, projection: str, hemisphere: str) -> List[Dataset]:
    """Return datasets from projection and hemisphere.

    Parameters
    ----------
    data_group
        HDF5 data group with data sets.
    projection
        ``"stereographic"`` or ``"lambert"`` projection.
    hemisphere
        ``"upper"``, ``"lower"`` or ``"both"`` hemisphere(s).

    Returns
    -------
    datasets
        List of HDF5 data sets.
    """
    hemisphere = hemisphere.lower()
    projection = projection.lower()

    projections = {"stereographic": "masterSP", "lambert": "mLP"}
    hemispheres = {"upper": "NH", "lower": "SH"}

    if projection not in projections.keys():
        raise ValueError(
            f"'projection' value {projection} must be one of {list(projections.keys())}"
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
            f"'hemisphere' value {hemisphere} must be one of {list(hemispheres.keys())}"
        )

    return datasets
