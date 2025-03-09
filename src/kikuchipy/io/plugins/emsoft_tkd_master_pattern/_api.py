# Copyright 2019-2024 The kikuchipy developers
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

"""Reader of simulated TKD master patterns from an EMsoft HDF5 file."""

from pathlib import Path

from kikuchipy._utils.vector import ValidHemispheres, ValidProjections
from kikuchipy.io.plugins._emsoft_master_pattern import EMsoftMasterPatternReader
from kikuchipy.io.plugins.emsoft_ebsd_master_pattern._api import (
    ENERGY_ARG,
    HEMISPHERE_ARG,
    PROJECTION_ARG,
)


class EMsoftTKDMasterPatternReader(EMsoftMasterPatternReader):
    @property
    def diffraction_type(self) -> str:
        return "TKD"

    @property
    def cl_parameters_group_name(self) -> str:
        return "MCCLfoil"  # Monte Carlo OpenCL

    @property
    def energy_string(self) -> str:
        return "EkeVs"


def file_reader(
    filename: str | Path,
    energy: range | None = None,
    projection: ValidProjections = "stereographic",
    hemisphere: ValidHemispheres = "upper",
    lazy: bool = False,
    **kwargs,
) -> list[dict]:
    """Read simulated transmission kikuchi diffraction master patterns
    from EMsoft's HDF5 file format :cite:`callahan2013dynamical`.

    Not meant to be used directly; use :func:`~kikuchipy.load`.

    Parameters
    ----------
    filename
        Full file path of the HDF file.
    energy
        %s
    projection
        %s
    hemisphere
        %s
    lazy
        Open the data lazily without actually reading the data from disk
        until requested. Allows opening datasets larger than available
        memory. Default is False.
    **kwargs
        Keyword arguments passed to :class:`h5py.File`.

    Returns
    -------
    signal_dict_list
        Data, axes, metadata and original metadata.
    """
    reader = EMsoftTKDMasterPatternReader(
        filename, energy, projection, hemisphere, lazy
    )
    return reader.read(**kwargs)


file_reader.__doc__ %= (ENERGY_ARG, PROJECTION_ARG, HEMISPHERE_ARG)
