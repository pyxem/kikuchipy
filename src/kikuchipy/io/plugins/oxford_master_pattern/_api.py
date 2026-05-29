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

"""Reader of EBSD master pattern simulations from an Oxford Instruments
SDF5 (HDF5) file.
"""

from pathlib import Path
from typing import Literal

import dask.array as da
from diffpy.structure import Lattice, Structure
import h5py
import numpy as np
from orix.crystal_map import Phase

from kikuchipy._utils.vector import ValidHemispheres, parse_hemisphere
from kikuchipy.io.plugins._h5ebsd import _hdf5group2dict
from kikuchipy.io.plugins.emsoft_ebsd_master_pattern._api import HEMISPHERE_ARG

ValidSimulationTypes = Literal["dynamical", "twobeam", "kinematic"]


class OxfordMasterPatternReader:
    def __init__(
        self,
        filename: str | Path,
        energy: range | None = None,
        hemisphere: ValidHemispheres = "both",
        simulation: ValidSimulationTypes = "dynamical",
        lazy: bool = False,
    ) -> None:
        self.filename = Path(filename)
        self.energy = energy
        self.hemisphere = parse_hemisphere(hemisphere)
        self.simulation = self.parse_simulation(simulation)
        self.lazy = lazy

    @staticmethod
    def check_file_format(file: h5py.File) -> None:
        if not "Proprietary/Source Info" in file:
            raise IOError(f"{file.filename!r} is not an Oxford Instruments SDF5 file")

    def get_axes(self, data_shape: tuple[int, ...]) -> list[dict]:
        sy, sx = data_shape[-2:]
        names = ["height", "width"]
        units = ["px", "px"]
        offsets = [-sy // 2, -sx // 2]
        if self.hemisphere == "both":
            names.insert(0, "hemisphere")
            offsets.insert(0, 0)
            units.insert(0, "")
        axes = []
        for i in range(len(data_shape)):
            axis = {
                "size": data_shape[i],
                "index_in_array": i,
                "name": names[i],
                "scale": 1,
                "offset": offsets[i],
                "units": units[i],
            }
            axes.append(axis)
        return axes

    def parse_data(self, group: h5py.Group) -> dict:
        d = _hdf5group2dict(group["Reflectors"])
        data_group = group[f"Master/{self.simulation}"]
        data_kwargs = {}
        if self.lazy:
            data_read_func = da.from_array
            data_stack_func = da.stack
            data_kwargs["chunks"] = "auto"
        else:
            data_read_func = np.asanyarray
            data_stack_func = np.stack
        if self.hemisphere == "upper":
            data = data_read_func(data_group["Upper"], **data_kwargs)
        elif self.hemisphere == "lower":
            data = data_read_func(data_group["Lower"], **data_kwargs)
        else:
            upper = data_read_func(data_group["Upper"], **data_kwargs)
            lower = data_read_func(data_group["Lower"], **data_kwargs)
            data = data_stack_func([upper, lower], axis=0)
        return {
            "data": data,
            "reflectors": {
                "extinction_distances": d["Extinction Distances"],
                "hkl": d["HKL"],
                "lattice_spacing": d["Lattice Spacing"],
                "normal_directions": d["Normal Directions"],
                "relative_intensities": d["Relative Intensities"],
            },
        }

    def parse_header(self, group: h5py.Group) -> dict:
        d = _hdf5group2dict(group)
        phase_info = self.parse_phase_info(group["Phase Info"])
        return {
            "beam_energy": d["Beam Voltage"],
            "debye_waller_factor": d["Debye-Waller Factor"],
            "minimum_intensity": d["Minimum Intensity"],
            "minimum_lattice_spacing": d["Minimum Lattice Spacing"],
            "phase": phase_info,
        }

    @staticmethod
    def parse_phase_info(group: h5py.Group) -> dict:
        d = _hdf5group2dict(group)
        abc = d["Lattice Dimensions"]
        # TODO: (a, b, c) unit options are Angstrom and ...? Figure out
        # the others, so we always return Ångstrøm.
        # dimension_unit = group["Lattice Dimensions"].attrs["Unit"]
        angles = d["Lattice Angles"]
        angle_unit = group["Lattice Angles"].attrs["Unit"]
        if angle_unit == "rad":
            angles = np.rad2deg(angles)
        return {
            "laue_group": d["Laue Group"],
            "name": d["Phase Name"],
            "reference": d["Reference"],
            "space_group": d["Space Group"],
            "structure": {
                "title": d["Phase Name"],
                "lattice": {
                    "a": abc[0],
                    "b": abc[1],
                    "c": abc[2],
                    "alpha": angles[0],
                    "beta": angles[1],
                    "gamma": angles[2],
                },
            },
        }

    @staticmethod
    def parse_simulation(simulation: ValidSimulationTypes) -> str:
        sim = simulation.lower()
        options = ["dynamical", "twobeam", "kinematic"]
        if sim not in options:
            raise ValueError(
                f"Unknown simulation type {simulation!r}. Options are "
                + ",".join(options)
                + "."
            )
        sim = sim.capitalize()
        if sim == "Twobeam":
            sim = "TwoBeam"
        return sim

    def read(self, **kwargs) -> list[dict]:
        file = h5py.File(self.filename, mode="r", **kwargs)
        self.check_file_format(file)
        header = self.parse_header(file["Header"])
        all_data = self.parse_data(file["Data"])
        md = {
            "Acquisition_instrument": {
                "SEM": {"beam_energy": header.pop("beam_energy")}
            },
            "General": {
                "original_filename": self.filename.name,
                "title": self.filename.stem,
            },
            "Signal": {"record_by": "image", "signal_type": "EBSDMasterPattern"},
        }
        phase_info = header.pop("phase")
        phase = Phase(
            name=phase_info["name"],
            space_group=int(phase_info["space_group"]),
            structure=Structure(
                title=phase_info["structure"]["title"],
                lattice=Lattice(**phase_info["structure"]["lattice"]),
            ),
        )
        data = all_data.pop("data")
        out = {
            "axes": self.get_axes(data.shape),
            "data": data,
            "hemisphere": "both",
            "metadata": md,
            "phase": phase,
            "projection": "stereographic",
        }
        omd = header
        omd.update(all_data)
        out["original_metadata"] = omd
        if not self.lazy:
            file.close()
        return [out]


def file_reader(
    filename: str | Path,
    energy: range | None = None,
    hemisphere: ValidHemispheres = "both",
    simulation: ValidSimulationTypes = "dynamical",
    lazy: bool = False,
    **kwargs,
) -> list[dict]:
    """Read simulated electron backscatter diffraction master patterns
    from Oxford Instruments' SDF5 (HDF5) file format.

    Not meant to be used directly; use :func:`~kikuchipy.load`.

    Parameters
    ----------
    filename
        Full file path of the SDF5 file.
    energy
        Desired beam energy. If not given, the simulation for the
        highest beam energy is returned.
    hemisphere
        Projection hemisphere(s) to read. Options are "both" (default),
        "upper", or "lower". If "both", these will be stacked in the
        vertical navigation axis.
    lazy
        Open the data lazily without actually reading the data from disk
        until requested. Allows opening datasets larger than available
        memory. Default is False.
    **kwargs
        Keyword arguments passed to :class:`h5py.File`.

    Returns
    -------
    signal_dict_list
        Data, axes, metadata, and original metadata.
    """
    reader = OxfordMasterPatternReader(filename, energy, hemisphere, simulation, lazy)
    return reader.read(**kwargs)
