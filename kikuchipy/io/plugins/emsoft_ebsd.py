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

"""Reader of simulated EBSD patterns from an EMsoft HDF5 file."""

import os
from pathlib import Path
from typing import List, Tuple, Union

import dask.array as da
from diffpy.structure import Atom, Lattice, Structure
from h5py import File
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation

from kikuchipy.detectors import EBSDDetector
from kikuchipy.io.plugins._h5ebsd import _hdf5group2dict


__all__ = ["file_reader"]


# Plugin characteristics
# ----------------------
format_name = "emsoft_ebsd"
description = (
    "Read support for dynamically simulated electron backscatter "
    "diffraction patterns stored in EMsoft's HDF5 file format produced "
    "by their EMEBSD.f90 program."
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
    filename: Union[str, Path],
    scan_size: Union[None, int, Tuple[int, ...]] = None,
    lazy: bool = False,
    **kwargs,
) -> List[dict]:
    """Read dynamically simulated electron backscatter diffraction
    patterns from EMsoft's format produced by their EMEBSD.f90 program.

    Not meant to be used directly; use :func:`~kikuchipy.load`.

    Parameters
    ----------
    filename
        Full file path of the HDF file.
    scan_size
        Scan size in number of patterns in width and height.
    lazy
        Open the data lazily without actually reading the data from disk
        until requested. Allows opening datasets larger than available
        memory. Default is ``False``.
    **kwargs
        Keyword arguments passed to :class:`h5py.File`.

    Returns
    -------
    signal_dict_list
        Data, axes, metadata and original metadata.
    """
    mode = kwargs.pop("mode", "r")
    f = File(filename, mode=mode, **kwargs)

    _check_file_format(f)

    group = f["/"]
    hd = _hdf5group2dict(group, data_dset_names=["EBSDPatterns"], recursive=True)
    nml_dict = hd["NMLparameters"]["EBSDNameList"]

    # --- Metadata
    fname = os.path.basename(filename).split(".")[0]
    metadata = {
        "Acquisition_instrument": {
            "SEM": {"beam_energy": nml_dict["energymax"]},
        },
        "General": {"original_filename": fname, "title": fname},
        "Signal": {"signal_type": "EBSD", "record_by": "image"},
    }
    scan = {"metadata": metadata, "original_metadata": hd}

    # --- Data
    dataset = f["EMData/EBSD/EBSDPatterns"]
    if lazy:
        chunks = "auto" if dataset.chunks is None else dataset.chunks
        data = da.from_array(dataset, chunks=chunks)
    else:
        data = np.asanyarray(dataset)
    # Reshape data if desired
    sy = nml_dict["numsy"]
    sx = nml_dict["numsx"]
    if scan_size is not None:
        if isinstance(scan_size, int):
            new_shape = (scan_size, sy, sx)
        else:
            new_shape = scan_size + (sy, sx)
        data = data.reshape(new_shape)
    scan["data"] = data

    # --- Axes
    px_size = nml_dict["delta"]
    units = ["px", "um", "um"]
    names = ["x", "dy", "dx"]
    scales = np.array([1, px_size, px_size])
    if data.ndim == 4:
        units = ["px"] + units
        names = ["y"] + names
        scales = np.append([1], scales)
    scan["axes"] = [
        {
            "size": data.shape[i],
            "index_in_array": i,
            "name": names[i],
            "scale": scales[i],
            "offset": 0,
            "units": units[i],
        }
        for i in range(data.ndim)
    ]

    # --- Crystal map
    phase = _crystaldata2phase(_hdf5group2dict(f["CrystalData"]))
    xtal_fname = f["EMData/EBSD/xtalname"][()][0].decode().split("/")[-1]
    phase.name, _ = os.path.splitext(xtal_fname)
    scan["xmap"] = CrystalMap(
        rotations=Rotation.from_euler(f["EMData/EBSD/EulerAngles"][()]),
        phase_list=PhaseList(phase),
    )

    # --- Detector
    scan["detector"] = EBSDDetector(
        shape=(sy, sx),
        binning=nml_dict["binning"],
        px_size=px_size,
        tilt=nml_dict["thetac"],
        pc=(nml_dict["xpc"], nml_dict["ypc"], nml_dict["L"]),
        convention="emsoft",
    )

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
            f"'{file.filename}' is not in EMsoft's format returned by their EMEBSD.f90 "
            "program."
        )


def _crystaldata2phase(dictionary: dict) -> Phase:
    """Return a :class:`~orix.crystal_map.Phase` object from a
    dictionary with EMsoft CrystalData group content.

    Parameters
    ----------
    dictionary
        Dictionary with phase information.

    Returns
    -------
    Phase
    """
    # TODO: Move this to orix.io.plugins.emsoft_h5ebsd as part of v0.6
    # Get list of atoms
    n_atoms = dictionary["Natomtypes"]
    atom_data = dictionary["AtomData"]
    atom_types = dictionary["Atomtypes"]
    if n_atoms == 1:
        atom_types = (atom_types,)  # Make iterable
    atoms = []
    for i in range(n_atoms):
        # TODO: Convert atom type integer to element name, like Ni for 26
        atoms.append(
            Atom(
                atype=atom_types[i],
                xyz=atom_data[:3, i],
                occupancy=atom_data[3, i],
                Uisoequiv=atom_data[4, i] / (8 * np.pi**2) * 1e2,  # Å^-2
            )
        )

    # TODO: Use space group setting
    return Phase(
        space_group=int(dictionary["SpaceGroupNumber"]),
        structure=Structure(
            lattice=Lattice(*dictionary["LatticeParameters"].T), atoms=atoms
        ),
    )
