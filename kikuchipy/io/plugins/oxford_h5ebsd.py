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

"""Reader of EBSD data from an Oxford Instruments h5ebsd (H5OINA) file.
"""

from pathlib import Path
from typing import List, Optional, Union
import warnings

import dask.array as da
import h5py
import numpy as np
from orix.crystal_map import CrystalMap

from kikuchipy.detectors import EBSDDetector
from kikuchipy.io.plugins._h5ebsd import _hdf5group2dict, H5EBSDReader


__all__ = ["file_reader"]


class OxfordH5EBSDReader(H5EBSDReader):
    """Oxford Instruments h5ebsd (H5OINA) file reader.

    Parameters
    ----------
    filename
        Full file path of the HDF5 file.
    **kwargs
        Keyword arguments passed to :class:`h5py.File`.
    """

    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)

    def scan2dict(self, group: h5py.Group, lazy: bool = False) -> dict:
        """Read (possibly lazily) patterns from group.

        Parameters
        ----------
        group
            Group with patterns.
        lazy
            Read dataset lazily (default is ``False``).

        Returns
        -------
        scan_dict
            Dictionary with keys ``"axes"``, ``"data"``, ``"metadata"``,
            ``"original_metadata"``, ``"detector"``,
            ``"static_background"``, and ``"xmap"``. This dictionary can
             be passed as keyword arguments to create an
             :class:`~kikuchipy.signals.EBSD` signal.

        Raises
        ------
        IOError
            If patterns are not acquired in a square grid.
        """
        hd = _hdf5group2dict(group["EBSD/Header"], recursive=True)
        dd = _hdf5group2dict(group["EBSD/Data"], data_dset_names=self.patterns_name)

        # Get data shapes
        ny, nx = hd["Y Cells"], hd["X Cells"]
        sy, sx = hd["Pattern Height"], hd["Pattern Width"]
        dy, dx = hd.get("Y Step", 1), hd.get("X Step", 1)
        px_size = 1.0

        # --- Metadata
        fname = self.filename.split("/")[-1].split(".")[0]
        title = fname + " " + group.name[1:].split("/")[0]
        if len(title) > 20:
            title = f"{title:.20}..."
        metadata = {
            "Acquisition_instrument": {
                "SEM": {
                    "beam_energy": hd.get("Beam Voltage"),
                    "magnification": hd.get("Magnification"),
                    "Stage": {
                        "rotation": hd.get("Stage Position", {}).get("Rotation"),
                        "tilt_alpha": hd.get("Stage Position", {}).get("Tilt"),
                        "x": hd.get("Stage Position", {}).get("X"),
                        "y": hd.get("Stage Position", {}).get("Y"),
                        "z": hd.get("Stage Position", {}).get("Z"),
                    },
                    "working_distance": hd.get("Working Distance"),
                },
            },
            "General": {
                "notes": hd.get("Site Notes"),
                "original_filename": fname,
                "title": title,
            },
            "Signal": {"signal_type": "EBSD", "record_by": "image"},
        }
        scan_dict = {"metadata": metadata}

        # --- Data
        # Get HDF5 dataset with pattern array
        try:
            data_dset = group["EBSD/Data/" + self.patterns_name]
        except KeyError:
            raise KeyError(
                "Could not find patterns in the expected dataset "
                f"'EBSD/Data/{self.patterns_name}'"
            )
        # Get array from dataset
        if lazy:
            if data_dset.chunks is None:
                chunks = "auto"
            else:
                chunks = data_dset.chunks
            data = da.from_array(data_dset, chunks=chunks)
        else:
            data = np.asanyarray(data_dset)
        # Reshape array
        try:
            data = data.reshape((ny, nx, sy, sx)).squeeze()
        except ValueError:
            warnings.warn(
                f"Pattern size ({sx} x {sy}) and scan size ({nx} x {ny}) larger than "
                "file size. Will attempt to load by zero padding incomplete frames"
            )
            # Data is stored image by image
            pw = [(0, ny * nx * sy * sx - data.size)]
            if lazy:
                data = da.pad(data.flatten(), pw)
            else:
                data = np.pad(data.flatten(), pw)
            data = data.reshape((ny, nx, sy, sx))
        scan_dict["data"] = data

        # --- Axes
        units = ["um"] * 4
        scales = np.ones(4)
        # Calibrate scan dimension and detector dimension
        scales[0] *= dy
        scales[1] *= dx
        scales[2] *= px_size
        scales[3] *= px_size
        # Set axes names
        names = ["y", "x", "dy", "dx"]
        if data.ndim == 3:
            if ny > nx:
                names.remove("x")
                scales = np.delete(scales, 1)
            else:
                names.remove("y")
                scales = np.delete(scales, 0)
        elif data.ndim == 2:
            names = names[2:]
            scales = scales[2:]
        # Create list of axis objects
        scan_dict["axes"] = [
            {
                "size": data.shape[i],
                "index_in_array": i,
                "name": names[i],
                "scale": scales[i],
                "offset": 0.0,
                "units": units[i],
            }
            for i in range(data.ndim)
        ]

        # --- Original metadata
        scan_dict["original_metadata"] = {
            "manufacturer": self.manufacturer,
            "version": self.version,
        }
        scan_dict["original_metadata"].update(hd)

        # --- Crystal map
        # TODO: Implement reader of Oxford Instruments h5ebsd crystal
        #  maps in orix
        xmap = CrystalMap.empty(shape=(ny, nx), step_sizes=(dy, dx))
        scan_dict["xmap"] = xmap

        # --- Detector
        pc = np.column_stack(
            (
                dd.get("Pattern Center X", 0.5),
                dd.get("Pattern Center Y", 0.5),
                dd.get("Detector Distance", 0.5),
            )
        )
        if pc.size > 3:
            pc = pc.reshape((ny, nx, 3))
        scan_dict["detector"] = EBSDDetector(
            shape=(sy, sx),
            px_size=px_size,
            sample_tilt=np.rad2deg(hd.get("Tilte Angle", np.deg2rad(70))),
            pc=pc,
            convention="oxford",
        )

        return scan_dict


def file_reader(
    filename: Union[str, Path],
    scan_group_names: Union[None, str, List[str]] = None,
    lazy: bool = False,
    **kwargs,
) -> List[dict]:
    """Read electron backscatter diffraction patterns, a crystal map,
    and an EBSD detector from an Oxford Instruments h5ebsd (H5OINA) file
    :cite:`jackson2014h5ebsd`.

    Parameters
    ----------
    filename
        Full file path of the HDF5 file.
    scan_group_names
        Name or a list of names of HDF5 top group(s) containing the
        scan(s) to return. If not given (default), the first scan in the
        file is returned.
    lazy
        Open the data lazily without actually reading the data from disk
        until required. Allows opening arbitrary sized datasets. Default
        is ``False``.
    **kwargs
        Keyword arguments passed to :class:`h5py.File`.

    Returns
    -------
    scan_dict_list
        List of one or more dictionaries with the keys ``"axes"``,
        ``"data"``, ``"metadata"``, ``"original_metadata"``,
        ``"detector"``, and ``"xmap"``. This
        dictionary can be passed as keyword arguments to create an
        :class:`~kikuchipy.signals.EBSD` signal.
    """
    reader = OxfordH5EBSDReader(filename, **kwargs)
    return reader.read(scan_group_names, lazy)
