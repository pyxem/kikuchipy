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

import os
from pathlib import Path
from typing import List, Union

import h5py
import numpy as np
from orix.crystal_map import CrystalMap

from kikuchipy.detectors import EBSDDetector
from kikuchipy.io.plugins._h5ebsd import _hdf5group2dict, H5EBSDReader


__all__ = ["file_reader"]


# Plugin characteristics
# ----------------------
format_name = "oxford_h5ebsd"
description = (
    "Read support for electron backscatter diffraction patterns stored "
    "in an HDF5 file formatted in Oxford Instruments' h5ebsd format, "
    "named H5OINA. The format is similar to the format described in "
    "Jackson et al.: h5ebsd: an archival data format for electron "
    "back-scatter diffraction data sets. Integrating Materials and "
    "Manufacturing Innovation 2014 3:4, doi: "
    "https://dx.doi.org/10.1186/2193-9772-3-4."
)
full_support = False
# Recognised file extension
file_extensions = ["h5oina"]
default_extension = 0
# Writing capabilities (signal dimensions, navigation dimensions)
writes = False


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
            Whether to read dataset lazily (default is ``False``).

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
        fname = os.path.basename(self.filename).split(".")[0]
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
        data = self.get_data(group, data_shape=(ny, nx, sy, sx), lazy=lazy)
        scan_dict["data"] = data

        # --- Axes
        scan_dict["axes"] = self.get_axes_list((ny, nx, sy, sx), (dy, dx, px_size))

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
