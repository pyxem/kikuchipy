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

"""Reader of EBSD data from an EDAX TSL h5ebsd file."""

from pathlib import Path
from typing import List, Union

import h5py
from orix.crystal_map import CrystalMap

from kikuchipy.detectors import EBSDDetector
from kikuchipy.io.plugins._h5ebsd import _hdf5group2dict, H5EBSDReader


__all__ = ["file_reader"]


# Plugin characteristics
# ----------------------
format_name = "edax_h5ebsd"
description = (
    "Read support for electron backscatter diffraction patterns stored "
    "in an HDF5 file formatted in EDAX TSL's h5ebsd format, similar to "
    "the format described in Jackson et al.: h5ebsd: an archival data "
    "format for electron back-scatter diffraction data sets. "
    "Integrating Materials and Manufacturing Innovation 2014 3:4, doi: "
    "https://dx.doi.org/10.1186/2193-9772-3-4."
)
full_support = False
# Recognised file extension
file_extensions = ["h5", "hdf5", "h5ebsd"]
default_extension = 0
# Writing capabilities (signal dimensions, navigation dimensions)
writes = False

# Unique HDF5 footprint
footprint = ["manufacturer", "version"]
manufacturer = "edax"


class EDAXH5EBSDReader(H5EBSDReader):
    """EDAX TSL h5ebsd file reader.

    The file contents are ment to be used for initializing a
    :class:`~kikuchipy.signals.EBSD` signal.

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
        if "SEM-PRIAS Images" in group.keys():
            sd = _hdf5group2dict(group["SEM-PRIAS Images"])
        else:
            sd = {}

        # Ensure file can be read
        grid_type = hd.get("Grid Type")
        if grid_type != "SqrGrid":
            raise IOError(f"Only square grids are supported, not {grid_type}")

        # Get data shapes
        ny, nx = hd["nRows"], hd["nColumns"]
        sy, sx = hd["Pattern Height"], hd["Pattern Width"]
        dy, dx = hd.get("Step Y", 1), hd.get("Step X", 1)
        px_size = 1.0

        # --- Metadata
        fname, title = self.get_metadata_filename_title(group.name)
        metadata = {
            "Acquisition_instrument": {
                "SEM": {
                    "working_distance": hd.get("Working Distance"),
                    "magnification": sd.get("Header", {}).get("Mag"),
                },
            },
            "General": {"original_filename": fname, "title": title},
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
        # TODO: Implement reader of EDAX h5ebsd crystal maps in orix
        xmap = CrystalMap.empty(shape=(ny, nx), step_sizes=(dy, dx))
        scan_dict["xmap"] = xmap

        # --- Detector
        scan_dict["detector"] = EBSDDetector(
            shape=(sy, sx),
            px_size=px_size,
            tilt=hd.get("Camera Elevation Angle", 0),
            azimuthal=hd.get("Camera Azimuthal Angle", 0),
            sample_tilt=hd.get("Sample Tilt", 70),
            pc=(
                hd.get("Pattern Center Calibration", {}).get("x-star", 0.5),
                hd.get("Pattern Center Calibration", {}).get("y-star", 0.5),
                hd.get("Pattern Center Calibration", {}).get("z-star", 0.5),
            ),
            convention="edax",
        )

        return scan_dict


def file_reader(
    filename: Union[str, Path],
    scan_group_names: Union[None, str, List[str]] = None,
    lazy: bool = False,
    **kwargs,
) -> List[dict]:
    """Read electron backscatter diffraction patterns, a crystal map,
    and an EBSD detector from an EDAX h5ebsd file
    :cite:`jackson2014h5ebsd`.

    Not meant to be used directly; use :func:`~kikuchipy.load`.

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
    reader = EDAXH5EBSDReader(filename, **kwargs)
    return reader.read(scan_group_names, lazy)
