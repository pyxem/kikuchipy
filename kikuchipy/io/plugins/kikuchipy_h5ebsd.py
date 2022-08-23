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

"""Reader and writer of EBSD data from a kikuchipy h5ebsd file."""

from pathlib import Path
from typing import List, Optional, Union
import warnings

import dask.array as da
import h5py
import numpy as np
from orix.crystal_map import CrystalMap

from kikuchipy.detectors import EBSDDetector
from kikuchipy.io.plugins._h5ebsd import _hdf5group2dict, H5EBSDReader, H5EBSDWriter


__all__ = ["file_reader", "file_writer"]


class KikuchipyH5EBSDReader(H5EBSDReader):
    """kikuchipy h5ebsd file reader.

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
        """
        # Get data shape
        hd = _hdf5group2dict(group["EBSD/Header"], recursive=True)
        # TODO: Make shape determination dependent on file version
        ny, nx = hd["n_rows"], hd["n_columns"]
        sy, sx = hd["pattern_height"], hd["pattern_width"]
        dy, dx = hd.get("step_y", 1), hd.get("step_x", 1)
        px_size = hd.get("detector_pixel_size", 1)

        # --- Metadata
        fname = self.filename.split("/")[-1].split(".")[0]
        title = fname + " " + group.name[1:].split("/")[0]
        if len(title) > 20:
            title = f"{title:.20}..."
        metadata = {
            "Acquisition_instrument": {
                "SEM": _hdf5group2dict(
                    group["SEM/Header"], data_dset_names=[self.patterns_name]
                ),
            },
            "General": {"original_filename": fname, "title": title},
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
                f"Pattern size ({sx} x {sy}) and scan size ({nx} x {ny}) larger than file "
                "size. Will attempt to load by zero padding incomplete frames"
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
        # TODO: Make shape determination dependent on file version
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
        # TODO: Make crystal map creation dependent on file version
        xmap = CrystalMap.empty(shape=(ny, nx), step_sizes=(dy, dx))
        scan_dict["xmap"] = xmap

        # --- Static background
        scan_dict["static_background"] = hd.get("static_background")

        # --- Detector
        # TODO: Make detector creation dependent on file version
        scan_dict["detector"] = EBSDDetector(
            shape=(sy, sx),
            px_size=px_size,
            binning=hd.get("binning", 1),
            tilt=hd.get("elevation_angle", 0),
            azimuthal=hd.get("azimuth_angle", 0),
            sample_tilt=hd.get("sample_tilt", 70),
        )

        return scan_dict


#         dset_required_data = [
#             "x",
#             "y",
#             "z",
#             "phi1",
#             "Phi",
#             "phi2",
#             "phase_id",
#             "id",
#             "is_in_data",
#         ]
#         dset_required_header = [
#             "grid_type",
#             "nz",
#             "ny",
#             "nx",
#             "z_step",
#             "y_step",
#             "x_step",
#             "rotations_per_point",
#             "scan_unit",
#         ]


class KikuchipyH5EBSDWriter(H5EBSDWriter):
    from kikuchipy.release import version as ver_signal

    manufacturer = "kikuchipy"
    version = ver_signal


def file_reader(
    filename: Union[str, Path],
    scan_group_names: Union[None, str, List[str]] = None,
    lazy: bool = False,
    **kwargs,
) -> List[dict]:
    """Read electron backscatter diffraction patterns, a crystal map,
    and an EBSD detector from a kikuchipy h5ebsd file
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
        ``"detector"``, ``"static_background"``, and ``"xmap"``. This
        dictionary can be passed as keyword arguments to create an
        :class:`~kikuchipy.signals.EBSD` signal.
    """
    reader = KikuchipyH5EBSDReader(filename, **kwargs)
    return reader.read(scan_group_names, lazy)


# def file_writer(
#    filename: str,
#    signal,
#    add_scan: Optional[bool] = None,
#    scan_number: int = 1,
#    **kwargs,
# ):
#    writer = H5EBSDWriter(filename=filename, signal=signal)
#    writer.write(add_scan=add_scan, scan_number=scan_number, **kwargs)
