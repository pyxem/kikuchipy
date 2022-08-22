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

import dask.array as da
import h5py
import numpy as np
from orix.crystal_map import CrystalMap

from kikuchipy.detectors import EBSDDetector
from kikuchipy.io.plugins._h5ebsd import hdf5group2dict, H5EBSDReader, H5EBSDWriter


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

    dset_name_patterns = "patterns"

    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)

    def scan2dict(self, scan_group: h5py.Group, lazy: bool = False) -> dict:
        """Read (possibly lazily) patterns from group.

        Parameters
        ----------
        scan_group
            HDF5 group with patterns.
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
        ebsd_header = hdf5group2dict(scan_group["EBSD/Header"], recursive=True)

        # --- Metadata
        title = (
            self.filename.split("/")[-1].split(".")[0]
            + " "
            + scan_group.name[1:].split("/")[0]
        )
        if len(title) > 20:
            title = f"{title:.20}..."
        metadata = {
            "General": {"title": title},
            "Acquisition_instrument": {
                "SEM": hdf5group2dict(
                    scan_group["SEM/Header"], data_dset_names=[self.dset_name_patterns]
                ),
            },
            "Signal": {"signal_type": "EBSD", "record_by": "image"},
        }
        scan_dict = {"metadata": metadata}

        # --- Data
        # Get HDF5 dataset with pattern array
        try:
            data_dset = scan_group["EBSD/Data/" + self.dset_name_patterns]
        except KeyError:
            raise KeyError(
                "Could not find patterns in the expected dataset "
                f"'EBSD/Data/{self.dset_name_patterns}'"
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
        # TODO: Make shape determination dependent on file version
        ny, nx = ebsd_header["n_rows"], ebsd_header["n_columns"]
        sy, sx = ebsd_header["pattern_height"], ebsd_header["pattern_width"]
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
        scales[0] *= ebsd_header["step_y"]
        scales[1] *= ebsd_header["step_x"]
        scales[2] *= ebsd_header["detector_pixel_size"]
        scales[3] *= ebsd_header["detector_pixel_size"]
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
        scan_dict["original_metadata"] = {}

        # --- Crystal map
        dset_required_data = [
            "x",
            "y",
            "z",
            "phi1",
            "Phi",
            "phi2",
            "phase_id",
            "id",
            "is_in_data",
        ]
        dset_required_header = [
            "grid_type",
            "nz",
            "ny",
            "nx",
            "z_step",
            "y_step",
            "x_step",
            "rotations_per_point",
            "scan_unit",
        ]

        # --- Static background

        # --- Detector

        return scan_dict


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
    """Read electron backscatter diffraction patterns and a crystal map
    from a kikuchipy h5ebsd file :cite:`jackson2014h5ebsd`.

    A valid h5ebsd file has at least one top group with the subgroup
    ``'EBSD'`` with the subgroups ``'Data'`` (patterns etc.) and
    ``'Header'`` (``metadata`` etc.).

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
        Data, axes, metadata and original metadata.
    """
    reader = KikuchipyH5EBSDReader(filename, **kwargs)

    return


def file_writer(
    filename: str,
    signal,
    add_scan: Optional[bool] = None,
    scan_number: int = 1,
    **kwargs,
):
    writer = H5EBSDWriter(filename=filename, signal=signal)
    writer.write(add_scan=add_scan, scan_number=scan_number, **kwargs)
