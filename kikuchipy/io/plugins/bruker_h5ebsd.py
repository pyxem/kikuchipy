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

"""Reader of EBSD data from a Bruker Nano h5ebsd file."""

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


class BrukerH5EBSDReader(H5EBSDReader):
    """Bruker Nano h5ebsd file reader.

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
        ValueError
            If a non-rectangular region of interest is used.
        """
        hd = _hdf5group2dict(group["EBSD/Header"], recursive=True)
        dd = _hdf5group2dict(group["EBSD/Data"], data_dset_names=self.patterns_name)

        # Ensure file can be read
        grid_type = hd.get("Grid Type")
        if grid_type != "isometric":
            raise IOError(f"Only square grids are supported, not {grid_type}")

        # Get region of interest (ROI, only rectangular shape supported)
        indices = None
        roi = False
        try:
            sd = _hdf5group2dict(group["EBSD/SEM"])
            iy = sd["IY"][()]
            ix = sd["IX"][()]
            roi = True
        except KeyError:
            ny = hd["NROWS"]
            nx = hd["NCOLS"]
        if roi:
            ny_roi, nx_roi, is_rectangular = _bruker_roi_is_rectangular(iy, ix)
            if is_rectangular:
                ny = ny_roi
                nx = nx_roi
                # Get indices of patterns in the 2D map
                idx = np.array([iy - iy.min(), ix - ix.min()])
                indices = np.ravel_multi_index(idx, (ny, nx)).argsort()
            else:
                raise ValueError("Only a rectangular region of interest is supported")

        # Get other data shapes
        sy, sx = hd["PatternHeight"], hd["PatternWidth"]
        dy, dx = hd["YSTEP"], hd["XSTEP"]
        px_size = hd.get("DetectorFullHeightMicrons", 1) / hd.get(
            "UnclippedPatternHeight", 1
        )

        # --- Metadata
        fname = self.filename.split("/")[-1].split(".")[0]
        title = fname + " " + group.name[1:].split("/")[0]
        if len(title) > 20:
            title = f"{title:.20}..."
        metadata = {
            "Acquisition_instrument": {
                "SEM": {
                    "beam_energy": hd.get("KV"),
                    "magnification": hd.get("Magnification"),
                    "working_distance": hd.get("WD"),
                },
            },
            "General": {
                "original_filename": hd.get("OriginalFile", fname),
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
            if indices is not None:
                data = data[indices]
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
        # TODO: Use reader from orix
        xmap = CrystalMap.empty(shape=(ny, nx), step_sizes=(dy, dx))
        scan_dict["xmap"] = xmap

        # --- Static background
        scan_dict["static_background"] = hd.get("StaticBackground")

        # --- Detector
        pc = np.column_stack(
            (dd.get("PCX", 0.5), dd.get("PCY", 0.5), dd.get("DD", 0.5))
        )
        if pc.size > 3:
            pc = pc.reshape((ny, nx, 3))
        scan_dict["detector"] = EBSDDetector(
            shape=(sy, sx),
            px_size=px_size,
            tilt=hd.get("CameraTilt", 0),
            sample_tilt=hd.get("Sample Tilt", 70),
            pc=pc,
        )

        return scan_dict


def _bruker_roi_is_rectangular(iy, ix):
    iy_unique, iy_unique_counts = np.unique(iy, return_counts=True)
    ix_unique, ix_unique_counts = np.unique(ix, return_counts=True)
    is_rectangular = (
        np.all(np.diff(np.sort(iy_unique)) == 1)
        and np.all(np.diff(np.sort(ix_unique)) == 1)
        and np.unique(iy_unique_counts).size == 1
        and np.unique(ix_unique_counts).size == 1
    )
    iy2 = np.max(iy) - np.min(iy) + 1
    ix2 = np.max(ix) - np.min(ix) + 1
    return iy2, ix2, is_rectangular


def file_reader(
    filename: Union[str, Path],
    scan_group_names: Union[None, str, List[str]] = None,
    lazy: bool = False,
    **kwargs,
) -> List[dict]:
    """Read electron backscatter diffraction patterns, a crystal map,
    and an EBSD detector from a Bruker h5ebsd file
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
    reader = BrukerH5EBSDReader(filename, **kwargs)
    return reader.read(scan_group_names, lazy)
