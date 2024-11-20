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

"""Reader of EBSD data from a Bruker Nano h5ebsd file."""

from pathlib import Path

import h5py
import numpy as np
from orix.crystal_map import CrystalMap

from kikuchipy.detectors.ebsd_detector import EBSDDetector
from kikuchipy.io.plugins._h5ebsd import H5EBSDReader, _hdf5group2dict


class BrukerH5EBSDReader(H5EBSDReader):
    """Bruker Nano h5ebsd file reader.

    Parameters
    ----------
    filename
        Full file path of the HDF5 file.
    **kwargs
        Keyword arguments passed to :class:`h5py.File`.

    Notes
    -----
    The file contents are meant to be used for initializing a
    :class:`~kikuchipy.signals.EBSD` signal.
    """

    def __init__(self, filename: str, **kwargs) -> None:
        super().__init__(filename, **kwargs)

    @property
    def patterns_name(self) -> str:
        return "RawPatterns"

    def scan2dict(self, group: h5py.Group, lazy: bool = False) -> dict:
        """Read (possibly lazily) patterns from group.

        Parameters
        ----------
        group
            Group with patterns.
        lazy
            Whether to read dataset lazily. Default is False.

        Returns
        -------
        scan_dict
            Dictionary with keys "axes", "data", "metadata",
            "original_metadata", "detector", "static_background", and
            "xmap". This dictionary can be passed as keyword arguments
            to create an :class:`~kikuchipy.signals.EBSD` signal.

        Raises
        ------
        IOError
            If patterns are not acquired in a square grid.
        KeyError
            If patterns cannot be found in the expected dataset.
        ValueError
            If a non-rectangular region of interest is used.

        Warns
        -----
        UserWarning
            If pattern array is smaller than the data shape determined
            from other datasets in the file.
        """
        header_dset = _hdf5group2dict(group["EBSD/Header"], recursive=True)
        data_dset = _hdf5group2dict(
            group["EBSD/Data"], data_dset_names=[self.patterns_name]
        )

        # Ensure file can be read
        grid_type = header_dset.get("Grid Type")
        if grid_type != "isometric":
            raise IOError(f"Only square grids are supported, not {grid_type!r}")

        # Get region of interest (ROI, only rectangular shape supported)
        indices = None
        try:
            sd = _hdf5group2dict(group["EBSD/SEM"])
            iy = sd["IY"][()]
            ix = sd["IX"][()]
            roi = True
        except KeyError:
            ny = header_dset["NROWS"]
            nx = header_dset["NCOLS"]
            roi = False
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
        sy, sx = header_dset["PatternHeight"], header_dset["PatternWidth"]
        dy, dx = header_dset["YSTEP"], header_dset["XSTEP"]
        px_size = header_dset.get("DetectorFullHeightMicrons", 1) / header_dset.get(
            "UnClippedPatternHeight", 1
        )

        # --- Metadata
        fname, title = self.get_metadata_filename_title(group.name)
        metadata = {
            "Acquisition_instrument": {
                "SEM": {
                    "beam_energy": header_dset.get("KV"),
                    "magnification": header_dset.get("Magnification"),
                    "working_distance": header_dset.get("WD"),
                },
            },
            "General": {
                "original_filename": header_dset.get("OriginalFile", fname),
                "title": title,
            },
            "Signal": {"signal_type": "EBSD", "record_by": "image"},
        }
        scan_dict = {"metadata": metadata}

        data = self.get_data(
            group,
            data_shape=(ny, nx, sy, sx),
            lazy=lazy,
            indices=indices,
        )
        scan_dict["data"] = data

        scan_dict["axes"] = self.get_axes_list((ny, nx, sy, sx), (dy, dx, px_size))

        scan_dict["original_metadata"] = {
            "manufacturer": self.manufacturer,
            "version": self.version,
        }
        scan_dict["original_metadata"].update(header_dset)

        # TODO: Use reader from orix
        xmap = CrystalMap.empty(shape=(ny, nx), step_sizes=(dy, dx))
        scan_dict["xmap"] = xmap

        scan_dict["static_background"] = header_dset.get("StaticBackground")

        pc = np.column_stack(
            (
                data_dset.get("PCX", 0.5),
                data_dset.get("PCY", 0.5),
                data_dset.get("DD", 0.5),
            )
        )
        if pc.size > 3:
            pc = pc.reshape((ny, nx, 3))
        scan_dict["detector"] = EBSDDetector(
            shape=(sy, sx),
            px_size=px_size,
            tilt=header_dset.get("CameraTilt", 0.0),
            sample_tilt=header_dset.get("Sample Tilt", 0.0),
            pc=pc,
        )

        return scan_dict


def _bruker_roi_is_rectangular(iy: int, ix: int) -> tuple[int, int, bool]:
    iy_unique, iy_unique_counts = np.unique(iy, return_counts=True)
    ix_unique, ix_unique_counts = np.unique(ix, return_counts=True)
    is_rectangular = bool(
        np.all(np.diff(np.sort(iy_unique)) == 1)
        and np.all(np.diff(np.sort(ix_unique)) == 1)
        and np.unique(iy_unique_counts).size == 1
        and np.unique(ix_unique_counts).size == 1
    )
    iy2 = np.max(iy) - np.min(iy) + 1
    ix2 = np.max(ix) - np.min(ix) + 1
    return iy2, ix2, is_rectangular


def file_reader(
    filename: str | Path,
    scan_group_names: str | list[str] | None = None,
    lazy: bool = False,
    **kwargs,
) -> list[dict]:
    """Return electron backscatter diffraction patterns, a crystal map,
    and an EBSD detector from a Bruker h5ebsd file
    :cite:`jackson2014h5ebsd`.

    Not meant to be used directly; use :func:`~kikuchipy.load` instead.

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
        is False.
    **kwargs
        Keyword arguments passed to :class:`h5py.File`.

    Returns
    -------
    scan_dict_list
        List of one or more dictionaries with the keys "axes", "data",
        "metadata", "original_metadata", "detector",
        "static_background", and "xmap". This dictionary can be passed
        as keyword arguments to create an
        :class:`~kikuchipy.signals.EBSD` signal.
    """
    reader = BrukerH5EBSDReader(filename, **kwargs)
    return reader.read(scan_group_names, lazy)
