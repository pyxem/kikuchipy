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

"""Reader of EBSD data from an Oxford Instruments h5ebsd (H5OINA) file."""

import logging
from pathlib import Path

import h5py
import numpy as np
from orix.crystal_map import CrystalMap

from kikuchipy.detectors.ebsd_detector import EBSDDetector
from kikuchipy.io.plugins._h5ebsd import H5EBSDReader, _hdf5group2dict

_logger = logging.getLogger(__name__)


class OxfordH5EBSDReader(H5EBSDReader):
    """Oxford Instruments h5ebsd (H5OINA) file reader.

    The file contents are meant to be used for initializing a
    :class:`~kikuchipy.signals.EBSD` signal.

    Parameters
    ----------
    filename
        Full file path of the HDF5 file.
    **kwargs
        Keyword arguments passed to :class:`h5py.File`.
    """

    pattern_dataset_names = ["Unprocessed Patterns", "Processed Patterns"]

    def __init__(self, filename: str | Path, processed: bool, **kwargs) -> None:
        super().__init__(filename, **kwargs)
        self._patterns_name = self.pattern_dataset_names[int(processed)]

    @property
    def patterns_name(self) -> str:
        return self._patterns_name

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
        """
        header_group = _hdf5group2dict(group["EBSD/Header"], recursive=True)
        data_group = _hdf5group2dict(
            group["EBSD/Data"], data_dset_names=self.pattern_dataset_names
        )

        # Get data shapes
        ny, nx = header_group["Y Cells"], header_group["X Cells"]
        sy, sx = header_group["Pattern Height"], header_group["Pattern Width"]
        dy, dx = header_group.get("Y Step", 1), header_group.get("X Step", 1)
        px_size = 1.0

        # --- Metadata
        fname, title = self.get_metadata_filename_title(group.name)
        metadata = {
            "Acquisition_instrument": {
                "SEM": {
                    "beam_energy": header_group.get("Beam Voltage"),
                    "magnification": header_group.get("Magnification"),
                    "working_distance": header_group.get("Working Distance"),
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
        scan_dict["original_metadata"].update(header_group)

        # --- Crystal map
        # TODO: Implement reader of Oxford Instruments h5ebsd crystal
        #  maps in orix
        xmap = CrystalMap.empty(shape=(ny, nx), step_sizes=(dy, dx))
        scan_dict["xmap"] = xmap

        # --- Static background
        scan_dict["static_background"] = header_group.get("Processed Static Background")

        # --- Detector
        pc = np.column_stack(
            (
                data_group.get("Pattern Center X", 0.5),
                data_group.get("Pattern Center Y", 0.5),
                data_group.get("Detector Distance", 0.5),
            )
        )
        if pc.size > 3:
            pc = pc.reshape((ny, nx, 3))
        detector_kw = dict(
            shape=(sy, sx),
            pc=pc,
            sample_tilt=np.rad2deg(header_group.get("Tilt Angle", np.deg2rad(70))),
            convention="oxford",
        )
        detector_tilt_euler = header_group.get("Detector Orientation Euler")
        try:
            detector_kw["tilt"] = np.rad2deg(detector_tilt_euler[1]) - 90
        except (IndexError, TypeError):  # pragma: no cover
            _logger.debug("Could not read detector tilt")
        binning_str = header_group.get("Camera Binning Mode")
        try:
            detector_kw["binning"] = int(binning_str.split("x")[0])
        except (IndexError, ValueError):  # pragma: no cover
            _logger.debug("Could not read detector binning")
        scan_dict["detector"] = EBSDDetector(**detector_kw)

        return scan_dict


def file_reader(
    filename: str | Path,
    scan_group_names: str | list[str] | None = None,
    lazy: bool = False,
    processed: bool = True,
    **kwargs,
) -> list[dict]:
    """Read electron backscatter diffraction patterns, a crystal map,
    and an EBSD detector from an Oxford Instruments h5ebsd (H5OINA) file
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
        is False.
    processed
        Whether to read processed patterns. Default is True. If False,
        try to read unprocessed patterns if available.
    **kwargs
        Keyword arguments passed to :class:`h5py.File`.

    Returns
    -------
    scan_dict_list
        List of one or more dictionaries with the keys "axes", "data",
        "metadata", "original_metadata", "detector", and "xmap". This
        dictionary can be passed as keyword arguments to create an
        :class:`~kikuchipy.signals.EBSD` signal.

    Raises
    ------
    KeyError
        If *processed* is False and unprocessed patterns are not
        available.
    """
    reader = OxfordH5EBSDReader(filename, processed, **kwargs)
    return reader.read(scan_group_names, lazy)
