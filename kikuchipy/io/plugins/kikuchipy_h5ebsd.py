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

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import warnings

import h5py
from hyperspy.io_plugins.hspy import overwrite_dataset
import numpy as np
from orix.crystal_map import CrystalMap
from orix.io.plugins.orix_hdf5 import crystalmap2dict, dict2crystalmap
from orix import __version__ as orix_version

from kikuchipy.detectors import EBSDDetector
from kikuchipy.io.plugins._h5ebsd import _dict2hdf5group, _hdf5group2dict, H5EBSDReader
from kikuchipy.io._util import _get_input_variable
from kikuchipy.release import version as kikuchipy_version
from kikuchipy.signals.util._crystal_map import _crystal_map_is_compatible_with_signal


__all__ = ["file_reader", "file_writer"]


# Plugin characteristics
# ----------------------
format_name = "kikuchipy_h5ebsd"
description = (
    "Read/write support for electron backscatter diffraction patterns "
    "stored in an HDF5 file formatted in kikuchipy's h5ebsd format, "
    "similar to the format described in Jackson et al.: h5ebsd: an "
    "archival data format for electron back-scatter diffraction data "
    "sets. Integrating Materials and Manufacturing Innovation 2014 3:4,"
    " doi: https://dx.doi.org/10.1186/2193-9772-3-4."
)
full_support = True
# Recognised file extension
file_extensions = ["h5", "hdf5", "h5ebsd"]
default_extension = 1
# Writing capabilities (signal dimensions, navigation dimensions)
writes = [(2, 2), (2, 1), (2, 0)]

# Unique HDF5 footprint
footprint = ["manufacturer", "version"]
manufacturer = "kikuchipy"


class KikuchipyH5EBSDReader(H5EBSDReader):
    """kikuchipy h5ebsd file reader.

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
        """
        hd = _hdf5group2dict(group["EBSD/Header"], recursive=True)

        # Note: When written, these were obtained from the
        # `axes_manager` attribute, not the `xmap` one
        ny, nx = hd["n_rows"], hd["n_columns"]
        sy, sx = hd["pattern_height"], hd["pattern_width"]
        dy, dx = hd.get("step_y", 1), hd.get("step_x", 1)
        px_size = hd.get("detector_pixel_size", 1)

        # --- Metadata
        fname, title = self.get_metadata_filename_title(group.name)
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
        if "CrystalMap" in group["EBSD"]:
            xmap_dict = _hdf5group2dict(
                group["EBSD/CrystalMap/crystal_map"], recursive=True
            )
            # TODO: Remove once orix v0.11.0 is released
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "Argument `z`", np.VisibleDeprecationWarning
                )
                xmap = dict2crystalmap(xmap_dict)
        else:
            xmap = CrystalMap.empty(shape=(ny, nx), step_sizes=(dy, dx))
        scan_dict["xmap"] = xmap

        # --- Static background
        scan_dict["static_background"] = hd.get("static_background")

        # --- Detector
        pc = np.column_stack(
            (hd.get("pcx", 0.5), hd.get("pcy", 0.5), hd.get("pcz", 0.5))
        )
        if pc.size > 3:
            pc = pc.reshape((ny, nx, 3))
        scan_dict["detector"] = EBSDDetector(
            shape=(sy, sx),
            px_size=px_size,
            binning=hd.get("binning", 1),
            tilt=hd.get("elevation_angle", 0),
            azimuthal=hd.get("azimuth_angle", 0),
            sample_tilt=hd.get("sample_tilt", 70),
            pc=pc,
        )

        return scan_dict


def file_reader(
    filename: Union[str, Path],
    scan_group_names: Union[None, str, List[str]] = None,
    lazy: bool = False,
    **kwargs,
) -> List[dict]:
    """Read electron backscatter diffraction patterns, a crystal map,
    and an EBSD detector from a kikuchipy h5ebsd file
    :cite:`jackson2014h5ebsd`.

    Not ment to be used directly; use :func:`~kikuchipy.load`.

    The file is closed after reading if ``lazy=False``.

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


class KikuchipyH5EBSDWriter:
    """kikuchipy h5ebsd file writer.

    Parameters
    ----------
    filename
        Full file path of the HDF5 file.
    signal : kikuchipy.signals.EBSD
        EBSD signal.
    add_scan
        Whether to add the signal to the file if it exists and is
        closed. If the file exists but this is not ``True``, the file
        will be overwritten.
    """

    def __init__(
        self, filename: Union[str, Path], signal, add_scan: Optional[bool] = None
    ):
        self.filename = filename
        self.signal = signal
        self.file_exists = os.path.isfile(filename)
        self.add_scan = add_scan

        if self.file_exists and self.add_scan:
            mode = "r+"
        else:
            mode = "w"
            self.file_exists = False
        try:
            self.file = h5py.File(self.filename, mode=mode)
        except OSError:
            raise OSError("Cannot write to an already open file")

        self.scan_groups = self.get_scan_groups()

        if self.file_exists:
            self.check_file()

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.filename}"

    @property
    def data_shape_scale(self) -> Tuple[tuple, tuple]:
        """Return the shape and scale of the data to write to file."""
        data_shape = [1] * 4  # (ny, nx, sy, sx)
        data_scales = [1] * 3  # (dy, dx, px_size)
        am = self.signal.axes_manager
        nav_axes = am.navigation_axes
        nav_dim = am.navigation_dimension
        if nav_dim == 1:
            nav_axis = nav_axes[0]
            if nav_axis.name == "y":
                data_shape[0] = nav_axis.size
                data_scales[0] = nav_axis.scale
            else:  # nav_axis.name == "x", or something else
                data_shape[1] = nav_axis.size
                data_scales[1] = nav_axis.scale
        elif nav_dim == 2:
            data_shape[:2] = [a.size for a in nav_axes][::-1]
            data_scales[:2] = [a.scale for a in nav_axes[::-1]]
        data_shape[2:] = am.signal_shape[::-1]
        data_scales[2] = am.signal_axes[0].scale
        return tuple(data_shape), tuple(data_scales)

    @property
    def scan_group_names(self) -> List[str]:
        """Return a list of available scan group names."""
        return [group.name.lstrip("/") for group in self.scan_groups]

    def check_file(self):
        """Check if the file, if it is old, is a valid kikuchipy h5ebsd
        file.

        Raises
        ------
        IOError
            If the file was not created with kikuchipy, or if there are
            no groups in the top group containing the datasets
            ``"EBSD/Data"`` and ``"EBSD/Header"``.
        """
        error = None
        top_groups = _hdf5group2dict(self.file["/"])
        if top_groups.get("manufacturer") != "kikuchipy":
            error = "it was not created with kikuchipy"
        elif not any(
            "EBSD/Data" in group and "EBSD/Header" in group
            for group in self.scan_groups
        ):
            error = (
                "no top groups with subgroup name 'EBSD' with subgroups 'Data' and"
                "'Header' were found"
            )
        if error is not None:
            raise IOError(
                f"{self.filename} is not a supported kikuchipy h5ebsd file, as {error}"
            )

    def get_scan_groups(self) -> list:
        """Return a list of groups with scans."""
        scan_groups = []
        if self.file_exists:
            for key in self.file.keys():
                if isinstance(self.file[key], h5py.Group) and key.startswith("Scan"):
                    scan_groups.append(self.file[key])
        return scan_groups

    def get_valid_scan_number(self, scan_number: int = 1) -> int:
        """Return a valid scan number.

        Parameters
        ----------
        scan_number
            Scan number in the file, e.g. 1 for "Scan 1" (default).

        Returns
        -------
        valid_scan_number
            ``scan_number`` or a new valid number if an existing scan in
            the file has this number.

        Raises
        ------
        IOError
            If asked for a new scan number and that scan number is not
            a number.
        """
        scan_nos = [int(name.split()[-1]) for name in self.scan_group_names]
        for i in scan_nos:
            if i == scan_number:
                q = f"Scan {i} already in file, enter another scan number:\n"
                scan_number = _get_input_variable(q, int)
                if scan_number in [None, i]:
                    raise IOError("Invalid scan number")
        return scan_number

    def get_xmap_dict(self) -> dict:
        """Return a dictionary produced from :attr:`signal.xmap` or an
        empty one.
        """
        (ny, nx, *_), (dy, dx, _) = self.data_shape_scale
        xmap = self.signal.xmap
        if xmap is None or not _crystal_map_is_compatible_with_signal(
            xmap, self.signal.axes_manager.navigation_axes[::-1]
        ):
            xmap = CrystalMap.empty(shape=(ny, nx), step_sizes=(dy, dx))
        return crystalmap2dict(xmap)

    def write(self, scan_number: int = 1, **kwargs):
        """Write an :class:`~kikuchipy.signals.EBSD` to file.

        The file is closed after writing.

        Parameters
        ----------
        scan_number
            Scan number in the file, e.g. "Scan 1" (default).
        **kwargs
            Keyword arguments passed to
            :meth:`h5py.Group.require_dataset`.

        Raises
        ------
        ValueError
            If the file exists but ``add_scan`` is ``None`` or
            ``False``.
        """
        if self.file_exists:
            valid_scan_number = self.get_valid_scan_number(scan_number)
        else:
            valid_scan_number = scan_number
            _dict2hdf5group(
                {"manufacturer": "kikuchipy", "version": kikuchipy_version},
                self.file["/"],
                **kwargs,
            )
        group = self.file.create_group(f"Scan {valid_scan_number}")

        # --- Patterns
        (ny, nx, sy, sx), (dy, dx, px_size) = self.data_shape_scale
        overwrite_dataset(
            group.create_group("EBSD/Data"),
            data=self.signal.data.reshape(ny * nx, sy, sx),
            key="patterns",
            signal_axes=(2, 1),
            **kwargs,
        )

        # --- Crystal map
        _dict2hdf5group(
            {
                "manufacturer": "orix",
                "version": orix_version,
                "crystal_map": self.get_xmap_dict(),
            },
            group.create_group("EBSD/CrystalMap"),
            **kwargs,
        )

        # --- Header
        static_bg = self.signal.static_background
        if static_bg is None:
            static_bg = -1
        detector = self.signal.detector
        _dict2hdf5group(
            {
                "azimuth_angle": detector.azimuthal,
                "binning": detector.binning,
                "elevation_angle": detector.tilt,
                "n_columns": nx,
                "n_rows": ny,
                "pattern_width": sx,
                "pattern_height": sy,
                "pcx": detector.pcx,
                "pcy": detector.pcy,
                "pcz": detector.pcz,
                "detector_pixel_size": detector.px_size,
                "sample_tilt": detector.sample_tilt,
                "static_background": static_bg,
                "step_x": dx,
                "step_y": dy,
            },
            group.create_group("EBSD/Header"),
            **kwargs,
        )

        # --- SEM Header
        md = self.signal.metadata.as_dictionary()
        md_sem = md.get("Acquisition_instrument", {}).get("SEM", {})
        _dict2hdf5group(
            {
                "beam_energy": md_sem.get("beam_energy", 0),
                "magnification": md_sem.get("magnification", 0),
                "microscope": md_sem.get("microscope", ""),
                "working_distance": md_sem.get("working_distance", 0),
            },
            group.create_group("SEM/Header"),
            **kwargs,
        )

        self.file.close()


def file_writer(
    filename: str,
    signal,
    add_scan: Optional[bool] = None,
    scan_number: int = 1,
    **kwargs,
):
    """Write an :class:`~kikuchipy.signals.EBSD` or
    :class:`~kikuchipy.signals.LazyEBSD` signal to an existing but not
    open or new h5ebsd file.

    Not meant to be used directly; use
    :func:`~kikuchipy.signals.EBSD.save`.

    The file is closed after writing.

    Parameters
    ----------
    filename
        Full path of HDF5 file.
    signal : kikuchipy.signals.EBSD or kikuchipy.signals.LazyEBSD
        Signal instance.
    add_scan
        Add signal to an existing, but not open, h5ebsd file. If it does
        not exist it is created and the signal is written to it.
    scan_number
        Scan number in name of HDF dataset when writing to an existing,
        but not open, h5ebsd file.
    **kwargs
        Keyword arguments passed to :meth:`h5py.Group.require_dataset`.
    """
    writer = KikuchipyH5EBSDWriter(filename, signal, add_scan)
    writer.write(scan_number, **kwargs)
