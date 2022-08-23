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

import dask.array as da
import h5py
from hyperspy.io_plugins.hspy import overwrite_dataset
import numpy as np
from orix.crystal_map import CrystalMap
from orix.io.plugins.orix_hdf5 import crystalmap2dict
from orix import __version__ as orix_version

from kikuchipy.detectors import EBSDDetector
from kikuchipy.io.plugins._h5ebsd import _dict2hdf5group, _hdf5group2dict, H5EBSDReader
from kikuchipy.io._util import _get_input_variable
from kikuchipy.release import version as kikuchipy_version
from kikuchipy.signals.util._crystal_map import _crystal_map_is_compatible_with_signal

__all__ = ["file_reader"]


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


class KikuchipyH5EBSDWriter:
    """kikuchipy h5ebsd file writer.

    Parameters
    ----------
    filename
        Full file path of the HDF5 file.
    signal : kikuchipy.signals.EBSD
        EBSD signal.
    """

    def __init__(self, filename: str, signal):
        self.filename = filename
        self.signal = signal
        self.file_exists = os.path.isfile(filename)

        if self.file_exists:
            mode = "r+"
        else:
            mode = "w"
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
        if not any(
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
                if isinstance(self.file[key], h5py.Group) and key.startswith("/Scan"):
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

    def write(self, add_scan: Optional[bool] = None, scan_number: int = 1, **kwargs):
        """Write an :class:`~kikuchipy.signals.EBSD` to file.

        Parameters
        ----------
        add_scan
            If the file to write to already exists, this must be
            ``True``.
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
        if self.file_exists and add_scan in [None, False]:
            raise ValueError("Set `add_scan=True` to write to an existing file")

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
        (ny, nx, sy, sx), _ = self.data_shape_scale
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
        detector = self.signal.detector
        _dict2hdf5group(
            {
                "azimuthal": detector.azimuthal,
                "binning": detector.binning,
                "px_size": detector.px_size,
                "tilt": detector.tilt,
                "sample_tilt": detector.sample_tilt,
                "static_background": self.signal.static_background,
                "pcx": detector.pcx,
                "pcy": detector.pcy,
                "pcz": detector.pcz,
            },
            group.create_group("EBSD/Header"),
            **kwargs,
        )

        # --- SEM Header
        md_sem = self.signal.metadata.get_item("Acquisition_instrument.SEM")
        md_sem = md_sem.as_dictionary()
        _dict2hdf5group(
            {
                "beam_energy": md_sem.get("beam_energy"),
                "magnification": md_sem.get("magnification"),
                "microscope": md_sem.get("microscope"),
                "working_distance": md_sem.get("working_distance"),
            },
            group.create_group("SEM/Header"),
            **kwargs,
        )

        self.file.close()
