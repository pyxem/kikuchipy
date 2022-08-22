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

"""Generic, private parent class for all h5ebsd file plugins."""

import os
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
import warnings

import dask.array as da
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.io_plugins.hspy import overwrite_dataset
import h5py
import numpy as np
from orix import __version__ as orix_ver
from orix.crystal_map import CrystalMap
from orix.io.plugins.orix_hdf5 import crystalmap2dict, dict2hdf5group

from kikuchipy.detectors import EBSDDetector
from kikuchipy.io._util import (
    _get_input_variable,
    _get_nested_dictionary,
    _delete_from_nested_dictionary,
)
from kikuchipy.signals.util._metadata import (
    ebsd_metadata,
    metadata_nodes,
    _phase_metadata,
    _update_phase_info,
)


__all__ = ["hdf5group2dict", "H5EBSDReader", "H5EBSDWriter"]


def hdf5group2dict(
    group: h5py.Group,
    dictionary: Union[None, dict] = None,
    recursive: bool = False,
    data_dset_names: Optional[list] = None,
) -> dict:
    """Return a dictionary with values from datasets in a group.

    Parameters
    ----------
    group
        HDF5 group object.
    dictionary
        To fill dataset values into.
    recursive
        Whether to add subgroups to ``dictionary`` (default is
        ``False``).
    data_dset_names
        List of names of HDF5 data sets with data to not read.

    Returns
    -------
    dictionary
        Dataset values in group (and subgroups if ``recursive=True``).
    """
    if data_dset_names is None:
        data_dset_names = []
    if dictionary is None:
        dictionary = {}
    for key, val in group.items():
        # Prepare value for entry in dictionary
        if isinstance(val, h5py.Dataset):
            if key not in data_dset_names:
                val = val[()]
            if isinstance(val, np.ndarray) and len(val) == 1:
                val = val[0]
                key = key.lstrip()  # EDAX has some leading whitespaces
            if isinstance(val, bytes):
                val = val.decode("latin-1")
        # Check whether to extract subgroup or write value to dictionary
        if isinstance(val, h5py.Group) and recursive:
            dictionary[key] = {}
            hdf5group2dict(
                group=group[key],
                dictionary=dictionary[key],
                data_dset_names=data_dset_names,
                recursive=recursive,
            )
        elif key in data_dset_names:
            pass
        else:
            dictionary[key] = val
    return dictionary


class H5EBSDReader:
    """Generic h5ebsd file reader to be extended in manufacturer
    specific readers.

    Parameters
    ----------
    filename
        Full file path of the HDF5 file.
    **kwargs
        Keyword arguments passed to :class:`h5py.File`.
    """

    manufacturer_patterns = {
        "bruker nano": "RawPatterns",
        "edax": "Pattern",
        "kikuchipy": "patterns",
    }

    def __init__(self, filename: str, **kwargs):
        self.filename = filename
        self.file = h5py.File(filename, **kwargs)
        self.scan_groups = self.get_scan_groups()
        self.manufacturer, self.version = self.check_file()
        self.patterns_name = self.manufacturer_patterns[self.manufacturer]

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.filename}"

    def check_file(self) -> Tuple[str, str]:
        """Check if the HDF5 file is a valid h5ebsd file by searching
        for datasets containing manufacturer, version and scans in the
        top group.

        Raises
        ------
        IOError
            If there are no datasets in the top group named
            ``"manufacturer"`` and ``"version"``.
        IOError
            If there are no groups in the top group containing the
            datasets ``"EBSD/Data"`` and ``"EBSD/Header"``.
        IOError
            If there is no reader for the file ``"manufacturer"``.

        Returns
        -------
        manufacturer
            File manufacturer.
        version
            File version.
        """
        error = None
        if len(self.scan_groups) != len(self.file["/"].keys()) - 2:
            error = "manufacturer and/or version could not be read from its top group"
        if not any(
            "EBSD/Data" in group and "EBSD/Header" in group
            for group in self.scan_groups
        ):
            error = (
                "no top groups with subgroup name 'EBSD' with subgroups 'Data' and "
                "'Header' was detected"
            )
        man, ver = self.get_manufacturer_version()
        man = man.lower()
        supported_manufacturers = list(self.manufacturer_patterns.keys())
        if man not in supported_manufacturers:
            error = (
                f"'{man}' is not among supported manufacturers "
                f"{supported_manufacturers}"
            )
        if error is not None:
            raise IOError(f"{self.filename} is not a supported h5ebsd file, as {error}")
        else:
            return man, ver

    def get_manufacturer_version(self) -> Tuple[str, str]:
        """Get manufacturer and version from h5ebsd file.

        Returns
        -------
        manufacturer
            File manufacturer.
        version
            File version.
        """
        manufacturer = None
        version = None
        for key, val in hdf5group2dict(group=self.file["/"]).items():
            if key.lower() == "manufacturer":
                manufacturer = val
            elif key.lower() == "version":
                version = val
        return manufacturer, version

    def get_scan_groups(self) -> List[h5py.Group]:
        """Return a list of the scan group names from an h5ebsd file.

        These are all groups in the top group not named
        ``"manufacturer"`` or ``"version"``.

        Returns
        -------
        scan_groups
            List of available scan groups.
        """
        scan_groups = []
        for key in self.file["/"].keys():
            if key.lstrip().lower() not in ["manufacturer", "version"]:
                scan_groups.append(self.file[key])
        return scan_groups

    def get_desired_scan_groups(
        self, scan_group_names: Union[None, str, List[str]] = None
    ) -> List[h5py.Group]:
        """Return desired HDF5 groups with scans within them.

        Parameters
        ----------
        scan_group_names
            Name or a list of names of the desired top HDF5 group(s). If
            not given, the first scan group is returned.

        Returns
        -------
        scan_groups
            A list of the desired scan group(s).
        """
        # Get desired scan groups
        scan_groups = []
        if scan_group_names is None:  # Return the first scan group
            scan_groups.append(self.scan_groups[0])
        else:
            if isinstance(scan_group_names, str):
                scan_group_names = [scan_group_names]
            for desired_scan in scan_group_names:
                scan_is_here = False
                for scan in self.scan_groups:
                    if desired_scan == scan.name.lstrip("/"):
                        scan_groups.append(scan)
                        scan_is_here = True
                        break
                if not scan_is_here:
                    error_str = (
                        f"Scan '{desired_scan}' is not among the available scans "
                        f"{self.scan_groups} in '{self.filename}'."
                    )
                    if len(scan_group_names) == 1:
                        raise IOError(error_str)
                    else:
                        warnings.warn(error_str)
        return scan_groups

    def read(
        self,
        lazy: bool,
        scan_group_names: Union[None, str, List[str]] = None,
    ) -> List[dict]:
        """Return a list of dictionaries which can be used to create
        :class:`~kikuchipy.signals.EBSD` signals.

        Parameters
        ----------
        lazy
            Read dataset lazily.
        scan_group_names
            Name or a list of names of the desired top HDF5 group(s). If
            not given, the first scan group is returned.

        Returns
        -------
        scan_list
            List of dictionaries with keys ``"axes"``, ``"data"``,
            ``"metadata"``, ``"original_metadata"``, ``"detector"``,
            ``"static_background"``, and ``"xmap"``.
        """
        scan_dict_list = []
        for scan in self.get_desired_scan_groups(scan_group_names):
            pass
        return scan_dict_list


class H5EBSDWriter:
    manufacturer = ""
    version = ""

    def __init__(self, filename: str, signal):
        self.filename = filename
        self.signal = signal
        self.old_file = os.path.isfile(filename)

    @property
    def all_scan_groups(self) -> list:
        f = self.file
        scan_group_list = []
        for k in f["/"].keys():
            if "Scan" in k:
                scan_group_list.append(f[k])
        return scan_group_list

    @property
    def data_shape_out(self):
        data_shape = [1] * 4  # (ny, nx, sy, sx)
        am = self.signal.axes_manager
        nav_axes = am.navigation_axes
        nav_dim = am.navigation_dimension
        if nav_dim == 1:
            nav_axis = nav_axes[0]
            if nav_axis.name == "y":
                data_shape[0] = nav_axis.size
            else:  # nav_axis.name == "x", or something else
                data_shape[1] = nav_axis.size
        elif nav_dim == 2:
            data_shape[:2] = [a.size for a in nav_axes][::-1]
        data_shape[2:] = am.signal_shape[::-1]
        return data_shape

    @property
    def ebsd_metadata(self) -> dict:
        md = self.signal_metadata
        return md["Acquisition_instrument"]["SEM"]["Detector"]["EBSD"]

    @property
    def scan_number(self) -> list:
        scan_number_list = []
        for gr in self.all_scan_groups:
            gr_name = int(gr.name.split()[-1])
            scan_number_list.append(gr_name)
        return scan_number_list

    @property
    def sem_metadata(self) -> dict:
        md = self.signal_metadata
        return md["Acquisition_instrument"]["SEM"]

    @property
    def signal_metadata(self) -> dict:
        return self.signal.metadata.as_dictionary()

    def check_file(self):
        """Check if the HDF5 file is a valid h5ebsd file by searching
        for datasets containing manufacturer, version and scans in the
        top group.

        Raises an IOError if the file is not valid.
        """
        f = self.file
        top_groups = list(f["/"].keys())
        scan_groups = get_scan_groups(f)
        n_groups = len(top_groups)
        if len(scan_groups) != n_groups - 2:
            raise IOError(
                f"'{f.filename}' is not an h5ebsd file, as manufacturer and/or"
                " version could not be read from its top group"
            )
        if not any(
            "EBSD/Data" in group and "EBSD/Header" in group for group in scan_groups
        ):
            raise IOError(
                f"'{f.filename}' is not an h5ebsd file, as no top groups with "
                "subgroup name 'EBSD' with subgroups 'Data' and 'Header' was "
                "detected"
            )

    def get_ebsd_header_dict(self):
        md = self.ebsd_metadata
        det = self.signal.detector
        return dict(
            scan_time=md["scan_time"],
            static_background=md["static_background"],
            Detector=dict(
                azimuth_angle=0,
                binning=det.binning,
                exposure_time=md["exposure_time"],
                frame_number=md["frame_number"],
                frame_rate=md["frame_rate"],
                gain=md["gain"],
                name=md["detector"],
                pc=det.pc,
                px_size=det.px_size,
                sample_tilt=det.sample_tilt,
                tilt=det.tilt,
            ),
        )

    def get_sem_header_dict(self):
        md = self.sem_metadata
        return dict(
            beam_energy=md["beam_energy"],
            magnification=md["magnification"],
            microscope=md["microscope"],
            working_distance=md["working_distance"],
        )

    def open_file(self, add_scan: bool):
        if self.old_file and add_scan:
            mode = "r+"
        else:
            mode = "w"
        try:
            self.file = h5py.File(self.filename, mode=mode)
        except OSError:
            raise OSError("Cannot write to an already open file")

    def set_manufacturer(self):
        for k, v in hdf5group2dict(group=self.file["/"]).items():
            if k.lower() == "manufacturer":
                self.manufacturer = v
                break
        else:
            self.manufacturer = None

    def set_version(self):
        for k, v in hdf5group2dict(group=self.file["/"]).items():
            if k.lower() == "version":
                self.version = v
                break
        else:
            self.version = None

    def write(self, add_scan: Optional[bool] = None, scan_number: int = 1, **kwargs):
        self.open_file(add_scan=add_scan)
        if self.old_file and add_scan:
            # File exists, try to add to it
            self.check_file()
            self.set_manufacturer()
            self.set_version()
            file_manufacturer = self.manufacturer
            if file_manufacturer.lower() != "kikuchipy":
                self.file.close()
                raise IOError(
                    "Only writing to kikuchipy's (and not "
                    f"{file_manufacturer}'s) h5ebsd format is supported"
                )
            for i in self.scan_number:
                if i == scan_number:
                    q = f"Scan {i} already in file, enter another scan number:\n"
                    scan_number = _get_input_variable(q, int)
                    if scan_number is None:
                        raise IOError("Invalid scan number.")
        else:
            # File did not exist
            dict2h5ebsdgroup(
                dict(manufacturer=self.manufacturer, version=self.version),
                self.file["/"],
                **kwargs,
            )

        # Create scan group
        scan_group = self.file.create_group(f"Scan {scan_number}")

        # Write metadata to scan group
        dict2h5ebsdgroup(
            dict(
                EBSD=dict(Header=self.get_ebsd_header_dict()),
                SEM=dict(Header=self.get_sem_header_dict()),
            ),
            scan_group,
        )

        # Write signal to file
        man_pats = manufacturer_pattern_names()
        dset_pattern_name = man_pats["kikuchipy"]
        ny, nx, sy, sx = self.data_shape_out
        overwrite_dataset(
            scan_group.create_group("EBSD/Data"),
            self.signal.data.reshape(nx * ny, sy, sx),
            dset_pattern_name,
            signal_axes=(2, 1),
            **kwargs,
        )

        # Write crystallographic data to scan group
        xmap = self.signal.xmap
        if xmap is not None:
            from orix import __version__ as ver_orix

            dict2hdf5group(
                dict(manufacturer="orix", version=ver_orix, crystal_map=xmap),
                scan_group.create_group("EBSD/CrystalMap"),
            )

        self.file.close()
