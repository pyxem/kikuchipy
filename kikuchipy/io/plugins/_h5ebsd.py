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

import abc
import os
from typing import Union, List, Tuple, Optional
import warnings

import dask.array as da
import h5py
import numpy as np


__all__ = ["_dict2hdf5group", "_hdf5group2dict", "H5EBSDReader"]


def _hdf5group2dict(
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
            _hdf5group2dict(
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


def _dict2hdf5group(dictionary: dict, group: h5py.Group, **kwargs):
    """Write a dictionary to datasets in a new group in an opened HDF5
    file format.

    Parameters
    ----------
    dictionary
        Dictionary with keys as dataset names.
    group
        HDF5 group to write dictionary to.
    **kwargs
        Keyword arguments passed to :meth:`h5py:Group.require_dataset`.
    """
    for key, val in dictionary.items():
        ddtype = type(val)
        dshape = (1,)
        if isinstance(val, dict):
            _dict2hdf5group(val, group.create_group(key), **kwargs)
            continue  # Jump to next item in dictionary
        elif isinstance(val, str):
            ddtype = "S" + str(len(val) + 1)
            val = val.encode()
        elif ddtype == np.dtype("O"):
            try:
                if isinstance(val, (np.ndarray, da.Array)):
                    ddtype = val.dtype
                else:
                    ddtype = val[0].dtype
                dshape = np.shape(val)
            except TypeError:
                warnings.warn(
                    "The HDF5 writer could not write the following (key, value) pair to"
                    f" file: ({key}, {val})"
                )
                break  # or continue?
        group.create_dataset(key, shape=dshape, dtype=ddtype, **kwargs)
        group[key][()] = val


class H5EBSDReader(abc.ABC):
    """Abstract class implementing a reader of an h5ebsd file in a
    format specific to each manufacturer.

    The file contents are ment to be used for initializing a
    :class:`~kikuchipy.signals.EBSD` signal.

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
        "oxford instruments": "Processed Patterns",
    }

    def __init__(self, filename: str, **kwargs):
        self.filename = filename
        self.file = h5py.File(filename, **kwargs)
        self.scan_groups = self.get_scan_groups()
        self.manufacturer, self.version = self.get_manufacturer_version()
        self.check_file()
        self.patterns_name = self.manufacturer_patterns[self.manufacturer]

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.version}): {self.filename}"

    @property
    def scan_group_names(self) -> List[str]:
        """Return a list of available scan group names."""
        return [group.name.lstrip("/") for group in self.scan_groups]

    def check_file(self):
        """Check if the file is a valid h5ebsd file by searching for
        datasets containing manufacturer, version and scans in the top
        group.

        Raises
        ------
        IOError
            If there are no datasets in the top group named
            ``"manufacturer"`` and ``"version"``, or if there are no
            groups in the top group containing the datasets
            ``"EBSD/Data"`` and ``"EBSD/Header"``, or if there is no
            reader for the file ``"manufacturer"``.
        """
        error = None
        if self.manufacturer is None or self.version is None:
            error = "manufacturer and/or version could not be read from its top group"
        elif not any(
            "EBSD/Data" in group and "EBSD/Header" in group
            for group in self.scan_groups
        ):
            error = (
                "no top groups with subgroup name 'EBSD' with subgroups 'Data' and "
                "'Header' were found"
            )
        man, ver = self.get_manufacturer_version()
        man = man.lower()
        supported_manufacturers = list(self.manufacturer_patterns.keys())
        if man not in supported_manufacturers and error is None:
            error = (
                f"'{man}' is not among supported manufacturers "
                f"{supported_manufacturers}"
            )
        if error is not None:
            raise IOError(f"{self.filename} is not a supported h5ebsd file, as {error}")

    def get_manufacturer_version(self) -> Tuple[str, str]:
        """Get manufacturer and version from the top group.

        Returns
        -------
        manufacturer
            File manufacturer.
        version
            File version.
        """
        manufacturer = None
        version = None
        for key, val in _hdf5group2dict(group=self.file["/"]).items():
            if key.lower() == "manufacturer":
                manufacturer = val.lower()
            elif key.lower() in ["version", "format version"]:
                version = val.lower()
        return manufacturer, version

    def get_scan_groups(self) -> List[h5py.Group]:
        """Return a list of the groups with scans.

        Assumes all top groups contain a scan.

        Returns
        -------
        scan_groups
            List of available scan groups.
        """
        scan_groups = []
        for key in self.file.keys():
            if isinstance(self.file[key], h5py.Group):
                scan_groups.append(self.file[key])
        return scan_groups

    def get_desired_scan_groups(
        self, group_names: Union[None, str, List[str]] = None
    ) -> List[h5py.Group]:
        """Return a list of the desired group(s) with scan(s).

        Parameters
        ----------
        group_names
            Name or a list of names of the desired top group(s). If not
            given, the first scan group is returned.

        Returns
        -------
        scan_groups
            A list of the desired scan group(s).
        """
        # Get desired scan groups
        scan_groups = []
        if group_names is None:  # Return the first scan group
            scan_groups.append(self.scan_groups[0])
        else:
            if isinstance(group_names, str):
                group_names = [group_names]
            for desired_name in group_names:
                scan_is_here = False
                for name, scan in zip(self.scan_group_names, self.scan_groups):
                    if desired_name == name:
                        scan_groups.append(scan)
                        scan_is_here = True
                        break
                if not scan_is_here:
                    error_str = (
                        f"Scan '{desired_name}' is not among the available scans "
                        f"{self.scan_group_names} in '{self.filename}'."
                    )
                    if len(group_names) == 1:
                        raise IOError(error_str)
                    else:
                        warnings.warn(error_str)
        return scan_groups

    def get_data(
        self,
        group: h5py.Group,
        data_shape: tuple,
        lazy: bool = False,
        indices: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, da.Array]:
        """Read and return patterns from file as a NumPy or Dask array.

        Parameters
        ----------
        group
            Group with patterns.
        data_shape
            Output shape of pattern array, (ny, nx, sy, sx) = (
            map rows, map columns, pattern rows, pattern columns).
        lazy
            Whether to read dataset lazily (default is ``False``).
        indices
            Mapping from pattern entry in the file to the 2D map, only
            used in the Bruker Nano reader.

        Returns
        -------
        data
            Patterns, possibly padded.

        Raises
        ------
        KeyError
            If patterns cannot be found in the expected dataset.

        Warns
        -----
        UserWarning
            If pattern array is smaller than the data shape determined
            from other datasets in the file.
        """
        ny, nx, sy, sx = data_shape

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

        return data

    @staticmethod
    def get_axes_list(data_shape: tuple, data_scale: tuple) -> List[dict]:
        """Return a description of each data axis.

        Parameters
        ----------
        data_shape
            4D shape of pattern array, ``(ny, nx, sy, sx)`` = (
            map rows, map columns, pattern rows, pattern columns).
        data_scale
            Map scale and detector pixel size, ``(dy, dx, px_size)``.

        Returns
        -------
        axes_list
            Description of each data axis as a list of dictionaries.
        """
        ny, nx, sy, sx = data_shape
        dy, dx, px_size = data_scale

        data_ndim = sum([ny != 1, nx != 1]) + 2

        units = ["um"] * 4
        scales = np.ones(4)

        # Calibrate scan dimension and detector dimension
        scales[0] *= dy
        scales[1] *= dx
        scales[2] *= px_size
        scales[3] *= px_size

        # Set axes names
        names = ["y", "x", "dy", "dx"]
        if data_ndim == 3:
            if ny > nx:
                names.remove("x")
                scales = np.delete(scales, 1)
                data_shape = np.delete(data_shape, 1)
            else:
                names.remove("y")
                scales = np.delete(scales, 0)
                data_shape = np.delete(data_shape, 0)
        elif data_ndim == 2:
            names = names[2:]
            scales = scales[2:]
            data_shape = data_shape[2:]

        # Create list of axis objects
        axes_list = [
            {
                "size": data_shape[i],
                "index_in_array": i,
                "name": names[i],
                "scale": scales[i],
                "offset": 0.0,
                "units": units[i],
            }
            for i in range(data_ndim)
        ]

        return axes_list

    def get_metadata_filename_title(self, group_name: str) -> Tuple[str, str]:
        """Return filename without full path and a scan title for the
        signal metadata.

        Parameters
        ----------
        group_name
            Name of scan group.

        Returns
        -------
        fname
            Filename without full path.
        title
            Scan title.
        """
        fname = os.path.basename(self.filename).split(".")[0]
        title = fname + " " + group_name[1:].split("/")[0]
        if len(title) > 20:
            title = f"{title:.20}..."
        return fname, title

    def read(
        self,
        group_names: Union[None, str, List[str]] = None,
        lazy: bool = False,
    ) -> List[dict]:
        """Return a list of dictionaries which can be used to create
        :class:`~kikuchipy.signals.EBSD` signals.

        The file is closed after reading if ``lazy=False``.

        Parameters
        ----------
        group_names
            Name or a list of names of the desired top HDF5 group(s). If
            not given, the first scan group is returned.
        lazy
            Read dataset lazily (default is ``False``). If ``False``,
            the file is closed after reading.

        Returns
        -------
        scan_list
            List of dictionaries with keys ``"axes"``, ``"data"``,
            ``"metadata"``, ``"original_metadata"``, ``"detector"``,
            (possibly) ``"static_background"``, and ``"xmap"``.
        """
        scan_dict_list = []
        for scan in self.get_desired_scan_groups(group_names):
            scan_dict_list.append(self.scan2dict(scan, lazy))

        if not lazy:
            self.file.close()

        return scan_dict_list

    @abc.abstractmethod
    def scan2dict(self, group: h5py.Group, lazy: bool = False) -> dict:
        """Read (possibly lazily) patterns from group.

        Parameters
        ----------
        group
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
        return NotImplemented  # pragma: no cover
