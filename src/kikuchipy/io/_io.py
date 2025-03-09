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

import glob
import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
from hyperspy.misc.utils import find_subclasses
from hyperspy.signal import BaseSignal
import numpy as np
import yaml

from kikuchipy.io._util import _ensure_directory, _get_input_bool, _overwrite
import kikuchipy.signals

PLUGINS: list = []
WRITE_EXTENSIONS = []
specification_paths = list(Path(__file__).parent.rglob("specification.yaml"))
for path in specification_paths:
    with open(path) as file:
        spec = yaml.safe_load(file)
        spec["api"] = ".".join(path.parts[-5:-1])
        PLUGINS.append(spec)
        if spec["writes"]:
            for ext in spec["file_extensions"]:
                WRITE_EXTENSIONS.append(ext)


if TYPE_CHECKING:  # pragma: no cover
    from kikuchipy.signals.ebsd import EBSD, LazyEBSD
    from kikuchipy.signals.ebsd_master_pattern import (
        EBSDMasterPattern,
        LazyEBSDMasterPattern,
    )
    from kikuchipy.signals.ecp_master_pattern import ECPMasterPattern


def load(
    filename: str | Path, lazy: bool = False, **kwargs
) -> "EBSD | EBSDMasterPattern | ECPMasterPattern | list[EBSD] | list[EBSDMasterPattern] | list[ECPMasterPattern]":
    """Load a supported signal from one of the
    :ref:`/tutorials/load_save_data.ipynb#Supported-file-formats`.

    Parameters
    ----------
    filename
        Name of file to load.
    lazy
        Open the data lazily without actually reading the data from disk
        until required. Allows opening arbitrary sized datasets. Default
        is False.
    **kwargs
        Keyword arguments passed to the corresponding kikuchipy reader.
        See their individual documentation for available options.

    Returns
    -------
    out
        Signal or a list of signals.

    Raises
    ------
    IOError
        If the file was not found or could not be read.

    Notes
    -----
    This function is a modified version of :func:`hyperspy.io.load`.

    Examples
    --------
    Import nine patterns from an HDF5 file in a directory DATA_PATH

    >>> import kikuchipy as kp
    >>> s = kp.load(DATA_PATH / "patterns.h5")
    >>> s
    <EBSD, title: patterns Scan 1, dimensions: (3, 3|60, 60)>
    """
    filename = str(filename)

    if not os.path.isfile(filename):
        is_wildcard = False
        filenames = glob.glob(filename)
        if len(filenames) > 0:
            is_wildcard = True
        if not is_wildcard:
            raise IOError(f"No filename matches {filename!r}")

    # Find matching reader for file extension
    extension = os.path.splitext(filename)[1][1:]
    readers = []
    for plugin in PLUGINS:
        if extension.lower() in plugin["file_extensions"]:
            readers.append(plugin)

    reader = None
    if len(readers) == 1:
        reader = readers[0]
    elif len(readers) > 1 and h5py.is_hdf5(filename):
        reader = _plugin_from_footprints(filename, plugins=readers)

    if len(readers) == 0 or reader is None:
        raise IOError(
            f"Could not read {filename!r}. If the file format is supported, please "
            "report this error"
        )

    # Get data and metadata (from potentially multiple signals if we're
    # reading from an h5ebsd file)
    file_reader = importlib.import_module(reader["api"]).file_reader
    signal_dicts = file_reader(filename, lazy=lazy, **kwargs)
    out = []
    for signal in signal_dicts:
        out.append(_dict2signal(signal, lazy=lazy))
        directory, filename = os.path.split(os.path.abspath(filename))
        filename, extension = os.path.splitext(filename)
        out[-1].tmp_parameters.folder = directory
        out[-1].tmp_parameters.filename = filename
        out[-1].tmp_parameters.extension = extension.replace(".", "")

    if len(out) == 1:
        out = out[0]

    return out


def _dict2signal(
    signal_dict: dict, lazy: bool = False
) -> "EBSD | LazyEBSD | EBSDMasterPattern | LazyEBSDMasterPattern":
    """Create a signal instance from a dictionary.

    Parameters
    ----------
    signal_dict
        Signal dictionary with "data", "metadata", and
        "original_metadata" keys.
    lazy
        Open the data lazily without actually reading the data from disk
        until required. Allows opening arbitrary sized datasets. Default
        is False.

    Returns
    -------
    signal
        Signal instance with "data", "metadata", and "original_metadata"
        keys.

    Notes
    -----
    This function is a modified version of
    :func:`hyperspy.io.dict2signal`.
    """
    signal_type = ""
    if "metadata" in signal_dict:
        md = signal_dict["metadata"]
        if "Signal" in md and "record_by" in md["Signal"]:
            record_by = md["Signal"]["record_by"]
            if record_by != "image":
                raise ValueError(
                    "kikuchipy only supports `record_by = image`, not " f"{record_by}"
                )
            del md["Signal"]["record_by"]
        if "Signal" in md and "signal_type" in md["Signal"]:
            signal_type = md["Signal"]["signal_type"]

    signal = _assign_signal_subclass(
        signal_dimension=2,
        signal_type=signal_type,
        dtype=signal_dict["data"].dtype,
        lazy=lazy,
    )(**signal_dict)

    if signal._lazy:
        signal.data = signal._lazy_data()

    return signal


def _plugin_from_footprints(filename: str, plugins) -> object:
    """Return correct HDF5 plugin from a list of potential plugins based
    on their unique footprints.

    The unique footprint is a list of strings that can take on either of
    two formats:

    * group/dataset names separated by "/", indicating nested
      groups/datasets
    * single group/dataset name indicating that the groups/datasets are
      in the top group

    Parameters
    ----------
    filename
        Input file name.
    plugins
        Potential plugins reading HDF5 files.

    Returns
    -------
    plugin
        One of the potential plugins, or None if no footprint was found.
    """

    def _hdf5group2dict(group):
        d = {}
        for key, val in group.items():
            key_lower = key.lstrip().lower()
            if isinstance(val, h5py.Group):
                d[key_lower] = _hdf5group2dict(val)
            elif key_lower == "manufacturer":
                d[key_lower] = key
            else:
                d[key_lower] = 1
        return d

    def _exists(obj: dict, chain: list[str]) -> dict | None:
        key = chain.pop(0)
        if key in obj:
            return _exists(obj[key], chain) if chain else obj[key]

    with h5py.File(filename) as file:
        top_group_keys = _hdf5group2dict(file["/"])

        matching_plugin = None

        plugins_matching_manufacturer = []
        # Find plugins matching the manufacturer dataset in the file
        for key, val in top_group_keys.items():
            if key == "manufacturer":
                manufacturer = file[val][()]
                if isinstance(manufacturer, np.ndarray) and len(manufacturer) == 1:
                    manufacturer = manufacturer[0]
                if isinstance(manufacturer, bytes):
                    manufacturer = manufacturer.decode("latin-1")
                for plugin in plugins:
                    if manufacturer.lower() == plugin["manufacturer"]:
                        plugins_matching_manufacturer.append(plugin)
                break

        if len(plugins_matching_manufacturer) == 1:
            matching_plugin = plugins_matching_manufacturer[0]
        else:
            # Search for a unique footprint
            plugins_matching_footprints = []
            for plugin in plugins:
                n_matches = 0
                n_desired_matches = len(plugin["footprints"])
                for footprint in plugin["footprints"]:
                    footprint = footprint.lower().split("/")
                    if _exists(top_group_keys, footprint) is not None:
                        n_matches += 1
                if n_matches > 0 and n_matches == n_desired_matches:
                    plugins_matching_footprints.append(plugin)
            if len(plugins_matching_footprints) == 1:
                matching_plugin = plugins_matching_footprints[0]

    return matching_plugin


def _assign_signal_subclass(
    dtype: np.dtype,
    signal_dimension: int,
    signal_type: str = "",
    lazy: bool = False,
) -> "EBSD | EBSDMasterPattern | ECPMasterPattern":
    """Return matching signal subclass given by *record_by* and
    *signal_type*.

    Parameters
    ----------
    dtype
        Data type of the signal data.
    signal_dimension
        Number of signal dimensions.
    signal_type
        Signal type. Options are '', 'EBSD', 'EBSDMasterPattern'.
    lazy
        Open the data lazily without actually reading the data from disc
        until required. Allows opening arbitrary sized datasets. Default
        is False.

    Returns
    -------
    Signal or subclass

    Notes
    -----
    This function is a modified version of
    :func:`hyperspy.io.assign_signal_subclass`.
    """
    # Check if parameter values are allowed
    if (
        "float" in dtype.name
        or "int" in dtype.name
        or "void" in dtype.name
        or "bool" in dtype.name
        or "object" in dtype.name
    ):
        dtype = "real"
    else:
        raise ValueError(f"Data type {dtype.name!r} not understood")
    if not isinstance(signal_dimension, int) or signal_dimension < 0:
        raise ValueError(
            f"Signal dimension must be a positive integer and not {signal_dimension!r}"
        )

    # Get possible signal classes
    signals = {}
    for k, v in find_subclasses(kikuchipy.signals, BaseSignal).items():
        if v._lazy == lazy:
            signals[k] = v

    # Get signals matching both input signal's dtype and signal dimension
    dtype_matches = [s for s in signals.values() if s._dtype == dtype]
    dtype_dim_matches = []
    for s in dtype_matches:
        if s._signal_dimension == signal_dimension:
            dtype_dim_matches.append(s)
    dtype_dim_type_matches = []
    for s in dtype_dim_matches:
        if signal_type == s._signal_type or signal_type in s._alias_signal_types:
            dtype_dim_type_matches.append(s)

    if len(dtype_dim_type_matches) == 1:
        matches = dtype_dim_type_matches
    else:
        raise ValueError(
            f"No kikuchipy signals match dtype {dtype!r}, signal dimension "
            f"'{signal_dimension}' and signal_type {signal_type!r}"
        )

    return matches[0]


def _save(
    filename: str | Path,
    signal: "EBSD | LazyEBSD",
    overwrite: bool | None = None,
    add_scan: bool | None = None,
    **kwargs,
) -> None:
    """Write a signal to file in a supported format.

    Parameters
    ----------
    filename
        File path including name of new file.
    signal
        Signal instance.
    overwrite
        Whether to overwrite file or not if it already exists.
    add_scan
        Whether to add the signal to an already existing h5ebsd file or
        not. If the file does not exist the signal is written to a new
        file.
    **kwargs
        Keyword arguments passed to the writer.

    Raises
    ------
    ValueError
        If file extension does not correspond to any supported format.
    ValueError
        If the file format cannot write the signal data.
    ValueError
        If the overwrite parameter is invalid.

    Notes
    -----
    This function is a modified version of :func:`hyperspy.io.save`.
    """
    filename = str(filename)

    ext = os.path.splitext(filename)[1][1:]
    if ext == "":  # Will write to kikuchipy's h5ebsd format
        ext = "h5"
        filename += "." + ext

    writer = None
    for plugin in PLUGINS:
        if ext.lower() in plugin["file_extensions"] and plugin["writes"]:
            writer = plugin
            break

    if writer is None:
        raise ValueError(
            f"{ext!r} does not correspond to any supported format. Supported file "
            f"extensions are: {WRITE_EXTENSIONS!r}"
        )
    else:
        sig_dim = signal.axes_manager.signal_dimension
        nav_dim = signal.axes_manager.navigation_dimension
        if writer["writes"] is not True and [sig_dim, nav_dim] not in writer["writes"]:
            compatible_plugin_names = []
            for plugin in PLUGINS:
                if (
                    plugin["writes"] is True
                    or plugin["writes"] is not False
                    and [sig_dim, nav_dim] in plugin["writes"]
                ):
                    compatible_plugin_names.append(plugin["name"])
            raise ValueError(
                f"Chosen IO plugin {writer['name']!r} cannot write this data. The "
                f"following plugins can: {compatible_plugin_names}"
            )

        _ensure_directory(filename)
        is_file = os.path.isfile(filename)

        # Check if we are to add signal to an already existing h5ebsd file
        if writer["name"] == "kikuchipy_h5ebsd" and overwrite is not True and is_file:
            if add_scan is None:
                add_scan = _get_input_bool(
                    f"Add scan to {filename!r} (y/n)?\n",
                    (
                        "Your terminal does not support raw input. Not adding scan. To "
                        "add the scan, pass 'add_scan=True'"
                    ),
                )
            if add_scan:
                # So that the 2nd statement below triggers
                overwrite = True
            kwargs["add_scan"] = add_scan

        # Determine if signal is to be written to file or not
        if overwrite is None:
            write = _overwrite(filename)  # Ask what to do
        elif overwrite is True or (overwrite is False and not is_file):
            write = True
        elif overwrite is False and is_file:
            write = False
        else:
            raise ValueError(
                "overwrite parameter can only be None, True, or False, and not "
                f"{overwrite}"
            )

        if write:
            file_writer = importlib.import_module(writer["api"]).file_writer
            file_writer(filename, signal, **kwargs)
            directory, filename = os.path.split(os.path.abspath(filename))
            signal.tmp_parameters.set_item("folder", directory)
            signal.tmp_parameters.set_item("filename", os.path.splitext(filename)[0])
            signal.tmp_parameters.set_item("extension", ext)
