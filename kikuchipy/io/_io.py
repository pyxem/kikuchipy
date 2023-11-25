# Copyright 2019-2023 The kikuchipy developers
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
from typing import List, Optional, Union

from hyperspy.misc.utils import strlist2enumeration, find_subclasses
from hyperspy.signal import BaseSignal
from h5py import File, is_hdf5, Group
import numpy as np
from rsciio import IO_PLUGINS
from rsciio.utils.tools import overwrite as overwrite_method
from rsciio.utils.tools import get_object_package_info
import yaml

import kikuchipy.signals
from kikuchipy.io._util import _get_input_bool, _ensure_directory


plugins = []
write_extensions = []
specification_paths = list(Path(__file__).parent.rglob("specification.yaml"))
for path in specification_paths:
    with open(path) as f:
        spec = yaml.safe_load(f)
        spec["api"] = f"kikuchipy.io.plugins.{path.parts[-2]}"
        plugins.append(spec)
        if spec["writes"]:
            for ext in spec["file_extensions"]:
                write_extensions.append(ext)
for plugin in IO_PLUGINS:
    if plugin["name"] in ["HSPY", "ZSPY"]:
        plugins.append(plugin)


def load(
    filename: Union[str, Path], lazy: bool = False, **kwargs
) -> Union[
    "EBSD",
    "EBSDMasterPattern",
    "ECPMasterPattern",
    List["EBSD"],
    List["EBSDMasterPattern"],
    List["ECPMasterPattern"],
]:
    """Load an :class:`~kikuchipy.signals.EBSD`,
    :class:`~kikuchipy.signals.EBSDMasterPattern` or
    :class:`~kikuchipy.signals.ECPMasterPattern` signal from one of the
    :ref:`/tutorials/load_save_data.ipynb#Supported-file-formats`.

    This function is a modified version of :func:`hyperspy.io.load`.

    Parameters
    ----------
    filename
        Name of file to load.
    lazy
        Open the data lazily without actually reading the data from disk
        until required. Allows opening arbitrary sized datasets. Default
        is ``False``.
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

    Examples
    --------
    Import nine patterns from an HDF5 file in a directory ``DATA_DIR``

    >>> import kikuchipy as kp
    >>> s = kp.load(DATA_DIR + "/patterns.h5")
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
            raise IOError(f"No filename matches '{filename}'.")

    # Find matching reader for file extension
    extension = os.path.splitext(filename)[1][1:]
    readers = []
    for plugin in plugins:
        if extension.lower() in plugin["file_extensions"]:
            readers.append(plugin)

    reader = None
    if len(readers) == 1:
        reader = readers[0]
    elif len(readers) > 1 and is_hdf5(filename):
        reader = _hdf5_plugin_from_footprints(filename, plugins=readers)

    if len(readers) == 0 or reader is None:
        raise IOError(
            f"Could not read '{filename}'. If the file format is supported, please "
            "report this error"
        )

    # Get data and metadata (from potentially multiple signals if an
    # h5ebsd file)
    signal_dicts = importlib.import_module(reader["api"]).file_reader(
        filename, lazy=lazy, **kwargs
    )
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


def _dict2signal(signal_dict: dict, lazy: bool = False):
    """Create a signal instance from a dictionary.

    This function is a modified version :func:`hyperspy.io.dict2signal`.

    Parameters
    ----------
    signal_dict
        Signal dictionary with ``data``, ``metadata`` and
        ``original_metadata``.
    lazy
        Open the data lazily without actually reading the data from disk
        until required. Allows opening arbitrary sized datasets. Default
        is False.

    Returns
    -------
    signal : EBSD, LazyEBSD, EBSDMasterPattern or LazyEBSDMasterPattern
        Signal instance with ``data``, ``metadata`` and
        ``original_metadata`` from dictionary.
    """
    signal_type = ""
    if "metadata" in signal_dict:
        md = signal_dict["metadata"]
        if "Signal" in md and "record_by" in md["Signal"]:
            record_by = md["Signal"]["record_by"]
            if record_by != "image":
                raise ValueError(
                    "kikuchipy only supports `record_by = image`, not " f"{record_by}."
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
        signal._make_lazy()

    return signal


def _hdf5_plugin_from_footprints(filename: str, plugins: list) -> Optional[dict]:
    """Get correct HDF5 plugin from a list of potential plugins based on
    their unique footprints.

    A plugin's unique footprint is a list of strings that can take on
    either of two formats:
        * group/dataset names separated by "/", indicating nested
          groups/datasets
        * single group/dataset name indicating that the groups/datasets
          are in the top group

    Parameters
    ----------
    filename
        Input file name.
    plugins
        Potential plugins reading HDF5 files.

    Returns
    -------
    plugin
        One of the potential plugins, or ``None`` if no footprint was
        found.
    """

    def _hdf5group2dict(group):
        d = {}
        for key, val in group.items():
            key_lower = key.lstrip().lower()
            if isinstance(val, Group):
                d[key_lower] = _hdf5group2dict(val)
            elif key_lower == "manufacturer":
                d[key_lower] = key
            else:
                d[key_lower] = 1
        return d

    def _exists(obj, chain):
        key = chain.pop(0)
        if key in obj:
            return _exists(obj[key], chain) if chain else obj[key]

    with File(filename) as f:
        d = _hdf5group2dict(f["/"])

        matching_plugin = None

        match_manufacturer = []
        # Search for the manufacturer (all h5ebsd files have this)
        for key, val in d.items():
            if key == "manufacturer":
                # Extracting the manufacturer is finicky
                man = f[val][()]
                if isinstance(man, np.ndarray) and len(man) == 1:
                    man = man[0]
                if isinstance(man, bytes):
                    man = man.decode("latin-1")
                for p in plugins:
                    if man.lower() == p["manufacturer"]:
                        match_manufacturer.append(p)

        if len(match_manufacturer) == 1:
            matching_plugin = match_manufacturer[0]
        else:
            # Search for a unique footprint
            match_footprint = []
            for p in plugins:
                n_matches = 0
                n_desired_matches = len(p["footprints"])
                for fp in p["footprints"]:
                    fp = fp.lower().split("/")
                    if _exists(d, fp) is not None:
                        n_matches += 1
                if n_matches > 0 and n_matches == n_desired_matches:
                    match_footprint.append(p)
            if len(match_footprint) == 1:
                matching_plugin = match_footprint[0]

    return matching_plugin


def _assign_signal_subclass(
    dtype: np.dtype,
    signal_dimension: int,
    signal_type: str = "",
    lazy: bool = False,
):
    """Given ``record_by`` and ``signal_type`` return the matching
    signal subclass.

    This function is a modified version of
    :func:`hyperspy.io.assign_signal_subclass`.

    Parameters
    ----------
    dtype
        Data type of signal data.
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
        raise ValueError(f"Data type '{dtype.name}' not understood")
    if not isinstance(signal_dimension, int) or signal_dimension < 0:
        raise ValueError(
            f"Signal dimension must be a positive integer and not '{signal_dimension}'"
        )

    # Get possible signal classes
    signals = {
        key: value
        for key, value in find_subclasses(kikuchipy.signals, BaseSignal).items()
        if value._lazy == lazy
    }

    # Get signals matching both input signal's dtype and signal dimension
    dtype_matches = [s for s in signals.values() if s._dtype == dtype]
    dtype_dim_matches = [
        s for s in dtype_matches if s._signal_dimension == signal_dimension
    ]
    dtype_dim_type_matches = [
        s
        for s in dtype_dim_matches
        if signal_type == s._signal_type or signal_type in s._alias_signal_types
    ]

    if len(dtype_dim_type_matches) == 1:
        matches = dtype_dim_type_matches
    else:
        raise ValueError(
            f"No kikuchipy signals match dtype '{dtype}', signal dimension "
            f"'{signal_dimension}' and signal_type '{signal_type}'."
        )

    return matches[0]


def _save(
    filename: Union[str, Path],
    signal,
    overwrite: Optional[bool] = None,
    add_scan: Optional[bool] = None,
    **kwargs,
):
    """Write signal to a file in a supported format.

    This function is a modified version of :func:`hyperspy.io.save`.

    Parameters
    ----------
    filename
        File path including name of new file.
    signal : EBSD or LazyEBSD
        Signal instance.
    overwrite
        Whether to overwrite file or not if it already exists.
    add_scan
        Whether to add the signal to an already existing h5ebsd file or
        not. If the file does not exist the signal is written to a new
        file.
    **kwargs :
        Keyword arguments passed to the writer.
    """
    filename = str(filename)

    ext = os.path.splitext(filename)[1][1:]
    if ext == "":  # Will write to kikuchipy's h5ebsd format
        ext = "h5"
        filename += "." + ext

    writer = None
    for plugin in plugins:
        if ext.lower() in plugin["file_extensions"] and plugin["writes"]:
            writer = plugin
            break

    if writer is None:
        raise ValueError(
            f"'{ext}' does not correspond to any supported format. Supported file "
            f"extensions are: '{write_extensions}'"
        )
    else:
        sd = signal.axes_manager.signal_dimension
        nd = signal.axes_manager.navigation_dimension
        if writer["writes"] is not True and [sd, nd] not in writer["writes"]:
            # Get writers that can write this data
            writing_plugins = []
            for plugin in plugins:
                if (
                    plugin["writes"] is True
                    or plugin["writes"] is not False
                    and (sd, nd) in plugin["writes"]
                ):
                    writing_plugins.append(plugin)
            raise ValueError(
                "This file format cannot write this data. The following formats can: "
                f"{strlist2enumeration(writing_plugins)}"
            )

        _ensure_directory(filename)
        is_file = os.path.isfile(filename)

        # Check if we are to add signal to an already existing h5ebsd file
        if writer["name"] == "kikuchipy_h5ebsd" and overwrite is not True and is_file:
            if add_scan is None:
                q = "Add scan to '{}' (y/n)?\n".format(filename)
                add_scan = _get_input_bool(q)
            if add_scan:
                overwrite = True  # So that the 2nd statement below triggers
            kwargs["add_scan"] = add_scan

        # Determine if signal is to be written to file or not
        if overwrite is None:
            write = overwrite_method(filename)  # Ask what to do
        elif overwrite is True or (overwrite is False and not is_file):
            write = True  # Write the file
        elif overwrite is False and is_file:
            write = False  # Don't write the file
        else:
            raise ValueError(
                "overwrite parameter can only be None, True, or False, and not "
                f"{overwrite}"
            )

        if write:
            if writer["name"] in ["HSPY", "ZSPY"]:
                signal_dict = signal._to_dictionary(add_models=True)
                signal_dict["package_info"] = get_object_package_info(signal)
                kwargs["signal"] = signal_dict
            else:
                kwargs["signal"] = signal

            importlib.import_module(writer["api"]).file_writer(filename, **kwargs)

            directory, filename = os.path.split(os.path.abspath(filename))
            signal.tmp_parameters.set_item("folder", directory)
            signal.tmp_parameters.set_item("filename", os.path.splitext(filename)[0])
            signal.tmp_parameters.set_item("extension", ext)
