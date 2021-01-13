# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
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

import os
from typing import Optional, Union

from hyperspy.io_plugins import hspy
from hyperspy.misc.io.tools import overwrite as overwrite_method
from hyperspy.misc.utils import strlist2enumeration, find_subclasses
from hyperspy.signal import BaseSignal
from h5py import File, is_hdf5, Group
import numpy as np

import kikuchipy.signals
from kikuchipy.io.plugins import (
    emsoft_ebsd,
    emsoft_ebsd_master_pattern,
    h5ebsd,
    nordif,
)
from kikuchipy.io._util import _get_input_bool, _ensure_directory


plugins = [
    emsoft_ebsd,
    emsoft_ebsd_master_pattern,
    hspy,
    h5ebsd,
    nordif,
]

default_write_ext = set()
for plugin in plugins:
    if plugin.writes:
        default_write_ext.add(plugin.file_extensions[plugin.default_extension])


def load(filename: str, lazy: bool = False, **kwargs):
    """Load an :class:`~kikuchipy.signals.EBSD` or
    :class:`~kikuchipy.signals.EBSDMasterPattern` object from a
    supported file format.

    This function is a modified version of :func:`hyperspy.io.load`.

    Parameters
    ----------
    filename
        Name of file to load.
    lazy
        Open the data lazily without actually reading the data from disk
        until required. Allows opening arbitrary sized datasets. Default
        is False.
    kwargs
        Keyword arguments passed to the corresponding kikuchipy reader.
        See their individual documentation for available options.

    Returns
    -------
    kikuchipy.signals.EBSD, kikuchipy.signals.EBSDMasterPattern, \
        list of kikuchipy.signals.EBSD or \
        list of kikuchipy.signals.EBSDMasterPattern

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.load("patterns.h5")
    >>> s
    <EBSD, title: , dimensions: (10, 20|60, 60)>
    """
    if not os.path.isfile(filename):
        raise IOError(f"No filename matches '{filename}'.")

    # Find matching reader for file extension
    extension = os.path.splitext(filename)[1][1:]
    readers = []
    for plugin in plugins:
        if extension.lower() in plugin.file_extensions:
            readers.append(plugin)
    if len(readers) == 0:
        raise IOError(
            f"Could not read '{filename}'. If the file format is supported, "
            "please report this error."
        )
    elif len(readers) > 1 and is_hdf5(filename):
        reader = _plugin_from_footprints(filename, plugins=readers)
    else:
        reader = readers[0]

    # Get data and metadata (from potentially multiple signals if an h5ebsd
    # file)
    signal_dicts = reader.file_reader(filename, lazy=lazy, **kwargs)
    signals = []
    for signal in signal_dicts:
        signals.append(_dict2signal(signal, lazy=lazy))
        directory, filename = os.path.split(os.path.abspath(filename))
        filename, extension = os.path.splitext(filename)
        signals[-1].tmp_parameters.folder = directory
        signals[-1].tmp_parameters.filename = filename
        signals[-1].tmp_parameters.extension = extension.replace(".", "")

    if len(signals) == 1:
        signals = signals[0]

    return signals


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
                    "kikuchipy only supports `record_by = image`, not "
                    f"{record_by}."
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


def _plugin_from_footprints(filename: str, plugins) -> Optional[object]:
    """Get HDF5 correct plugin from a list of potential plugins based on
    their unique footprints.

    The unique footprint is a list of strings that can take on either of
    two formats:
        * group/dataset names separated by "/", indicating nested
          groups/datasets
        * single group/dataset name indicating that the groups/datasets
          are in the top group

    Parameters
    ----------
    filename
        Input file name.
    plugins
        Potential plugins.

    Returns
    -------
    plugin
        One of the potential plugins, or None if no footprint was found.
    """

    def _hdfgroups2dict(group):
        d = {}
        for key, val in group.items():
            key = key.lstrip().lower()
            if isinstance(val, Group):
                d[key] = _hdfgroups2dict(val)
            else:
                d[key] = 1
        return d

    def _exists(obj, chain):
        key = chain.pop(0)
        if key in obj:
            return _exists(obj[key], chain) if chain else obj[key]

    f = File(filename, mode="r")
    d = _hdfgroups2dict(f["/"])

    plugin = None
    plugins_with_footprints = [p for p in plugins if hasattr(p, "footprint")]
    for p in plugins_with_footprints:
        n_matches = 0
        n_desired_matches = len(p.footprint)
        for fp in p.footprint:
            fp = fp.lower().split("/")
            if _exists(d, fp) is not None:
                n_matches += 1
        if n_matches == n_desired_matches:
            plugin = p

    f.close()

    return plugin


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
        Signal type. Options are ""/"EBSD"/"EBSDMasterPattern".
    lazy
        Open the data lazily without actually reading the data from disk
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
        raise ValueError(f"Data type '{dtype.name}' not understood.")
    if not isinstance(signal_dimension, int) or signal_dimension < 0:
        raise ValueError(
            "Signal dimension must be a positive integer and not "
            f"'{signal_dimension}'."
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
    filename: str,
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
    ext = os.path.splitext(filename)[1][1:]
    if ext == "":  # Will write to kikuchipy's h5ebsd format
        ext = "h5"
        filename = filename + "." + ext

    writer = None
    for plugin in plugins:
        if ext.lower() in plugin.file_extensions and plugin.writes:
            writer = plugin
            break

    if writer is None:
        raise ValueError(
            f"'{ext}' does not correspond to any supported format. Supported "
            f"file extensions are: '{strlist2enumeration(default_write_ext)}'"
        )
    else:
        sd = signal.axes_manager.signal_dimension
        nd = signal.axes_manager.navigation_dimension
        if writer.writes is not True and (sd, nd) not in writer.writes:
            # Get writers that can write this data
            writing_plugins = []
            for plugin in plugins:
                if (
                    plugin.writes is True
                    or plugin.writes is not False
                    and (sd, nd) in plugin.writes
                ):
                    writing_plugins.append(plugin)
            raise ValueError(
                "This file format cannot write this data. The following "
                f"formats can: {strlist2enumeration(writing_plugins)}"
            )

        _ensure_directory(filename)
        is_file = os.path.isfile(filename)

        # Check if we are to add signal to an already existing h5ebsd file
        if writer.format_name == "h5ebsd" and overwrite is not True and is_file:
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
                "overwrite parameter can only be None, True or False, and "
                f"not {overwrite}"
            )

        # Finally, write file
        if write:
            writer.file_writer(filename, signal, **kwargs)
            directory, filename = os.path.split(os.path.abspath(filename))
            signal.tmp_parameters.set_item("folder", directory)
            signal.tmp_parameters.set_item(
                "filename", os.path.splitext(filename)[0]
            )
            signal.tmp_parameters.set_item("extension", ext)
