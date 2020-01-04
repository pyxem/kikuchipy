# -*- coding: utf-8 -*-
# Copyright 2019-2020 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

import os
import logging

from hyperspy.io_plugins import hspy
from hyperspy.misc.io.tools import ensure_directory
from hyperspy.misc.io.tools import overwrite as overwrite_method
from hyperspy.misc.utils import strlist2enumeration, find_subclasses
from hyperspy.signal import BaseSignal

import kikuchipy.signals
from kikuchipy.io.plugins import h5ebsd, nordif
from kikuchipy.util.io import _get_input_bool

_logger = logging.getLogger(__name__)

plugins = [hspy, h5ebsd, nordif]

default_write_ext = set()
for plugin in plugins:
    if plugin.writes:
        default_write_ext.add(plugin.file_extensions[plugin.default_extension])


def load(filename, lazy=False, **kwargs):
    """Load an EBSD object from a supported file.

    This function is a modified version of :func:`hyperspy.io.load`.

    Parameters
    ----------
    filename : str
        Name of file to load.
    lazy : bool, optional
        Open the data lazily without actually reading the data from disk
        until required. Allows opening arbitrary sized datasets. Default
        is ``False``.
    **kwargs :
        Keyword arguments passed to the corresponding KikuchiPy reader.
        See their individual documentation for available options.
    """

    if not os.path.isfile(filename):
        raise IOError(f"No filename matches '{filename}'.")

    # Find matching reader for file extension
    extension = os.path.splitext(filename)[1][1:]
    reader = None
    for plugin in plugins:
        if extension.lower() in plugin.file_extensions:
            reader = plugin
            break
    if reader is None:
        raise IOError(
            f"Could not read '{filename}'. If the file format is supported, "
            "please report this error."
        )

    # Get data and metadata (from potentially multiple scans if an h5ebsd file)
    scan_dicts = reader.file_reader(filename, lazy=lazy, **kwargs)
    scans = []
    for scan in scan_dicts:
        scans.append(_dict2signal(scan, lazy=lazy))
        directory, filename = os.path.split(os.path.abspath(filename))
        filename, extension = os.path.splitext(filename)
        scans[-1].tmp_parameters.folder = directory
        scans[-1].tmp_parameters.filename = filename
        scans[-1].tmp_parameters.extension = extension.replace(".", "")

    if len(scans) == 1:
        scans = scans[0]

    return scans


def _dict2signal(signal_dict, lazy=False):
    """Create a signal instance from a dictionary.

    This function is a modified version :func:`hyperspy.io.dict2signal`.

    Parameters
    ----------
    signal_dict : dict
        Signal dictionary with ``data``, ``metadata`` and
        ``original_metadata``.
    lazy : bool, optional
        Open the data lazily without actually reading the data from disk
        until required. Allows opening arbitrary sized datasets. Default
        is ``False``.

    Returns
    -------
    signal : kikuchipy.signals.ebsd.EBSD or\
            kikuchipy.signals.ebsd.LazyEBSD
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
                    "KikuchiPy only supports `record_by = image`, not "
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


def _assign_signal_subclass(
    dtype, signal_dimension, signal_type="", lazy=False
):
    """Given ``record_by`` and ``signal_type`` return the matching
    signal subclass.

    This function is a modified version of
    :func:`hyperspy.io.assign_signal_subclass`.

    Parameters
    ----------
    dtype : numpy.dtype
        Data type of signal data.
    signal_dimension : int
        Number of signal dimensions.
    signal_type : '' or 'EBSD', optional
        Signal type.
    lazy : bool, optional
        Open the data lazily without actually reading the data from disk
        until required. Allows opening arbitrary sized datasets. Default
        is ``False``.

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
            f"No KikuchiPy signals match dtype '{dtype}', signal dimension "
            f"'{signal_dimension}' and signal_type '{signal_type}'."
        )

    return matches[0]


def save(filename, signal, overwrite=None, add_scan=None, **kwargs):
    """Write signal to a file in a supported format.

    This function is a modified version of :func:`hyperspy.io.save`.

    Parameters
    ----------
    filename : str
        File path including name of new file.
    signal : kikuchipy.signals.ebsd.EBSD or\
            kikuchipy.signals.ebsd.LazyEBSD
        Signal instance.
    overwrite : bool or None, optional
        Whether to overwrite file or not if it already exists.
    add_scan : bool or None, optional
        Whether to add the signal to an already existing h5ebsd file or
        not. If the file does not exist the signal is written to a new
        file.
    **kwargs :
        Keyword arguments passed to the writer.
    """

    ext = os.path.splitext(filename)[1][1:]
    if ext == "":  # Will write to KikuchiPy's h5ebsd format
        ext = "h5"
        filename = filename + "." + ext

    writer = None
    for plugin in plugins:
        if ext.lower() in plugin.file_extensions:
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

        ensure_directory(filename)
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
