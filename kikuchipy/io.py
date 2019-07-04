# -*- coding: utf-8 -*-
# Copyright 2019 The KikuchiPy developers
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
import glob

import numpy as np
from natsort import natsorted
from hyperspy.ui_registry import get_gui
from hyperspy.misc.utils import stack as stack_method
from hyperspy.misc.io.tools import ensure_directory
from hyperspy.misc.io.tools import overwrite as overwrite_method
from hyperspy.drawing.marker import markers_metadata_dict_to_markers
from hyperspy.misc.utils import strlist2enumeration, find_subclasses

from kikuchipy.io_plugins import io_plugins, default_write_ext

f_error_fmt = (
    "\tFile %d:\n"
    "\t\t%d _signals\n"
    "\t\tPath: %s")


def load(filenames=None, signal_type=None, stack=False, stack_axis=None,
         new_axis_name='stack_element', lazy=False, convert_units=False,
         **kwargs):
    """Load potentially multiple supported files into a KikuchiPy
    structure.

    Supported formats: hspy or hdf5 (HDF5) and NORDIF dat.

    Any extra keyword is passed to the corresponding reader. For
    available options see their individual documentation, which may be
    found in either KikuchiPy or HyperSpy.

    Parameters
    ----------
    filenames : None, str or list of strings, optional
        The filename to be loaded. If None, a window will open to
        select a file to load. If a valid filename is passed in that
        single file is loaded. If multiple file names are passed in a
        list, a list of objects or a single object containing the data
        of the individual files stacked are returned. This behaviour is
        controlled by the `stack` parameter (see bellow). Multiple
        files can be loaded by using simple shell-style wildcards,
        e.g. 'my_file*.dat' loads all the files that starts
        by 'my_file' and has the '.dat' extension.
    signal_type : {None, 'EBSD', 'RadonTransform', '', str}, optional
        The name or acronym that identifies the signal type. The value
        provided may determine the Signal subclass assigned to the
        data. If None the value is read/guessed from the file. Any
        other value overrides the value stored in the file if any. If ''
        (empty string) the value is not read from the file and is
        considered undefined.
    stack : bool, optional
        If True and multiple filenames are passed in, stacking all
        the data into a single object is attempted. All files must
        match in shape. If each file contains multiple (N) _signals, N
        stacks will be created, with the requirement that each file
        contains the same number of _signals.
    stack_axis : {None, int, str}, optional
        If None, the _signals are stacked over a new axis. The data must
        have the same dimensions. Otherwise the _signals are stacked
        over the axis given by its integer index or its name. The data
        must have the same shape, except in the dimension corresponding
        to `axis`.
    new_axis_name : string, optional
        The name of the new axis when `axis` is None. If an axis with
        this name already exists it automatically appends '-i', where
        `i` are integers, until it finds a name that is not yet in use.
    lazy : {None, bool}, optional
        Open the data lazily - i.e. without actually reading the data
        from the disk until required. Allows opening arbitrary-sized
        datasets. The default is `False`.
    convert_units : bool, optional
        If True, convert the units using the `convert_to_units` method
        of the `axes_manager`. If False (default), nothing is done.
    **kwargs :
        Keyword arguments to be passed to the corresponding reader.

    Returns
    -------
    objects : signal instance or list of signal instances

    Examples
    --------
    Loading a single file providing the signal type:
    >>> s = kp.load('Pattern.dat', signal_type='EBSD')
    Loading multiple files:
    >>> s = kp.load(['Pattern1.dat', 'Pattern2.dat'])
    Loading multiple files matching the pattern:
    >>> s = kp.load('Pattern*.dat')
    Loading (potentially larger than the available memory) files lazily
    and stacking:
    >>> s = kp.load('Pattern*.dat', lazy=True, stack=True)
    """
    kwargs['signal_type'] = signal_type
    kwargs['convert_units'] = convert_units

    if filenames is None:
        from hyperspy.signal_tools import Load
        load_ui = Load()
        get_gui(load_ui, toolkey='load')
        if load_ui.filename:
            filenames = load_ui.filename
            lazy = load_ui.lazy
        if filenames is None:
            raise ValueError("No file provided to reader")

    if isinstance(filenames, str):
        filenames = natsorted([f for f in glob.glob(filenames)
                               if os.path.isfile(f)])
        if not filenames:
            raise ValueError("No file name matches this pattern")
    elif not isinstance(filenames, (list, tuple)):
        raise ValueError("The filenames parameter must be a list, tuple, "
                         "string or None")

    if not filenames:
        raise ValueError("No file provided to reader")
    else:
        if stack is True:
            # We are loading a stack!
            # Note that while each file might contain several signals, all
            # files are required to contain the same number of signals. We
            # therefore use the first file to determine the number of signals.
            for i, filename in enumerate(filenames):
                obj = load_single_file(filename, lazy=lazy, **kwargs)
                if i == 0:
                    # First iteration, determine number of _signals, if several
                    n = 1
                    if isinstance(obj, (list, tuple)):
                        n = len(obj)
                    # Initialize signal 2D list
                    signals = [[] for j in range(n)]
                else:
                    # Check tat number of _signals per file doesn't change for
                    # other files
                    if isinstance(obj, (list, tuple)):
                        if n != len(obj):
                            raise ValueError(
                                "The number of signals per file does not "
                                "match:\n" +
                                (f_error_fmt % (1, n, filenames[0])) +
                                (f_error_fmt % (i, len(obj), filename)))
                # Append loaded _signals to 2D list
                if n == 1:
                    signals[0].append(obj)
                elif n > 1:
                    for j in range(n):
                        signals[j].append(obj[j])
            # Next, merge the signals in the `stack_axis` direction.
            # When each file has N signals, we create N stacks.
            objects = []
            for i in range(n):
                signal = signals[i]  # Sublist, with len = len(filenames)
                signal = stack_method(signal, axis=stack_axis,
                                      new_axis_name=new_axis_name, lazy=lazy)
                signal.metadata.General.title = os.path.split(
                    os.path.split(os.path.abspath(filenames[0]))[0])[1]
                objects.append(signal)
        else:  # No stack, so simply load all _signals in all files separately
            objects = [load_single_file(filename, lazy=lazy, **kwargs)
                       for filename in filenames]

        if len(objects) == 1:
            objects = objects[0]

    return objects


def load_single_file(filename, **kwargs):
    """Load any supported file into a KikuchiPy structure.

    Supported formats: hspy or hdf5 (HDF5) and NORDIF dat.

    Parameters
    ----------
    filename : str
        File name with extension.
    """
    extension = os.path.splitext(filename)[1][1:]

    i = 0
    while extension.lower() not in io_plugins[i].file_extensions and \
            i < len(io_plugins) - 1:
        i += 1
    if i == len(io_plugins):
        # Try to load it with the Python Image Library (PIL)
        try:
            from hyperspy.io_plugins import image
            reader = image
            return load_with_reader(filename, reader, **kwargs)
        except BaseException:
            raise IOError("If the file format is supported please report "
                          "this error.")
    else:
        reader = io_plugins[i]
        return load_with_reader(filename=filename, reader=reader, **kwargs)


def load_with_reader(filename, reader, signal_type=None, convert_units=False,
                     **kwargs):
    """Read data, axes, metadata and original metadata from specified reader.

    Parameters
    ----------
    filename : str
    reader : io_plugin
    signal_type : {None, 'ElectronBackscatterDiffraction',
        'RadonTransform', '', str}, optional
    convert_units : bool, optional

    Returns
    -------
    objects : signal instance or list of signal instances
    """
    lazy = kwargs.get('lazy', False)
    file_data_list = reader.file_reader(filename, **kwargs)

    objects = []

    for signal_dict in file_data_list:
        if 'metadata' in signal_dict:
            if 'Signal' not in signal_dict['metadata']:
                signal_dict['metadata']['Signal'] = {}
            if signal_type is not None:
                signal_dict['metadata']['Signal']['signal_type'] = signal_type
            objects.append(dict2signal(signal_dict, lazy=lazy))
            folder, filename = os.path.split(os.path.abspath(filename))
            filename, extension = os.path.splitext(filename)
            objects[-1].tmp_parameters.folder = folder
            objects[-1].tmp_parameters.filename = filename
            objects[-1].tmp_parameters.extension = extension.replace('.', '')
            if convert_units:
                objects[-1].axes_manager.convert_units()
        else:  # It's a standalone model
            continue

    if len(objects) == 1:
        objects = objects[0]

    return objects


def dict2signal(signal_dict, lazy=False):
    """Create a signal (or subclass) instance defined by a dictionary.

    Parameters
    ----------
    signal_dict : dictionary
    lazy : bool, optional

    Returns
    -------
    s : Signal or subclass
    """
    signal_dimension = -1  # Undefined
    signal_type = ''
    if 'metadata' in signal_dict:
        md = signal_dict['metadata']
        if 'Signal' in md and 'record_by' in md['Signal']:
            record_by = md['Signal']['record_by']
            if record_by == 'spectrum':  # Throw ValueError here since KikuchiPy
                signal_dimension = 1     # does not support spectra?
            elif record_by == 'image':
                signal_dimension = 2
            del md['Signal']['record_by']
        if 'Signal' in md and 'signal_type' in md['Signal']:
            signal_type = md['Signal']['signal_type']
    if 'attributes' in signal_dict and '_lazy' in signal_dict['attributes']:
        lazy = signal_dict['attributes']['_lazy']
    # 'Estimate' signal_dimension from axes. It takes precedence over record_by
    if ('axes' in signal_dict and
        len(signal_dict['axes']) == len(
                [axis for axis in signal_dict['axes'] if 'navigate' in axis])):
        # If navigate is defined for all axes
        signal_dimension = len(
            [axis for axis in signal_dict['axes'] if not axis['navigate']])
    elif signal_dimension == -1:
        # If not defined, all dimensions are categorised as signal
        signal_dimension = signal_dict['data'].ndim
    signal = assign_signal_subclass(signal_dimension=signal_dimension,
                                    signal_type=signal_type,
                                    dtype=signal_dict['data'].dtype,
                                    lazy=lazy)(**signal_dict)
    if signal._lazy:
        signal._make_lazy()
    if signal.axes_manager.signal_dimension != signal_dimension:
        # This may happen when the signal dimension couldn't be matched with
        # any specialised subclass.
        signal.axes_manager.set_signal_dimension(signal_dimension)
    if 'post_process' in signal_dict:
        for f in signal_dict['post_process']:
            signal = f(signal)
    if 'mapping' in signal_dict:
        for opattr, (mpattr, function) in signal_dict['mapping'].items():
            if opattr in signal.original_metadata:
                value = signal.original_metadata.get_item(opattr)
                if function is not None:
                    value = function(value)
                if value is not None:
                    signal.metadata.set_item(mpattr, value)
    if 'metadata' in signal_dict and 'Markers' in md:
        markers_dict = markers_metadata_dict_to_markers(
            md['Markers'],
            axes_manager=signal.axes_manager)
        del signal.metadata.Markers
        signal.metadata.Markers = markers_dict

    return signal


def assign_signal_subclass(dtype, signal_dimension, signal_type='', lazy=False):
    """Given record_by and signal_type return the matching signal
    subclass.

    Parameters
    ----------
    dtype : :class:`~.numpy.dtype`
    signal_dimension : int
    signal_type : {'', 'EBSD', 'RadonTransform', str}, optional
    lazy : bool, optional

    Returns
    -------
    Signal or subclass
    """
    import kikuchipy.signals
    import kikuchipy.lazy_signals
    from hyperspy.signal import BaseSignal
    from hyperspy.signals import Signal1D, Signal2D
    from hyperspy._lazy_signals import LazySignal, LazySignal1D, LazySignal2D

    # Check if parameter values are allowed:
    if np.issubdtype(dtype, np.complexfloating):
        dtype = 'complex'
    elif ('float' in dtype.name or 'int' in dtype.name or
          'void' in dtype.name or 'bool' in dtype.name or
          'object' in dtype.name):
        dtype = 'real'
    else:
        raise ValueError("Data type '{}' not understood!".format(dtype.name))

    if not isinstance(signal_dimension, int) or signal_dimension < 0:
        raise ValueError("signal_dimension must be a positive integer.")

    base_signals = find_subclasses(kikuchipy.signals, BaseSignal)
    lazy_signals = find_subclasses(kikuchipy.lazy_signals, LazySignal)

    # Add HyperSpy's BaseSignal, Signal1D, Signal2D and their lazy partners for
    # when signal_type = ''
    base_signals.update({'BaseSignal': BaseSignal, 'Signal1D': Signal1D,
                         'Signal2D': Signal2D})
    lazy_signals.update({'LazySignal1D': LazySignal1D,
                         'LazySignal2D': LazySignal2D})

    if lazy:
        signals = lazy_signals
    else:
        signals = {
            k: v for k,
            v in base_signals.items() if k not in lazy_signals}
    dtype_matches = [s for s in signals.values() if dtype == s._dtype]
    dtype_dim_matches = [s for s in dtype_matches
                         if signal_dimension == s._signal_dimension]
    dtype_dim_type_matches = [s for s in dtype_dim_matches
                              if signal_type == s._signal_type
                              or signal_type in s._alias_signal_types]

    if dtype_dim_type_matches:  # Perfect match found, return it
        return dtype_dim_type_matches[0]
    elif [s for s in dtype_dim_matches if s._signal_type == '']:
        # Just signal_dimension and dtype matches
        # Return a general class for the given signal dimension
        return [s for s in dtype_dim_matches if s._signal_type == ''][0]
    else:
        # No signal_dimension match either, hence return the general subclass
        # for correct dtype
        return [s for s in dtype_matches if s._signal_dimension == - 1
                and s._signal_type == ''][0]


def save(filename, signal, overwrite=None, **kwargs):
    """Save EBSD or Radon Transform data.
    """
    extension = os.path.splitext(filename)[1][1:]
    if extension == '':
        extension = 'hspy'
        filename = filename + '.' + extension
    writer = None
    for plugin in io_plugins:
        if extension.lower() in plugin.file_extensions:
            writer = plugin
            break

    if writer is None:
        raise ValueError(
            (".%s does not correspond to any supported format. Supported "
             "file extensions are: %s") %
            (extension, strlist2enumeration(default_write_ext)))
    else:  # Check if the writer can write
        sd = signal.axes_manager.signal_dimension
        nd = signal.axes_manager.navigation_dimension
        if writer.writes is False:
            raise ValueError("Writing to this format is not "
                             "supported, supported file extensions are: %s " %
                             strlist2enumeration(default_write_ext))
        if writer.writes is not True and (sd, nd) not in writer.writes:
            yes_we_can = [plugin.format_name for plugin in io_plugins
                          if plugin.writes is True or
                          plugin.writes is not False and
                          (sd, nd) in plugin.writes]
            raise IOError("This file format cannot write this data. "
                          "The following formats can: %s" %
                          strlist2enumeration(yes_we_can))
        ensure_directory(filename)
        is_file = os.path.isfile(filename)
        if overwrite is None:
            write = overwrite_method(filename)  # Ask what to do
        elif overwrite is True or (overwrite is False and not is_file):
            write = True  # Write the file
        elif overwrite is False and is_file:
            write = False  # Don't write the file
        else:
            raise ValueError("`overwrite` parameter can only be None, True or "
                             "False")
        if write:
            writer.file_writer(filename, signal, **kwargs)
            folder, filename = os.path.split(os.path.abspath(filename))
            signal.tmp_parameters.set_item('folder', folder)
            signal.tmp_parameters.set_item('filename',
                                           os.path.splitext(filename)[0])
            signal.tmp_parameters.set_item('extension', extension)
