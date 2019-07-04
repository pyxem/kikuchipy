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
import re
import warnings
import time
import datetime

import numpy as np
from hyperspy.misc.utils import DictionaryTreeBrowser
from kikuchipy.utils.io_utils import _ebsd_metadata

# Plugin characteristics
# ----------------------
format_name = 'NORDIF'
description = 'Read/write support for NORDIF Pattern.dat files'
full_support = True
# Recognised file extension
file_extensions = ['dat']
default_extension = 0
# Writing capabilities
writes = [(2, 2)]

# Set common strings
SEM_str = 'Acquisition_instrument.SEM.'
EBSD_str = SEM_str + 'Detector.EBSD.'


def get_string(content, expression, line_no, file):
    """Get relevant part of binary string using regular expressions.

    Parameters
    ----------
    content : list of str
        Input file content to search through for expression.
    expression : str
        String with regular expression.
    line_no : int
        Line number in content to search.
    file : file handle

    Returns
    -------
    string_out : str
        Output string with relevant value.
    """
    try:
        match = re.search(expression, content[line_no])
        string_out = match.group(1)
    except AttributeError:
        raise ValueError("Failed to read line no '{}' (0 indexing) in "
                         "settings file '{}' using the regular expression "
                         "'{}'".format(line_no, file, expression))
    return string_out


def get_settings_from_file(filename):
    """Metadata with scan size, pattern size and other relevant
    parameters from the NORDIF Setting.txt file.

    Parameters
    ----------
    filename : str
        Full file path of NORDIF setting file.

    Returns
    -------
    md : DictionaryTreeBrowser
        Metadata complying with HyperSpy's metadata structure.
    """
    try:
        f = open(filename, 'r', encoding='latin-1')
    except ValueError:
        warnings.warn("Settings file '{}' does not exist".format(filename))

    content = f.read().splitlines()

    # Get line numbers of setting blocks
    blocks = {'[Microscope]': -1, '[EBSD detector]': -1,
              '[Detector angles]': -1, '[Acquisition settings]': -1,
              '[Area]': -1}
    for i, line in enumerate(content):
        for block in blocks:
            if block in line:
                blocks[block] = i
    line_mic = blocks['[Microscope]']
    line_det = blocks['[EBSD detector]']
    line_ang = blocks['[Detector angles]']
    line_acq = blocks['[Acquisition settings]']
    line_area = blocks['[Area]']

    # Populate EBSD metadata structure
    md = _ebsd_metadata()

    # Microscope
    manufacturer = get_string(content, 'Manufacturer\t(.*)\t', line_mic + 1, f)
    model = get_string(content, 'Model\t(.*)\t', line_mic + 2, f)
    md.set_item(SEM_str + 'microscope', manufacturer + ' ' + model)

    # Magnification
    magnification = get_string(content, 'Magnification\t(.*)\t#', line_mic + 3,
                               f)
    md.set_item(SEM_str + 'magnification', int(magnification))

    # Beam energy
    beam_energy = get_string(content, 'Accelerating voltage\t(.*)\tkV',
                             line_mic + 5, f)
    md.set_item(SEM_str + 'beam_energy', float(beam_energy))

    # Working distance
    working_distance = get_string(content, 'Working distance\t(.*)\tmm',
                                  line_mic + 6, f)
    md.set_item(SEM_str + 'working_distance', float(working_distance))

    # Sample tilt
    sample_tilt = get_string(content, 'Tilt angle\t(.*)\t', line_mic + 7, f)
    md.set_item(EBSD_str + 'sample_tilt', float(sample_tilt))

    # Detector
    detector = get_string(content, 'Model\t(.*)\t', line_det + 1, f)
    detector = re.sub('[^a-zA-Z0-9]', repl='', string=detector)
    md.set_item(EBSD_str + 'detector', 'NORDIF ' + detector)

    # Azimuth angle
    azimuth_angle = get_string(content, 'Azimuthal\t(.*)\t', line_ang + 4, f)
    md.set_item(EBSD_str + 'azimuth_angle', azimuth_angle)

    # Elevation angle
    elevation_angle = get_string(content, 'Elevation\t(.*)\t', line_ang + 5, f)
    md.set_item(EBSD_str + 'elevation_angle', elevation_angle)

    # Acquisition frame rate
    frame_rate = get_string(content, 'Frame rate\t(.*)\tfps', line_acq + 1, f)
    md.set_item(EBSD_str + 'frame_rate', int(frame_rate))

    # Acquisition resolution (pattern size)
    pattern_size = get_string(content, 'Resolution\t(.*)\tpx', line_acq + 2, f)
    SX, SY = [int(i) for i in pattern_size.split('x')]
    md.set_item(EBSD_str + 'pattern_width', SX)
    md.set_item(EBSD_str + 'pattern_height', SY)

    # Acquisition exposure time
    exposure_time = get_string(content, 'Exposure time\t(.*)\t', line_acq + 3,
                               f)
    md.set_item(EBSD_str + 'exposure_time', float(exposure_time) / 1e6)

    # Acquisition gain
    gain = get_string(content, 'Gain\t(.*)\t', line_acq + 4, f)
    md.set_item(EBSD_str + 'gain', int(gain))

    # Scan size
    scan_size = get_string(content, 'Number of samples\t(.*)\t#',
                           line_area + 6, f)
    NY, NX = [int(i) for i in scan_size.split('x')]
    md.set_item(EBSD_str + 'n_columns', NX)
    md.set_item(EBSD_str + 'n_rows', NY)

    # Step size
    step_size = get_string(content, 'Step size\t(.*)\t', line_area + 5, f)
    md.set_item(EBSD_str + 'step_x', float(step_size))
    md.set_item(EBSD_str + 'step_y', float(step_size))

    # Scan time
    scan_time = get_string(content, 'Scan time\t(.*)\t', line_area + 7, f)
    scan_time = time.strptime(scan_time, '%H:%M:%S')
    scan_time = datetime.timedelta(hours=scan_time.tm_hour,
                                   minutes=scan_time.tm_min,
                                   seconds=scan_time.tm_sec).total_seconds()
    md.set_item(EBSD_str + 'scan_time', int(scan_time))

    # Grid type
    md.set_item(EBSD_str + 'grid_type', 'SqrGrid')

    return md


def file_reader(filename, mmap_mode=None, lazy=False, scan_size=None,
                pattern_size=None, setting_file='Setting.txt'):
    """Read electron backscatter patterns from a NORDIF data file.

    Parameters
    ----------
    filename : str
        Full file path of NORDIF data file.
    mmap_mode : str, optional
    lazy : bool, optional
    scan_size : {None, tuple}, optional
        Tuple with size of scan region of interest in pixels
        (patterns), width x height.
    pattern_size : {None, tuple}, optional
        Tuple with size of patterns in pixels, width x height.
    setting_file : {'Setting.txt', str}, optional
        File name of NORDIF settings file (default is Setting.txt).

    Returns
    -------
    dictionary : dict
        Data, axes, metadata from settings file.
    """

    if mmap_mode is None:
        mmap_mode = 'r' if lazy else 'c'

    # Make sure we open in correct mode
    if '+' in mmap_mode or ('write' in mmap_mode and
                            'copyonwrite' != mmap_mode):
        if lazy:
            raise ValueError("Lazy loading does not support in-place writing")
        f = open(filename, 'r+b')
    else:
        f = open(filename, 'rb')

    # Get metadata with scan size and pattern size from settings file
    folder, _ = os.path.split(filename)
    try:
        md = get_settings_from_file(os.path.join(folder, setting_file))
        if scan_size is None:
            scan_size = (md.get_item(EBSD_str + 'n_columns'),
                         md.get_item(EBSD_str + 'n_rows'))
        if pattern_size is None:
            pattern_size = (md.get_item(EBSD_str + 'pattern_width'),
                            md.get_item(EBSD_str + 'pattern_height'))
    except BaseException:
        warnings.warn("Reading the NORDIF settings file failed")

    if scan_size is None and pattern_size is None:
        raise ValueError("No scan size and pattern size provided")

    # Set scan size and pattern size
    (NX, NY) = scan_size
    (SX, SY) = pattern_size

    # Read data from file
    data_size = NY * NX * SX * SY
    if not lazy:
        f.seek(0)
        data = np.fromfile(f, dtype='uint8', count=data_size)
    else:
        data = np.memmap(f, mode=mmap_mode, dtype='uint8')

    try:
        data = data.reshape((NY, NX, SX, SY), order='C').squeeze()
    except ValueError:
        warnings.warn("Pattern size and scan size larger than file size! "
                      "Will attempt to load by zero padding incomplete "
                      "frames.")
        # Data is stored pattern by pattern
        pw = [(0, NY * NX * SX * SY - data.size)]
        data = np.pad(data, pw, mode='constant')
        data = data.reshape((NY, NX, SX, SY))

    units = [u'\u03BC'+'m', u'\u03BC'+'m', 'A^{-1}', 'A^{-1}']
    names = ['y', 'x', 'dx', 'dy']
    scales = np.ones(4)

    # Calibrate scan dimension
    try:
        scales[:2] = scales[:2]*md.get_item(EBSD_str + 'step_x')
    except BaseException:
        warnings.warn("Could not calibrate scan dimension, this can be done "
                      "using set_scan_calibration()")

    # Set relevant values in metadata
    md.set_item('General.original_filename', filename)
    md.set_item('Signal.signal_type', 'electron_backscatter_diffraction')
    md.set_item('Signal.record_by', 'image')

    # Create axis objects for each axis
    dim = data.ndim
    axes = [
        {
            'size': data.shape[i],
            'index_in_array': i,
            'name': names[i],
            'scale': scales[i],
            'offset': 0.0,
            'units': units[i], }
        for i in range(dim)]

    dictionary = {'data': data,
                  'axes': axes,
                  'metadata': md.as_dictionary()}
    f.close()

    return [dictionary, ]


def file_writer(filename, signal):
    """Write electron backscatter diffraction patterns to a NORDIF data
    binary .dat file.

    Writing dask arrays to a binary file is not yet supported.

    Parameters
    ----------
    filename : str
        Full file path of NORDIF data file.
    signal : :obj:`kikuchipy.signals.EBSD` or
             :obj:`kikuchipy.signals.LazyEBSD`
    """
    with open(filename, 'wb') as f:
        if signal._lazy:
            raise ValueError("Writing lazily to NORDIF .dat file is not yet "
                             "supported")
        else:
            for pattern in signal._iterate_signal():
                pattern.flatten().tofile(f)
