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
from kikuchipy.utils.io_utils import ebsd_metadata


# Plugin characteristics
# ----------------------
format_name = 'NORDIF'
description = 'Read/write support for NORDIF pattern and setting files'
full_support = False
# Recognised file extension
file_extensions = ['dat', 'DAT']
default_extension = 0
# Writing capabilities
writes = [(2, 2), (2, 1), (2, 0)]

# Set common strings
SEM_str = 'Acquisition_instrument.SEM.'
EBSD_str = SEM_str + 'Detector.EBSD.'


def get_string(content, expression, line_no, file):
    """Get relevant part of string using regular expression.

    Parameters
    ----------
    content : list
        File content to search in for the regular expression.
    expression : str
        Regular expression.
    line_no : int
        Content line number to search in. Need this because setting file
        contains many similar lines.
    file : file handle

    Returns
    -------
    string_out : str
        Output string with relevant value.
    """
    match = re.search(expression, content[line_no])
    if match is None:
        string_out = '-1'
        warnings.warn("Failed to read line no {} in settings file '{}' using "
                      "the regular expression '{}'".format(line_no - 1,
                                                           file.name, expression))
    else:
        string_out = match.group(1)
    return string_out


def get_settings_from_file(filename):
    """Metadata with relevant parameters from NORDIF setting file.

    Parameters
    ----------
    filename : str
        File path of NORDIF setting file.

    Returns
    -------
    md : DictionaryTreeBrowser
        Metadata complying with HyperSpy's metadata structure.
    omd : DictionaryTreeBrowser
        Metadata that does not fit into HyperSpy's metadata structure.
    """
    f = open(filename, 'r', encoding='latin-1')  # Avoid byte strings
    content = f.read().splitlines()

    # Get line numbers of setting blocks
    blocks = {'[Microscope]': -1, '[EBSD detector]': -1,
              '[Detector angles]': -1, '[Acquisition settings]': -1,
              '[Area]': -1}
    for i, line in enumerate(content):
        for block in blocks:
            if block in line:
                blocks[block] = i
    l_mic = blocks['[Microscope]']
    l_det = blocks['[EBSD detector]']
    l_ang = blocks['[Detector angles]']
    l_acq = blocks['[Acquisition settings]']
    l_area = blocks['[Area]']

    # Create metadata and original metadata structures
    md = ebsd_metadata()
    omd = DictionaryTreeBrowser()
    omd.set_item('nordif_header', content)

    # Get values
    manufacturer = get_string(content, 'Manufacturer\t(.*)\t', l_mic + 1, f)
    model = get_string(content, 'Model\t(.*)\t', l_mic + 2, f)
    magnification = get_string(content, 'Magnification\t(.*)\t#', l_mic + 3, f)
    beam_energy = get_string(content, 'Accelerating voltage\t(.*)\tkV',
                             l_mic + 5, f)
    working_distance = get_string(content, 'Working distance\t(.*)\tmm',
                                  l_mic + 6, f)
    sample_tilt = get_string(content, 'Tilt angle\t(.*)\t', l_mic + 7, f)
    detector = get_string(content, 'Model\t(.*)\t', l_det + 1, f)
    detector = re.sub('[^a-zA-Z0-9]', repl='', string=detector)
    azimuth_angle = get_string(content, 'Azimuthal\t(.*)\t', l_ang + 4, f)
    elevation_angle = get_string(content, 'Elevation\t(.*)\t', l_ang + 5, f)
    frame_rate = get_string(content, 'Frame rate\t(.*)\tfps', l_acq + 1, f)
    pattern_size = get_string(content, 'Resolution\t(.*)\tpx', l_acq + 2, f)
    sx, sy = [int(i) for i in pattern_size.split('x')]
    exposure_time = get_string(content, 'Exposure time\t(.*)\t', l_acq + 3, f)
    gain = get_string(content, 'Gain\t(.*)\t', l_acq + 4, f)
    scan_size = get_string(content, 'Number of samples\t(.*)\t#', l_area + 6, f)
    ny, nx = [int(i) for i in scan_size.split('x')]
    step_size = get_string(content, 'Step size\t(.*)\t', l_area + 5, f)
    scan_time = get_string(content, 'Scan time\t(.*)\t', l_area + 7, f)
    scan_time = time.strptime(scan_time, '%H:%M:%S')
    scan_time = datetime.timedelta(hours=scan_time.tm_hour,
                                   minutes=scan_time.tm_min,
                                   seconds=scan_time.tm_sec).total_seconds()

    # Populate metadata
    md.set_item(SEM_str + 'microscope', manufacturer + ' ' + model)
    md.set_item(SEM_str + 'magnification', int(magnification))
    md.set_item(SEM_str + 'beam_energy', float(beam_energy))
    md.set_item(SEM_str + 'working_distance', float(working_distance))
    md.set_item(EBSD_str + 'sample_tilt', float(sample_tilt))
    md.set_item(EBSD_str + 'detector', 'NORDIF ' + detector)
    md.set_item(EBSD_str + 'azimuth_angle', azimuth_angle)
    md.set_item(EBSD_str + 'elevation_angle', elevation_angle)
    md.set_item(EBSD_str + 'frame_rate', int(frame_rate))
    md.set_item(EBSD_str + 'pattern_width', sx)
    md.set_item(EBSD_str + 'pattern_height', sy)
    md.set_item(EBSD_str + 'exposure_time', float(exposure_time) / 1e6)
    md.set_item(EBSD_str + 'gain', int(gain))
    md.set_item(EBSD_str + 'n_columns', nx)
    md.set_item(EBSD_str + 'n_rows', ny)
    md.set_item(EBSD_str + 'step_x', float(step_size))
    md.set_item(EBSD_str + 'step_y', float(step_size))
    md.set_item(EBSD_str + 'scan_time', int(scan_time))
    md.set_item(EBSD_str + 'grid_type', 'SqrGrid')

    return md, omd


def file_reader(filename, mmap_mode=None, lazy=False, scan_size=None,
                pattern_size=None, setting_file=None):
    """Read electron backscatter patterns from a NORDIF data file.

    Parameters
    ----------
    filename : str
        File path to NORDIF data file.
    mmap_mode : str, optional
    lazy : bool, optional
    scan_size : {None, tuple}, optional
        Scan size in number of patterns in width and height.
    pattern_size : {None, tuple}, optional
        Pattern size in detector pixels in width and height.
    setting_file : {None, str}, optional
        File path to NORDIF setting file (default is Setting.txt in
        same directory as `filename`).

    Returns
    -------
    dictionary : dict
        Data, axes, metadata and original metadata.
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

    # Get metadata from setting file
    folder, _ = os.path.split(filename)
    if setting_file is None:
        setting_file = os.path.join(folder, 'Setting.txt')
    setting_file_exists = os.path.isfile(setting_file)
    if setting_file_exists:
        md, omd = get_settings_from_file(setting_file)
        scan_size = (md.get_item(EBSD_str + 'n_columns'),
                     md.get_item(EBSD_str + 'n_rows'))
        pattern_size = (md.get_item(EBSD_str + 'pattern_width'),
                        md.get_item(EBSD_str + 'pattern_height'))
    else:
        warnings.warn("No setting file found, will attempt to use values for "
                      "scan_size and pattern_size from input arguments.")
        md = ebsd_metadata()
        omd = DictionaryTreeBrowser()
        try:
            md.set_item(EBSD_str + 'n_columns', scan_size[0])
            md.set_item(EBSD_str + 'n_rows', scan_size[1])
            md.set_item(EBSD_str + 'pattern_width', pattern_size[0])
            md.set_item(EBSD_str + 'pattern_height', pattern_size[1])
        except BaseException:
            raise ValueError("No setting file found, and no scan_size and "
                             "pattern_size provided in input arguments.")

    # Set required and other parameters in metadata
    md.set_item('General.original_filename', filename)
    md.set_item('General.title', os.path.split(folder)[-1])
    md.set_item('Signal.signal_type', 'EBSD')
    md.set_item('Signal.record_by', 'image')

    # Set scan size and pattern size
    nx, ny = scan_size
    sx, sy = pattern_size

    # Read data from file
    data_size = ny * nx * sx * sy
    if not lazy:
        f.seek(0)
        data = np.fromfile(f, dtype='uint8', count=data_size)
    else:
        data = np.memmap(f, mode=mmap_mode, dtype='uint8')

    try:
        data = data.reshape((ny, nx, sx, sy), order='C').squeeze()
    except ValueError:
        warnings.warn("Pattern size and scan size larger than file size! "
                      "Will attempt to load by zero padding incomplete "
                      "frames.")
        # Data is stored pattern by pattern
        pw = [(0, ny * nx * sx * sy - data.size)]
        data = np.pad(data, pw, mode='constant')
        data = data.reshape((ny, nx, sx, sy))

    units = [u'\u03BC'+'m', u'\u03BC'+'m', 'A^{-1}', 'A^{-1}']
    names = ['y', 'x', 'dx', 'dy']
    scales = np.ones(4)

    # Calibrate scan dimension
    try:
        scales[:2] = scales[:2]*md.get_item(EBSD_str + 'step_x')
    except BaseException:
        warnings.warn("Could not calibrate scan dimensions, this can be done "
                      "using set_scan_calibration()")

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
                  'metadata': md.as_dictionary(),
                  'original_metadata': omd.as_dictionary()}

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
