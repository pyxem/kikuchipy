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
import matplotlib.pyplot as plt
from hyperspy.misc.utils import DictionaryTreeBrowser
from kikuchipy.utils.io_utils import kikuchipy_metadata, metadata_nodes


# Plugin characteristics
# ----------------------
format_name = 'NORDIF'
description = 'Read/write support for NORDIF pattern and setting files'
full_support = False
# Recognised file extension
file_extensions = ['dat']
default_extension = 0
# Writing capabilities
writes = [(2, 2), (2, 1), (2, 0)]


def file_reader(filename, mmap_mode=None, scan_size=None,
                pattern_size=None, setting_file=None, lazy=False):
    """Read electron backscatter patterns from a NORDIF data file.

    Parameters
    ----------
    filename : str
        File path to NORDIF data file.
    mmap_mode : str, optional
    scan_size : {None, tuple}, optional
        Scan size in number of patterns in width and height.
    pattern_size : {None, tuple}, optional
        Pattern size in detector pixels in width and height.
    setting_file : {None, str}, optional
        File path to NORDIF setting file (default is Setting.txt in
        same directory as `filename`).
    lazy : bool, optional

    Returns
    -------
    scan : dict
        Data, axes, metadata and original metadata.
    """
    if mmap_mode is None:
        mmap_mode = 'r' if lazy else 'c'

    scan = {'attributes': {}}

    # Make sure we open in correct mode
    if '+' in mmap_mode or ('write' in mmap_mode and
                            'copyonwrite' != mmap_mode):
        if lazy:
            raise ValueError("Lazy loading does not support in-place writing")
        f = open(filename, 'r+b')
        scan['attributes']['_lazy'] = True
    else:
        f = open(filename, 'rb')

    # Get metadata from setting file
    sem_node, ebsd_node = metadata_nodes()
    folder, fname = os.path.split(filename)
    if setting_file is None:
        setting_file = os.path.join(folder, 'Setting.txt')
    setting_file_exists = os.path.isfile(setting_file)
    if setting_file_exists:
        md, omd = get_settings_from_file(setting_file)
        scan_size = (md.get_item(ebsd_node + '.n_columns'),
                     md.get_item(ebsd_node + '.n_rows'))
        pattern_size = (md.get_item(ebsd_node + '.pattern_width'),
                        md.get_item(ebsd_node + '.pattern_height'))
    else:
        warnings.warn("No setting file found, will attempt to use values for "
                      "scan_size and pattern_size from input arguments.")
        md = kikuchipy_metadata()
        omd = DictionaryTreeBrowser()
        try:
            md.set_item(ebsd_node + '.n_columns', scan_size[0])
            md.set_item(ebsd_node + '.n_rows', scan_size[1])
            md.set_item(ebsd_node + '.pattern_width', pattern_size[0])
            md.set_item(ebsd_node + '.pattern_height', pattern_size[1])
        except (NameError, IndexError):
            raise IOError("No setting file found, and no scan_size and "
                          "pattern_size provided in input arguments.")

    # Read static background pattern into metadata
    static_bg_file = os.path.join(folder, 'Background acquisition pattern.bmp')
    try:
        md.set_item(ebsd_node + '.static_background',
                    plt.imread(static_bg_file))
    except FileNotFoundError:
        warnings.warn("Could not read static background pattern '{}', however "
                      "it can be added using set_experimental_parameters()."
                      "".format(static_bg_file))

    # Set required and other parameters in metadata
    md.set_item('General.original_filename', filename)
    md.set_item('General.title',
                os.path.splitext(os.path.split(filename)[1])[0])
    md.set_item('Signal.signal_type', 'EBSD')
    md.set_item('Signal.record_by', 'image')
    scan['metadata'] = md.as_dictionary()
    scan['original_metadata'] = omd.as_dictionary()

    # Set scan size and pattern size
    nx, ny = scan_size
    sx, sy = pattern_size

    # Read data from file
    data_size = ny * nx * sy * sx
    if not lazy:
        f.seek(0)
        data = np.fromfile(f, dtype='uint8', count=data_size)
    else:
        data = np.memmap(f, mode=mmap_mode, dtype='uint8')

    try:
        data = data.reshape((ny, nx, sy, sx), order='C').squeeze()
    except ValueError:
        warnings.warn("Pattern size and scan size larger than file size! "
                      "Will attempt to load by zero padding incomplete "
                      "frames.")
        # Data is stored pattern by pattern
        pw = [(0, ny * nx * sy * sx - data.size)]
        data = np.pad(data, pw, mode='constant')
        data = data.reshape((ny, nx, sy, sx))
    scan['data'] = data

    units = [u'\u03BC'+'m', u'\u03BC'+'m', 'A^{-1}', 'A^{-1}']
    names = ['x', 'y', 'dx', 'dy']
    scales = np.ones(4)

    # Calibrate scan dimension
    try:
        scales[:2] = scales[:2]*md.get_item(ebsd_node + '.step_x')
    except BaseException:
        warnings.warn("Could not calibrate scan dimensions, this can be done "
                      "using set_scan_calibration()")

    # Create axis objects for each axis
    axes = [{'size': data.shape[i], 'index_in_array': i, 'name': names[i],
             'scale': scales[i], 'offset': 0.0, 'units': units[i]}
            for i in range(data.ndim)]
    scan['axes'] = axes

    f.close()

    return [scan, ]


def get_settings_from_file(filename):
    """Return metadata with parameters from NORDIF setting file.

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
    blocks = {'[NORDIF]': -1, '[Microscope]': -1, '[EBSD detector]': -1,
              '[Detector angles]': -1, '[Acquisition settings]': -1,
              '[Area]': -1}
    for i, line in enumerate(content):
        for block in blocks:
            if block in line:
                blocks[block] = i
    l_nordif = blocks['[NORDIF]']
    l_mic = blocks['[Microscope]']
    l_det = blocks['[EBSD detector]']
    l_ang = blocks['[Detector angles]']
    l_acq = blocks['[Acquisition settings]']
    l_area = blocks['[Area]']

    # Create metadata and original metadata structures
    md = kikuchipy_metadata()
    omd = DictionaryTreeBrowser()
    omd.set_item('nordif_header', content)

    # Get values
    azimuth_angle = get_string(content, 'Azimuthal\t(.*)\t', l_ang + 4, f)
    beam_energy = get_string(content, 'Accelerating voltage\t(.*)\tkV',
                             l_mic + 5, f)
    detector = get_string(content, 'Model\t(.*)\t', l_det + 1, f)
    detector = re.sub('[^a-zA-Z0-9]', repl='', string=detector)
    elevation_angle = get_string(content, 'Elevation\t(.*)\t', l_ang + 5, f)
    exposure_time = get_string(content, 'Exposure time\t(.*)\t', l_acq + 3, f)
    frame_rate = get_string(content, 'Frame rate\t(.*)\tfps', l_acq + 1, f)
    gain = get_string(content, 'Gain\t(.*)\t', l_acq + 4, f)
    magnification = get_string(content, 'Magnification\t(.*)\t#', l_mic + 3, f)
    mic_manufacturer = get_string(content, 'Manufacturer\t(.*)\t', l_mic + 1, f)
    mic_model = get_string(content, 'Model\t(.*)\t', l_mic + 2, f)
    pattern_size = get_string(content, 'Resolution\t(.*)\tpx', l_acq + 2, f)
    scan_size = get_string(content, 'Number of samples\t(.*)\t#', l_area + 6, f)
    ny, nx = [int(i) for i in scan_size.split('x')]
    step_size = get_string(content, 'Step size\t(.*)\t', l_area + 5, f)
    sample_tilt = get_string(content, 'Tilt angle\t(.*)\t', l_mic + 7, f)
    scan_time = get_string(content, 'Scan time\t(.*)\t', l_area + 7, f)
    scan_time = time.strptime(scan_time, '%H:%M:%S')
    scan_time = datetime.timedelta(hours=scan_time.tm_hour,
                                   minutes=scan_time.tm_min,
                                   seconds=scan_time.tm_sec).total_seconds()
    sx, sy = [int(i) for i in pattern_size.split('x')]
    version = get_string(content, 'Software version\t(.*)\t', l_nordif + 1, f)
    working_distance = get_string(content, 'Working distance\t(.*)\tmm',
                                  l_mic + 6, f)

    # Populate metadata
    sem_node, ebsd_node = metadata_nodes()
    md.set_item(ebsd_node + '.azimuth_angle', float(azimuth_angle))
    md.set_item(sem_node + '.beam_energy', float(beam_energy))
    md.set_item(ebsd_node + '.detector', 'NORDIF ' + detector)
    md.set_item(ebsd_node + '.elevation_angle', float(elevation_angle))
    md.set_item(ebsd_node + '.exposure_time', float(exposure_time) / 1e6)
    md.set_item(ebsd_node + '.frame_rate', int(frame_rate))
    md.set_item(ebsd_node + '.gain', int(gain))
    md.set_item(ebsd_node + '.grid_type', 'square')
    md.set_item(sem_node + '.magnification', int(magnification))
    md.set_item(ebsd_node + '.manufacturer', 'NORDIF')
    md.set_item(sem_node + '.microscope', mic_manufacturer + ' ' + mic_model)
    md.set_item(ebsd_node + '.n_columns', int(nx))
    md.set_item(ebsd_node + '.n_rows', int(ny))
    md.set_item(ebsd_node + '.pattern_width', int(sx))
    md.set_item(ebsd_node + '.pattern_height', int(sy))
    md.set_item(ebsd_node + '.sample_tilt', float(sample_tilt))
    md.set_item(ebsd_node + '.scan_time', int(scan_time))
    md.set_item(ebsd_node + '.step_x', float(step_size))
    md.set_item(ebsd_node + '.step_y', float(step_size))
    md.set_item(ebsd_node + '.version', version)
    md.set_item(sem_node + '.working_distance', float(working_distance))

    return md, omd


def get_string(content, expression, line_no, file):
    """Get relevant part of string using regular expression.

    Parameters
    ----------
    content : list
        File content to search in for the regular expression.
    expression : str
        Regular expression.
    line_no : int
        Line number to search in.
    file : file handle

    Returns
    -------
    str
        Output string with relevant value.
    """
    match = re.search(expression, content[line_no])
    if match is None:
        warnings.warn("Failed to read line {} in settings file '{}' using "
                      "regular expression '{}'".format(line_no - 1,
                                                       file.name, expression))
        return '-1'
    else:
        return match.group(1)


def file_writer(filename, signal):
    """Write electron backscatter patterns to NORDIF binary file.

    Parameters
    ----------
    filename : str
        File path of NORDIF data file.
    signal : kikuchipy.signals.EBSD
    """
    with open(filename, 'wb') as f:
        if signal._lazy:
            raise ValueError("Writing lazily to NORDIF file format is not yet "
                             "supported")
        else:
            for pattern in signal._iterate_signal():
                pattern.flatten().tofile(f)
