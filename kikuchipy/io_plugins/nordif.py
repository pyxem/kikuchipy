# -*- coding: utf-8 -*-

import os
import re
import warnings
import time
import datetime

import numpy as np
from hyperspy.misc.utils import DictionaryTreeBrowser

# Plugin characteristics
# ----------------------
format_name = 'NORDIF'
description = 'Read/write support for NORDIF Pattern.dat files'
full_support = False
# Recognised file extension
file_extensions = ['dat', 'DAT']
default_extension = 0
# Writing capabilities
writes = [(2, 2), (2, 1), (2, 0)]
magics = []


def get_string(content, expression, lineno, file):
    """Get relevant part of binary string using regular expressions.

    Parameters
    ----------
    content : str
        Input file content to search through for expression.
    expression : binary str
        Binary string with regular expression.
    lineno : int
        Line number in content to search.
    file :
        File handle.

    Returns
    -------
    string_out : str
        Output string with relevant value.
    """
    try:
        match = re.search(expression.encode('ascii'), content[lineno])
        string_out = match.group(1).decode('ascii')
    except AttributeError:
        raise ValueError("Failed to read line no '{}' (0 indexing) in "
                         "settings file '{}' using the regular expression "
                         "'{}'".format(lineno, file, expression))
    return string_out


def get_settings_from_file(filename):
    """Get scan size, pattern size and other relevant parameters from
    the NORDIF Setting.txt file.

    Parameters
    ----------
    filename : str
        Full file path of NORDIF settings file (default is
        Setting.txt).

    Returns
    -------
    md : DictionaryTreeBrowser
        Metadata complying with HyperSpy's metadata structure.
    omd : DictionaryTreeBrowser
        Original metadata that does not fit into HyperSpy's metadata
        structure.
    """
    try:
        f = open(filename, 'rb')
    except ValueError:
        warnings.warn("Settings file '{}' does not exist".format(filename))

    content = f.read().splitlines()

    # Set up metadata and original_metadata structure
    md = DictionaryTreeBrowser()
    omd = DictionaryTreeBrowser()
    sem = 'Acquisition_instrument.SEM'
    ebsd = sem + '.Detector.EBSD'

    # Microscope
    manufacturer = get_string(content, 'Manufacturer\t(.*)\t', 4, f)
    model = get_string(content, 'Model\t(.*)\t', 5, f)
    md.set_item(sem + '.microscope', manufacturer + ' ' + model)

    # Magnification
    magnification = get_string(content, 'Magnification\t(.*)\t#', 6, f)
    md.set_item(sem + '.magnification', int(magnification))

    # Beam energy
    beam_energy = get_string(content, 'Accelerating voltage\t(.*)\tkV', 8, f)
    md.set_item(sem + '.beam_energy', float(beam_energy))

    # Working distance
    working_distance = get_string(content, 'Working distance\t(.*)\tmm', 9, f)
    md.set_item(sem + '.working_distance', float(working_distance))

    # Tilt angle
    tilt_angle = get_string(content, 'Tilt angle\t(.*)\t', 10, f)
    omd.set_item(ebsd + '.tilt_angle', float(tilt_angle))

    # Acquisition frame rate
    frame_rate = get_string(content, 'Frame rate\t(.*)\tfps', 46, f)
    omd.set_item(ebsd + '.frame_rate', int(frame_rate))

    # Acquisition resolution (pattern size)
    pattern_size = get_string(content, 'Resolution\t(.*)\tpx', 47, f)
    SX, SY = [int(i) for i in pattern_size.split('x')]
    omd.set_item(ebsd + '.SX', SX)
    omd.set_item(ebsd + '.SY', SY)

    # Acquisition exposure time
    exposure_time = get_string(content, 'Exposure time\t(.*)\t', 48, f)
    omd.set_item(ebsd + '.exposure_time', float(exposure_time) / 1e6)

    # Acquisition gain
    gain = get_string(content, 'Gain\t(.*)\t', 49, f)
    omd.set_item(ebsd + '.gain', int(gain))

    # Scan size
    scan_size = get_string(content, 'Number of samples\t(.*)\t#', 79, f)
    NY, NX = [int(i) for i in scan_size.split('x')]
    omd.set_item(ebsd + '.NX', NX)
    omd.set_item(ebsd + '.NY', NY)

    # Step size
    step_size = get_string(content, 'Step size\t(.*)\t', 78, f)
    omd.set_item(ebsd + '.step_size', float(step_size))

    # Scan time
    scan_time = get_string(content, 'Scan time\t(.*)\t', 80, f)
    scan_time = time.strptime(scan_time, '%H:%M:%S')
    scan_time = datetime.timedelta(hours=scan_time.tm_hour,
                                   minutes=scan_time.tm_min,
                                   seconds=scan_time.tm_sec).total_seconds()
    omd.set_item(ebsd + '.scan_time', int(scan_time))

    return md, omd


def file_reader(filename, mmap_mode=None, lazy=False, scan_size=None,
                pattern_size=None, settings_file='Setting.txt', **kwargs):
    """Read electron backscatter diffraction patterns from a NORDIF
    data file.

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
    settings_file : {'Setting.txt', str}, optional
        File name of NORDIF settings file (default is Setting.txt).

    Returns
    -------
    dictionary : dict
        Data, axes, metadata and original metadata from settings file.
    """
    if mmap_mode is None:
        mmap_mode = 'r' if lazy else 'c'

    # Make sure we open in right mode
    if '+' in mmap_mode or ('write' in mmap_mode and
                            'copyonwrite' != mmap_mode):
        if lazy:
            raise ValueError("Lazy loading does not support in-place writing")
        f = open(filename, 'r+b')
    else:
        f = open(filename, 'rb')

    # Get original metadata, scan size and pattern size from settings file
    folder, filename = os.path.split(filename)
    try:
        md, omd = get_settings_from_file(os.path.join(folder, settings_file))
        ebsd = 'Acquisition_instrument.SEM.Detector.EBSD'
        scan_size = (omd.get_item(ebsd + '.NX'), omd.get_item(ebsd + '.NY'))
        pattern_size = (omd.get_item(ebsd + '.SX'), omd.get_item(ebsd + '.SY'))
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
        f.seek(-data_size, 2)
        data = np.fromfile(f, dtype='uint8')
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
    scales = [1, 1, 1, 1]

    # Set relevant values in metadata and original_metadata
    md.set_item('General.original_filename', os.path.split(filename)[1])
    md.set_item('Signal.signal_type', 'electron_backscatter_diffraction')
    md.set_item('Signal.record_by', 'image')
    omd.set_item('General.original_filepath', folder)

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


def file_writer(filename, signal, **kwargs):
    """Write electron backscatter diffraction patterns to a NORDIF data file.
    """
    with open(filename, 'wb') as f:
        pass
