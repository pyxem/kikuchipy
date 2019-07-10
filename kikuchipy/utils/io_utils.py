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

import logging
from hyperspy.misc.utils import DictionaryTreeBrowser

_logger = logging.getLogger(__name__)


def kikuchipy_metadata():
    """Return a dictionary in HyperSpy's DictionaryTreeBrowser format
    with the default KikuchiPy metadata.

    Returns
    -------
    DictionaryTreeBrowser
    """
    ebsd = {'manufacturer': '',  # File manufacturer
            'version': '',  # File version
            'detector': '',  # Detector used to acquire the data
            'azimuth_angle': -1,  # Detector azimuth angle
            'elevation_angle': -1,  # Detector elevation angle
            'sample_tilt': -1,  # Sample tilt
            'binning': -1,  # Camera binning
            'detector_pixel_size': -1,  # [um]
            'exposure_time': -1,  # [us]
            'frame_number': -1,  # Number of frames averaged
            'frame_rate': -1,  # Frames per second
            'scan_time': -1,  # Total scan time [s]
            'gain': -1,  # Camera gain [dB]
            'grid_type': '',  # In which patterns are acquired (only square)
            'n_columns': -1,  # Number of patterns in horizontal direction
            'n_rows': -1,  # Number of patterns in vertical direction
            'xpc': -1,  # Pattern centre horizontal coordinate with respect to
                        # detector centre
            'ypc': -1,  # Pattern centre vertical coordinate with respect to
                        # detector centre
            'zpc': -1,  # Specimen to scintillator distance
            'pattern_height': -1,  # In pixels
            'pattern_width': -1,  # In pixels
            'step_x': -1,  # Beam step in horizontal direction on sample [um]
            'step_y': -1}  # Beam step in vertical direction on sample [um]
    sem = {'microscope': '',
           'magnification': -1,  # Microscope magnification [x]
           'beam_energy': -1,  # Acceleration voltage [kV]
           'working_distance': -1,  # Distance from pole piece to sample
           'Detector': {'EBSD': ebsd}}
    return DictionaryTreeBrowser({'Acquisition_instrument': {'SEM': sem}})


def user_input(question):
    """Get input from user on boolean choice, returning the answer.

    Parameters
    ----------
    question : str
        Question to ask user.
    """
    try:
        answer = input(question)
        answer = answer.lower()
        while (answer != 'y') and (answer != 'n'):
            print("Please answer y or n.")
            answer = input(question)
        if answer.lower() == 'y':
            return True
        elif answer.lower() == 'n':
            return False
    except BaseException:
        # Running in an IPython notebook that does not support raw_input
        _logger.info("Your terminal does not support raw input. Not adding "
                     "scan. To add the scan use `add_scan=True`")
        return False
    else:
        return True
