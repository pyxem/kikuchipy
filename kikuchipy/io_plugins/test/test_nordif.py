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
import pytest
import tempfile
import gc
import numpy as np
from kikuchipy.io_plugins.nordif import (
    file_reader, get_settings_from_file, get_string, file_writer)


DIR_PATH = os.path.dirname(__file__)
NORDIF_PATH = os.path.join(DIR_PATH, '../../data/nordif')

# Settings content
METADATA = {
    'Acquisition_instrument': {
        'SEM': {
            'microscope': 'Hitachi SU-6600', 'magnification': 200,
            'beam_energy': 20.0, 'working_distance': 24.7, 'Detector': {
                'EBSD': {
                    'azimuth_angle': 0.0, 'binning': 1,
                    'detector': 'NORDIF UF1100', 'elevation_angle': 0.0,
                    'exposure_time': 0.0035, 'frame_number': 1,
                    'frame_rate': 202, 'gain': 0.0, 'grid_type': 'square',
                    'sample_tilt': 70.0, 'scan_time': 148,
                    'static_background': 1, 'xpc': 1.0, 'ypc': 1.0, 'zpc': 1.0,
                    'version': '3.1.2', 'manufacturer': 'NORDIF'}}}},
    'Sample': {
        'Phases': {
            '1': {
                'atom_coordinates': {
                    '1': {
                        'atom': '', 'coordinates': np.array([0., 0., 0.]),
                        'site_occupation': 0.0,
                        'debye_waller_factor': 0.0}},
                'formula': '', 'info': '',
                'lattice_constants': np.array([0., 0., 0., 0., 0., 0.]),
                'laue_group': '', 'material_name': 'Ni', 'point_group': '',
                'setting': 0, 'space_group': 0, 'symmetry': 0}}}}
ORIGINAL_METADATA = {
    'nordif_header': [
        '[NORDIF]\t\t', 'Software version\t3.1.2\t', '\t\t',
        '[Microscope]\t\t', 'Manufacturer\tHitachi\t', 'Model\tSU-6600\t',
        'Magnification\t200\t#', 'Scan direction\tDirect\t',
        'Accelerating voltage\t20\tkV', 'Working distance\t24.7\tmm',
        'Tilt angle\t70\t°', '\t\t', '[Signal voltages]\t\t',
        'Minimum\t0.0\tV', 'Maximum\t1.0\tV', '\t\t',
        '[Deflection voltages]\t\t', 'Minimum\t-5.5\tV', 'Maximum\t5.5\tV',
        '\t\t', '[Electron image]\t\t', 'Frame rate\t0.25\tfps',
        'Resolution\t1000x1000\tpx', 'Rotation\t0\t°', 'Flip x-axis\tFalse\t',
        'Flip y-axis\tFalse\t', 'Calibration factor\t7273\tµm/V',
        'Tilt axis\tx-axis\t', '\t\t', '[Aspect ratio]\t\t',
        'X-axis\t1.000\t', 'Y-axis\t1.000\t', '\t\t', '[EBSD detector]\t\t',
        'Model\tUF1100\t', 'Port position\t90\t', 'Jumbo frames\tFalse\t',
        '\t\t', '[Detector angles]\t\t', 'Euler 1\t0\t°', 'Euler 2\t0\t°',
        'Euler 3\t0\t°', 'Azimuthal\t0\t°', 'Elevation\t0\t°', '\t\t',
        '[Acquisition settings]\t\t', 'Frame rate\t202\tfps',
        'Resolution\t60x60\tpx', 'Exposure time\t3500\tµs', 'Gain\t0\t', '\t\t',
        '[Calibration settings]\t\t', 'Frame rate\t10\tfps',
        'Resolution\t480x480\tpx', 'Exposure time\t99950\tµs', 'Gain\t8\t',
        '\t\t', '[Specimen]\t\t', 'Name\tNi\t',
        'Mounting\t1. ND||EB TD||TA\t', '\t\t', '[Phase 1]\t\t', 'Name\t\t',
        'Pearson S.T.\t\t', 'IT\t\t', '\t\t', '[Phase 2]\t\t', 'Name\t\t',
        'Pearson S.T.\t\t', 'IT\t\t', '\t\t', '[Region of interest]\t\t',
        '\t\t', '[Area]\t\t', 'Top\t89.200 (223)\tµm (px)',
        'Left\t60.384 (152)\tµm (px)', 'Width\t4.500 (11)\tµm (px)',
        'Height\t4.500 (11)\tµm (px)', 'Step size\t1.500\tµm',
        'Number of samples\t3x3\t#', 'Scan time\t00:02:28\t', '\t\t',
        '[Points of interest]\t\t', '\t\t', '[Acquisition patterns]\t\t',
        'Acquisition (507,500)\t507,500\tpx',
        'Acquisition (393,501)\t393,501\tpx',
        'Acquisition (440,448)\t440,448\tpx', '\t\t',
        '[Calibration patterns]\t\t', 'Calibration (425,447)\t425,447\tpx',
        'Calibration (294,532)\t294,532\tpx',
        'Calibration (573,543)\t573,543\tpx',
        'Calibration (596,378)\t596,378\tpx',
        'Calibration (308,369)\t308,369\tpx',
        'Calibration (171,632)\t171,632\tpx',
        'Calibration (704,668)\t704,668\tpx',
        'Calibration (696,269)\t696,269\tpx',
        'Calibration (152,247)\t152,247\tpx']}
AXES_MANAGER = {
    'nx': 3, 'ny': 3, 'sx': 60, 'sy': 60, 'step_x': 1.5, 'step_y': 1.5}


@pytest.fixture()
def save_path():
    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, 'nordif', 'save_temp.dat')
        yield file_path
        gc.collect()


class TestNORDIF:

    def test_get_settings_from_file(self):
        setting_file = os.path.join(NORDIF_PATH, 'Setting.txt')
        settings = get_settings_from_file(setting_file)
        answers = [METADATA, ORIGINAL_METADATA, AXES_MANAGER]
        assert len(settings) == len(answers)
        for setting_read, answer in zip(settings, answers):
            np.testing.assert_equal(setting_read.as_dictionary(), answer)
