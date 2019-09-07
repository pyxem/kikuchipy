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

from kikuchipy.signals import EBSD
from kikuchipy.utils.io_utils import metadata_nodes, kikuchipy_metadata
from hyperspy.misc.utils import DictionaryTreeBrowser
import numpy as np
import pytest


def assert_dictionary(input_dict, output_dict):
    if isinstance(input_dict, DictionaryTreeBrowser):
        input_dict = input_dict.as_dictionary()
        output_dict = output_dict.as_dictionary()
    for key in output_dict.keys():
        if isinstance(output_dict[key], dict):
            assert_dictionary(input_dict[key], output_dict[key])
        else:
            if isinstance(output_dict[key], list)\
                    or isinstance(input_dict[key], list):
                output_dict[key] = np.array(output_dict[key])
                input_dict[key] = np.array(input_dict[key])
            if isinstance(output_dict[key], np.ndarray):
                assert input_dict[key].all() == output_dict[key].all()
            else:
                assert input_dict[key] == output_dict[key]


class TestEBSD:

    def test_init(self):
        # Signal shape
        array0 = np.zeros(shape=(10, 10, 10, 10))
        s0 = EBSD(array0)
        assert array0.shape == s0.axes_manager.shape
        # Cannot initialise signal with one signal dimension
        with pytest.raises(ValueError):
            EBSD(np.zeros(10))
        # Shape of one-pattern signal
        array1 = np.zeros(shape=(10, 10))
        s1 = EBSD(array1)
        assert array1.shape == s1.axes_manager.shape
        # SEM metadata
        kp_md = kikuchipy_metadata()
        sem_node = metadata_nodes(ebsd=False)
        assert_dictionary(kp_md.get_item(sem_node),
                          s1.metadata.get_item(sem_node))
        # Phases metadata
        assert s1.metadata.has_item('Sample.Phases')

    def test_set_experimental_parameters(self, signal):
        p = {'detector': 'NORDIF UF-1100', 'azimuth_angle': 1.0,
             'elevation_angle': 1.0, 'sample_tilt': 70.0,
             'working_distance': 23.2, 'binning': 8, 'exposure_time': 0.01,
             'grid_type': 'square', 'gain': 10, 'frame_number': 4,
             'frame_rate': 100, 'scan_time': 60.0, 'beam_energy': 20.0,
             'xpc': 0.5, 'ypc': 0.5, 'zpc': 15000.0,
             'static_background': np.ones(shape=(10, 10)),
             'manufacturer': 'NORDIF', 'version': '3.1.2',
             'microscope': 'Hitachi SU-6600', 'magnification': 500}
        signal.set_experimental_parameters(**p)
        ebsd_node = metadata_nodes(sem=False)
        md_dict = signal.metadata.get_item(ebsd_node).as_dictionary()
        assert_dictionary(p, md_dict)

    def test_set_phase_parameters(self, signal):
        p = {'number': 1,
             'atom_coordinates': {
                 '1': {'atom': 'Ni',
                       'coordinates': [0, 0, 0],
                       'site_occupation': 1,
                       'debye_waller_factor': 0.0035}},
             'formula': 'Ni',
             'info': 'Some sample info',
             'lattice_constants': [0.35236, 0.35236, 0.35236, 90, 90, 90],
             'laue_group': 'm3m',
             'material_name': 'Ni',
             'point_group': '432',
             'space_group': 225,
             'setting': 1,
             'symmetry': 43}
        signal.set_phase_parameters(**p)
        md_dict = signal.metadata.get_item('Sample.Phases.1').as_dictionary()
        p.pop('number')
        assert_dictionary(p, md_dict)

    def test_set_scan_calibration(self, signal):
        (new_step_x, new_step_y) = (2, 3)
        signal.set_scan_calibration(step_x=new_step_x, step_y=new_step_y)
        x, y = signal.axes_manager.navigation_axes
        assert (x.name, y.name) == ('x', 'y')
        assert (x.scale, y.scale) == (new_step_x, new_step_y)
        assert x.units, y.units == u'\u03BC' + 'm'

    def test_set_detector_calibration(self, signal):
        delta = 70
        signal.set_detector_calibration(delta=delta)
        dx, dy = signal.axes_manager.signal_axes
        centre = np.array(signal.axes_manager.signal_shape) / 2 * delta
        assert dx.units, dy.units == u'\u03BC' + 'm'
        assert dx.scale, dy.scale == delta
        assert dx.offset, dy.offset == -centre

    @pytest.mark.parametrize(
        'operation, relative', [('subtract', True), ('subtract', False),
                                ('divide', True), ('divide', False)])
    def test_static_background_correction(
            self, signal, background_pattern, operation, relative):
        signal.static_background_correction(
            operation=operation, relative=relative,
            static_bg=background_pattern)
