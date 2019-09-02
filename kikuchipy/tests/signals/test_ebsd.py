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
from kikuchipy.utils.io_utils import metadata_nodes
import numpy as np
import pytest


def assert_dictionary(input_dict, output_dict):
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
        array0 = np.zeros(shape=(10, 10, 10, 10))
        s0 = EBSD(array0)
        assert array0.shape == s0.axes_manager.shape

        with pytest.raises(ValueError):
            EBSD(np.zeros(10))

        array1 = np.zeros(shape=(10, 10))
        s1 = EBSD(array1)
        assert array1.shape == s1.axes_manager.shape

    def test_set_experimental_parameters(self):
        s = EBSD(np.zeros((10, 10, 10, 10)))
        p = {'detector': 'NORDIF UF-1100', 'azimuth_angle': 1.0,
             'elevation_angle': 1.0, 'sample_tilt': 70.0,
             'working_distance': 23.2, 'binning': 8, 'exposure_time': 0.01,
             'grid_type': 'square', 'gain': 10, 'frame_number': 4,
             'frame_rate': 100, 'scan_time': 60.0, 'beam_energy': 20.0,
             'xpc': 0.5, 'ypc': 0.5, 'zpc': 15000.0,
             'static_background': np.ones(shape=(10, 10)),
             'manufacturer': 'NORDIF', 'version': '3.1.2',
             'microscope': 'Hitachi SU-6600', 'magnification': 500}
        s.set_experimental_parameters(**p)
        ebsd_node = metadata_nodes(sem=False)
        md_dict = s.metadata.get_item(ebsd_node).as_dictionary()
        assert_dictionary(p, md_dict)

    def test_set_phase_parameters(self):
        s = EBSD(np.zeros((10, 10, 10, 10)))
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
        s.set_phase_parameters(**p)
        md_dict = s.metadata.get_item('Sample.Phases.1').as_dictionary()
        p.pop('number')
        assert_dictionary(p, md_dict)
