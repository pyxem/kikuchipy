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
        array = np.zeros(shape=(10, 10, 10, 10))
        s = EBSD(array)
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
        for k, v in md_dict.items():
            if isinstance(p[k], np.ndarray):
                assert md_dict[k].all() == p[k].all()
            else:
                assert md_dict[k] == p[k]
