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
import gc
import tempfile
import pytest
import h5py
import numpy as np
import dask.array as da
import kikuchipy as kp

DIR_PATH = os.path.dirname(__file__)
PATTERN_FILE = os.path.join(DIR_PATH, '../../data/kikuchipy/patterns.h5')
BG_FILE = os.path.join(
    DIR_PATH, '../../data/nordif/Background acquisition pattern.bmp')
AXES_MANAGER = {
    'axis-0': {
        'name': 'y', 'scale': 1.5, 'offset': 0.0, 'size': 3, 'units': 'μm',
        'navigate': True},
    'axis-1': {
        'name': 'x', 'scale': 1.5, 'offset': 0.0, 'size': 3, 'units': 'μm',
        'navigate': True},
    'axis-2': {
        'name': 'dy', 'scale': 1.0, 'offset': 0.0, 'size': 60, 'units': 'μm',
        'navigate': False},
    'axis-3': {
        'name': 'dx', 'scale': 1.0, 'offset': 0.0, 'size': 60, 'units': 'μm',
        'navigate': False}}


@pytest.fixture()
def save_path():
    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, 'kikuchipy', 'patterns_temp.h5')
        yield file_path
        gc.collect()


class Testh5ebsd:

    def test_load_kikuchipy(self):
        s = kp.load(PATTERN_FILE)

        assert s.data.shape == (3, 3, 60, 60)
        assert s.axes_manager.as_dictionary() == AXES_MANAGER

#    def test_load_edax(self):
#        return 0

#    def test_load_bruker(self):
#        return 0

    def test_load_save_cycle(self, save_path):
        s = kp.load(PATTERN_FILE)

        # Check that metadata is read correctly
        assert s.metadata.Acquisition_instrument.SEM.Detector.EBSD.xpc == -5.64
        assert s.metadata.General.title == 'patterns Scan 1'

        s.save(save_path, overwrite=True)
        s_reload = kp.load(save_path)
        np.testing.assert_equal(s.data, s_reload.data)

        # Change data set name to make metadata equal
        s_reload.metadata.General.title = s.metadata.General.title
        np.testing.assert_equal(
            s_reload.metadata.as_dictionary(), s.metadata.as_dictionary())

    @pytest.mark.parametrize('scans', ([1, 2], [1, 2, 3]))
    def test_load_multiple(self, scans):
        if len(scans) == 3:
            with pytest.warns(UserWarning):
                s1, s2 = kp.load(PATTERN_FILE, scans=scans)
        else:
            s1, s2 = kp.load(PATTERN_FILE, scans=scans)

        np.testing.assert_equal(s1.data, s2.data)
        with pytest.raises(
                AssertionError,
                match="\nItems are not equal:\nkey='title'\nkey='General'\n\n "
                      "ACTUAL: 'patterns Scan 1'\n DESIRED: 'patterns Scan 2'"):
            np.testing.assert_equal(
                s1.metadata.as_dictionary(), s2.metadata.as_dictionary())
        s2.metadata.General.title = s1.metadata.General.title
        np.testing.assert_equal(
            s1.metadata.as_dictionary(), s2.metadata.as_dictionary())

    def test_load_save_lazy(self, save_path):
        s = kp.load(PATTERN_FILE, lazy=True)
        assert isinstance(s.data, da.Array)
        s.save(save_path, overwrite=True)
        s_reload = kp.load(save_path, lazy=True)
        assert s.data.shape == s_reload.data.shape

    def test_load_readonly(self):
        s = kp.load(PATTERN_FILE, lazy=True)
        k = next(filter(lambda x: isinstance(x, str) and
                        x.startswith("array-original"),
                        s.data.dask.keys()))
        mm = s.data.dask[k]
        assert isinstance(mm, h5py.Dataset)
        with pytest.raises(NotImplementedError):
            s.data[:] = 23

    def test_save_fresh(self, save_path):
        scan_size = (10, 3)
        pattern_size = (5, 5)
        data_shape = scan_size + pattern_size
        s = kp.signals.EBSD(
            (255 * np.random.rand(*data_shape)).astype(np.uint8))
        s.save(save_path, overwrite=True)
        s_reload = kp.load(save_path)
        np.testing.assert_equal(s.data, s_reload.data)

    @pytest.mark.parametrize('scan_number', (1, 2))
    def test_save_multiple(self, save_path, scan_number):
        s1, s2 = kp.load(PATTERN_FILE, scans=[1, 2])
        s1.save(save_path)
        error = 'Invalid scan number'
        with pytest.raises(OSError, match=error):
            s2.save(save_path, add_scan=True)
        if scan_number == 1:
            with pytest.raises(OSError, match=error):
                s2.save(save_path, add_scan=True, scan_number=scan_number)
        else:
            s2.save(save_path, add_scan=True, scan_number=scan_number)
