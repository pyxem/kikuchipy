# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.

import os

import dask.array as da
import numpy as np
import pytest

import kikuchipy as kp


DIR_PATH = os.path.dirname(__file__)
OXFORD_PATH = os.path.join(DIR_PATH, "../../../data/oxford_binary")
OXFORD_FILE = os.path.join(OXFORD_PATH, "patterns.ebsp")


class TestOxfordBinaryReader:
    def test_load(self):
        """Load into memory."""
        s = kp.load(OXFORD_FILE)
        s2 = kp.data.nickel_ebsd_small()

        assert isinstance(s, kp.signals.EBSD)
        assert np.allclose(s.data, s2.data)

    def test_load_lazy(self):
        """Load lazily."""
        s = kp.load(OXFORD_FILE, lazy=True)
        s2 = kp.data.nickel_ebsd_small()

        assert isinstance(s, kp.signals.LazyEBSD)
        assert isinstance(s.data, da.Array)
        s.compute()
        assert np.allclose(s.data, s2.data)

    @pytest.mark.parametrize(
        "oxford_binary_file",
        [((2, 3), (60, 60), np.uint8, 2, True, True)],
        indirect=["oxford_binary_file"],
    )
    def test_compressed_patterns_raises(self, oxford_binary_file):
        """Ensure explanatory error message is raised when a file we
        cannot read is tried to be read from.
        """
        with pytest.raises(NotImplementedError, match="Cannot read compressed"):
            _ = kp.load(oxford_binary_file.name)

    @pytest.mark.parametrize(
        "oxford_binary_file, dtype",
        [
            (((2, 3), (60, 60), np.uint8, 2, False, True), np.uint8),
            (((2, 3), (60, 60), np.uint16, 2, False, True), np.uint16),
        ],
        indirect=["oxford_binary_file"],
    )
    def test_dtype(self, oxford_binary_file, dtype):
        """Ensure both uint8 and uint16 patterns can be read."""
        s = kp.load(oxford_binary_file.name)
        assert np.issubdtype(s.data.dtype, dtype)

    @pytest.mark.parametrize(
        "oxford_binary_file",
        [((2, 3), (60, 60), np.uint8, 2, False, False)],
        indirect=["oxford_binary_file"],
    )
    def test_not_all_patterns_present(self, oxford_binary_file):
        """Ensure files with only non-indexed patterns can be read."""
        s = kp.load(oxford_binary_file.name)
        assert s.axes_manager.navigation_shape == (5,)
        # (2, 2) is missing
        assert np.allclose(s.original_metadata.beam_y, [0, 1, 1, 1, 0])
        assert np.allclose(s.original_metadata.beam_x, [2, 0, 1, 2, 0])

    @pytest.mark.parametrize(
        "oxford_binary_file, ver, desired_nav_shape",
        [
            (((2, 3), (60, 60), np.uint8, 2, False, True), 2, (2, 3)),
            (((2, 3), (60, 60), np.uint16, 1, False, True), 1, (2, 3)),
            (((2, 3), (60, 60), np.uint8, 0, False, True), 0, (6,)),
        ],
        indirect=["oxford_binary_file"],
    )
    def test_versions(self, oxford_binary_file, ver, desired_nav_shape):
        """Ensure that versions 0, 1 and > 1 can be read."""
        s = kp.load(oxford_binary_file.name)
        assert s.axes_manager.navigation_shape[::-1] == desired_nav_shape
        if ver > 0:
            assert s.original_metadata.has_item("beam_x")
            assert s.original_metadata.has_item("beam_y")

    @pytest.mark.parametrize(
        "oxford_binary_file, n_patterns",
        [
            (((2, 3), (60, 60), np.uint8, 2, False, True), 6),
            (((3, 4), (62, 73), np.uint8, 2, False, True), 12),
        ],
        indirect=["oxford_binary_file"],
    )
    def test_guess_number_of_patterns(self, oxford_binary_file, n_patterns):
        """Ensure that the function guessing the number of patterns in
        the file works.
        """
        with open(oxford_binary_file.name, mode="rb") as f:
            fox = kp.io.plugins.oxford_binary.OxfordBinaryFileReader(f)
            assert fox.n_patterns == n_patterns
