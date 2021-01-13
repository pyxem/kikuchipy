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
from h5py import File
import numpy as np
from orix.crystal_map import CrystalMap
import pytest

from kikuchipy.io._io import load
from kikuchipy.io.plugins.emsoft_ebsd import (
    _check_file_format,
    _crystaldata2phase,
)
from kikuchipy.io.plugins.h5ebsd import hdf5group2dict

DIR_PATH = os.path.dirname(__file__)
EMSOFT_FILE = os.path.join(
    DIR_PATH, "../../../data/emsoft_ebsd/simulated_ebsd.h5"
)


class TestEMsoftEBSDReader:
    def test_file_reader(self):
        """Test correct data shape, axes properties and metadata."""
        s = load(EMSOFT_FILE)

        assert isinstance(s.xmap, CrystalMap)
        assert s.data.shape == (10, 10, 10)
        assert s.axes_manager["dx"].scale == 70
        assert s.axes_manager["dx"].units == "um"
        assert s.axes_manager["x"].units == "px"

    @pytest.mark.parametrize("scan_size", [10, (1, 10), (5, 2)])
    def test_scan_size(self, scan_size):
        """Scan size parameter works as expected."""
        s = load(EMSOFT_FILE, scan_size=scan_size)

        sy, sx = (10, 10)
        if isinstance(scan_size, int):
            desired_shape = (scan_size, sy, sx)
        else:
            desired_shape = scan_size + (sy, sx)

        assert s.data.shape == desired_shape

    def test_read_lazy(self):
        """Lazy parameter works as expected."""
        s = load(EMSOFT_FILE, lazy=True)

        assert isinstance(s.data, da.Array)

    def test_check_file_format(self, save_path_hdf5):
        """Wrong file format raises an error."""
        with File(save_path_hdf5, mode="w") as f:
            g1 = f.create_group("EMheader")
            g2 = g1.create_group("EBSD")
            g2.create_dataset(
                "ProgramName", data=np.array([b"EMEBSDD.f90"], dtype="S11"),
            )
            with pytest.raises(IOError, match=".* is not in EMsoft's format "):
                _ = _check_file_format(f)

    def test_crystaldata2phase(self):
        """A Phase object is correctly returned."""
        with File(EMSOFT_FILE, mode="r") as f:
            xtal_dict = hdf5group2dict(f["CrystalData"])
        phase = _crystaldata2phase(xtal_dict)

        assert phase.name == ""
        assert phase.space_group.number == 140
        assert phase.color == "tab:blue"

        structure = phase.structure
        assert np.allclose(
            structure.lattice.abcABG(), [0.5949, 0.5949, 0.5821, 90, 90, 90]
        )
        assert np.allclose(
            structure.xyz, [[0.1587, 0.6587, 0], [0, 0, 0.25]], atol=1e-4
        )
        assert np.allclose(structure.occupancy, [1, 1])
        assert np.allclose(structure.Bisoequiv, [0.5] * 2)
        assert np.compare_chararrays(
            structure.element,
            np.array(["13", "29"], dtype="|S2"),
            "==",
            rstrip=False,
        ).all()

    def test_crystaldata2phase_single_atom(self):
        """A Phase object is correctly returned when there is only one
        atom present.
        """
        with File(EMSOFT_FILE, mode="r") as f:
            xtal_dict = hdf5group2dict(f["CrystalData"])
        xtal_dict["Natomtypes"] = 1
        xtal_dict["AtomData"] = xtal_dict["AtomData"][:, 0][..., np.newaxis]
        xtal_dict["Atomtypes"] = xtal_dict["Atomtypes"][0]

        phase = _crystaldata2phase(xtal_dict)

        assert len(phase.structure) == 1
