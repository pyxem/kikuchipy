# -*- coding: utf-8 -*-
# Copyright 2019-2020 The kikuchipy developers
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
import pytest

from kikuchipy.io._io import load
from kikuchipy.io.plugins.emsoft_ebsd import _check_file_format
from kikuchipy.signals.tests.test_ebsd import assert_dictionary

DIR_PATH = os.path.dirname(__file__)
EMSOFT_FILE = os.path.join(
    DIR_PATH, "../../../data/emsoft_ebsd/simulated_ebsd.h5"
)

METADATA = {
    "General": {
        "original_filename": "simulated_ebsd.h5",
        "title": "simulated_ebsd",
    },
    "Sample": {
        "Phases": {
            "1": {
                "atom_coordinates": {
                    "1": {
                        "atom": 13,
                        "coordinates": np.array([0.1587, 0.6587, 0]),
                        "site_occupation": 1.0,
                        "debye_waller_factor": 0.005,
                    },
                    "2": {
                        "atom": 29,
                        "coordinates": np.array([0, 0, 0.25]),
                        "site_occupation": 1.0,
                        "debye_waller_factor": 0.005,
                    },
                },
                "lattice_constants": np.array(
                    [0.5949, 0.5949, 0.5821, 90, 90, 90]
                ),
                "setting": 1,
                "source": "Su Y.C., Yan J., Lu P.T., Su J.T.: Thermodynamic...",
                "space_group": 140,
            }
        }
    },
    "Signal": {"binned": False, "signal_type": "EBSD"},
}

AXES_MANAGER = {
    "y": {
        "name": "y",
        "scale": 1.0,
        "offset": 0.0,
        "size": 1,
        "units": "px",
        "navigate": True,
    },
    "x": {
        "name": "x",
        "scale": 1.0,
        "offset": 0.0,
        "size": 10,
        "units": "px",
        "navigate": True,
    },
    "dy": {
        "name": "dy",
        "scale": 70,
        "offset": 0,
        "size": 10,
        "units": "um",
        "navigate": False,
    },
    "dx": {
        "name": "dx",
        "scale": 70,
        "offset": 0,
        "size": 10,
        "units": "um",
        "navigate": False,
    },
}


def setup_axes_manager(axes=None):
    if axes is None:
        axes = ["y", "x", "dy", "dx"]
    d = {}
    for i, a in enumerate(axes):
        d["axis-" + str(i)] = AXES_MANAGER[a]
    return d


class TestEMsoftEBSDReader:
    def test_file_reader(self):
        s = load(EMSOFT_FILE)

        assert s.data.shape == (10, 10, 10)
        assert_dictionary(s.metadata.as_dictionary(), METADATA)

    @pytest.mark.parametrize("scan_size", [10, (1, 10), (5, 2)])
    def test_scan_size(self, scan_size):
        s = load(EMSOFT_FILE, scan_size=scan_size)

        sy, sx = (10, 10)
        if isinstance(scan_size, int):
            desired_shape = (scan_size, sy, sx)
        else:
            desired_shape = scan_size + (sy, sx)

        assert s.data.shape == desired_shape

        if len(desired_shape) == 3:
            axes_manager = setup_axes_manager(["x", "dy", "dx"])
            axes_manager["axis-0"]["size"] = desired_shape[0]
        else:
            axes_manager = setup_axes_manager()
            axes_manager["axis-0"]["size"] = desired_shape[0]
            axes_manager["axis-1"]["size"] = desired_shape[1]

        assert s.axes_manager.as_dictionary() == axes_manager
        assert_dictionary(s.metadata.as_dictionary(), METADATA)

    def test_read_lazy(self):
        s = load(EMSOFT_FILE, lazy=True)

        assert isinstance(s.data, da.Array)

    def test_check_file_format(self, save_path_hdf5):
        with File(save_path_hdf5, mode="w") as f:
            g1 = f.create_group("EMheader")
            g2 = g1.create_group("EBSD")
            g2.create_dataset(
                "ProgramName", data=np.array([b"EMEBSDD.f90"], dtype="S11"),
            )
            with pytest.raises(IOError, match=".* is not in EMsoft's h5ebsd "):
                _ = _check_file_format(f)
