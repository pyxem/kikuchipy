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

import gc
import os
import tempfile

from diffpy.structure import Atom, Lattice, Structure
import numpy as np
from orix.crystal_map import Phase
from orix.quaternion.rotation import Rotation
from orix.vector import Vector3d, neo_euler
import pytest

from kikuchipy.crystallography import ReciprocalLatticePoint
from kikuchipy.detectors import EBSDDetector
from kikuchipy.signals import EBSD


@pytest.fixture
def dummy_signal():
    """Dummy signal of shape <3, 3|3, 3>. If this is changed, all tests
    using this signal will fail since they compare the output from
    methods using this signal (as input) to hard-coded outputs.
    """
    # fmt: off
    dummy_array = np.array(
        [
            5, 6, 5, 7, 6, 5, 6, 1, 0, 9, 7, 8, 7, 0, 8, 8, 7, 6, 0, 3, 3, 5, 2,
            9, 3, 3, 9, 8, 1, 7, 6, 4, 8, 8, 2, 2, 4, 0, 9, 0, 1, 0, 2, 2, 5, 8,
            6, 0, 4, 7, 7, 7, 6, 0, 4, 1, 6, 3, 4, 0, 1, 1, 0, 5, 9, 8, 4, 6, 0,
            2, 9, 2, 9, 4, 3, 6, 5, 6, 2, 5, 9
        ],
        dtype=np.uint8
    ).reshape((3, 3, 3, 3))
    # fmt: on
    return EBSD(dummy_array)


@pytest.fixture
def dummy_background():
    """Dummy static background image for the dummy signal. If this is
    changed, all tests using this background will fail since they
    compare the output from methods using this background (as input) to
    hard-coded outputs.
    """
    return np.array([5, 4, 5, 4, 3, 4, 4, 4, 3], dtype=np.uint8).reshape((3, 3))


@pytest.fixture(params=["h5"])
def save_path_hdf5(request):
    """Temporary file in a temporary directory for use when tests need
    to write, and sometimes read again, a signal to, and from, a file.
    """
    ext = request.param
    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, "patterns_temp." + ext)
        yield file_path
        gc.collect()


@pytest.fixture
def nickel_structure():
    """A diffpy.structure with a Nickel crystal structure."""
    return Structure(
        atoms=[Atom("Ni", [0, 0, 0])],
        lattice=Lattice(3.5236, 3.5236, 3.5236, 90, 90, 90),
    )


@pytest.fixture
def nickel_phase(nickel_structure):
    """A orix.crystal_map.Phase with a Nickel crystal structure and
    symmetry operations.
    """
    return Phase(name="ni", structure=nickel_structure, space_group=225)


@pytest.fixture(params=[1.3])
def nickel_rlp(request, nickel_phase):
    """A set of reciprocal lattice points for a Nickel crystal
    structure with a minimum interplanar spacing.
    """
    return ReciprocalLatticePoint.from_min_dspacing(
        phase=nickel_phase, min_dspacing=request.param,
    )


@pytest.fixture
def pc1():
    """One projection center (PC) in TSL convention."""
    return [0.4210, 0.7794, 0.5049]


@pytest.fixture(params=[(1,)])
def nordif_detector(request, pc1):
    """A NORDIF UF1100 EBSD detector with a TSL PC."""
    return EBSDDetector(
        shape=(60, 60),
        binning=8,
        px_size=70,
        pc=np.ones(request.param + (3,)) * pc1,
        sample_tilt=70,
        tilt=0,
        convention="tsl",
    )


@pytest.fixture
def nickel_rotations():
    """A set of 25 rotations in a TSL sample reference frame (RD-TD-ND).

    The rotations are from an EMsoft indexing of patterns in the region
    of interest (row0:row1, col0:col1) = (79:84, 134:139) of the first
    Nickel data set in this set of scans:
    https://zenodo.org/record/3265037.
    """
    return Rotation.from_euler(
        np.array(
            [
                [0.8662, 0.2033, -0.3483, -0.2951],
                [0.8888, 0.3188, -0.2961, -0.1439],
                [0.8883, 0.3188, -0.2973, -0.1444],
                [0.8884, 0.3187, -0.2975, -0.1437],
                [0.9525, 0.1163, -0.218, -0.1782],
                [0.8658, 0.2031, -0.3486, -0.296],
                [0.8661, 0.203, -0.3486, -0.2954],
                [0.8888, 0.3179, -0.297, -0.1439],
                [0.9728, -0.1634, 0.0677, 0.1494],
                [0.9526, 0.1143, -0.2165, -0.1804],
                [0.8659, 0.2033, -0.3483, -0.2958],
                [0.8663, 0.2029, -0.348, -0.2955],
                [0.8675, 0.1979, -0.3455, -0.298],
                [0.9728, -0.1633, 0.0685, 0.1494],
                [0.9726, -0.1634, 0.0684, 0.1506],
                [0.8657, 0.2031, -0.3481, -0.297],
                [0.8666, 0.2033, -0.3475, -0.2949],
                [0.9111, 0.3315, -0.1267, -0.2095],
                [0.9727, -0.1635, 0.0681, 0.1497],
                [0.9727, -0.1641, 0.0682, 0.1495],
                [0.8657, 0.2024, -0.3471, -0.2986],
                [0.9109, 0.3318, -0.1257, -0.2105],
                [0.9113, 0.3305, -0.1257, -0.2112],
                [0.9725, -0.1643, 0.0691, 0.1497],
                [0.9727, -0.1633, 0.0685, 0.1499],
            ]
        )
    )


@pytest.fixture
def r_tsl2bruker():
    """A rotation from the TSL to bruker sample reference frame."""
    return Rotation.from_neo_euler(
        neo_euler.AxAngle.from_axes_angles(Vector3d.zvector(), np.pi / 2)
    )
