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

from kikuchipy.signals.util._metadata import (
    ebsd_metadata,
    ebsd_master_pattern_metadata,
    metadata_nodes,
)


class TestMetadata:
    def test_ebsd_metadata(self):
        sem_node, ebsd_node = metadata_nodes(["sem", "ebsd"])
        md = ebsd_metadata()
        assert md.get_item(sem_node + ".microscope") == ""
        assert md.get_item(ebsd_node + ".xpc") == -1.0

    def test_metadata_nodes(self):
        sem_node = "Acquisition_instrument.SEM"
        ebsd_node = sem_node + ".Detector.EBSD"
        simulation_node = "Simulation"
        ebsd_master_pattern_node = simulation_node + ".EBSD_master_pattern"

        assert metadata_nodes("sem") == sem_node
        assert metadata_nodes("ebsd") == ebsd_node
        assert metadata_nodes() == [
            sem_node,
            ebsd_node,
            ebsd_master_pattern_node,
        ]
        assert metadata_nodes(["ebsd", "sem"]) == [
            ebsd_node,
            sem_node,
        ]
        assert metadata_nodes(("sem", "ebsd_master_pattern")) == [
            sem_node,
            ebsd_master_pattern_node,
        ]

    def test_ebsd_masterpattern_metadata(self):
        ebsd_mp_node = metadata_nodes("ebsd_master_pattern")
        md = ebsd_master_pattern_metadata()

        assert md.get_item(ebsd_mp_node + ".BSE_simulation.mode") == ""
        assert (
            md.get_item(
                ebsd_mp_node + ".Master_pattern.smallest_interplanar_spacing"
            )
            == -1.0
        )
