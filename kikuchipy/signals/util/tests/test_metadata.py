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

import pytest

from kikuchipy.signals.util._metadata import (
    ebsd_metadata,
    metadata_nodes,
    _set_metadata_from_mapping,
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

        assert metadata_nodes("sem") == sem_node
        assert metadata_nodes("ebsd") == ebsd_node
        assert metadata_nodes() == [sem_node, ebsd_node]
        assert metadata_nodes(["ebsd", "sem"]) == [ebsd_node, sem_node]

    def test_set_metadata_from_mapping(self):
        """Updating DictionaryTreeBrowser with values from a dictionary
        via a mapping.
        """
        sem_node, ebsd_node = metadata_nodes(["sem", "ebsd"])
        md = ebsd_metadata()

        old_val1, new_val1 = (-1, 20)
        old_val2, new_val2 = (-1, 2)
        omd = {"V": new_val1, "xpc": {"xpc2": {"xpc3": new_val2}}}
        key1 = f"{sem_node}.beam_energy"
        key2 = f"{ebsd_node}.xpc"
        assert md.get_item(key1) == old_val1
        assert md.get_item(key2) == old_val2

        mapping = {key1: "V", key2: ["xpc", "xpc2", "xpc3"]}
        _set_metadata_from_mapping(omd, md, mapping)
        assert md.get_item(key1) == new_val1
        assert md.get_item(key2) == new_val2

        with pytest.warns(UserWarning, match="Could not read"):
            _ = _set_metadata_from_mapping(omd, md, {"a": "b"})
