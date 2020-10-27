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

import numpy as np
from orix.crystal_map import CrystalMap
from orix.quaternion import Rotation

from kikuchipy import load
from kikuchipy.signals import EBSD

from kikuchipy.indexing.static_dictionary_indexing import (
    StaticDictionaryIndexing,
)

DIR_PATH = os.path.dirname(__file__)
KIKUCHIPY_FILE = os.path.join(DIR_PATH, "../../data/kikuchipy/patterns.h5")


class TestStaticDictionaryIndexing:
    def test_init(self):
        s = load(KIKUCHIPY_FILE)

        self.sd = StaticDictionaryIndexing(s)
        pass

    def test_init_index(self):
        pass

    def test_index(self):
        s = load(KIKUCHIPY_FILE)
        s_dict1 = EBSD(s.data.reshape(-1, 60, 60))
        s_dict2 = EBSD(s.data.reshape(-1, 60, 60))
        s_dict1._xmap = CrystalMap(Rotation(np.zeros((9, 4))), x=np.arange(9))
        s_dict2._xmap = CrystalMap(Rotation(np.zeros((9, 4))), x=np.arange(9))
        s_dict1.xmap.phases._dict[0].name = "1"
        s_dict2.xmap.phases._dict[0].name = "2"
        sd = StaticDictionaryIndexing([s_dict1, s_dict2])
        res = sd.index(s)
        cm1, _, _ = res
        assert np.allclose(cm1.metric_results, 1)
        # np.isin(["metric_results","template_indices","osm"],list(cm.prop.keys()))
        assert np.all(
            [
                "metric_results" in cm.prop
                and "template_indices" in cm.prop
                and "osm" in cm.prop
                for cm in res
            ]
        )
