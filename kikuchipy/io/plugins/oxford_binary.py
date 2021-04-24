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

"""Read support for EBSD patterns in Oxford Instrument's binary .ebsp
format.
"""

from typing import Dict, List


# Plugin characteristics
# ----------------------
format_name = "Oxford"
description = "Read support for Oxford Instrument's .ebsp file."
full_support = False
# Recognised file extension
file_extensions = ["ebsp"]
default_extension = 0
# Writing capabilities (signal dimensions, navigation dimensions)
writes = False


def file_reader(filename: str) -> List[Dict]:
    return
