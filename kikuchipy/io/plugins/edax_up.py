# Copyright 2019-2022 The kikuchipy developers
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

"""Reader of EBSD data from EDAX TSL up1 and up2 files."""

# __all__ = ["file_reader"]


# Plugin characteristics
# ----------------------
format_name = "edax_up"
description = (
    "Read support for electron backscatter diffraction patterns stored "
    "in a binary file formatted in EDAX TSL's up1/up2 format with file "
    "extension '.up1' or '.up2'."
)
full_support = False
# Recognised file extension
file_extensions = ["up1", "up2"]
default_extension = 0
# Writing capabilities (signal dimensions, navigation dimensions)
writes = False


class EDAXBinaryFileReader:
    """EDAX TSL's binary .up1 and .up2 file reader."""

    def __init__(self):
        pass
