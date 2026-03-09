#
# Copyright 2019-2026 the kikuchipy developers
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
#

"""Utilities for importing objects and working with RosettaSciIO.

Used in at least the IO and signals modules.
"""

from typing import Callable

import numpy as np
from packaging.version import Version

from kikuchipy._constants import dependency_version

if dependency_version["rosettasciio"] >= Version("0.12"):
    from rsciio.utils.file import memmap_distributed
    from rsciio.utils.rgb import RGB_DTYPES
else:
    from rsciio.utils.distributed import memmap_distributed
    from rsciio.utils.rgb_tools import rgb_dtypes as RGB_DTYPES

RGB_DTYPES: dict[str, np.dtype]
memmap_distributed: Callable

__all__ = [
    "RGB_DTYPES",
    "memmap_distributed",
]
