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

"""Private tools for refinement of crystal orientations and projection
centers by optimizing the similarity between experimental and simulated
patterns.

This module and documentation is only relevant for kikuchipy developers,
not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with no
    warning.
"""

# fmt: off
SUPPORTED_OPTIMIZATION_METHODS = {
    # Local
    "minimize": {
        "type": "local",
        "supports_bounds": True
    },
    # Global
    "basinhopping": {
        "type": "global",
        "supports_bounds": False
    },
    "differential_evolution": {
        "type": "global",
        "supports_bounds": True
    },
    "dual_annealing": {
        "type": "global",
        "supports_bounds": True
    },
    "shgo": {
        "type": "global",
        "supports_bounds": True
    },
}
# fmt: on
