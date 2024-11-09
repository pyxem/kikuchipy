# Copyright 2019-2024 The kikuchipy developers
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

"""Vector-related tools useful across modules."""

from typing import Literal

from kikuchipy._utils.exceptions import UnknownHemisphereError, UnknownProjectionError

ValidHemispheres = Literal["upper", "lower", "both"]
ValidProjections = Literal["stereographic", "lambert"]


def poles_from_hemisphere(hemisphere: ValidHemispheres) -> list[int]:
    """Return pole(s) (-1, 1) for the given hemisphere(s) (upper, lower,
    or both).

    Raises
    ------
    ValueError
        If an unknown hemisphere is given.
    """
    hemi = parse_hemisphere(hemisphere)
    match hemi:
        case "upper":
            return [-1]
        case "lower":
            return [1]
        case _:
            return [-1, 1]


def parse_hemisphere(hemisphere: ValidHemispheres) -> str:
    hemi = hemisphere.lower()
    if hemi not in ["upper", "lower", "both"]:
        raise UnknownHemisphereError(hemisphere)
    else:
        return hemi


def parse_projection(projection: ValidProjections) -> str:
    proj = projection.lower()
    if proj not in ["stereographic", "lambert"]:
        raise UnknownProjectionError(projection)
    else:
        return proj
