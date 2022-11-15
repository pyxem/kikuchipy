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

"""Internal tools for overwriting HyperSpy class methods."""

import inspect
from typing import Callable, List, Union


def get_parameters(
    method: Callable, params_of_interest: List[str], args: tuple, kwargs: dict
) -> Union[dict, None]:
    sig = inspect.signature(method)

    params = {}
    # Set keys with default values
    for k, v in sig.parameters.items():
        if k != "self":
            default = v.default
            if default is inspect.Signature.empty:
                default = None
            params[k] = default

    # Update any positional arguments
    if len(args) != 0:
        keys = list(params.keys())
        for i, a in enumerate(args):
            k = keys[i]
            params[k] = a

    # Update any keyword arguments
    for k, v in kwargs.items():
        if k in params.keys():
            params[k] = v

    # Extract parameters of interest, returning None if the parameter
    # could not be found
    try:
        out = {k: params[k] for k in params_of_interest}
    except KeyError:
        out = None

    return out
