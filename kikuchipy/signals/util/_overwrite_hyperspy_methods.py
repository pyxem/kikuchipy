# Copyright 2019-2023 The kikuchipy developers
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

import functools
import inspect
import re
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


class insert_doc_disclaimer:
    """Decorator to add a disclaimer to methods inhertied from HyperSpy.

    Adopted from Dask's insertion of disclaimers into methods they
    inherit from NumPy.

    Parameters
    ----------
    cls
        Class, typically Signal2D.
    meth
        Method, e.g. rebin.
    """

    def __init__(self, cls, meth):
        self.cls_name = cls.__name__
        self.doc = meth.__doc__
        self.name = meth.__name__

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__doc__ = self._insert_doc_disclaimer()
        return wrapper

    def _insert_doc_disclaimer(self):
        doc = self.doc
        if doc is None:
            return doc
        i = doc.find("\n\n")
        if i != -1:
            # Disclaimer
            l1 = (
                "This docstring was copied from HyperSpy's "
                f"{self.cls_name}.{self.name}.\n"
            )
            l2 = "Some inconsistencies with the kikuchipy version may exist."

            # Parts
            head = doc[: i + 2]
            tail = doc[i + 2 :]
            indent = re.match(r"\s*", tail).group(0)
            parts = [head, indent, l1, indent, l2, "\n\n", tail]
            doc = "".join(parts)

        return doc
