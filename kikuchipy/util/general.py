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

from typing import NoReturn, Union, List, Optional, Any
from functools import reduce

from hyperspy.misc.utils import DictionaryTreeBrowser


def _write_parameters_to_dictionary(
    parameters: dict, dictionary: DictionaryTreeBrowser, node: str
) -> NoReturn:
    """Write dictionary of parameters to DictionaryTreeBrowser.

    Parameters
    ----------
    parameters
        Dictionary of parameters to write to dictionary.
    dictionary
        Dictionary to write parameters to.
    node
        String like 'Acquisition_instrument.SEM' etc. with dictionary
        nodes to write parameters to.

    """

    for key, val in parameters.items():
        if val is not None:
            dictionary.set_item(node + "." + key, val)


def _delete_from_nested_dictionary(
    dictionary: Union[dict, DictionaryTreeBrowser], keys: List[str],
) -> Union[dict, DictionaryTreeBrowser]:
    """Delete key(s) from a nested dictionary.

    Parameters
    ----------
    dictionary
        Dictionary to delete key(s) from.
    keys
        Key(s) to delete.

    Returns
    -------
    modified_dict
        Dictionary without deleted keys.

    """

    dict_type = type(dictionary)
    if isinstance(dictionary, DictionaryTreeBrowser):
        dictionary = dictionary.as_dictionary()
    modified_dict = {}
    for key, val in dictionary.items():
        if key not in keys:
            if isinstance(val, dict):
                modified_dict[key] = _delete_from_nested_dictionary(val, keys)
            else:
                modified_dict[key] = val
    if dict_type != dict:  # Revert to DictionaryTreeBrowser
        modified_dict = DictionaryTreeBrowser(modified_dict)
    return modified_dict


def _get_nested_dictionary(
    dictionary: Union[dict, DictionaryTreeBrowser],
    keys: List[str],
    default: Optional[Any] = None,
) -> dict:
    """Get key from a nested dictionary, returning a default value if
    not found.

    Parameters
    ----------
    dictionary
        Dictionary to search through
    keys
        List of keys to get values from.
    default
        Default value to return if `keys` are not found.

    """

    if isinstance(dictionary, DictionaryTreeBrowser):
        dictionary = dictionary.as_dictionary()
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
        keys.split("."),
        dictionary,
    )
