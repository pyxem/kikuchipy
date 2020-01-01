# -*- coding: utf-8 -*-
# Copyright 2019-2020 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

from functools import reduce

from hyperspy.misc.utils import DictionaryTreeBrowser


def _write_parameters_to_dictionary(parameters, dictionary, node):
    """Write dictionary of parameters to DictionaryTreeBrowser.

    Parameters
    ----------
    parameters : dictionary
        Dictionary of parameters to write to dictionary.
    dictionary : DictionaryTreeBrowser
        Dictionary to write parameters to.
    node : str
        String like 'Acquisition_instrument.SEM' etc. with dictionary
        nodes to write parameters to.
    """

    for key, val in parameters.items():
        if val is not None:
            dictionary.set_item(node + "." + key, val)


def _delete_from_nested_dictionary(dictionary, keys):
    """Delete key(s) from a nested dictionary.

    Parameters
    ----------
    dictionary : dictionary or DictionaryTreeBrowser
        Dictionary to delete key(s) from.
    keys : dict_values
        Key(s) to delete.

    Returns
    -------
    modified_dict : dictionary or DictionaryTreeBrowser
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


def _get_nested_dictionary(dictionary, keys, default=None):
    """Get key from a nested dictionary, returning a default value if
    not found.

    Parameters
    ----------
    dictionary : dictionary or DictionaryTreeBrowser
        Dictionary to search through
    keys : list
        List of keys to get values from.
    default : optional
        Default value to return if `keys` are not found.
    """

    if isinstance(dictionary, DictionaryTreeBrowser):
        dictionary = dictionary.as_dictionary()
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
        keys.split("."),
        dictionary,
    )
