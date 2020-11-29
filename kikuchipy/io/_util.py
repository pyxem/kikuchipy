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

from functools import reduce
import os
from typing import Union, Any, List, Optional
import warnings

from hyperspy.misc.utils import DictionaryTreeBrowser


def _get_input_bool(question: str) -> bool:
    """Get input from user on boolean choice, returning the answer.

    Parameters
    ----------
    question
        Question to ask user.
    """
    try:
        answer = input(question)
        answer = answer.lower()
        while (answer != "y") and (answer != "n"):
            print("Please answer y or n.")
            answer = input(question)
        if answer.lower() == "y":
            return True
        elif answer.lower() == "n":
            return False
    except OSError:
        warnings.warn(
            "Your terminal does not support raw input. Not adding scan. To add "
            "the scan use `add_scan=True`"
        )
        return False


def _get_input_variable(question: str, var_type: Any) -> Union[None, Any]:
    """Get variable input from user, returning the variable.

    Parameters
    ----------
    question
        Question to ask user.
    var_type
        Type of variable to return.
    """
    try:
        answer = input(question)
        while type(answer) != var_type:
            try:
                answer = var_type(answer)
            except (TypeError, ValueError):
                print(f"Please enter a variable of type {var_type}:\n")
                answer = input(question)
        return answer
    except OSError:
        warnings.warn(
            "Your terminal does not support raw input. Not adding scan. To add "
            "the scan use `add_scan=True`"
        )
        return None


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
    keys: Union[str, List[str]],
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
    if not isinstance(keys, list):
        keys = keys.split(".")
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
        keys,
        dictionary,
    )


def _ensure_directory(filename: str):
    """Check if the filename path exists, create it if not."""
    directory = os.path.split(filename)[0]
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
