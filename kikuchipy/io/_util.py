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

import os
from typing import Union, Any
import warnings


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
            "Your terminal does not support raw input. Not adding scan. To add the scan"
            " use `add_scan=True`"
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
            "Your terminal does not support raw input. Not adding scan. To add the scan"
            " use `add_scan=True`"
        )
        return None


def _ensure_directory(filename: str):
    """Check if the filename path exists, create it if not."""
    directory = os.path.split(filename)[0]
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
