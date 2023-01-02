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

import kikuchipy as kp
from kikuchipy.signals.util._overwrite_hyperspy_methods import (
    get_parameters,
    insert_doc_disclaimer,
)


def test_get_parameters():
    params1 = ["method", "params_of_interest", "args", "kwargs"]
    out1 = get_parameters(get_parameters, params1, (), {})
    assert list(out1.keys()) == params1
    assert all(v is None for v in out1.values())

    params2 = ["operation", "scale_bg", "static_bg"]
    out2 = get_parameters(
        kp.signals.EBSD.remove_static_background,
        params2,
        args=("subtract",),
        kwargs={"static_bg": [1, 2, 3]},
    )
    assert list(out2.keys()) == params2
    assert out2["operation"] == "subtract"
    assert out2["static_bg"] == [1, 2, 3]
    assert not out2["scale_bg"]

    out3 = get_parameters(
        kp.signals.EBSD.remove_static_background, ["operation", "relative"], (), {}
    )
    assert out3 is None


def test_insert_doc_disclaimer():
    class A:
        def __init__(self, number):
            self.number = number

        def add(self, number):
            """A method.

            Parameters
            ----------
            number

            Returns
            -------
            new_number
            """
            return self.number + number

        def subtract(self, number):
            return self.number - number

    class B(A):
        @insert_doc_disclaimer(A, A.add)
        def add(self, number):
            return super().add(number)

        @insert_doc_disclaimer(A, A.subtract)
        def subtract(self, number):
            return super().subtract(number)

    b = B(1)

    # Adds disclaimer
    assert b.add(1) == 2
    assert "HyperSpy" in b.add.__doc__

    # Does not add disclaimer
    assert b.subtract(1) == 0
    assert b.subtract.__doc__ is None
