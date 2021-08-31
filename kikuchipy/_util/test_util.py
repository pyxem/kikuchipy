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

import numpy as np
import pytest

from kikuchipy._util import deprecated


class TestDeprecationWarning:
    def test_deprecation_since(self):
        """Ensure functions decorated with the custom deprecated
        decorator returns desired output, raises a desired warning, and
        gets the desired additions to their docstring.
        """

        @deprecated(since=0.7, alternative="bar", removal=0.8)
        def foo(n):
            """Some docstring."""
            return n + 1

        with pytest.warns(np.VisibleDeprecationWarning) as record:
            assert foo(4) == 5
        desired_msg = (
            "Function `foo()` is deprecated and will be removed in version 0.8. Use "
            "`bar()` instead."
        )
        assert str(record[0].message) == desired_msg
        assert foo.__doc__ == (
            "[*Deprecated*] Some docstring.\n"
            "\nNotes\n-----\n"
            ".. deprecated:: 0.7\n"
            f"   {desired_msg}"
        )

        @deprecated(since=1.9)
        def foo2(n):
            """Another docstring.

            Notes
            -----
            Some existing notes.
            """
            return n + 2

        with pytest.warns(np.VisibleDeprecationWarning) as record2:
            assert foo2(4) == 6
        desired_msg2 = "Function `foo2()` is deprecated."
        assert str(record2[0].message) == desired_msg2
        assert foo2.__doc__ == (
            "[*Deprecated*] Another docstring.\n"
            "\nNotes\n-----\n"
            "Some existing notes.\n\n"
            ".. deprecated:: 1.9\n"
            f"   {desired_msg2}"
        )

    def test_deprecation_no_old_doc(self):
        @deprecated(since=0.7, alternative="bar", removal=0.8)
        def foo(n):
            return n + 1

        with pytest.warns(np.VisibleDeprecationWarning) as record:
            assert foo(4) == 5
        desired_msg = (
            "Function `foo()` is deprecated and will be removed in version 0.8. Use "
            "`bar()` instead."
        )
        assert str(record[0].message) == desired_msg
        assert foo.__doc__ == (
            "[*Deprecated*] \n"
            "\nNotes\n-----\n"
            ".. deprecated:: 0.7\n"
            f"   {desired_msg}"
        )
