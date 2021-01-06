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

import io
import sys
from contextlib import contextmanager

import pytest

from kikuchipy.io._util import _get_input_bool, _get_input_variable


@contextmanager
def replace_stdin(target):
    orig = sys.stdin
    sys.stdin = target
    yield
    sys.stdin = orig


class TestIO:
    @pytest.mark.parametrize(
        "answer, should_return", [("y", True), ("n", False), ("m", True)]
    )
    def test_get_input_bool(self, answer, should_return):
        question = "Ehm, hello? ... is, is there anybody out there?"
        if answer == "m":
            with replace_stdin(io.StringIO(answer)):
                with pytest.raises(EOFError):
                    _ = _get_input_bool(question)
            return 0
        else:
            with replace_stdin(io.StringIO(answer)):
                returns = _get_input_bool(question)
        assert returns == should_return

    @pytest.mark.parametrize("var_type", (int, 1))
    def test_get_input_variable(self, var_type):
        question = "How few are too few coffee cups?"
        answer = "1"
        with replace_stdin(io.StringIO(answer)):
            if isinstance(var_type, int):
                with pytest.raises(EOFError):
                    _ = _get_input_variable(question, var_type)
                return 0
            else:
                returns = _get_input_variable(question, var_type)
        assert returns == var_type(answer)
