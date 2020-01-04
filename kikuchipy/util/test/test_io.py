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

import io
import sys
from contextlib import contextmanager

import pytest

import kikuchipy as kp


@contextmanager
def replace_stdin(target):
    orig = sys.stdin
    sys.stdin = target
    yield
    sys.stdin = orig


class TestIO:
    def test_kikuchipy_metadata(self):
        sem_node, ebsd_node = kp.util.io.metadata_nodes()
        md = kp.util.io.kikuchipy_metadata()
        assert md.get_item(sem_node + ".microscope") == ""
        assert md.get_item(ebsd_node + ".xpc") == 1.0

    def test_metadata_nodes(self):
        sem_node = kp.util.io.metadata_nodes(ebsd=False)
        assert sem_node == "Acquisition_instrument.SEM"
        ebsd_node = kp.util.io.metadata_nodes(sem=False)
        assert ebsd_node == sem_node + ".Detector.EBSD"
        sem_node, ebsd_node = kp.util.io.metadata_nodes()

    @pytest.mark.parametrize(
        "answer, should_return", [("y", True), ("n", False), ("m", True)]
    )
    def test_get_input_bool(self, answer, should_return):
        question = "Ehm, hello? ... is, is there anybody out there?"
        if answer == "m":
            with replace_stdin(io.StringIO(answer)):
                with pytest.raises(EOFError):
                    kp.util.io._get_input_bool(question)
            return 0
        else:
            with replace_stdin(io.StringIO(answer)):
                returns = kp.util.io._get_input_bool(question)
        assert returns == should_return

    @pytest.mark.parametrize("var_type", (int, 1))
    def test_get_input_variable(self, var_type):
        question = "How few are too few coffee cups?"
        answer = "1"
        with replace_stdin(io.StringIO(answer)):
            if isinstance(var_type, int):
                with pytest.raises(EOFError):
                    kp.util.io._get_input_variable(question, var_type)
                return 0
            else:
                returns = kp.util.io._get_input_variable(question, var_type)
        assert returns == var_type(answer)
