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

import logging

import kikuchipy as kp


class TestLogging:
    def test_logging(self, caplog, dummy_signal):
        logger = logging.getLogger("kikuchipy")
        assert logger.level == 0  # Warning

        # No info messages are logged
        _ = dummy_signal.deepcopy()
        assert len(caplog.records) == 0

        kp.set_log_level("DEBUG")
        assert logger.level == 10

        # Info messages are logged
        _ = dummy_signal.deepcopy()
        for record in caplog.records:
            assert record.levelname == "DEBUG"
