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

import logging

import kikuchipy as kp


class TestLogging:
    def test_logging(self, caplog, dummy_signal):
        logger = logging.getLogger("kikuchipy")
        assert logger.level == 0  # Warning

        dummy_signal.set_signal_type("EBSDMasterPattern")
        dummy_signal2 = dummy_signal.deepcopy()

        # No info messages are logged
        dummy_signal.set_signal_type("EBSD")
        assert len(caplog.records) == 0

        kp.set_log_level("DEBUG")
        assert logger.level == 10

        # Info messages are logged
        dummy_signal2.set_signal_type("EBSD")
        assert len(caplog.records) > 0
        for record in caplog.records:
            assert record.levelname == "DEBUG"
