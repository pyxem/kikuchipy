#
# Copyright 2019-2026 the kikuchipy developers
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
#

import matplotlib.pyplot as plt
import numpy as np

import kikuchipy as kp


class TestPlotPatternProcessingResult:
    def test_plot(self, pattern_processing_result):
        fig = kp.draw.plot_pattern_processing_result(pattern_processing_result)

        # Pattern and histogram per pattern
        assert len(fig.axes) == 2 * 3
        titles = []
        for ax in fig.axes[:3]:
            titles.append(ax.get_title())
        assert titles == [
            "Raw\nIQ=0.100, NCC=0.4000",
            "DBR\nIQ=0.200, NCC=0.5000",
            "DBR + Bandpass\nIQ=0.300, NCC=0.6000",
        ]
        assert np.allclose(fig.get_size_inches(), (3 * 3, 4.5))

        plt.close(fig)

    def test_plot_reference_figsize(self, pattern_processing_result):
        reference = np.zeros((10, 12), dtype=np.uint8)
        fig = kp.draw.plot_pattern_processing_result(
            pattern_processing_result, reference=reference, figsize=(8, 3)
        )
        assert len(fig.axes) == 2 * 4
        assert fig.axes[0].get_title() == "Reference"
        assert np.allclose(fig.get_size_inches(), (8, 3))

        plt.close(fig)

    def test_plot_signal_mask(self, pattern_processing_result):
        """Masked out pixels are hidden in the patterns and excluded
        from the histograms.
        """
        signal_mask = np.zeros((10, 12), dtype=bool)
        signal_mask[:2] = True
        pattern_processing_result["signal_mask"] = signal_mask
        fig = kp.draw.plot_pattern_processing_result(pattern_processing_result)

        image = fig.axes[0].get_images()[0].get_array()
        assert np.array_equal(image.mask, signal_mask)

        n_pixels_hist = 0
        for patch in fig.axes[3].patches:
            n_pixels_hist += patch.get_height()
        assert n_pixels_hist == np.sum(~signal_mask)

        plt.close(fig)
