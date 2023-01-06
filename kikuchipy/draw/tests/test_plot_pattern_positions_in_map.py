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

import matplotlib.pyplot as plt
import numpy as np

import kikuchipy as kp


class TestPlotPatternPositionsInMap:
    # Original metadata similar to the original metadata returned with
    # the Ni gain 0 calibration pattern dataset
    omd = {
        "roi": {
            "origin": (185, 73),
            "origin_scaled": (49, 19),
            "shape": (558, 750),
            "shape_scaled": (149, 200),
        },
        "area": {"shape": (1000, 1000), "shape_scaled": (267, 267)},
        "area_image": np.ones((1000, 1000), dtype="uint8"),
        "calibration_patterns": {
            "indices": np.array(
                [
                    [447, 425],
                    [532, 294],
                    [543, 573],
                    [378, 596],
                    [369, 308],
                    [632, 171],
                    [668, 704],
                    [269, 696],
                    [247, 152],
                ]
            ),
            "indices_scaled": np.array(
                [
                    [119, 113],
                    [142, 78],
                    [145, 153],
                    [101, 159],
                    [98, 82],
                    [169, 46],
                    [178, 188],
                    [72, 186],
                    [66, 41],
                ]
            ),
        },
    }

    def test_plot_pos_no_area(self):
        rc = self.omd["calibration_patterns"]["indices_scaled"]
        rc_shifted = rc - np.array(self.omd["roi"]["origin_scaled"])

        fig = kp.draw.plot_pattern_positions_in_map(
            rc=rc,
            roi_shape=self.omd["roi"]["shape_scaled"],
            roi_origin=self.omd["roi"]["origin_scaled"],
            return_figure=True,
        )
        ax = fig.axes[0]

        assert ax.get_xlabel() == "Column"
        assert ax.get_ylabel() == "Row"
        assert np.allclose(ax.get_xlim(), (0, 200))
        assert np.allclose(ax.get_ylim(), (149, 0))  # Checks inversion

        coll = ax.collections
        assert len(coll) == len(rc)
        # Ensure it's a cross
        assert np.allclose(
            coll[0].get_paths()[0].vertices, [[-0.5, 0], [0.5, 0], [0, -0.5], [0, 0.5]]
        )

        texts = ax.texts
        assert len(texts) == 9
        for i, t in enumerate(texts):
            assert int(t.get_text()) == i
            assert np.allclose(t.get_position(), rc_shifted[i][::-1])
            assert t.get_color() == "k"

        plt.close(fig)

    def test_plot_pos_area(self):
        rc = self.omd["calibration_patterns"]["indices"]

        area_shape = self.omd["area"]["shape"]
        area_size = int(np.prod(area_shape))

        fig = kp.draw.plot_pattern_positions_in_map(
            rc=rc,
            roi_shape=self.omd["roi"]["shape"],
            roi_origin=self.omd["roi"]["origin"],
            area_shape=area_shape,
            area_image=np.random.random(area_size).reshape(area_shape),
            roi_image=np.zeros(self.omd["roi"]["shape"]),
            return_figure=True,
            color="w",
        )
        ax = fig.axes[0]

        assert np.allclose(ax.get_xlim(), (0, 1000))
        assert np.allclose(ax.get_ylim(), (1000, 0))

        images = ax.images
        assert images[0].get_array().shape == area_shape
        assert images[1].get_array().shape == self.omd["roi"]["shape"]

        # Marker labels plus "Area" and "Region of interest"
        texts = ax.texts
        assert len(texts) == 11
        assert texts[0].get_color() == "black"
        assert texts[2].get_color() == "w"
        assert texts[0].get_text() == "Area"
        assert texts[1].get_text() == "Region of interest"

        # Red rectangle for the ROI
        rectangle = ax.patches[0]
        assert np.allclose(rectangle.get_edgecolor(), [1, 0, 0, 1])
        assert rectangle.get_xy() == self.omd["roi"]["origin"][::-1]

        plt.close(fig)

    def test_plot_pos_existing_axis(self):
        fig = plt.figure()
        ax = fig.add_subplot()

        kp.draw.plot_pattern_positions_in_map(
            rc=self.omd["calibration_patterns"]["indices_scaled"],
            roi_shape=self.omd["roi"]["shape_scaled"],
            roi_origin=self.omd["roi"]["origin_scaled"],
            axis=ax,
        )

        assert len(ax.collections) == len(self.omd["calibration_patterns"]["indices"])
        assert len(ax.texts) == 9

        plt.close(fig)
