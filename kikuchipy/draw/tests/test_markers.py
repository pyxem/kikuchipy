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

from hyperspy.utils.markers import line_segment, point, text
import numpy as np
import pytest

from kikuchipy.draw.markers import (
    get_line_segment_list,
    get_point_list,
    get_text_list,
)


class TestMarkers:
    @pytest.mark.parametrize("n_lines", [1, 2, 3])
    def test_get_line_segment_list0d(self, n_lines):
        """Lines in 0D () navigation space."""
        nav_shape = ()
        size = int(np.prod(nav_shape) * n_lines * 4)
        lines = np.random.random(size=size).reshape(nav_shape + (n_lines, 4))
        kwargs = dict(linewidth=1, color="red", alpha=1, zorder=1)
        line_markers = get_line_segment_list(list(lines), **kwargs)

        # Number of markers
        assert isinstance(line_markers, list)
        assert len(line_markers) == n_lines
        assert isinstance(line_markers[0], line_segment)

        # Coordinates, data shape and marker properties
        for line, marker in zip(lines, line_markers):
            assert np.allclose(
                [
                    marker.get_data_position("x1"),
                    marker.get_data_position("y1"),
                    marker.get_data_position("x2"),
                    marker.get_data_position("y2"),
                ],
                line,
            )
            assert marker._get_data_shape() == nav_shape
            for k, v in kwargs.items():
                assert marker.marker_properties[k] == v

    def test_get_line_segment_0d_2(self):
        """A line in 0d but no (1,) navigation shape."""
        lines = np.random.random(size=4).reshape((4,))
        line_markers = get_line_segment_list(lines)

        assert len(line_markers) == 1
        assert np.allclose(
            [
                line_markers[0].get_data_position("x1"),
                line_markers[0].get_data_position("y1"),
                line_markers[0].get_data_position("x2"),
                line_markers[0].get_data_position("y2"),
            ],
            lines,
        )

    def test_get_line_segment_list1d(self):
        """Lines in 1D (2,) navigation space."""
        nav_shape = (2,)
        n_lines = 2
        size = int(np.prod(nav_shape) * n_lines * 4)
        lines = np.random.random(size=size).reshape(nav_shape + (n_lines, 4))
        line_markers = get_line_segment_list(lines)

        assert len(line_markers) == n_lines

        # Iterate over lines
        for i in range(n_lines):
            assert line_markers[i]._get_data_shape() == nav_shape  # 1d
            assert np.allclose(
                np.dstack(line_markers[i].data.tolist()[:4]), lines[:, i],
            )

    def test_get_line_segment_list2d(self):
        """Lines in 2D (2, 3) navigation space."""
        nav_shape = (2, 3)
        n_lines = 3
        size = int(np.prod(nav_shape) * n_lines * 4)
        lines = np.random.random(size=size).reshape(nav_shape + (n_lines, 4))
        line_markers = get_line_segment_list(lines)

        assert len(line_markers) == n_lines

        # Iterate over lines
        for i in range(n_lines):
            assert line_markers[i]._get_data_shape() == nav_shape
            assert np.allclose(
                np.dstack(line_markers[i].data.tolist()[:4]), lines[:, :, i],
            )

    def test_get_line_segment_list_nans(self):
        lines = np.ones((2, 3, 4)) * np.nan
        assert len(get_line_segment_list(lines)) == 0

    @pytest.mark.parametrize("n_points", [1, 2, 3])
    def test_get_point_list0d(self, n_points):
        """Points in 0D () navigation space."""
        nav_shape = ()
        size = int(np.prod(nav_shape) * n_points * 2)
        points = np.random.random(size=size).reshape(nav_shape + (n_points, 2))
        kwargs = dict(
            s=40, marker="o", facecolor="w", edgecolor="k", zorder=5, alpha=1
        )
        point_markers = get_point_list(list(points), **kwargs)

        # Number of markers
        assert isinstance(point_markers, list)
        assert len(point_markers) == n_points
        assert isinstance(point_markers[0], point)

        # Coordinates, data shape and marker properties
        for i, marker in zip(points, point_markers):
            assert np.allclose(
                [
                    marker.get_data_position("x1"),
                    marker.get_data_position("y1"),
                ],
                i,
            )
            assert marker._get_data_shape() == nav_shape
            for k, v in kwargs.items():
                assert marker.marker_properties[k] == v

    def test_get_point_list0d_2(self):
        """One point in 0d but no (1,) navigation shape."""
        points = np.random.random(size=2).reshape((2,))
        point_marker = get_point_list(points)

        assert len(point_marker) == 1
        assert np.allclose(
            [
                point_marker[0].get_data_position("x1"),
                point_marker[0].get_data_position("y1"),
            ],
            points,
        )

    def test_get_point_list1d(self):
        """Points in 1D (2,) navigation space."""
        nav_shape = (2,)
        n_points = 2
        size = int(np.prod(nav_shape) * n_points * 2)
        points = np.random.random(size=size).reshape(nav_shape + (n_points, 2))
        point_markers = get_point_list(points)

        assert len(point_markers) == n_points

        # Iterate over points
        for i in range(n_points):
            assert point_markers[i]._get_data_shape() == nav_shape  # 1d
            assert np.allclose(
                np.dstack(point_markers[i].data.tolist()[:2]), points[:, i],
            )

    def test_get_point_list2d(self):
        """Points in 2D (2, 3) navigation space."""
        nav_shape = (2, 3)
        n_points = 3
        size = int(np.prod(nav_shape) * n_points * 2)
        points = np.random.random(size=size).reshape(nav_shape + (n_points, 2))
        point_markers = get_point_list(points)

        assert len(point_markers) == n_points

        # Iterate over points
        for i in range(n_points):
            assert point_markers[i]._get_data_shape() == nav_shape
            assert np.allclose(
                np.dstack(point_markers[i].data.tolist()[:2]), points[:, :, i],
            )

    def test_get_point_list_nans(self):
        points = np.ones((2, 3, 2)) * np.nan
        assert len(get_point_list(points)) == 0

    @pytest.mark.parametrize("n_labels", [1, 2, 3])
    def test_get_text_list0d(self, n_labels):
        """Text labels in 0D () navigation space."""
        nav_shape = ()
        size = int(np.prod(nav_shape) * n_labels * 2)
        texts = ["111", "220", "-220"][:n_labels]
        text_coords = np.random.random(size=size).reshape(
            nav_shape + (n_labels, 2)
        )
        kwargs = dict(
            color="k",
            zorder=5,
            ha="center",
            bbox=dict(
                facecolor="w",
                edgecolor="k",
                boxstyle="round, rounding_size=0.2",
                pad=0.1,
                alpha=0.1,
            ),
        )
        text_markers = get_text_list(
            texts=texts, coordinates=list(text_coords), **kwargs
        )

        # Number of markers
        assert isinstance(text_markers, list)
        assert len(text_markers) == n_labels
        assert isinstance(text_markers[0], text)

        # Coordinates, data shape and marker properties
        for i, (t, marker) in enumerate(zip(text_coords, text_markers)):
            assert np.allclose(
                [
                    marker.get_data_position("x1"),
                    marker.get_data_position("y1"),
                ],
                t,
            )
            assert marker.get_data_position("text") == texts[i]
            assert marker._get_data_shape() == nav_shape
            for k, v in kwargs.items():
                assert marker.marker_properties[k] == v

    def test_get_text_list0d_2(self):
        """One text label in 0d but no (1,) navigation shape."""
        text_coords = np.random.random(size=2).reshape((2,))
        texts = ["123"]
        text_markers = get_text_list(texts=texts, coordinates=text_coords)

        assert len(text_markers) == 1
        assert np.allclose(
            [
                text_markers[0].get_data_position("x1"),
                text_markers[0].get_data_position("y1"),
            ],
            text_coords,
        )
        assert text_markers[0].get_data_position("text") == texts[0]

    def test_get_text_list1d(self):
        """Text labels in 1D (2,) navigation space."""
        nav_shape = (2,)
        n_labels = 2
        size = int(np.prod(nav_shape) * n_labels * 2)
        texts = ["111", "-220"]
        text_coords = np.random.random(size=size).reshape(
            nav_shape + (n_labels, 2)
        )
        text_markers = get_text_list(texts=texts, coordinates=text_coords)

        assert len(text_markers) == n_labels

        # Iterate over text labels
        for i in range(n_labels):
            assert text_markers[i]._get_data_shape() == nav_shape
            assert np.allclose(
                np.dstack(text_markers[i].data.tolist()[:2]), text_coords[:, i],
            )
            assert text_markers[i].data["text"] == texts[i]

    def test_get_text_list2d(self):
        """Text labels in 2D (2, 3) navigation space."""
        nav_shape = (2, 3)
        n_labels = 3
        size = int(np.prod(nav_shape) * n_labels * 2)
        texts = ["111", "-2-20", "311"]
        text_coords = np.random.random(size=size).reshape(
            nav_shape + (n_labels, 2)
        )
        text_markers = get_text_list(texts=texts, coordinates=text_coords)

        assert len(text_markers) == n_labels

        # Iterate over text labels
        for i in range(n_labels):
            assert text_markers[i]._get_data_shape() == nav_shape
            assert np.allclose(
                np.dstack(text_markers[i].data.tolist()[:2]),
                text_coords[:, :, i],
            )
            assert text_markers[i].data["text"] == texts[i]

    def test_get_text_list_nans(self):
        """Returns"""
        text_coords = np.ones((2, 3, 2)) * np.nan
        assert (
            len(
                get_text_list(
                    texts=["111", "200", "220"], coordinates=text_coords
                )
            )
            == 0
        )
