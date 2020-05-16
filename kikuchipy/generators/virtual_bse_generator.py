# -*- coding: utf-8 -*-
# Copyright 2019-2020 The kikuchipy developers
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

from typing import List, Optional, Tuple, Union

from dask.array import Array
from hyperspy.roi import BaseInteractiveROI, RectangularROI
from hyperspy.drawing._markers.horizontal_line import HorizontalLine
from hyperspy.drawing._markers.vertical_line import VerticalLine
from hyperspy.drawing._markers.rectangle import Rectangle
from hyperspy.drawing._markers.text import Text
import numpy as np

from kikuchipy.signals import EBSD, LazyEBSD
from kikuchipy.signals import VirtualBSEImage
from kikuchipy.generators.util import (
    get_rgb_image as get_rgb_image_from_arrays,
    _transfer_navigation_axes_to_signal_axes,
)
from kikuchipy.pattern import rescale_intensity


class VirtualBSEGenerator:
    """Generates virtual backscatter electron (BSE) images for a
    specified electron backscatter diffraction (EBSD) signal and a set
    of EBSD detector areas.

    Attributes
    ----------
    signal : kikuchipy.signals.EBSD
    grid_shape : Tuple[int]
    """

    def __init__(self, signal: Union[EBSD, LazyEBSD]):
        self.signal = signal
        self.grid_shape = (5, 5)

    def __repr__(self):
        return f"VirtualBSEGenerator for {self.signal}"

    @property
    def grid_rows(self) -> np.ndarray:
        """Return detector grid rows, defined by `grid_shape`."""
        gy = self.grid_shape[0]
        sy = self.signal.axes_manager.signal_shape[1]
        return np.linspace(0, sy, gy + 1)

    @property
    def grid_cols(self) -> np.ndarray:
        """Return detector grid columns, defined by `grid_shape`."""
        gx = self.grid_shape[1]
        sx = self.signal.axes_manager.signal_shape[0]
        return np.linspace(0, sx, gx + 1)

    def get_rgb_image(
        self,
        rois: Union[List[BaseInteractiveROI], List[Tuple]],
        percentile: Optional[Tuple] = None,
        normalize: bool = True,
        alpha: Optional[np.ndarray] = None,
        dtype_out: Union[np.uint8, np.uint16] = np.uint8,
        **kwargs,
    ) -> VirtualBSEImage:
        """Return an in-memory RGB virtual BSE image from three regions
        of interest (ROIs) on the EBSD detector, with a potential "alpha
        channel" in which all three arrays are multiplied by a fourth.

        Parameters
        ----------
        rois
            A list of three ROIs or tuples with detector grid indices to
            integrate the intensity within for the red, green and blue
            channel, respectively.
        normalize
            Whether to normalize the individual images (channels) before
            RGB image creation.
        alpha
            "Alpha channel". If None (default), no "alpha channel" is
            added to the image.
        percentile
            Whether to apply contrast stretching with a given percentile
            tuple with percentages, e.g. (0.5, 99.5), after creating the
            RGB image. If None (default), no contrast stretching is
            performed.
        dtype_out
            Output data type, either np.uint16 or np.uint8 (default).
        kwargs :
            Keyword arguments passed to
            :func:` ~kikuchipy.generators.util.virtual_bse.normalize_image`.

        Returns
        -------
        vbse_rgb_image : VirtualBSEImage
            Virtual RGB image in memory.

        See Also
        --------
        kikuchipy.signals.EBSD.virtual_bse_imaging,
        kikuchipy.signals.EBSD.get_virtual_bse_image,
        kikuchipy.generators.util.get_rgb_image
        """
        if isinstance(rois[0], tuple):
            rois = [self.roi_from_grid(row, col) for row, col in rois]

        channels = []
        for roi in rois[:3]:
            image = self.signal.get_virtual_bse_image(roi).data
            if isinstance(image, Array):
                image = image.compute()
            channels.append(image)

        rgb_image = get_rgb_image_from_arrays(
            channels=channels,
            normalize=normalize,
            alpha=alpha,
            percentile=percentile,
            dtype_out=dtype_out,
            **kwargs,
        )

        rgb_image = rgb_image.astype(dtype_out)
        vbse_rgb_image = VirtualBSEImage(rgb_image).transpose(signal_axes=1)
        vbse_rgb_image.axes_manager = _transfer_navigation_axes_to_signal_axes(
            new_axes=vbse_rgb_image.axes_manager,
            old_axes=self.signal.axes_manager,
        )

        dtype_rgb = "rgb" + str(8 * np.iinfo(dtype_out).dtype.itemsize)
        vbse_rgb_image.change_dtype(dtype_rgb)

        return vbse_rgb_image

    def get_images_from_grid(
        self, normalize: bool = True, dtype_out: np.dtype = np.uint32,
    ) -> VirtualBSEImage:
        """Return an in-memory signal with a stack of virtual
        backscatter electron (BSE) images by integrating the intensities
        within regions of interest (ROI) defined by the detector
        `grid_shape`.

        Parameters
        ----------
        normalize
            Whether to normalize the images, keeping relative
            intensities. Default is True.
        dtype_out
            Output data type, default is uint32.

        Returns
        -------
        vbse_images : VirtualBSEImage
            In-memory signal with virtual BSE images.

        Examples
        --------
        >>> s
        <EBSD, title: Pattern, dimensions: (200, 149|60, 60)>
        >>> vbse_gen = VirtualBSEGenerator(s)
        >>> vbse_gen.grid_shape = (5, 5)
        >>> vbse = vbse_gen.get_images_from_grid()
        >>> vbse
        <VirtualBSEImage, title: , dimensions: (5, 5|200, 149)>
        """
        grid_shape = self.grid_shape
        new_shape = grid_shape + self.signal.axes_manager.navigation_shape[::-1]
        images = np.zeros(new_shape, dtype=dtype_out)
        for row, col in np.ndindex(grid_shape):
            roi = self.roi_from_grid(row, col)
            images[row, col] = self.signal.get_virtual_bse_image(roi).data

        vbse_images = VirtualBSEImage(images)
        vbse_images.axes_manager = _transfer_navigation_axes_to_signal_axes(
            new_axes=vbse_images.axes_manager, old_axes=self.signal.axes_manager
        )
        vbse_images.change_dtype(dtype_out)

        if normalize:
            images_min = vbse_images.data.min()
            images_max = vbse_images.data.max()
            for idx in np.ndindex(grid_shape):
                img = vbse_images.data[idx]
                vbse_images.data[idx] = rescale_intensity(
                    img, in_range=(images_min, images_max), dtype_out=dtype_out,
                )

        return vbse_images

    def roi_from_grid(self, row: int = 0, col: int = 0):
        """Return a rectangular region of interest (ROI) on the EBSD
        detector from giving the row and column in the generator grid.

        Parameters
        ----------
        row
            Detector grid row index.
        col
            Detector grid column index.

        Returns
        -------
        roi : hyperspy.roi.RectangularROI
            ROI defined by the grid indices.
        """
        rows = self.grid_rows
        cols = self.grid_cols
        return RectangularROI(
            left=cols[col],
            top=rows[row],
            right=cols[col] + cols[1],
            bottom=rows[row] + rows[1],
        )

    def plot_grid(
        self,
        pattern_idx: Optional[Tuple[int, ...]] = None,
        rgb_channels: Optional[List[Tuple]] = None,
        visible_indices: bool = True,
        **kwargs,
    ):
        """Plot a pattern with the detector grid superimposed,
        potentially coloring the edges of three grid tiles red, green
        and blue.

        Parameters
        ----------
        pattern_idx
            A tuple of integers defining the pattern to superimpose the
            grid on. If None (default), the first pattern is used.
        rgb_channels
            A list of tuple indices defining three detector grid tiles
            which edges to color red, green and blue. If None (default),
            no tiles' edges are colored.
        visible_indices
            Whether to show grid indices. Default is True.
        kwargs :
            Keyword arguments passed to
            :func:`matplotlib.pyplot.axhline` and `axvline`, used by
            HyperSpy to draw lines.

        Returns
        -------
        pattern : kikuchipy.signals.EBSD
            A single pattern with the markers added.
        """
        # Get detector scales
        axes_manager = self.signal.axes_manager
        dx, dy = [i.scale for i in axes_manager.signal_axes]

        # Set lines
        rows = self.grid_rows
        cols = self.grid_cols
        markers = [HorizontalLine((i - 0.5) * dy, **kwargs) for i in rows]
        markers += [VerticalLine((j - 0.5) * dx, **kwargs) for j in cols]

        # Set grid tile indices
        if visible_indices:
            color = kwargs.pop("color", "r")
            for row, col in np.ndindex(self.grid_shape):
                markers.append(
                    Text(
                        x=cols[col],
                        y=rows[row] + (0.1 * rows[1]),
                        text=f"{row,col}",
                        color=color,
                    )
                )

        # Color RGB tiles
        if rgb_channels is not None:
            for (row, col), color in zip(rgb_channels, ["r", "g", "b"]):
                kwargs.update({"color": color, "zorder": 3, "linewidth": 2})
                roi = self.roi_from_grid(row, col)
                markers += [
                    Rectangle(
                        x1=(roi.left - 0.5) * dx,
                        y1=(roi.top - 0.5) * dx,
                        x2=(roi.right - 0.5) * dy,
                        y2=(roi.bottom - 0.5) * dy,
                        **kwargs,
                    )
                ]

        # Get pattern and add list of markers
        if pattern_idx is None:
            pattern_idx = (0,) * axes_manager.navigation_dimension
        pattern = self.signal.inav[pattern_idx]
        pattern.add_marker(markers, permanent=True)

        return pattern
