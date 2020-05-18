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
from hyperspy.drawing._markers.horizontal_line import HorizontalLine
from hyperspy.drawing._markers.vertical_line import VerticalLine
from hyperspy.drawing._markers.rectangle import Rectangle
from hyperspy.drawing._markers.text import Text
from hyperspy.roi import BaseInteractiveROI, RectangularROI
from hyperspy._signals.signal2d import Signal2D
import numpy as np

from kikuchipy.signals import EBSD, LazyEBSD
from kikuchipy.signals import VirtualBSEImage
from kikuchipy.generators.util import (
    get_rgb_image as get_rgb_image_from_arrays,
    _transfer_navigation_axes_to_signal_axes,
)


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
        r: Union[
            BaseInteractiveROI, Tuple, List[BaseInteractiveROI], List[Tuple],
        ],
        g: Union[
            BaseInteractiveROI, Tuple, List[BaseInteractiveROI], List[Tuple],
        ],
        b: Union[
            BaseInteractiveROI, Tuple, List[BaseInteractiveROI], List[Tuple],
        ],
        percentiles: Optional[Tuple] = None,
        normalize: bool = True,
        alpha: Union[None, np.ndarray, VirtualBSEImage] = None,
        dtype_out: Union[np.uint8, np.uint16] = np.uint8,
        **kwargs,
    ) -> VirtualBSEImage:
        """Return an in-memory RGB virtual BSE image from three regions
        of interest (ROIs) on the EBSD detector, with a potential "alpha
        channel" in which all three arrays are multiplied by a fourth.

        Parameters
        ----------
        r
            One ROI or a list of ROIs, or one tuple or a list of tuples
            with detector grid indices specifying one or more ROI(s).
            Intensities within the specified ROI(s) are summed up to
            form the red color channel.
        g
            One ROI or a list of ROIs, or one tuple or a list of tuples
            with detector grid indices specifying one or more ROI(s).
            Intensities within the specified ROI(s) are summed up to
            form the green color channel.
        b
            One ROI or a list of ROIs, or one tuple or a list of tuples
            with detector grid indices specifying one or more ROI(s).
            Intensities within the specified ROI(s) are summed up to
            form the blue color channel.
        normalize
            Whether to normalize the individual images (channels) before
            RGB image creation.
        alpha
            "Alpha channel". If None (default), no "alpha channel" is
            added to the image.
        percentiles
            Whether to apply contrast stretching with a given percentile
            tuple with percentages, e.g. (0.5, 99.5), after creating the
            RGB image. If None (default), no contrast stretching is
            performed.
        dtype_out
            Output data type, either np.uint16 or np.uint8 (default).
        kwargs
            Keyword arguments passed to
            :func:` ~kikuchipy.generators.util.virtual_bse.get_rgb_image`.

        Returns
        -------
        vbse_rgb_image : VirtualBSEImage
            Virtual RGB image in memory.

        See Also
        --------
        kikuchipy.signals.EBSD.plot_virtual_bse_intensity,
        kikuchipy.signals.EBSD.get_virtual_bse_intensity,
        kikuchipy.generators.util.get_rgb_image

        Notes
        -----
        HyperSpy only allows for RGB signal dimensions with data types
        unsigned 8 or 16 bit.
        """
        channels = []
        for rois in [r, g, b]:
            if isinstance(rois, tuple) or hasattr(rois, "__iter__") is False:
                rois = (rois,)

            image = np.zeros(
                self.signal.axes_manager.navigation_shape[::-1],
                dtype=np.float64,
            )
            for roi in rois:
                if isinstance(roi, tuple):
                    roi = self.roi_from_grid(roi)
                roi_image = self.signal.get_virtual_bse_intensity(roi).data
                if isinstance(roi_image, Array):
                    roi_image = roi_image.compute()
                image += roi_image

            channels.append(image)

        if alpha is not None and isinstance(alpha, Signal2D):
            alpha = alpha.data

        rgb_image = get_rgb_image_from_arrays(
            channels=channels,
            normalize=normalize,
            alpha=alpha,
            percentiles=percentiles,
            dtype_out=dtype_out,
            **kwargs,
        )

        rgb_image = rgb_image.astype(dtype_out)
        vbse_rgb_image = VirtualBSEImage(rgb_image).transpose(signal_axes=1)

        dtype_rgb = "rgb" + str(8 * np.iinfo(dtype_out).dtype.itemsize)
        vbse_rgb_image.change_dtype(dtype_rgb)

        vbse_rgb_image.axes_manager = _transfer_navigation_axes_to_signal_axes(
            new_axes=vbse_rgb_image.axes_manager,
            old_axes=self.signal.axes_manager,
        )

        return vbse_rgb_image

    def get_images_from_grid(
        self, dtype_out: np.dtype = np.float32,
    ) -> VirtualBSEImage:
        """Return an in-memory signal with a stack of virtual
        backscatter electron (BSE) images by integrating the intensities
        within regions of interest (ROI) defined by the detector
        `grid_shape`.

        Parameters
        ----------
        dtype_out
            Output data type, default is float32.

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
            roi = self.roi_from_grid((row, col))
            images[row, col] = self.signal.get_virtual_bse_intensity(roi).data

        vbse_images = VirtualBSEImage(images)
        # TODO: Transfer signal's detector axes to new navigation axes with
        #  proper binning
        vbse_images.axes_manager = _transfer_navigation_axes_to_signal_axes(
            new_axes=vbse_images.axes_manager, old_axes=self.signal.axes_manager
        )

        return vbse_images

    def roi_from_grid(self, index: Union[Tuple, List[Tuple]]) -> RectangularROI:
        """Return a rectangular region of interest (ROI) on the EBSD
        detector from one or multiple generator grid tile indices as
        row(s) and column(s).

        Parameters
        ----------
        index
            Row and column of one or multiple grid tiles as a tuple or a
            list of tuples.

        Returns
        -------
        roi : hyperspy.roi.RectangularROI
            ROI defined by the grid indices.
        """
        rows = self.grid_rows
        cols = self.grid_cols

        if isinstance(index, tuple):
            index = (index,)
        index = np.array(index)

        min_col = cols[min(index[:, 1])]
        max_col = cols[max(index[:, 1])] + cols[1]
        min_row = rows[min(index[:, 0])]
        max_row = rows[max(index[:, 0])] + rows[1]

        return RectangularROI(
            left=min_col, top=min_row, right=max_col, bottom=max_row,
        )

    def plot_grid(
        self,
        pattern_idx: Optional[Tuple[int, ...]] = None,
        rgb_channels: Union[None, List[Tuple], List[List[Tuple]]] = None,
        visible_indices: bool = True,
        **kwargs,
    ) -> EBSD:
        """Plot a pattern with the detector grid superimposed,
        potentially coloring the edges of three grid tiles red, green
        and blue.

        Parameters
        ----------
        pattern_idx
            A tuple of integers defining the pattern to superimpose the
            grid on. If None (default), the first pattern is used.
        rgb_channels
            A list of tuple indices defining three or more detector grid
            tiles which edges to color red, green and blue. If None (default),
            no tiles' edges are colored.
        visible_indices
            Whether to show grid indices. Default is True.
        kwargs
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

        rows = self.grid_rows
        cols = self.grid_cols

        # Set grid tile indices
        markers = []
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

        # Set lines
        kwargs.setdefault("color", "w")
        markers += [HorizontalLine((i - 0.5) * dy, **kwargs) for i in rows]
        markers += [VerticalLine((j - 0.5) * dx, **kwargs) for j in cols]

        # Color RGB tiles
        if rgb_channels is not None:
            for channels, color in zip(rgb_channels, ["r", "g", "b"]):
                if isinstance(channels, tuple):
                    channels = (channels,)
                for (row, col) in channels:
                    kwargs.update({"color": color, "zorder": 3, "linewidth": 2})
                    roi = self.roi_from_grid((row, col))
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
