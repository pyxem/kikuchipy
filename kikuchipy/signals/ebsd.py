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

from __future__ import annotations
import copy
import datetime
import gc
import numbers
import os
from typing import Union, List, Optional, Tuple, Iterable
import warnings

import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import hyperspy.api as hs
from hyperspy.learn.mva import LearningResults
from hyperspy.roi import BaseInteractiveROI
from h5py import File
import numpy as np
from orix.crystal_map import CrystalMap
from scipy.ndimage import correlate, gaussian_filter
from skimage.util.dtype import dtype_range

from kikuchipy.detectors import EBSDDetector
from kikuchipy.filters.fft_barnes import _fft_filter, _fft_filter_setup
from kikuchipy.filters.window import Window
from kikuchipy.indexing._dictionary_indexing import _dictionary_indexing
from kikuchipy.indexing._refinement._refinement import (
    _refine_orientation,
    _refine_orientation_projection_center,
    _refine_projection_center,
)
from kikuchipy.indexing.similarity_metrics import (
    SimilarityMetric,
    NormalizedCrossCorrelationMetric,
    NormalizedDotProductMetric,
)
from kikuchipy.io._io import _save
from kikuchipy.pattern import chunk
from kikuchipy.pattern.chunk import _average_neighbour_patterns
from kikuchipy.pattern._pattern import (
    fft_frequency_vectors,
    fft_filter,
    _dynamic_background_frequency_space_setup,
    _get_image_quality,
    _remove_static_background_subtract,
    _remove_static_background_divide,
    _remove_dynamic_background,
)
from kikuchipy.signals.util._dask import (
    get_dask_array,
    get_chunking,
    _get_chunk_overlap_depth,
    _rechunk_learning_results,
    _update_learning_results,
)
from kikuchipy.signals.util._detector import _detector_is_compatible_with_signal
from kikuchipy.signals.util._crystal_map import _crystal_map_is_compatible_with_signal
from kikuchipy.signals.util._map_helper import (
    _get_neighbour_dot_product_matrices,
    _get_average_dot_product_map,
)
from kikuchipy.signals._kikuchipy_signal import KikuchipySignal2D, LazyKikuchipySignal2D
from kikuchipy.signals.virtual_bse_image import VirtualBSEImage
from kikuchipy._util import deprecated_argument


class EBSD(KikuchipySignal2D):
    """Scan of Electron Backscatter Diffraction (EBSD) patterns.

    This class extends HyperSpy's Signal2D class for EBSD patterns. See
    the docstring of :class:`~hyperspy._signals.signal2d.Signal2D` for
    the list of inherited attributes and methods.

    Parameters
    ----------
    *args
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    detector : EBSDDetector, optional
        Detector describing the EBSD detector-sample geometry. If not
        given, this is a default detector (see :class:`EBSDDetector`).
    static_background : ~numpy.ndarray or ~dask.array.Array, optional
        Static background pattern. If not given, this is ``None``.
    xmap : ~orix.crystal_map.CrystalMap
        Crystal map containing the phases, unit cell rotations and
        auxiliary properties of the EBSD dataset. If not given, this is
        ``None``.
    **kwargs
        See :class:`~hyperspy._signals.signal2d.Signal2D`.

    See Also
    --------
    kikuchipy.data.nickel_ebsd_small :
        An EBSD signal with ``(3, 3)`` experimental nickel patterns.
    kikuchipy.data.nickel_ebsd_large :
        An EBSD signal with ``(55, 75)`` experimental nickel patterns.
    kikuchipy.data.silicon_ebsd_moving_screen_in :
        An EBSD signal with one experimental silicon pattern.
    kikuchipy.data.silicon_ebsd_moving_screen_out5mm :
        An EBSD signal with one experimental silicon pattern.
    kikuchipy.data.silicon_ebsd_moving_screen_out10mm :
        An EBSD signal with one experimental silicon pattern.

    Examples
    --------
    Load one of the example datasets and inspect some properties

    >>> import kikuchipy as kp
    >>> s = kp.data.nickel_ebsd_small()
    >>> s
    <EBSD, title: patterns My awes0m4 ..., dimensions: (3, 3|60, 60)>
    >>> s.detector
    EBSDDetector (60, 60), px_size 1.0 um, binning 8, tilt 0.0, azimuthal 0.0, pc (0.5, 0.5, 0.5)
    >>> s.static_background
    array([[84, 87, 90, ..., 27, 29, 30],
           [87, 90, 93, ..., 27, 28, 30],
           [92, 94, 97, ..., 39, 28, 29],
           ...,
           [80, 82, 84, ..., 36, 30, 26],
           [79, 80, 82, ..., 28, 26, 26],
           [76, 78, 80, ..., 26, 26, 25]], dtype=uint8)
    >>> s.xmap
    Phase  Orientations  Name  Space group  Point group  Proper point group     Color
        0    9 (100.0%)  None         None         None                None  tab:blue
    Properties:
    Scan unit: px
    """

    _signal_type = "EBSD"
    _alias_signal_types = ["electron_backscatter_diffraction"]
    _custom_properties = ["detector", "static_background", "xmap"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._detector = kwargs.get(
            "detector",
            EBSDDetector(
                shape=self.axes_manager.signal_shape,
                px_size=self.axes_manager.signal_axes[0].scale,
            ),
        )
        self._static_background = kwargs.get("static_background")
        self._xmap = kwargs.get("xmap")

    # ---------------------- Custom properties ----------------------- #

    @property
    def detector(self) -> EBSDDetector:
        """Return or set the detector describing the EBSD
        detector-sample geometry.

        Parameters
        ----------
        value : EBSDDetector
            EBSD detector.
        """
        return self._detector

    @detector.setter
    def detector(self, value: EBSDDetector):
        if _detector_is_compatible_with_signal(
            detector=value,
            navigation_shape=self.axes_manager.navigation_shape[::-1],
            signal_shape=self.axes_manager.signal_shape[::-1],
            raise_if_not=True,
        ):
            self._detector = value

    @property
    def xmap(self) -> Union[CrystalMap, None]:
        """Return or set the crystal map containing the phases, unit
        cell rotations and auxiliary properties of the EBSD dataset.

        Parameters
        ----------
        value : ~orix.crystal_map.CrystalMap
            Crystal map with the same shape as the signal navigation
            shape.
        """
        return self._xmap

    @xmap.setter
    def xmap(self, value: CrystalMap):
        if _crystal_map_is_compatible_with_signal(
            value, self.axes_manager.navigation_axes[::-1], raise_if_not=True
        ):
            self._xmap = value

    @property
    def static_background(self) -> Union[np.ndarray, da.Array, None]:
        """Return or set the static background pattern.

        Parameters
        ----------
        value : ~numpy.ndarray or ~dask.array.Array
            Static background pattern with the same (signal) shape and
            data type as the EBSD signal.
        """
        return self._static_background

    @static_background.setter
    def static_background(self, value: Union[np.ndarray, da.Array]):
        if value.dtype != self.data.dtype:
            warnings.warn("Background pattern has different data type from patterns")
        if value.shape != self.axes_manager.signal_shape[::-1]:
            warnings.warn("Background pattern has different shape from patterns")
        self._static_background = value

    # ------------------------ Custom methods ------------------------ #

    def set_scan_calibration(
        self, step_x: Union[int, float] = 1.0, step_y: Union[int, float] = 1.0
    ) -> None:
        """Set the step size in microns.

        Parameters
        ----------
        step_x
            Scan step size in um per pixel in horizontal direction.
        step_y
            Scan step size in um per pixel in vertical direction.

        See Also
        --------
        set_detector_calibration

        Examples
        --------
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s.axes_manager['x'].scale
        1.5
        >>> s.set_scan_calibration(step_x=2)  # Microns
        >>> s.axes_manager['x'].scale
        2.0
        """
        x, y = self.axes_manager.navigation_axes
        x.name, y.name = ("x", "y")
        x.scale, y.scale = (step_x, step_y)
        x.units, y.units = ["um"] * 2

    def set_detector_calibration(self, delta: Union[int, float]) -> None:
        """Set detector pixel size in microns. The offset is set to the
        the detector center.

        Parameters
        ----------
        delta
            Detector pixel size in microns.

        See Also
        --------
        set_scan_calibration

        Examples
        --------
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s.axes_manager['dx'].scale  # Default value
        1.0
        >>> s.set_detector_calibration(delta=70.)
        >>> s.axes_manager['dx'].scale
        70.0
        """
        center = delta * np.array(self.axes_manager.signal_shape) / 2
        dx, dy = self.axes_manager.signal_axes
        dx.units, dy.units = ["um"] * 2
        dx.scale, dy.scale = (delta, delta)
        dx.offset, dy.offset = -center

    def remove_static_background(
        self,
        operation: str = "subtract",
        static_bg: Union[np.ndarray, da.Array, None] = None,
        scale_bg: bool = False,
        show_progressbar: Optional[bool] = None,
    ) -> None:
        """Remove the static background inplace.

        The removal is performed by subtracting or dividing by a static
        background pattern. Resulting pattern intensities are rescaled
        loosing relative intensities and stretched to fill the available
        grey levels in the patterns' data type range.

        Parameters
        ----------
        operation
            Whether to ``"subtract"`` (default) or ``"divide"`` by the
            static background pattern.
        static_bg
            Static background pattern. If not given, the background is
            obtained from the ``EBSD.static_background`` property.
        scale_bg
            Whether to scale the static background pattern to each
            individual pattern's data range before removal. Default is
            ``False``.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.

        See Also
        --------
        remove_dynamic_background

        Examples
        --------
        It is assumed that a static background pattern of the same shape
        and data type (e.g. 8-bit unsigned integer, ``uint8``) as the
        patterns is available in signal metadata:

        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s.static_background
        array([[84, 87, 90, ..., 27, 29, 30],
               [87, 90, 93, ..., 27, 28, 30],
               [92, 94, 97, ..., 39, 28, 29],
               ...,
               [80, 82, 84, ..., 36, 30, 26],
               [79, 80, 82, ..., 28, 26, 26],
               [76, 78, 80, ..., 26, 26, 25]], dtype=uint8)

        The static background can be removed by subtracting or dividing
        this background from each pattern:

        >>> s.remove_static_background(operation="divide")

        If the ``static_background`` property is ``None``, this must be
        passed in the ``static_bg`` parameter as a ``numpy`` or ``dask``
        array.
        """
        dtype = np.float32  # During processing
        dtype_out = self.data.dtype.type
        omin, omax = dtype_range[dtype_out]

        # Get background pattern
        if static_bg is None:
            static_bg = self.static_background
            try:
                if not isinstance(static_bg, (np.ndarray, da.Array)):
                    raise ValueError
            except (AttributeError, ValueError):
                raise ValueError("`EBSD.static_background` is not a valid array")
        if isinstance(static_bg, da.Array):
            static_bg = static_bg.compute()
        if dtype_out != static_bg.dtype:
            raise ValueError(
                f"Static background dtype_out {static_bg.dtype} is not the same as "
                f"pattern dtype_out {dtype_out}"
            )
        pat_shape = self.axes_manager.signal_shape[::-1]  # xy -> ij
        bg_shape = static_bg.shape
        if bg_shape != pat_shape:
            raise ValueError(
                f"Signal {pat_shape} and static background {bg_shape} shapes are not "
                "the same"
            )
        static_bg = static_bg.astype(dtype)

        # Remove background and rescale to input data type
        if operation == "subtract":
            operation_func = _remove_static_background_subtract
        else:
            operation_func = _remove_static_background_divide

        properties = self._get_custom_properties()
        self.map(
            operation_func,
            show_progressbar=show_progressbar,
            parallel=True,
            output_dtype=dtype_out,
            static_bg=static_bg,
            dtype_out=dtype_out,
            omin=omin,
            omax=omax,
            scale_bg=scale_bg,
        )
        self._set_custom_properties(properties)

    def remove_dynamic_background(
        self,
        operation: str = "subtract",
        filter_domain: str = "frequency",
        std: Union[int, float, None] = None,
        truncate: Union[int, float] = 4.0,
        show_progressbar: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Remove the dynamic background in an EBSD scan inplace.

        The removal is performed by subtracting or dividing by a
        Gaussian blurred version of each pattern. Resulting pattern
        intensities are rescaled to fill the input patterns' data type
        range individually.

        Parameters
        ----------
        operation
            Whether to ``"subtract"`` (default) or ``"divide"`` by the
            dynamic background pattern.
        filter_domain
            Whether to obtain the dynamic background by applying a
            Gaussian convolution filter in the ``"frequency"`` (default)
            or ``"spatial"`` domain.
        std
            Standard deviation of the Gaussian window. If None
            (default), it is set to width/8.
        truncate
            Truncate the Gaussian window at this many standard
            deviations. Default is ``4.0``.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.
        **kwargs
            Keyword arguments passed to the Gaussian blurring function
            determined from ``filter_domain``.

        See Also
        --------
        remove_static_background,
        get_dynamic_background,
        kikuchipy.pattern.remove_dynamic_background,
        kikuchipy.pattern.get_dynamic_background

        Examples
        --------
        Remove the static and dynamic background

        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s.remove_static_background()
        >>> s.remove_dynamic_background(operation="divide", std=5)
        """
        if std is None:
            std = self.axes_manager.signal_shape[0] / 8

        # Get filter function and set up necessary keyword arguments
        if filter_domain == "frequency":
            # FFT filter setup for Connelly Barnes' algorithm
            filter_func = _fft_filter
            (
                kwargs["fft_shape"],
                kwargs["window_shape"],
                kwargs["transfer_function"],
                kwargs["offset_before_fft"],
                kwargs["offset_after_ifft"],
            ) = _dynamic_background_frequency_space_setup(
                pattern_shape=self.axes_manager.signal_shape[::-1],
                std=std,
                truncate=truncate,
            )
        elif filter_domain == "spatial":
            filter_func = gaussian_filter
            kwargs["sigma"] = std
            kwargs["truncate"] = truncate
        else:
            filter_domains = ["frequency", "spatial"]
            raise ValueError(f"{filter_domain} must be either of {filter_domains}.")

        map_func = _remove_dynamic_background

        dtype_out = self.data.dtype.type
        omin, omax = dtype_range[dtype_out]

        properties = self._get_custom_properties()
        self.map(
            map_func,
            show_progressbar=show_progressbar,
            parallel=True,
            output_dtype=dtype_out,
            filter_func=filter_func,
            operation=operation,
            dtype_out=dtype_out,
            omin=omin,
            omax=omax,
            **kwargs,
        )
        self._set_custom_properties(properties)

    def get_dynamic_background(
        self,
        filter_domain: str = "frequency",
        std: Union[int, float, None] = None,
        truncate: Union[int, float] = 4.0,
        dtype_out: Union[str, np.dtype, type, None] = None,
        show_progressbar: Optional[bool] = None,
        **kwargs,
    ) -> Union[EBSD, LazyEBSD]:
        """Get the dynamic background per EBSD pattern in a scan.

        Parameters
        ----------
        filter_domain
            Whether to apply a Gaussian convolution filter in the
            ``"frequency"`` (default) or ``"spatial"`` domain.
        std
            Standard deviation of the Gaussian window. If not given, it
            is set to width/8.
        truncate
            Truncate the Gaussian filter at this many standard
            deviations. Default is ``4.0``.
        dtype_out
            Data type of the background patterns. If not given, it is
            set to the same data type as the input pattern.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.
        **kwargs
            Keyword arguments passed to the Gaussian blurring function
            determined from ``filter_domain``.

        Returns
        -------
        background_signal
            Signal with the large scale variations across the detector.
        """
        if std is None:
            std = self.axes_manager.signal_shape[-1] / 8

        # Get filter function and set up necessary keyword arguments
        if filter_domain == "frequency":
            filter_func = _fft_filter
            # FFT filter setup for Connelly Barnes' algorithm
            (
                kwargs["fft_shape"],
                kwargs["window_shape"],
                kwargs["transfer_function"],
                kwargs["offset_before_fft"],
                kwargs["offset_after_ifft"],
            ) = _dynamic_background_frequency_space_setup(
                pattern_shape=self.axes_manager.signal_shape[::-1],
                std=std,
                truncate=truncate,
            )
        elif filter_domain == "spatial":
            filter_func = gaussian_filter
            kwargs["sigma"] = std
            kwargs["truncate"] = truncate
        else:
            filter_domains = ["frequency", "spatial"]
            raise ValueError(f"{filter_domain} must be either of {filter_domains}.")

        if dtype_out is None:
            dtype_out = self.data.dtype
        else:
            dtype_out = np.dtype(dtype_out)
        dask_array = get_dask_array(self, dtype=dtype_out)

        background_patterns = dask_array.map_blocks(
            func=chunk.get_dynamic_background,
            filter_func=filter_func,
            dtype_out=dtype_out,
            dtype=dtype_out,
            **kwargs,
        )

        if not self._lazy:
            background_return = np.empty(
                shape=background_patterns.shape, dtype=dtype_out
            )

            pbar = ProgressBar()
            if show_progressbar or (
                show_progressbar is None and hs.preferences.General.show_progressbar
            ):
                pbar.register()

            background_patterns.store(background_return, compute=True)
            background_signal = EBSD(background_return)

            try:
                pbar.unregister()
            except KeyError:
                pass
        else:
            background_signal = LazyEBSD(background_patterns)

        return background_signal

    def adaptive_histogram_equalization(
        self,
        kernel_size: Optional[Union[Tuple[int, int], List[int]]] = None,
        clip_limit: Union[int, float] = 0,
        nbins: int = 128,
        show_progressbar: Optional[bool] = None,
    ) -> None:
        """Enhance the local contrast in an EBSD scan inplace using
        adaptive histogram equalization.

        This method uses :func:`skimage.exposure.equalize_adapthist`.

        Parameters
        ----------
        kernel_size
            Shape of contextual regions for adaptive histogram
            equalization, default is 1/4 of image height and 1/4 of
            image width.
        clip_limit
            Clipping limit, normalized between 0 and 1 (higher values
            give more contrast). Default is ``0``.
        nbins
            Number of gray bins for histogram ("data range"), default is
            ``128``.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.

        See Also
        --------
        kikuchipy.signals.EBSD.rescale_intensity,
        kikuchipy.signals.EBSD.normalize_intensity

        Notes
        -----
        It is recommended to perform adaptive histogram equalization
        only *after* static and dynamic background corrections,
        otherwise some unwanted darkening towards the edges might
        occur.

        The default window size might not fit all pattern sizes, so it
        may be necessary to search for the optimal window size.

        Examples
        --------
        Load one pattern from the small nickel dataset, remove the
        background and perform adaptive histogram equalization. A copy
        without equalization is kept for comparison.

        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small().inav[0, 0]
        >>> s.remove_static_background()
        >>> s.remove_dynamic_background()
        >>> s2 = s.deepcopy()
        >>> s2.adaptive_histogram_equalization()

        Compute the intensity histograms and plot the patterns and
        histograms

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> hist, _ = np.histogram(s.data, range=(0, 255))
        >>> hist2, _ = np.histogram(s2.data, range=(0, 255))
        >>> _, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
        >>> _ = ax0.imshow(s.data)
        >>> _ = ax1.imshow(s2.data)
        >>> _ = ax2.plot(hist)
        >>> _ = ax3.plot(hist2)
        """
        # Determine window size (shape of contextual region)
        sig_shape = self.axes_manager.signal_shape
        if kernel_size is None:
            kernel_size = (sig_shape[0] // 4, sig_shape[1] // 4)
        elif isinstance(kernel_size, numbers.Number):
            kernel_size = (kernel_size,) * self.axes_manager.signal_dimension
        elif len(kernel_size) != self.axes_manager.signal_dimension:
            raise ValueError(f"Incorrect value of `shape`: {kernel_size}")
        kernel_size = [int(k) for k in kernel_size]

        # Create dask array of signal patterns and do processing on this
        dask_array = get_dask_array(signal=self)

        # Local contrast enhancement
        equalized_patterns = dask_array.map_blocks(
            func=chunk.adaptive_histogram_equalization,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=nbins,
            dtype=self.data.dtype,
        )

        # Overwrite signal patterns
        if not self._lazy:
            pbar = ProgressBar()
            if show_progressbar or (
                show_progressbar is None and hs.preferences.General.show_progressbar
            ):
                pbar.register()

            equalized_patterns.store(self.data, compute=True)

            try:
                pbar.unregister()
            except KeyError:
                pass
        else:
            self.data = equalized_patterns

    def get_image_quality(
        self,
        normalize: bool = True,
        show_progressbar: Optional[bool] = None,
    ) -> Union[np.ndarray, da.Array]:
        """Compute the image quality map of patterns in an EBSD scan.

        The image quality :math:`Q` is calculated based on the procedure
        defined by Krieger Lassen :cite:`lassen1994automated`.

        Parameters
        ----------
        normalize
            Whether to normalize patterns to a mean of zero and standard
            deviation of 1 before calculating the image quality. Default
            is ``True``.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.

        Returns
        -------
        image_quality_map
            Image quality map of same shape as navigation axes. This is
            a Dask array if the signal is lazy.

        See Also
        --------
        kikuchipy.pattern.get_image_quality

        Examples
        --------
        Load an example dataset, remove the static and dynamic
        background and compute :math:`Q`

        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> s
        <EBSD, title: patterns My awes0m4 ..., dimensions: (3, 3|60, 60)>
        >>> s.remove_static_background()
        >>> s.remove_dynamic_background()
        >>> iq = s.get_image_quality()
        >>> iq
        array([[0.19935645, 0.16657268, 0.18803978],
               [0.19040637, 0.1616931 , 0.17834103],
               [0.19411428, 0.16031407, 0.18413563]], dtype=float32)
        """
        # Calculate frequency vectors
        sx, sy = self.axes_manager.signal_shape
        frequency_vectors = fft_frequency_vectors((sy, sx))
        inertia_max = np.sum(frequency_vectors) / (sy * sx)

        image_quality_map = self.map(
            _get_image_quality,
            show_progressbar=show_progressbar,
            parallel=True,
            inplace=False,
            output_dtype=np.float32,
            normalize=normalize,
            frequency_vectors=frequency_vectors,
            inertia_max=inertia_max,
        )

        return image_quality_map.data

    def dictionary_indexing(
        self,
        dictionary: EBSD,
        metric: Union[SimilarityMetric, str] = "ncc",
        keep_n: int = 20,
        n_per_iteration: Optional[int] = None,
        signal_mask: Optional[np.ndarray] = None,
        rechunk: bool = False,
        dtype: Union[str, np.dtype, type, None] = None,
    ) -> CrystalMap:
        """Match each experimental pattern to a dictionary of simulated
        patterns of known orientations to index them
        :cite:`chen2015dictionary,jackson2019dictionary`.

        Parameters
        ----------
        dictionary
            EBSD signal with dictionary patterns. The signal must have a
            1D navigation axis, an :attr:`xmap` property with crystal
            orientations set, and equal detector shape.
        metric
            Similarity metric, by default ``"ncc"`` (normalized
            cross-correlation). ``"ndp"`` (normalized dot product) is
            also available. A valid user-defined similarity metric
            may be used instead. The metric must be a class implementing
            the :class:`~kikuchipy.indexing.SimilarityMetric` abstract
            class methods. See
            :class:`~kikuchipy.indexing.NormalizedCrossCorrelationMetric`
            and :class:`~kikuchipy.indexing.NormalizedDotProductMetric`
            for examples.
        keep_n
            Number of best matches to keep, by default 20 or the number
            of dictionary patterns if fewer than 20 are available.
        n_per_iteration
            Number of dictionary patterns to compare to all experimental
            patterns in each indexing iteration. If not given, and the
            dictionary is a ``LazyEBSD`` signal, it is equal to the
            chunk size of the first pattern array axis, while if if is
            an ``EBSD`` signal, it is set equal to the number of
            dictionary patterns, yielding only one iteration. This
            parameter can be increased to use less memory during
            indexing, but this will increase the computation time.

            .. versionadded:: 0.5
        signal_mask
            A boolean mask equal to the experimental patterns' detector
            shape ``(n rows, n columns)``, where only pixels equal to
            ``False`` are matched. If not given, all pixels are used.

            .. versionadded:: 0.5
        rechunk
            Whether ``metric`` is allowed to rechunk experimental and
            dictionary patterns before matching. Default is ``False``.
            Rechunking usually makes indexing faster, but uses more
            memory. If a custom ``metric`` is passed, whatever
            :attr:`~kikuchipy.indexing.SimilarityMetric.rechunk` is set
            to will be used.

            .. versionadded:: 0.5
        dtype
            Which data type ``metric`` shall cast the patterns to before
            matching. If not given, ``"float32"`` will be used unless a
            custom ``metric`` is passed and it has set the
            :attr:`~kikuchipy.indexing.SimilarityMetric.dtype`, which
            will then be used instead. ``"float32"`` and ``"float64"``
            are allowed for the available ``"ncc"`` and ``"ndp"``
            metrics.

            .. versionadded:: 0.5

        Returns
        -------
        xmap
            A crystal map with ``keep_n`` rotations per point with the
            sorted best matching orientations in the dictionary. The
            corresponding best scores and indices into the dictionary
            are stored in the ``xmap.prop`` dictionary as ``"scores"``
            and ``"simulation_indices"``.

        See Also
        --------
        kikuchipy.indexing.SimilarityMetric
        kikuchipy.indexing.NormalizedCrossCorrelationMetric
        kikuchipy.indexing.NormalizedDotProductMetric
        kikuchipy.indexing.merge_crystal_maps :
            Merge multiple single phase crystal maps into one multi
            phase map.
        kikuchipy.indexing.orientation_similarity_map :
            Calculate an orientation similarity map.

        Notes
        -----
        .. versionchanged:: 0.5
           Only one dictionary can be passed and the
           ``return_merged_crystal_map`` and
           ``get_orientation_similarity_map`` parameters were removed.
        """
        exp_am = self.axes_manager
        dict_am = dictionary.axes_manager
        dict_size = dict_am.navigation_size

        if n_per_iteration is None:
            if isinstance(dictionary.data, da.Array):
                n_per_iteration = dictionary.data.chunksize[0]
            else:
                n_per_iteration = dict_size

        exp_sig_shape = exp_am.signal_shape[::-1]
        dict_sig_shape = dict_am.signal_shape[::-1]
        if exp_sig_shape != dict_sig_shape:
            raise ValueError(
                f"Experimental {exp_sig_shape} and dictionary {dict_sig_shape} signal "
                "shapes must be identical"
            )

        dict_xmap = dictionary.xmap
        if dict_xmap is None or dict_xmap.shape != (dict_size,):
            raise ValueError(
                "Dictionary signal must have a non-empty `EBSD.xmap` property of equal "
                "size as the number of dictionary patterns, and both the signal and"
                "crystal map must have only one navigation dimension"
            )

        metric = self._prepare_metric(metric, signal_mask, dtype, rechunk, dict_size)

        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            return _dictionary_indexing(
                experimental=self.data,
                experimental_nav_shape=exp_am.navigation_shape[::-1],
                dictionary=dictionary.data,
                step_sizes=tuple(a.scale for a in exp_am.navigation_axes[::-1]),
                dictionary_xmap=dictionary.xmap,
                keep_n=keep_n,
                n_per_iteration=n_per_iteration,
                metric=metric,
            )

    @deprecated_argument(
        name="mask", since="0.7.0", removal="0.8.0", alternative="signal_mask"
    )
    def refine_orientation(
        self,
        xmap: CrystalMap,
        detector: EBSDDetector,
        master_pattern: "EBSDMasterPattern",
        energy: Union[int, float],
        mask: Optional[np.ndarray] = None,
        signal_mask: Optional[np.ndarray] = None,
        method: Optional[str] = "minimize",
        method_kwargs: Optional[dict] = None,
        trust_region: Optional[list] = None,
        compute: bool = True,
        rechunk: bool = True,
        chunk_kwargs: Optional[dict] = None,
    ) -> Union[CrystalMap, da.Array]:
        r"""Refine orientations by searching orientation space around
        the best indexed solution using fixed projection centers.

        Refinement attempts to maximize the similarity between patterns
        in this signal and simulated patterns projected from a master
        pattern. The similarity metric used is the normalized
        cross-correlation (NCC). The orientation, represented by a
        Rodrigues-Frank vector (:math:`R_x`, :math:`R_y`, :math:`R_z`),
        is optimized during refinement, while the sample-detector
        geometry, represented by the three projection center (PC)
        parameters (PCx, PCy, PCz), is fixed.

        A subset of the optimization methods in SciPy are available:
            - Local optimization via :func:`~scipy.optimize.minimize`
              (includes Nelder-Mead, Powell etc.)
            - Global optimization:
                - :func:`~scipy.optimize.differential_evolution`
                - :func:`~scipy.optimize.dual_annealing`
                - :func:`~scipy.optimize.basinhopping`
                - :func:`~scipy.optimize.shgo`

        Parameters
        ----------
        xmap
            Single phase crystal map with at least one orientation per
            point. The orientations are assumed to be relative to the
            EDAX TSL sample reference frame RD-TD-ND.
        detector
            Detector describing the detector-sample geometry with either
            one PC to be used for all map points or one for each point.
        master_pattern
            Master pattern in the square Lambert projection of the same
            phase as the one in the crystal map.
        energy
            Accelerating voltage of the electron beam in kV specifying
            which master pattern energy to use during projection of
            simulated patterns.
        mask
            A boolean mask equal to the experimental patterns' detector
            shape ``(n rows, n columns)``, where only pixels equal to
            ``False`` are matched. If not given, all pixels are used.

            .. deprecated:: 0.7.0
                Use ``signal_mask`` instead.
        signal_mask
            A boolean mask equal to the experimental patterns' detector
            shape ``(n rows, n columns)``, where only pixels equal to
            ``False`` are matched. If not given, all pixels are used.
        method
            Name of the :mod:`scipy.optimize` optimization method, among
            ``"minimize"``, ``"differential_evolution"``,
            ``"dual_annealing"``, ``"basinhopping"``, and ``"shgo"``.
            Default is ``"minimize"``, which by default performs local
            optimization with the Nelder-Mead method unless another
            ``"minimize"`` method is passed to ``method_kwargs``.
        method_kwargs
            Keyword arguments passed to the :mod:`scipy.optimize`
            ``method``. For example, to perform refinement with the
            modified Powell algorithm, pass ``method="minimize"`` and
            ``method_kwargs=dict(method="Powell")``.
        trust_region
            List of +/- angular deviation in degrees as bound
            constraints on the three Rodrigues-Frank vector components.
            If not given and ``method`` requires bounds, they are set to
            ``[1, 1, 1]``. If given, ``method`` is assumed to support
            bounds and they are passed to ``method``.
        compute
            Whether to refine now (``True``) or later (``False``).
            Default is ``True``. See :meth:`~dask.array.Array.compute`
            for more details.
        rechunk
            If ``True`` (default), rechunk the dask array with patterns
            used in refinement (not the signal data inplace) if it is
            returned from :func:`~kikuchipy.signals.util.get_dask_array`
            in a single chunk. This ensures small data sets are
            rechunked so as to utilize multiple CPUs.
        chunk_kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.signals.util.get_chunking` if
            ``rechunk=True`` and the dask array with patterns used in
            refinement is returned from
            :func:`~kikuchipy.signals.util.get_dask_array` in a single
            chunk.

        Returns
        -------
        out
            Crystal map with refined orientations and similarity metrics
            in a ``"scores"`` property if ``compute=True``. If
            ``compute=False``, a dask array of navigation shape + (4,)
            is returned, to be computed later. See
            :func:`~kikuchipy.indexing.compute_refine_orientation_results`.
            Each navigation point has the optimized score and the three
            Euler angles in radians in element 0, 1, 2, and 3,
            respectively.

        See Also
        --------
        scipy.optimize, refine_projection_center,
        refine_orientation_projection_center
        """
        if mask is not None:
            signal_mask = mask
        self._check_refinement_parameters(
            xmap=xmap, detector=detector, signal_mask=signal_mask
        )
        patterns, signal_mask = self._prepare_patterns_for_refinement(
            signal_mask=signal_mask, rechunk=rechunk, chunk_kwargs=chunk_kwargs
        )
        return _refine_orientation(
            xmap=xmap,
            detector=detector,
            master_pattern=master_pattern,
            energy=energy,
            patterns=patterns,
            signal_mask=signal_mask,
            method=method,
            method_kwargs=method_kwargs,
            trust_region=trust_region,
            compute=compute,
        )

    @deprecated_argument(
        name="mask", since="0.7.0", removal="0.8.0", alternative="signal_mask"
    )
    def refine_projection_center(
        self,
        xmap: CrystalMap,
        detector: EBSDDetector,
        master_pattern: "EBSDMasterPattern",
        energy: Union[int, float],
        mask: Optional[np.ndarray] = None,
        signal_mask: Optional[np.ndarray] = None,
        method: Optional[str] = "minimize",
        method_kwargs: Optional[dict] = None,
        trust_region: Optional[list] = None,
        compute: bool = True,
        rechunk: bool = True,
        chunk_kwargs: Optional[dict] = None,
    ) -> Union[Tuple[np.ndarray, EBSDDetector], da.Array]:
        """Refine projection centers by searching the parameter space
        using fixed orientations.

        Refinement attempts to maximize the similarity between patterns
        in this signal and simulated patterns projected from a master
        pattern. The similarity metric used is the normalized
        cross-correlation (NCC). The sample-detector geometry,
        represented by the three projection center (PC) parameters
        (PCx, PCy, PCz), is updated during refinement, while the
        orientations are fixed.

        A subset of the optimization methods in SciPy are available:
            - Local optimization:
                - :func:`~scipy.optimize.minimize`
                  (includes Nelder-Mead, Powell etc.)
            - Global optimization:
                - :func:`~scipy.optimize.differential_evolution`
                - :func:`~scipy.optimize.dual_annealing`
                - :func:`~scipy.optimize.basinhopping`
                - :func:`~scipy.optimize.shgo`

        Parameters
        ----------
        xmap
            Single phase crystal map with at least one orientation per
            point. The orientations are assumed to be relative to the
            EDAX TSL sample reference frame RD-TD-ND.
        detector
            Detector describing the detector-sample geometry with either
            one PC to be used for all map points or one for each point.
        master_pattern
            Master pattern in the square Lambert projection of the same
            phase as the one in the crystal map.
        energy
            Accelerating voltage of the electron beam in kV specifying
            which master pattern energy to use during projection of
            simulated patterns.
        mask
            A boolean mask equal to the experimental patterns' detector
            shape ``(n rows, n columns)``, where only pixels equal to
            ``False`` are matched. If not given, all pixels are used.

            .. deprecated:: 0.7.0
                Use ``signal_mask`` instead.
        signal_mask
            A boolean mask equal to the experimental patterns' detector
            shape ``(n rows, n columns)``, where only pixels equal to
            ``False`` are matched. If not given, all pixels are used.
        method
            Name of the :mod:`scipy.optimize` optimization method, among
            ``"minimize"``, ``"differential_evolution"``,
            ``"dual_annealing"``, ``"basinhopping"``, and ``"shgo"``.
            Default is ``"minimize"``, which by default performs local
            optimization with the Nelder-Mead method unless another
            ``"minimize"`` method is passed to ``method_kwargs``.
        method_kwargs
            Keyword arguments passed to the :mod:`scipy.optimize`
            ``method``. For example, to perform refinement with the
            modified Powell algorithm, pass ``method="minimize"`` and
            ``method_kwargs=dict(method="Powell")``.
        trust_region
            List of +/- percentage deviations as bound constraints on
            the PC parameters in the Bruker convention. The parameter
            range is [0, 1]. If not given and ``method`` requires
            bounds, they are set to ``[0.05, 0.05, 0.05]``. If given,
            ``method`` is assumed to support bounds and they are passed
            to ``method``.
        compute
            Whether to refine now (``True``) or later (``False``).
            Default is ``True``. See :meth:`~dask.array.Array.compute`
            for more details.
        rechunk
            If ``True`` (default), rechunk the dask array with patterns
            used in refinement (not the signal data inplace) if it is
            returned from :func:`~kikuchipy.signals.util.get_dask_array`
            in a single chunk. This ensures small data sets are
            rechunked so as to utilize multiple CPUs.
        chunk_kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.signals.util.get_chunking` if
            ``rechunk=True`` and the dask array with patterns used in
            refinement is returned from
            :func:`~kikuchipy.signals.util.get_dask_array` in a single
            chunk.

        Returns
        -------
        out
            New similarity metrics and a new EBSD detector instance with
            the refined PCs if ``compute=True``. If ``compute=False``,
            a dask array of navigation shape + (4,) is returned, to be
            computed later. See
            :func:`~kikuchipy.indexing.compute_refine_projection_center_results`.
            Each navigation point has the optimized score and the three
            PC parameters in the Bruker convention in element 0, 1, 2,
            and 3, respectively.

        See Also
        --------
        scipy.optimize, refine_orientation,
        refine_orientation_projection_center
        """
        if mask is not None:
            signal_mask = mask
        self._check_refinement_parameters(
            xmap=xmap, detector=detector, signal_mask=signal_mask
        )
        patterns, signal_mask = self._prepare_patterns_for_refinement(
            signal_mask=signal_mask, rechunk=rechunk, chunk_kwargs=chunk_kwargs
        )
        return _refine_projection_center(
            xmap=xmap,
            detector=detector,
            master_pattern=master_pattern,
            energy=energy,
            patterns=patterns,
            signal_mask=signal_mask,
            method=method,
            method_kwargs=method_kwargs,
            trust_region=trust_region,
            compute=compute,
        )

    @deprecated_argument(
        name="mask", since="0.7.0", removal="0.8.0", alternative="signal_mask"
    )
    def refine_orientation_projection_center(
        self,
        xmap: CrystalMap,
        detector: EBSDDetector,
        master_pattern: "EBSDMasterPattern",
        energy: Union[int, float],
        mask: Optional[np.ndarray] = None,
        signal_mask: Optional[np.ndarray] = None,
        method: Optional[str] = "minimize",
        method_kwargs: Optional[dict] = None,
        trust_region: Optional[list] = None,
        compute: bool = True,
        rechunk: bool = True,
        chunk_kwargs: Optional[dict] = None,
    ) -> Union[Tuple[CrystalMap, EBSDDetector], da.Array]:
        r"""Refine orientations and projection centers simultaneously by
        searching the orientation and PC parameter space.

        Refinement attempts to maximize the similarity between patterns
        in this signal and simulated patterns projected from a master
        pattern. The only supported similarity metric is the normalized
        cross-correlation (NCC). The orientation, represented by a
        Rodrigues-Frank vector (:math:`R_x`, :math:`R_y`, :math:`R_z`),
        and the sample-detector geometry, represented by the three
        projection center (PC) parameters (PCx, PCy, PCz), are updated
        during refinement.

        A subset of the optimization methods in SciPy are available:
            - Local optimization:
                - :func:`~scipy.optimize.minimize`
                  (includes Nelder-Mead, Powell etc.)
            - Global optimization:
                - :func:`~scipy.optimize.differential_evolution`
                - :func:`~scipy.optimize.dual_annealing`
                - :func:`~scipy.optimize.basinhopping`
                - :func:`~scipy.optimize.shgo`

        Parameters
        ----------
        xmap
            Single phase crystal map with at least one orientation per
            point. The orientations are assumed to be relative to the
            EDAX TSL sample reference frame RD-TD-ND.
        detector
            Detector describing the detector-sample geometry with either
            one PC to be used for all map points or one for each point.
        master_pattern
            Master pattern in the square Lambert projection of the same
            phase as the one in the crystal map.
        energy
            Accelerating voltage of the electron beam in kV specifying
            which master pattern energy to use during projection of
            simulated patterns.
        mask
            A boolean mask equal to the experimental patterns' detector
            shape ``(n rows, n columns)``, where only pixels equal to
            ``False`` are matched. If not given, all pixels are used.

            .. deprecated:: 0.7.0
                Use ``signal_mask`` instead.
        signal_mask
            A boolean mask equal to the experimental patterns' detector
            shape ``(n rows, n columns)``, where only pixels equal to
            ``False`` are matched. If not given, all pixels are used.
        method
            Name of the :mod:`scipy.optimize` optimization method, among
            ``"minimize"``, ``"differential_evolution"``,
            ``"dual_annealing"``, ``"basinhopping"``, and ``"shgo"``.
            Default is ``"minimize"``, which by default performs local
            optimization with the Nelder-Mead method unless another
            ``"minimize"`` method is passed to ``method_kwargs``.
        method_kwargs
            Keyword arguments passed to the :mod:`scipy.optimize`
            ``method``. For example, to perform refinement with the
            modified Powell algorithm, pass ``method="minimize"`` and
            ``method_kwargs=dict(method="Powell")``.
        trust_region
            List of +/- angular deviations in degrees as bound
            constraints on the three Rodrigues-Frank vector components
            and +/- percentage deviations as bound constraints on the PC
            parameters in the Bruker convention. The latter parameter
            range is [0, 1]. If not given and ``method`` requires
            bounds, they are set to ``[1, 1, 1, 0.05, 0.05, 0.05]``. If
            given, ``method`` is assumed to support bounds and they are
            passed to ``method``.
        compute
            Whether to refine now (``True``) or later (``False``).
            Default is ``True``. See :meth:`~dask.array.Array.compute`
            for more details.
        rechunk
            If ``True`` (default), rechunk the dask array with patterns
            used in refinement (not the signal data inplace) if it is
            returned from :func:`~kikuchipy.signals.util.get_dask_array`
            in a single chunk. This ensures small data sets are
            rechunked so as to utilize multiple CPUs.
        chunk_kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.signals.util.get_chunking` if
            ``rechunk=True`` and the dask array with patterns used in
            refinement is returned from
            :func:`~kikuchipy.signals.util.get_dask_array` in a single
            chunk.

        Returns
        -------
        out
            Crystal map with refined orientations and a new EBSD
            detector instance with the refined PCs, if ``compute=True``.
            If ``compute=False``, a dask array of navigation shape +
            (7,) is returned, to be computed later. See
            :func:`~kikuchipy.indexing.compute_refine_orientation_projection_center_results`.
            Each navigation point has the optimized score, the three
            Rodriues-Frank vector components in radians, and the three
            PC parameters in the Bruker convention in element 0, 1, 2,
            3, 4, 5, and 6, respectively.

        See Also
        --------
        scipy.optimize, refine_orientation, refine_projection_center

        Notes
        -----
        The method attempts to refine the orientations and projection
        center at the same time for each map point. The optimization
        landscape is sloppy :cite:`pang2020optimization`, where the
        orientation and PC can make up for each other. Thus, it is
        possible that the parameters that yield the highest similarity
        are incorrect. As always, it is left to the user to ensure that
        the output is reasonable.
        """
        if mask is not None:
            signal_mask = mask
        self._check_refinement_parameters(
            xmap=xmap, detector=detector, signal_mask=signal_mask
        )
        patterns, signal_mask = self._prepare_patterns_for_refinement(
            signal_mask=signal_mask, rechunk=rechunk, chunk_kwargs=chunk_kwargs
        )
        return _refine_orientation_projection_center(
            xmap=xmap,
            detector=detector,
            master_pattern=master_pattern,
            energy=energy,
            patterns=patterns,
            signal_mask=signal_mask,
            method=method,
            method_kwargs=method_kwargs,
            trust_region=trust_region,
            compute=compute,
        )

    def fft_filter(
        self,
        transfer_function: Union[np.ndarray, Window],
        function_domain: str,
        shift: bool = False,
        show_progressbar: Optional[bool] = None,
    ) -> None:
        """Filter an EBSD scan inplace in the frequency domain.

        Patterns are transformed via the Fast Fourier Transform (FFT) to
        the frequency domain, where their spectrum is multiplied by the
        ``transfer_function``, and the filtered spectrum is subsequently
        transformed to the spatial domain via the inverse FFT (IFFT).
        Filtered patterns are rescaled to input data type range.

        Note that if ``function_domain`` is ``"spatial"``, only real
        valued FFT and IFFT is used.

        Parameters
        ----------
        transfer_function
            Filter to apply to patterns. This can either be a transfer
            function in the frequency domain of pattern shape or a
            kernel in the spatial domain. What is passed is determined
            from ``function_domain``.
        function_domain
            Options are ``"frequency"`` and ``"spatial"``, indicating,
            respectively, whether the filter function passed to
            ``filter_function`` is a transfer function in the frequency
            domain or a kernel in the spatial domain.
        shift
            Whether to shift the zero-frequency component to the center.
            Default is ``False``. This is only used when
            ``function_domain="frequency"``.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.

        See Also
        --------
        kikuchipy.filters.Window

        Examples
        --------
        Applying a Gaussian low pass filter with a cutoff frequency of
        20:

        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> pattern_shape = s.axes_manager.signal_shape[::-1]
        >>> w = kp.filters.Window(
        ...     "lowpass", cutoff=20, shape=pattern_shape
        ... )
        >>> s.fft_filter(
        ...     transfer_function=w,
        ...     function_domain="frequency",
        ...     shift=True,
        ... )
        """
        dtype_out = self.data.dtype.type

        dtype = np.float32
        dask_array = get_dask_array(signal=self, dtype=dtype)

        kwargs = {}
        if function_domain == "frequency":
            filter_func = fft_filter
            kwargs["shift"] = shift
        elif function_domain == "spatial":
            filter_func = _fft_filter  # Barnes
            kwargs["window_shape"] = transfer_function.shape

            # FFT filter setup
            (
                kwargs["fft_shape"],
                transfer_function,  # Padded
                kwargs["offset_before_fft"],
                kwargs["offset_after_ifft"],
            ) = _fft_filter_setup(
                image_shape=self.axes_manager.signal_shape[::-1],
                window=transfer_function,
            )
        else:
            function_domains = ["frequency", "spatial"]
            raise ValueError(f"{function_domain} must be either of {function_domains}.")

        filtered_patterns = dask_array.map_blocks(
            func=chunk.fft_filter,
            filter_func=filter_func,
            transfer_function=transfer_function,
            dtype_out=dtype_out,
            dtype=dtype_out,
            **kwargs,
        )

        # Overwrite signal patterns
        if not self._lazy:
            pbar = ProgressBar()
            if show_progressbar or (
                show_progressbar is None and hs.preferences.General.show_progressbar
            ):
                pbar.register()

            filtered_patterns.store(self.data, compute=True)

            try:
                pbar.unregister()
            except KeyError:
                pass
        else:
            self.data = filtered_patterns

    def get_neighbour_dot_product_matrices(
        self,
        window: Optional[Window] = None,
        zero_mean: bool = True,
        normalize: bool = True,
        dtype_out: Union[str, np.dtype, type] = "float32",
        show_progressbar: Optional[bool] = None,
    ) -> Union[np.ndarray, da.Array]:
        """Get an array with dot products of a pattern and its
        neighbours within a window.

        Parameters
        ----------
        window
            Window with integer coefficients defining the neighbours to
            calculate the dot products with. If not given, the four
            nearest neighbours are used. Must have the same number of
            dimensions as signal navigation dimensions.
        zero_mean
            Whether to subtract the mean of each pattern individually to
            center the intensities about zero before calculating the
            dot products. Default is ``True``.
        normalize
            Whether to normalize the pattern intensities to a standard
            deviation of 1 before calculating the dot products. This
            operation is performed after centering the intensities if
            ``zero_mean=True``. Default is ``True``.
        dtype_out
            Data type of the output map. Default is ``"float32"``.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.

        Returns
        -------
        dp_matrices
            Dot products between a pattern and its nearest neighbours.
        """
        if self.axes_manager.navigation_dimension == 0:
            raise ValueError("Signal must have at least one navigation dimension")

        # Create dask array of signal patterns and do processing on this
        dask_array = get_dask_array(signal=self)

        # Default to the nearest neighbours
        nav_dim = self.axes_manager.navigation_dimension
        if window is None:
            window = Window(window="circular", shape=(3, 3)[:nav_dim])

        # Set overlap depth between navigation chunks equal to the max.
        # number of nearest neighbours in each navigation axis
        overlap_depth = _get_chunk_overlap_depth(
            window=window,
            axes_manager=self.axes_manager,
            chunksize=dask_array.chunksize,
        )

        dtype_out = np.dtype(dtype_out)

        dp_matrices = dask_array.map_overlap(
            _get_neighbour_dot_product_matrices,
            window=window,
            sig_dim=self.axes_manager.signal_dimension,
            sig_size=self.axes_manager.signal_size,
            zero_mean=zero_mean,
            normalize=normalize,
            dtype_out=dtype_out,
            drop_axis=self.axes_manager.signal_indices_in_array[::-1],
            new_axis=tuple(np.arange(window.ndim) + nav_dim),
            dtype=dtype_out,
            depth=overlap_depth,
            boundary="none",
        )

        if not self._lazy:
            pbar = ProgressBar()
            if show_progressbar or (
                show_progressbar is None and hs.preferences.General.show_progressbar
            ):
                pbar.register()

            dp_matrices = dp_matrices.compute()

            try:
                pbar.unregister()
            except KeyError:
                pass

        return dp_matrices

    def get_average_neighbour_dot_product_map(
        self,
        window: Optional[Window] = None,
        zero_mean: bool = True,
        normalize: bool = True,
        dtype_out: Union[str, np.dtype, type] = "float32",
        dp_matrices: Optional[np.ndarray] = None,
        show_progressbar: Optional[bool] = None,
    ) -> Union[np.ndarray, da.Array]:
        """Get a map of the average dot product between patterns and
        their neighbours within an averaging window.

        Parameters
        ----------
        window
            Window with integer coefficients defining the neighbours to
            calculate the average with. If not given, the four nearest
            neighbours are used. Must have the same number of dimensions
            as signal navigation dimensions.
        zero_mean
            Whether to subtract the mean of each pattern individually to
            center the intensities about zero before calculating the
            dot products. Default is ``True``.
        normalize
            Whether to normalize the pattern intensities to a standard
            deviation of 1 before calculating the dot products. This
            operation is performed after centering the intensities if
            ``zero_mean=True``. Default is ``True``.
        dtype_out
            Data type of the output map. Default is ``"float32"``.
        dp_matrices
            Optional pre-calculated dot product matrices, by default
            ``None``. If an array is passed, the average dot product map
            is calculated from this array. The ``dp_matrices`` array can
            be obtained from :meth:`get_neighbour_dot_product_matrices`.
            Its shape must correspond to the signal's navigation shape
            and the window's shape.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.

        Returns
        -------
        adp
            Average dot product map.
        """
        if self.axes_manager.navigation_dimension == 0:
            raise ValueError("Signal must have at least one navigation dimension")

        # Default to the nearest neighbours
        nav_dim = self.axes_manager.navigation_dimension
        if window is None:
            window = Window(window="circular", shape=(3, 3)[:nav_dim])

        if dp_matrices is not None:
            nan_slices = [slice(None) for _ in range(nav_dim)]
            nan_slices += [slice(i, i + 1) for i in window.origin]
            dp_matrices2 = dp_matrices.copy()
            dp_matrices2[tuple(nan_slices)] = np.nan
            if nav_dim == 1:
                mean_axis = 1
            else:  # == 2
                mean_axis = (2, 3)
            return np.nanmean(dp_matrices2, axis=mean_axis)

        # Create dask array of signal data and do processing on this
        dask_array = get_dask_array(signal=self)

        # Set overlap depth between navigation chunks equal to the max.
        # number of nearest neighbours in each navigation axis
        overlap_depth = _get_chunk_overlap_depth(
            window=window,
            axes_manager=self.axes_manager,
            chunksize=dask_array.chunksize,
        )

        dtype_out = np.dtype(dtype_out)

        adp = dask_array.map_overlap(
            _get_average_dot_product_map,
            window=window,
            sig_dim=self.axes_manager.signal_dimension,
            sig_size=self.axes_manager.signal_size,
            zero_mean=zero_mean,
            normalize=normalize,
            dtype_out=dtype_out,
            drop_axis=self.axes_manager.signal_indices_in_array,
            dtype=dtype_out,
            depth=overlap_depth,
            boundary="none",
        )
        chunks = get_chunking(
            data_shape=self.axes_manager.navigation_shape[::-1],
            nav_dim=nav_dim,
            sig_dim=0,
            dtype=dtype_out,
        )
        adp = adp.rechunk(chunks=chunks)

        if not self._lazy:
            pbar = ProgressBar()
            if show_progressbar or (
                show_progressbar is None and hs.preferences.General.show_progressbar
            ):
                pbar.register()

            adp = adp.compute()

            try:
                pbar.unregister()
            except KeyError:
                pass

        return adp

    def average_neighbour_patterns(
        self,
        window: Union[str, np.ndarray, da.Array, Window] = "circular",
        window_shape: Tuple[int, ...] = (3, 3),
        show_progressbar: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Average patterns inplace with its neighbours within a window.

        The amount of averaging is specified by the window coefficients.
        All patterns are averaged with the same window. Map borders are
        extended with zeros. Resulting pattern intensities are rescaled
        to fill the input patterns' data type range individually.

        Averaging is accomplished by correlating the window with the
        extended array of patterns using
        :func:`scipy.ndimage.correlate`.

        Parameters
        ----------
        window
            Name of averaging window or an array. Available types are
            listed in :func:`scipy.signal.windows.get_window`, in
            addition to a ``"circular"`` window (default) filled with
            ones in which corner coefficients are set to zero. A window
            element is considered to be in a corner if its radial
            distance to the origin (window center) is shorter or equal
            to the half width of the window's longest axis. A 1D or 2D
            :class:`~numpy.ndarray`, :class:`~dask.array.Array` or
            :class:`~kikuchipy.filters.Window` can also be passed.
        window_shape
            Shape of averaging window. Not used if a custom window or
            :class:`~kikuchipy.filters.Window` is passed to ``window``.
            This can be either 1D or 2D, and can be asymmetrical.
            Default is ``(3, 3)``.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.
        **kwargs
            Keyword arguments passed to the available window type listed
            in :func:`~scipy.signal.windows.get_window`. If not given,
            the default values of that particular window are used.

        See Also
        --------
        kikuchipy.filters.Window, scipy.signal.windows.get_window,
        scipy.ndimage.correlate
        """
        if isinstance(window, Window) and window.is_valid:
            averaging_window = copy.copy(window)
        else:
            averaging_window = Window(window=window, shape=window_shape, **kwargs)

        nav_shape = self.axes_manager.navigation_shape[::-1]
        window_shape = averaging_window.shape
        if window_shape in [(1,), (1, 1)]:
            # Do nothing if a window of shape (1,) or (1, 1) is passed
            return warnings.warn(
                f"A window of shape {window_shape} was passed, no averaging is "
                "therefore performed."
            )
        elif len(nav_shape) > len(window_shape):
            averaging_window = averaging_window.reshape(window_shape + (1,))

        # Get sum of window data for each pattern, to normalize with
        # after correlation
        window_sums = correlate(
            input=np.ones(nav_shape, dtype=int),
            weights=averaging_window,
            mode="constant",
        )

        # Add signal dimensions to window array to enable its use with
        # Dask's map_overlap()
        sig_dim = self.axes_manager.signal_dimension
        averaging_window = averaging_window.reshape(
            averaging_window.shape + (1,) * sig_dim
        )

        # Create dask array of signal patterns and do processing on this
        if self._lazy:
            old_chunks = self.data.chunks
        dask_array = get_dask_array(signal=self, chunk_bytes=8e6, rechunk=True)

        # Add signal dimensions to array be able to use with Dask's
        # map_overlap()
        nav_dim = self.axes_manager.navigation_dimension
        for i in range(sig_dim):
            window_sums = np.expand_dims(window_sums, axis=window_sums.ndim)
        window_sums = da.from_array(
            window_sums, chunks=dask_array.chunks[:nav_dim] + (1,) * sig_dim
        )

        # Create overlap between chunks to enable correlation with the
        # window using Dask's map_overlap()
        window_dim = averaging_window.ndim
        overlap_depth = {}
        for i in range(nav_dim):
            if i < window_dim and dask_array.chunks[i][0] < dask_array.shape[i]:
                overlap_depth[i] = (window_shape[i] // 2) + 1
            else:
                overlap_depth[i] = 1
        overlap_depth.update(
            {i: 0 for i in self.axes_manager.signal_indices_in_array[::-1]}
        )

        dtype_out = self.data.dtype
        omin, omax = dtype_range[dtype_out.type]
        averaged_patterns = da.overlap.map_overlap(
            _average_neighbour_patterns,
            dask_array,
            window_sums,
            window=averaging_window,
            dtype_out=dtype_out,
            omin=omin,
            omax=omax,
            dtype=dtype_out,
            depth=overlap_depth,
            boundary="none",
        )

        # Overwrite signal patterns
        if not self._lazy:
            pbar = ProgressBar()
            if show_progressbar or (
                show_progressbar is None and hs.preferences.General.show_progressbar
            ):
                pbar.register()

            averaged_patterns.store(self.data, compute=True)

            try:
                pbar.unregister()
            except KeyError:
                pass
        else:
            # Revert original chunks
            averaged_patterns = averaged_patterns.rechunk(old_chunks)

            self.data = averaged_patterns

        # Don't sink
        gc.collect()

    def plot_virtual_bse_intensity(
        self,
        roi: BaseInteractiveROI,
        out_signal_axes: Union[Iterable[int], Iterable[str], None] = None,
        **kwargs,
    ) -> None:
        """Plot an interactive virtual backscatter electron (VBSE)
        image formed from intensities within a specified and adjustable
        region of interest (ROI) on the detector.

        Adapted from
        :meth:`pyxem.signals.common_diffraction.CommonDiffraction.plot_integrated_intensity`.

        Parameters
        ----------
        roi
            Any interactive ROI detailed in HyperSpy.
        out_signal_axes
            Which navigation axes to use as signal axes in the virtual
            image. If not given, the first two navigation axes are used.
        **kwargs:
            Keyword arguments passed to the ``plot()`` method of the
            virtual image.

        See Also
        --------
        get_virtual_bse_intensity

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> rect_roi = hs.roi.RectangularROI(
        ...     left=0, right=5, top=0, bottom=5
        ... )
        >>> s.plot_virtual_bse_intensity(rect_roi)
        """
        # Plot signal if necessary
        if self._plot is None or not self._plot.is_active:
            self.plot()

        # Get the sliced signal from the ROI
        sliced_signal = roi.interactive(self, axes=self.axes_manager.signal_axes)

        # Create an output signal for the virtual backscatter electron
        # calculation
        out = self._get_sum_signal(self, out_signal_axes)
        out.metadata.General.title = "Virtual backscatter electron intensity"

        # Create the interactive signal
        hs.interactive(
            f=sliced_signal.nansum,
            axis=sliced_signal.axes_manager.signal_axes,
            event=roi.events.changed,
            recompute_out_event=None,
            out=out,
        )

        # Plot the result
        out.plot(**kwargs)

    def get_virtual_bse_intensity(
        self,
        roi: BaseInteractiveROI,
        out_signal_axes: Union[Iterable[int], Iterable[str], None] = None,
    ) -> VirtualBSEImage:
        """Get a virtual backscatter electron (VBSE) image formed from
        intensities within a region of interest (ROI) on the detector.

        Adapted from
        :meth:`pyxem.signals.common_diffraction.CommonDiffraction.get_integrated_intensity`.

        Parameters
        ----------
        roi
            Any interactive ROI detailed in HyperSpy.
        out_signal_axes
            Which navigation axes to use as signal axes in the virtual
            image. If not given, the first two navigation axes are used.

        Returns
        -------
        virtual_image
            VBSE image formed from detector intensities within an ROI
            on the detector.

        See Also
        --------
        plot_virtual_bse_intensity

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> import kikuchipy as kp
        >>> rect_roi = hs.roi.RectangularROI(
        ...     left=0, right=5, top=0, bottom=5
        ... )
        >>> s = kp.data.nickel_ebsd_small()
        >>> vbse_image = s.get_virtual_bse_intensity(rect_roi)
        """
        vbse = roi(self, axes=self.axes_manager.signal_axes)
        vbse_sum = self._get_sum_signal(vbse, out_signal_axes)
        vbse_sum.metadata.General.title = "Virtual backscatter electron image"
        vbse_sum.set_signal_type("VirtualBSEImage")
        return vbse_sum

    # ------ Methods overwritten from hyperspy.signals.Signal2D ------ #

    def save(
        self,
        filename: Optional[str] = None,
        overwrite: Optional[bool] = None,
        extension: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Write the signal to file in the specified format.

        The function gets the format from the extension: ``h5``,
        ``hdf5`` or ``h5ebsd`` for kikuchipy's specification of the
        h5ebsd format, ``dat`` for the NORDIF binary format or ``hspy``
        for HyperSpy's HDF5 specification. If no extension is provided
        the signal is written to a file in kikuchipy's h5ebsd format.
        Each format accepts a different set of parameters.

        This method is a modified version of HyperSpy's function
        :meth:`hyperspy.signal.BaseSignal.save`.

        Parameters
        ----------
        filename
            If not given and ``tmp_parameters.filename`` and
            ``tmp_parameters.folder`` in signal metadata are defined,
            the filename and path will be taken from there. A valid
            extension can be provided e.g. ``"data.h5"``, see
            ``extension``.
        overwrite
            If not given and the file exists, it will query the user. If
            ``True`` (``False``) it (does not) overwrite the file if it
            exists.
        extension
            Extension of the file that defines the file format. Options
            are ``"h5"``, ``"hdf5"``, ``"h5ebsd"``, ``"dat"``,
            ``"hspy"``. ``"h5"``, ``"hdf5"``, and ``"h5ebsd"`` are
            equivalent. If not given, the extension is determined from
            the following list in this order: i) the filename, ii)
            ``tmp_parameters.extension`` or iii) ``"h5"`` (kikuchipy's
            h5ebsd format).
        **kwargs
            Keyword arguments passed to the writer.

        See Also
        --------
        kikuchipy.io.plugins
        """
        if filename is None:
            tmp_params = self.tmp_parameters
            if tmp_params.has_item("filename") and tmp_params.has_item("folder"):
                filename = os.path.join(tmp_params.folder, tmp_params.filename)
                extension = tmp_params.extension if not extension else extension
            elif self.metadata.has_item("General.original_filename"):
                filename = self.metadata.General.original_filename
            else:
                raise ValueError("Filename not defined.")
        if extension is not None:
            basename, _ = os.path.splitext(filename)
            filename = basename + "." + extension
        _save(filename, self, overwrite=overwrite, **kwargs)

    def get_decomposition_model(
        self,
        components: Union[int, List[int], None] = None,
        dtype_out: Union[str, np.dtype, type] = "float32",
    ) -> Union[EBSD, LazyEBSD]:
        """Get the model signal generated with the selected number of
        principal components from a decomposition.

        Calls HyperSpy's
        :meth:`~hyperspy.learn.mva.MVA.get_decomposition_model`.
        Learning results are preconditioned before this call, doing the
        following:

        1. Set :class:`numpy.dtype` to desired ``dtype_out``.
        2. Remove unwanted components.
        3. Rechunk to suitable chunks if :class:`~dask.array.Array`.

        Parameters
        ----------
        components
            If not given, rebuilds the signal from all components.
            If ``int``, rebuilds signal from ``components`` in range
            0-given ``int``. If list of ``ints``, rebuilds signal from
            only ``components`` in given list.
        dtype_out
            Data type to cast learning results to (default is
            ``"float32``). Note that HyperSpy casts to ``"float64"``.

        Returns
        -------
        s_model
            Model signal.
        """
        # Keep original results to revert back after updating
        factors_orig = self.learning_results.factors.copy()
        loadings_orig = self.learning_results.loadings.copy()

        # Change data type, keep desired components and rechunk if lazy
        dtype_out = np.dtype(dtype_out)
        (
            self.learning_results.factors,
            self.learning_results.loadings,
        ) = _update_learning_results(
            learning_results=self.learning_results,
            dtype_out=dtype_out,
            components=components,
        )

        # Call HyperSpy's function
        s_model = super().get_decomposition_model()

        # Revert learning results to original results
        self.learning_results.factors = factors_orig
        self.learning_results.loadings = loadings_orig

        # Remove learning results from model signal
        s_model.learning_results = LearningResults()

        return s_model

    # ------------------------ Private methods ----------------------- #

    def _check_refinement_parameters(
        self,
        xmap: CrystalMap,
        detector: EBSDDetector,
        signal_mask: Optional[np.ndarray] = None,
    ):
        """Raise ValueError if EBSD refinement input is invalid."""
        _crystal_map_is_compatible_with_signal(
            xmap=xmap,
            navigation_axes=self.axes_manager.navigation_axes[::-1],
            raise_if_not=True,
        )
        sig_shape = self.axes_manager.signal_shape[::-1]
        _detector_is_compatible_with_signal(
            detector=detector,
            navigation_shape=self.axes_manager.navigation_shape[::-1],
            signal_shape=sig_shape,
            raise_if_not=True,
        )
        if len(xmap.phases.ids) != 1:
            raise ValueError("Crystal map must have exactly one phase")
        if signal_mask is not None and sig_shape != signal_mask.shape:
            raise ValueError("Signal mask and signal axes must have the same shape")

    def _prepare_patterns_for_refinement(
        self,
        signal_mask: Union[np.ndarray, None],
        rechunk: bool,
        chunk_kwargs: Optional[dict] = None,
    ) -> Tuple[da.Array, Union[np.ndarray, bool]]:
        """Prepare pattern array and mask for refinement.

        Parameters
        ----------
        signal_mask
            A boolean mask equal to the experimental patterns' detector
            shape ``(n rows, n columns)``, where only pixels equal to
            ``False`` are matched. If not given, all pixels are used.
        rechunk
            Whether to allow rechunking of Dask array produced from the
            signal patterns if the array has only one chunk.
        chunk_kwargs
            Keyword arguments passed to
            :func:`kikuchipy.signals.util._dask.get_chunking`.

        Returns
        -------
        patterns
            Dask array.
        mask
            If ``mask`` is not ``None``, a boolean mask is returned,
            else ``False`` is returned.
        """
        patterns = get_dask_array(signal=self)

        # Flatten signal dimensions
        patterns = patterns.reshape(patterns.shape[:-2] + (-1,))

        # Prepare mask
        if signal_mask is None:
            signal_mask = np.ones(self.axes_manager.signal_size, dtype=bool)
        else:
            signal_mask = ~signal_mask.ravel()

        if (patterns.chunksize == patterns.shape) and rechunk:
            if chunk_kwargs is None:
                chunk_kwargs = dict(chunk_shape=16, chunk_bytes=None)
            chunks = get_chunking(
                data_shape=patterns.shape,
                nav_dim=self.axes_manager.navigation_dimension,
                sig_dim=1,
                dtype=self.data.dtype,
                **chunk_kwargs,
            )
            patterns = patterns.rechunk(chunks)

        return patterns, signal_mask

    @staticmethod
    def _get_sum_signal(signal, out_signal_axes: Optional[List] = None):
        out = signal.nansum(signal.axes_manager.signal_axes)
        if out_signal_axes is None:
            out_signal_axes = list(
                np.arange(min(signal.axes_manager.navigation_dimension, 2))
            )
        if len(out_signal_axes) > signal.axes_manager.navigation_dimension:
            raise ValueError(
                "The length of 'out_signal_axes' cannot be longer than the navigation "
                "dimension of the signal."
            )
        out.set_signal_type("")
        return out.transpose(out_signal_axes)

    def _prepare_metric(
        self,
        metric: Union[SimilarityMetric, str],
        signal_mask: Union[np.ndarray, None],
        dtype: Union[str, np.dtype, type, None],
        rechunk: bool,
        n_dictionary_patterns: int,
    ) -> SimilarityMetric:
        metrics = {
            "ncc": NormalizedCrossCorrelationMetric,
            "ndp": NormalizedDotProductMetric,
        }
        if isinstance(metric, str) and metric in metrics:
            metric_class = metrics[metric]
            metric = metric_class()
            metric.rechunk = rechunk
        if not isinstance(metric, SimilarityMetric):
            raise ValueError(
                f"'{metric}' must be either of {metrics.keys()} or a custom metric "
                "class inheriting from SimilarityMetric. See "
                "kikuchipy.indexing.SimilarityMetric"
            )
        metric.n_experimental_patterns = max(self.axes_manager.navigation_size, 1)
        metric.n_dictionary_patterns = max(n_dictionary_patterns, 1)
        if signal_mask is not None:
            metric.signal_mask = ~signal_mask
        if dtype is not None:
            metric.dtype = dtype
        metric.raise_error_if_invalid()
        return metric

    # --- Inherited methods from KikuchipySignal2D overwritten here for
    # documentation purposes

    def rescale_intensity(
        self,
        relative: bool = False,
        in_range: Union[Tuple[int, int], Tuple[float, float], None] = None,
        out_range: Union[Tuple[int, int], Tuple[float, float], None] = None,
        dtype_out: Union[
            str, np.dtype, type, Tuple[int, int], Tuple[float, float], None
        ] = None,
        percentiles: Union[Tuple[int, int], Tuple[float, float], None] = None,
        show_progressbar: Optional[bool] = None,
    ) -> None:
        super().rescale_intensity(
            relative,
            in_range,
            out_range,
            dtype_out,
            percentiles,
            show_progressbar,
        )

    def normalize_intensity(
        self,
        num_std: int = 1,
        divide_by_square_root: bool = False,
        dtype_out: Union[str, np.dtype, type, None] = None,
        show_progressbar: Optional[bool] = None,
    ) -> None:
        super().normalize_intensity(
            num_std, divide_by_square_root, dtype_out, show_progressbar
        )

    def as_lazy(self, *args, **kwargs) -> LazyEBSD:
        return super().as_lazy(*args, **kwargs)

    def change_dtype(self, *args, **kwargs) -> None:
        super().change_dtype(*args, **kwargs)
        if isinstance(self, EBSD) and isinstance(
            self._static_background, (np.ndarray, da.Array)
        ):
            self._static_background = self._static_background.astype(self.data.dtype)

    def deepcopy(self) -> EBSD:
        return super().deepcopy()


class LazyEBSD(LazyKikuchipySignal2D, EBSD):
    """Lazy implementation of :class:`~kikuchipy.signals.EBSD`.

    See the documentation of ``EBSD`` for attributes and methods.

    This class extends HyperSpy's
    :class:`~hyperspy._signals.signal2d.LazySignal2D` class for EBSD
    patterns. See the documentation of that class for how to create
    this signal and the list of inherited attributes and methods.
    """

    def compute(self, *args, **kwargs) -> None:
        super().compute(*args, **kwargs)

    def get_decomposition_model_write(
        self,
        components: Union[int, List[int], None] = None,
        dtype_learn: Union[str, np.dtype, type] = "float32",
        mbytes_chunk: int = 100,
        dir_out: Optional[str] = None,
        fname_out: Optional[str] = None,
    ) -> None:
        """Write the model signal generated from the selected number of
        principal components directly to an ``.hspy`` file.

        The model signal intensities are rescaled to the original
        signals' data type range, keeping relative intensities.

        Parameters
        ----------
        components
            If not given, rebuilds the signal from all ``components``.
            If ``int``, rebuilds signal from ``components`` in range
            0-given ``int``. If list of ``int``, rebuilds signal from
            only ``components`` in given list.
        dtype_learn
            Data type to set learning results to (default is
            ``"float32"``) before multiplication.
        mbytes_chunk
            Size of learning results chunks in MB, default is 100 MB as
            suggested in the Dask documentation.
        dir_out
            Directory to place output signal in.
        fname_out
            Name of output signal file.

        Notes
        -----
        Multiplying the learning results' factors and loadings in memory
        to create the model signal cannot sometimes be done due to too
        large matrices. Here, instead, learning results are written to
        file, read into dask arrays and multiplied using
        :func:`dask.array.matmul`, out of core.
        """
        dtype_learn = np.dtype(dtype_learn)

        # Change data type, keep desired components and rechunk if lazy
        factors, loadings = _update_learning_results(
            self.learning_results, components=components, dtype_out=dtype_learn
        )

        # Write learning results to HDF5 file
        if dir_out is None:
            try:
                dir_out = self.original_metadata.General.original_filepath
            except AttributeError:
                raise AttributeError("Output directory has to be specified")

        t_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        file_learn = os.path.join(dir_out, "learn_" + t_str + ".h5")
        with File(file_learn, mode="w") as f:
            f.create_dataset(name="factors", data=factors)
            f.create_dataset(name="loadings", data=loadings)

        # Matrix multiplication
        with File(file_learn) as f:
            # Read learning results from HDF5 file
            chunks = _rechunk_learning_results(
                factors=factors, loadings=loadings, mbytes_chunk=mbytes_chunk
            )
            factors = da.from_array(f["factors"], chunks=chunks[0])
            loadings = da.from_array(f["loadings"], chunks=chunks[1])

            # Perform matrix multiplication
            loadings = loadings.T
            res = factors @ loadings
            res = res.T  # Transpose

            # Create new signal from result of matrix multiplication
            s_model = self.deepcopy()
            s_model.learning_results = LearningResults()
            s_model.data = res.reshape(s_model.data.shape)
            s_model.data = s_model.data.rechunk(chunks=(1, 1, -1, -1))

            # Rescale intensities
            s_model.rescale_intensity(dtype_out=self.data.dtype, relative=True)

            # Write signal to file
            if fname_out is None:
                fname_out = "model_" + t_str
            file_model = os.path.join(dir_out, fname_out)
            s_model.save(file_model)

        # Delete temporary files
        os.remove(file_learn)
        gc.collect()  # Don't sink
