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

from copy import deepcopy
import gc
import logging
import numbers
import os
from typing import Any, List, Optional, Union, Tuple
import warnings

import dask.array as da
from hyperspy.signals import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from hyperspy.misc.rgb_tools import rgb_dtypes
import numpy as np
from skimage.util.dtype import dtype_range
import yaml

from kikuchipy.pattern import normalize_intensity, rescale_intensity
from kikuchipy.pattern._pattern import _adaptive_histogram_equalization
from kikuchipy.signals.util._overwrite_hyperspy_methods import insert_doc_disclaimer


_logger = logging.getLogger(__name__)


# Signal type aliases defined in hyperspy_extension.yaml
DIR_PATH = os.path.dirname(__file__)
SIGNAL_TYPES = []
with open(os.path.join(DIR_PATH, "../hyperspy_extension.yaml")) as f:
    content = yaml.safe_load(f.read())
    for _, info in content["signals"].items():
        for k, v in info.items():
            if k == "signal_type":
                SIGNAL_TYPES.append(v)
            elif k == "signal_type_aliases":
                SIGNAL_TYPES.extend(v)


class KikuchipySignal2D(Signal2D):
    """General class for image signals in kikuchipy, extending
    HyperSpy's Signal2D class with some methods for carrying over custom
    properties and some methods for intensity manipulation.

    Not meant to be used directly, see derived classes like
    :class:`~kikuchipy.signals.EBSD`.
    """

    _custom_attributes = []

    @property
    def _signal_shape_rc(self) -> tuple:
        """Return the signal's signal shape as (row, column)."""
        return self.axes_manager.signal_shape[::-1]

    @property
    def _navigation_shape_rc(self) -> tuple:
        """Return the signal's navigation shape as (row, column)."""
        return self.axes_manager.navigation_shape[::-1]

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
        inplace: bool = True,
        lazy_output: Optional[bool] = None,
    ) -> Union[None, Any]:
        """Rescale image intensities.

        Output min./max. intensity is determined from ``out_range`` or
        the data type range of the :class:`numpy.dtype` passed to
        ``dtype_out`` if ``out_range`` is ``None``.

        This method is based on
        :func:`skimage.exposure.rescale_intensity`.

        Parameters
        ----------
        relative
            Whether to keep relative intensities between images (default
            is ``False``). If ``True``, ``in_range`` must be ``None``,
            because ``in_range`` is in this case set to the global
            min./max. intensity. Use with care, as this requires the
            computation of the min./max. intensity of the signal before
            rescaling.
        in_range
            Min./max. intensity of input images. If not given,
            ``in_range`` is set to pattern min./max intensity. Contrast
            stretching is performed when ``in_range`` is set to a
            narrower intensity range than the input patterns. Must be
            ``None`` if ``relative=True`` or ``percentiles`` are passed.
        out_range
            Min./max. intensity of output images. If not given,
            ``out_range`` is set to ``dtype_out`` min./max according to
            ``skimage.util.dtype.dtype_range``.
        dtype_out
            Data type of rescaled images, default is input images' data
            type.
        percentiles
            Disregard intensities outside these percentiles. Calculated
            per image. Must be ``None`` if ``in_range`` or ``relative``
            is passed. Default is ``None``.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.
        inplace
            Whether to operate on the current signal or return a new
            one. Default is ``True``.
        lazy_output
            Whether the returned signal is lazy. If not given this
            follows from the current signal. Can only be ``True`` if
            ``inplace=False``.

        Returns
        -------
        s_out
            Rescaled signal, returned if ``inplace=False``. Whether
            it is lazy is determined from ``lazy_output``.

        See Also
        --------
        :func:`skimage.exposure.rescale_intensity`

        Notes
        -----
        Rescaling RGB images is not possible. Use RGB channel
        normalization when creating the image instead.

        Examples
        --------
        >>> import numpy as np
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()

        Image intensities are stretched to fill the available grey
        levels in the input images' data type range or any data type
        range passed to ``dtype_out``, either keeping relative
        intensities between images or not

        >>> print(
        ...     s.data.dtype, s.data.min(), s.data.max(),
        ...     s.inav[0, 0].data.min(), s.inav[0, 0].data.max()
        ... )
        uint8 23 246 26 245
        >>> s2 = s.deepcopy()
        >>> s.rescale_intensity(dtype_out=np.uint16)
        >>> print(
        ...     s.data.dtype, s.data.min(), s.data.max(),
        ...     s.inav[0, 0].data.min(), s.inav[0, 0].data.max()
        ... )
        uint16 0 65535 0 65535
        >>> s2.rescale_intensity(relative=True)
        >>> print(
        ...     s2.data.dtype, s2.data.min(), s2.data.max(),
        ...     s2.inav[0, 0].data.min(), s2.inav[0, 0].data.max()
        ... )
        uint8 0 255 3 253

        Contrast stretching can be performed by passing percentiles

        >>> s.rescale_intensity(percentiles=(1, 99))

        Here, the darkest and brightest pixels within the 1% percentile
        are set to the ends of the data type range, e.g. 0 and 255
        respectively for images of ``uint8`` data type.
        """
        if lazy_output and inplace:
            raise ValueError("`lazy_output=True` requires `inplace=False`")

        if self.data.dtype in rgb_dtypes.values():
            raise NotImplementedError(
                "Use RGB channel normalization when creating the image instead."
            )

        # Determine min./max. intensity of input image to rescale to
        if in_range is not None and percentiles is not None:
            raise ValueError("'percentiles' must be None if 'in_range' is not None.")
        elif relative is True and in_range is not None:
            raise ValueError("'in_range' must be None if 'relative' is True.")
        elif relative:  # Scale relative to min./max. intensity in images
            in_range = (self.data.min(), self.data.max())
            in_range = tuple(da.compute(in_range)[0])

        if dtype_out is None:
            dtype_out = self.data.dtype
        else:
            dtype_out = np.dtype(dtype_out)

        if out_range is None:
            out_range = dtype_range[dtype_out.type]

        map_kw = dict(
            show_progressbar=show_progressbar,
            parallel=True,
            output_dtype=dtype_out,
            in_range=in_range,
            out_range=out_range,
            dtype_out=dtype_out,
            percentiles=percentiles,
        )
        attrs = self._get_custom_attributes()
        if inplace:
            self.map(rescale_intensity, inplace=True, **map_kw)
            self._set_custom_attributes(attrs)
        else:
            s_out = self.map(
                rescale_intensity, inplace=False, lazy_output=lazy_output, **map_kw
            )
            s_out._set_custom_attributes(attrs)
            return s_out

    def normalize_intensity(
        self,
        num_std: int = 1,
        divide_by_square_root: bool = False,
        dtype_out: Union[str, np.dtype, type, None] = None,
        show_progressbar: Optional[bool] = None,
        inplace: bool = True,
        lazy_output: Optional[bool] = None,
    ) -> Union[None, Any]:
        """Normalize image intensities to a mean of zero with a given
        standard deviation.

        Parameters
        ----------
        num_std
            Number of standard deviations of the output intensities.
            Default is ``1``.
        divide_by_square_root
            Whether to divide output intensities by the square root of
            the signal dimension size. Default is ``False``.
        dtype_out
            Data type of normalized images. If not given, the input
            images' data type is used.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.
        inplace
            Whether to operate on the current signal or return a new
            one. Default is ``True``.
        lazy_output
            Whether the returned signal is lazy. If not given this
            follows from the current signal. Can only be ``True`` if
            ``inplace=False``.

        Returns
        -------
        s_out
            Normalized signal, returned if ``inplace=False``. Whether
            it is lazy is determined from ``lazy_output``.

        Notes
        -----
        Data type should always be changed to floating point, e.g.
        ``float32`` with
        :meth:`~hyperspy.signal.BaseSignal.change_dtype`, before
        normalizing the intensities.

        Rescaling RGB images is not possible. Use RGB channel
        normalization when creating the image instead.

        Examples
        --------
        >>> import numpy as np
        >>> import kikuchipy as kp
        >>> s = kp.data.nickel_ebsd_small()
        >>> np.mean(s.data)
        146.0670987654321
        >>> s.normalize_intensity(dtype_out=np.float32)
        >>> np.mean(s.data)
        2.6373216e-08
        """
        if lazy_output and inplace:
            raise ValueError("`lazy_output=True` requires `inplace=False`")

        if self.data.dtype in rgb_dtypes.values():
            raise NotImplementedError(
                "Use RGB channel normalization when creating the image instead."
            )

        if dtype_out is None:
            dtype_out = self.data.dtype
        else:
            dtype_out = np.dtype(dtype_out)

        map_kw = dict(
            show_progressbar=show_progressbar,
            parallel=True,
            output_dtype=dtype_out,
            num_std=num_std,
            divide_by_square_root=divide_by_square_root,
            dtype_out=dtype_out,
        )
        attrs = self._get_custom_attributes()
        if inplace:
            self.map(normalize_intensity, inplace=True, **map_kw)
            self._set_custom_attributes(attrs)
        else:
            s_out = self.map(
                normalize_intensity, inplace=False, lazy_output=lazy_output, **map_kw
            )
            s_out._set_custom_attributes(attrs)
            return s_out

    def adaptive_histogram_equalization(
        self,
        kernel_size: Optional[Union[Tuple[int, int], List[int]]] = None,
        clip_limit: Union[int, float] = 0,
        nbins: int = 128,
        show_progressbar: Optional[bool] = None,
        inplace: bool = True,
        lazy_output: Optional[bool] = None,
    ) -> Union[None, Any]:
        """Enhance the local contrast using adaptive histogram
        equalization.

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
        inplace
            Whether to operate on the current signal or return a new
            one. Default is ``True``.
        lazy_output
            Whether the returned signal is lazy. If not given this
            follows from the current signal. Can only be ``True`` if
            ``inplace=False``.

        Returns
        -------
        s_out
            Equalized signal, returned if ``inplace=False``. Whether it
            is lazy is determined from ``lazy_output``.

        See Also
        --------
        rescale_intensity,
        normalize_intensity

        Notes
        -----
        It is recommended to perform adaptive histogram equalization
        only *after* static and dynamic background corrections of EBSD
        patterns, otherwise some unwanted darkening towards the edges
        might occur.

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
        if lazy_output and inplace:
            raise ValueError("`lazy_output=True` requires `inplace=False`")

        dtype_out = self.data.dtype
        if np.issubdtype(dtype_out, np.floating):
            warnings.warn(
                (
                    "Equalization of signals with floating point data type has been "
                    "shown to give bad results. Rescaling intensities to integer "
                    "intensities is recommended."
                ),
                UserWarning,
            )
        if not self._lazy and np.isnan(self.data).any():
            warnings.warn(
                (
                    "Equalization of signals with NaN data has been shown to give bad "
                    "results"
                ),
                UserWarning,
            )

        # Determine window size (shape of contextual region)
        sig_shape = self.axes_manager.signal_shape
        if kernel_size is None:
            kernel_size = (sig_shape[0] // 4, sig_shape[1] // 4)
        elif isinstance(kernel_size, numbers.Number):
            kernel_size = (kernel_size,) * self.axes_manager.signal_dimension
        elif len(kernel_size) != self.axes_manager.signal_dimension:
            raise ValueError(f"Incorrect value of `shape`: {kernel_size}")
        kernel_size = [int(k) for k in kernel_size]

        map_kw = dict(
            show_progressbar=show_progressbar,
            parallel=True,
            output_dtype=dtype_out,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=nbins,
        )
        attrs = self._get_custom_attributes()
        if inplace:
            self.map(_adaptive_histogram_equalization, inplace=True, **map_kw)
            self._set_custom_attributes(attrs)
        else:
            s_out = self.map(
                _adaptive_histogram_equalization,
                inplace=False,
                lazy_output=lazy_output,
                **map_kw,
            )
            s_out._set_custom_attributes(attrs)
            return s_out

    def _get_custom_attributes(self, make_deepcopy: bool = False) -> dict:
        """Return a dictionary of attributes not in ``Signal2D``.

        This is a quick way to get all custom attributes of a class
        before calling a method of ``Signal2D`` which returns a new
        instance or operates in place on the current instance and does
        not carry over these attributes, like with ``deepcopy()``.

        Parameters
        ----------
        make_deepcopy
            Whether the returned dictionary should contain deep copies
            of each attribute. Default is ``False``.

        Returns
        -------
        dictionary
            Dictionary with custom attributes.
        """
        dictionary = {}
        for name in self._custom_attributes:
            attr = self.__getattribute__(name)
            if make_deepcopy:
                try:
                    dictionary[name] = deepcopy(attr)
                except ValueError:  # pragma: no cover
                    _logger.debug(f"Could not deepcopy attribute {name}")
                    dictionary[name] = attr
            else:
                dictionary[name] = attr
        return dictionary

    def _set_custom_attributes(
        self,
        attributes: dict,
        make_deepcopy: bool = False,
        make_lazy: bool = False,
        unmake_lazy: bool = False,
    ):
        """Set custom attributes not in ``Signal2D``.

        This is a quick way to set all custom attributes of a class
        after calling a method of ``Signal2D`` which returns a new
        instance or operates in place on the current instance and does
        not carry over these attributes, like with ``deepcopy()``.

        Parameters
        ----------
        attributes
            Dictionary of custom attributes.
        make_deepcopy
            Whether to make a deepcopy of all attributes before setting
            them in this instance. Default is ``False``.
        make_lazy
            Whether to cast attributes which are
            :class:`~numpy.ndarray` to :class:`~dask.array.Array` before
            setting them. Default is ``False``.
        unmake_lazy
            Whether to cast attributes which are
            :class:`~dask.array.Array` to :class:`~numpy.ndarray` before
            setting them. Default is ``False``. Ignored if both this and
            ``make_lazy`` are ``True``.
        """
        for name, value in attributes.items():
            if name in self._custom_attributes:
                try:
                    if make_lazy and isinstance(value, np.ndarray):
                        value = da.from_array(value)
                    elif unmake_lazy and isinstance(value, da.Array):
                        value = value.compute()
                    if make_deepcopy:
                        value = deepcopy(value)
                    self.__setattr__("_" + name, value)
                except ValueError:  # pragma: no cover
                    _logger.debug(f"Could not set attribute {name}")

    # --- Inherited methods from Signal2D overwritten

    @insert_doc_disclaimer(cls=Signal2D, meth=Signal2D.as_lazy)
    def as_lazy(self, *args, **kwargs) -> Any:
        s_new = super().as_lazy(*args, **kwargs)

        if s_new._signal_type in SIGNAL_TYPES:
            attrs = self._get_custom_attributes()
            s_new._set_custom_attributes(attrs, make_lazy=True)

        return s_new

    @insert_doc_disclaimer(cls=Signal2D, meth=Signal2D.change_dtype)
    def change_dtype(self, *args, **kwargs) -> None:
        attrs = self._get_custom_attributes()

        super().change_dtype(*args, **kwargs)

        if self._signal_type in SIGNAL_TYPES:
            self._set_custom_attributes(attrs)
        else:
            for name in attrs.keys():
                try:
                    self.__delattr__("_" + name)
                except AttributeError:
                    pass

    def deepcopy(self) -> Any:
        s_new = super().deepcopy()

        if s_new._signal_type in SIGNAL_TYPES:
            attrs = self._get_custom_attributes()
            s_new._set_custom_attributes(attrs, make_deepcopy=True)

        return s_new

    def _assign_subclass(self):
        attrs = self._custom_attributes

        super()._assign_subclass()

        if self._signal_type not in SIGNAL_TYPES:
            for name in attrs:
                try:
                    self.__delattr__("_" + name)
                except AttributeError:  # pragma: no cover
                    pass


class LazyKikuchipySignal2D(LazySignal2D, KikuchipySignal2D):
    """General class for lazy image signals in kikuchipy, extending
    HyperSpy's LazySignal2D class with some methods for carrying over
    custom properties and some methods for intensity manipulation.

    Not meant to be used directly, see derived classes like
    :class:`~kikuchipy.signals.LazyEBSD`.
    """

    @insert_doc_disclaimer(cls=LazySignal2D, meth=LazySignal2D.compute)
    def compute(self, *args, **kwargs) -> None:
        attrs = self._get_custom_attributes()
        super().compute(*args, **kwargs)
        self._set_custom_attributes(attrs, unmake_lazy=True)
        gc.collect()
