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
import os
from typing import Any, Union, Tuple, Optional

import dask.array as da
from dask.diagnostics import ProgressBar
import hyperspy.api as hs
from hyperspy.signals import Signal2D
from hyperspy._lazy_signals import LazySignal2D
from hyperspy.misc.rgb_tools import rgb_dtypes
import numpy as np
from skimage.util.dtype import dtype_range
import yaml

from kikuchipy.pattern import chunk
from kikuchipy.signals.util._dask import get_dask_array
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
        """Rescale image intensities inplace.

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
            min./max. intensity.
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

        if dtype_out is None:
            dtype_out = self.data.dtype
        else:
            dtype_out = np.dtype(dtype_out)

        if out_range is None:
            out_range = dtype_range[dtype_out.type]

        # Create dask array of signal images and do processing on this
        dask_array = get_dask_array(signal=self)

        # Rescale images
        rescaled_images = dask_array.map_blocks(
            func=chunk.rescale_intensity,
            in_range=in_range,
            out_range=out_range,
            dtype_out=dtype_out,
            percentiles=percentiles,
            dtype=dtype_out,
        )

        # Overwrite signal images
        if not self._lazy:
            pbar = ProgressBar()
            if show_progressbar or (
                show_progressbar is None and hs.preferences.General.show_progressbar
            ):
                pbar.register()

            if self.data.dtype != rescaled_images.dtype:
                self.change_dtype(dtype_out)
            rescaled_images.store(self.data, compute=True)

            try:
                pbar.unregister()
            except KeyError:
                pass
        else:
            self.data = rescaled_images

    def normalize_intensity(
        self,
        num_std: int = 1,
        divide_by_square_root: bool = False,
        dtype_out: Union[str, np.dtype, type, None] = None,
        show_progressbar: Optional[bool] = None,
    ) -> None:
        """Normalize image intensities in inplace to a mean of zero with
        a given standard deviation.

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
        if self.data.dtype in rgb_dtypes.values():
            raise NotImplementedError(
                "Use RGB channel normalization when creating the image instead."
            )

        if dtype_out is None:
            dtype_out = self.data.dtype
        else:
            dtype_out = np.dtype(dtype_out)

        dask_array = get_dask_array(self, dtype=np.float32)

        normalized_images = dask_array.map_blocks(
            func=chunk.normalize_intensity,
            num_std=num_std,
            divide_by_square_root=divide_by_square_root,
            dtype_out=dtype_out,
            dtype=dtype_out,
        )

        # Change data type if requested
        if dtype_out != self.data.dtype:
            self.change_dtype(dtype_out)

        # Overwrite signal patterns
        if not self._lazy:
            pbar = ProgressBar()
            if show_progressbar or (
                show_progressbar is None and hs.preferences.General.show_progressbar
            ):
                pbar.register()

            normalized_images.store(self.data, compute=True)

            try:
                pbar.unregister()
            except KeyError:
                pass
        else:
            self.data = normalized_images

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
        gc.collect()
        self._set_custom_attributes(attrs, unmake_lazy=True)
