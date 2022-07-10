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

import os
from pathlib import Path
from typing import Optional

import hyperspy.api as hs
import pooch

from kikuchipy.signals import EBSD, EBSDMasterPattern
from kikuchipy import load
from kikuchipy.release import version
from kikuchipy.data._registry import registry_hashes, registry_urls


_fetcher = pooch.create(
    path=pooch.os_cache("kikuchipy"),
    base_url="",
    version=version.replace(".dev", "+"),
    version_dev="develop",
    env="KIKUCHIPY_DATA_DIR",
    registry=registry_hashes,
    urls=registry_urls,
)


def _fetch(filename: str, allow_download: bool = False, show_progressbar=None) -> Path:
    fname = "data/" + filename
    expected_hash = registry_hashes[fname]
    file_in_package = Path(os.path.dirname(__file__)) / ".." / fname
    if file_in_package.exists() and pooch.file_hash(file_in_package) == expected_hash:
        # Bypass pooch
        file_path = file_in_package
    else:
        file_in_cache = Path(_fetcher.path) / fname
        if file_in_cache.exists():
            allow_download = True
        if allow_download:
            if show_progressbar is None:
                show_progressbar = hs.preferences.General.show_progressbar
            downloader = pooch.HTTPDownloader(progressbar=show_progressbar)
            file_path = _fetcher.fetch(fname, downloader=downloader)
        else:
            raise ValueError(
                f"Dataset {filename} must be (re)downloaded from the kikuchipy-data "
                "repository on GitHub (https://github.com/pyxem/kikuchipy-data) to your"
                " local cache with the pooch Python package. Pass `allow_download=True`"
                " to allow this download."
            )
    return file_path


def nickel_ebsd_small(**kwargs) -> EBSD:
    """9 EBSD patterns in a (3, 3) navigation shape of (60, 60) detector
    pixels from Nickel, acquired on a NORDIF UF-1100 detector
    :cite:`aanes2019electron`.

    Carries a CC BY 4.0 license.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to :func:`~kikuchipy.load`.

    Returns
    -------
    ebsd_signal
        EBSD signal.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.nickel_ebsd_small()
    >>> s
    <EBSD, title: patterns My awes0m4 ..., dimensions: (3, 3|60, 60)>
    >>> s.plot()
    """
    fname = _fetch("kikuchipy_h5ebsd/patterns.h5")
    return load(fname, **kwargs)


def nickel_ebsd_master_pattern_small(**kwargs) -> EBSDMasterPattern:
    """(401, 401) ``uint8`` square Lambert or stereographic projection
    of the northern and southern hemisphere of a Nickel master pattern
    at 20 keV accelerating voltage.

    Carries a CC BY 4.0 license.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to :func:`~kikuchipy.load`.

    Returns
    -------
    ebsd_master_pattern_signal
        EBSD master pattern signal.

    Notes
    -----
    Initially generated using the EMsoft EMMCOpenCL and EMEBSDMaster
    programs. The included file was rewritten to disk with
    :mod:`h5py`, where the master patterns' data type is converted from
    ``float32`` to ``uint8`` with
    :meth:`~kikuchipy.signals.EBSDMasterPattern.rescale_intensity`, all
    datasets were written with
    :meth:`~kikuchipy.io.plugins.h5ebsd.dict2h5ebsdgroup` with
    keyword arguments ``compression="gzip"`` and ``compression_opts=9``.
    All other HDF5 groups and datasets are the same as in the original
    file.

    Examples
    --------
    Import master pattern in the stereographic projection

    >>> import kikuchipy as kp
    >>> s = kp.data.nickel_ebsd_master_pattern_small()
    >>> s
    <EBSDMasterPattern, title: ni_mc_mp_20kv_uint8_gzip_opts9, dimensions: (|401, 401)>
    >>> s.projection
    'stereographic'

    Import master pattern in the square Lambert projection and plot it

    >>> s2 = kp.data.nickel_ebsd_master_pattern_small(projection="lambert")
    >>> s2.projection
    'lambert'
    >>> s2.plot()
    """
    fname = _fetch("emsoft_ebsd_master_pattern/ni_mc_mp_20kv_uint8_gzip_opts9.h5")
    return load(fname, **kwargs)


def nickel_ebsd_large(
    allow_download: bool = False, show_progressbar: Optional[bool] = None, **kwargs
) -> EBSD:
    """4125 EBSD patterns in a (55, 75) navigation shape of (60, 60)
    detector pixels from Nickel, acquired on a NORDIF UF-1100 detector
    :cite:`aanes2019electron`.

    Carries a CC BY 4.0 license.

    Parameters
    ----------
    allow_download
        Whether to allow downloading the dataset from the kikuchipy-data
        GitHub repository (https://github.com/pyxem/kikuchipy-data) to
        the local cache with the pooch Python package. Default is
        ``False``.
    show_progressbar
        Whether to show a progressbar when downloading. If not given,
        the value of
        :obj:`hyperspy.api.preferences.General.show_progressbar` is
        used.
    **kwargs
        Keyword arguments passed to :func:`~kikuchipy.load`.

    Returns
    -------
    ebsd_signal
        EBSD signal.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.nickel_ebsd_large(allow_download=True)
    >>> s
    <EBSD, title: patterns Scan 1, dimensions: (75, 55|60, 60)>
    >>> s.plot()
    """
    fname = _fetch("nickel_ebsd_large/patterns.h5", allow_download, show_progressbar)
    return load(fname, **kwargs)


def silicon_ebsd_moving_screen_in(
    allow_download: bool = False, show_progressbar: Optional[bool] = None, **kwargs
) -> EBSD:
    """One EBSD pattern of (480, 480) detector pixels from a single
    crystal Silicon sample, acquired on a NORDIF UF-420 detector.

    This pattern and two other patterns from the same sample position
    but with 5 mm and 10 mm greater sample-screen-distances were
    acquired to test the moving-screen projection center estimation
    technique :cite:`hjelen1991electron`.

    Carries a CC BY 4.0 license.

    Parameters
    ----------
    allow_download
        Whether to allow downloading the dataset from the kikuchipy-data
        GitHub repository (https://github.com/pyxem/kikuchipy-data) to
        the local cache with the pooch Python package. Default is
        ``False``.
    show_progressbar
        Whether to show a progressbar when downloading. If not given,
        the value of
        :obj:`hyperspy.api.preferences.General.show_progressbar` is
        used.
    **kwargs
        Keyword arguments passed to :func:`~kikuchipy.load`.

    Returns
    -------
    ebsd_signal
        EBSD signal.

    See Also
    --------
    silicon_ebsd_moving_screen_out5mm, silicon_ebsd_moving_screen_out10mm

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.silicon_ebsd_moving_screen_in(allow_download=True)
    >>> s
    <EBSD, title: si_in Scan 1, dimensions: (|480, 480)>
    >>> s.plot()
    """
    fname = _fetch(
        "silicon_ebsd_moving_screen/si_in.h5", allow_download, show_progressbar
    )
    return load(fname, **kwargs)


def silicon_ebsd_moving_screen_out5mm(
    allow_download: bool = False, show_progressbar: Optional[bool] = None, **kwargs
) -> EBSD:
    """One EBSD pattern of (480, 480) detector pixels from a single
    crystal Silicon sample, acquired on a NORDIF UF-420 detector.

    This pattern and two other patterns from the same sample position
    but with sample-screen-distances 5 mm shorter
    (:func:`silicon_ebsd_moving_screen_in`) and 5 mm greater
    (:func:`silicon_ebsd_moving_screen_out10mm`) were acquired to test
    the moving-screen projection center estimation technique
    :cite:`hjelen1991electron`.

    Carries a CC BY 4.0 license.

    Parameters
    ----------
    allow_download
        Whether to allow downloading the dataset from the kikuchipy-data
        GitHub repository (https://github.com/pyxem/kikuchipy-data) to
        the local cache with the pooch Python package. Default is
        ``False``.
    show_progressbar
        Whether to show a progressbar when downloading. If not given,
        the value of
        :obj:`hyperspy.api.preferences.General.show_progressbar` is
        used.
    **kwargs
        Keyword arguments passed to :func:`~kikuchipy.load`.

    Returns
    -------
    ebsd_signal
        EBSD signal.

    See Also
    --------
    silicon_ebsd_moving_screen_in, silicon_ebsd_moving_screen_out10mm

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.silicon_ebsd_moving_screen_out5mm(allow_download=True)
    >>> s
    <EBSD, title: si_out5mm Scan 1, dimensions: (|480, 480)>
    >>> s.plot()
    """
    fname = _fetch(
        "silicon_ebsd_moving_screen/si_out5mm.h5", allow_download, show_progressbar
    )
    return load(fname, **kwargs)


def silicon_ebsd_moving_screen_out10mm(
    allow_download: bool = False, show_progressbar: Optional[bool] = None, **kwargs
) -> EBSD:
    """One EBSD pattern of (480, 480) detector pixels from a single
    crystal Silicon sample, acquired on a NORDIF UF-420 detector.

    This pattern and two other patterns from the same sample position
    but with sample-screen-distances 10 mm shorter
    (:func:`silicon_ebsd_moving_screen_in`) and 5 mm shorter
    (:func:`silicon_ebsd_moving_screen_out5mm`) were acquired to test
    the moving-screen projection center estimation technique
    :cite:`hjelen1991electron`.

    Carries a CC BY 4.0 license.

    Parameters
    ----------
    allow_download
        Whether to allow downloading the dataset from the kikuchipy-data
        GitHub repository (https://github.com/pyxem/kikuchipy-data) to
        the local cache with the pooch Python package. Default is
        ``False``.
    show_progressbar
        Whether to show a progressbar when downloading. If not given,
        the value of
        :obj:`hyperspy.api.preferences.General.show_progressbar` is
        used.
    **kwargs
        Keyword arguments passed to :func:`~kikuchipy.load`.

    Returns
    -------
    ebsd_signal
        EBSD signal.

    See Also
    --------
    silicon_ebsd_moving_screen_in, silicon_ebsd_moving_screen_out5mm

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.silicon_ebsd_moving_screen_out10mm(allow_download=True)
    >>> s
    <EBSD, title: si_out10mm Scan 1, dimensions: (|480, 480)>
    >>> s.plot()
    """
    fname = _fetch(
        "silicon_ebsd_moving_screen/si_out10mm.h5", allow_download, show_progressbar
    )
    return load(fname, **kwargs)
