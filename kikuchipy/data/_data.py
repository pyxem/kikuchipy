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

import os
from pathlib import Path
from typing import Optional, Union

import hyperspy.api as hs
import pooch

from kikuchipy.signals import EBSD, EBSDMasterPattern
from kikuchipy import load
from kikuchipy.release import version
from kikuchipy.data._registry import registry_hashes, registry_urls
from kikuchipy._util import deprecated


marshall = pooch.create(
    path=pooch.os_cache("kikuchipy"),
    base_url="",
    version=version.replace(".dev", "+"),
    version_dev="develop",
    env="KIKUCHIPY_DATA_DIR",
    registry=registry_hashes,
    urls=registry_urls,
)


# ----------------------- Experimental datasets ---------------------- #


def nickel_ebsd_small(**kwargs) -> EBSD:
    """Ni EBSD patterns in a (3, 3) navigation shape of (60, 60) pixels
    from nickel, acquired on a NORDIF UF-1100 detector
    :cite:`aanes2019electron`.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to :func:`~kikuchipy.load`.

    Returns
    -------
    ebsd_signal
        EBSD signal.

    Notes
    -----
    The dataset carries a CC BY 4.0 license.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.nickel_ebsd_small()
    >>> s
    <EBSD, title: patterns Scan 1, dimensions: (3, 3|60, 60)>
    >>> s.plot()
    """
    NiEBSDSmall = Dataset("kikuchipy_h5ebsd/patterns.h5")
    file_path = NiEBSDSmall.fetch_file_path()
    return load(file_path, **kwargs)


def nickel_ebsd_large(
    allow_download: bool = False, show_progressbar: Optional[bool] = None, **kwargs
) -> EBSD:
    """4125 EBSD patterns in a (55, 75) navigation shape of (60, 60)
    pixels from nickel, acquired on a NORDIF UF-1100 detector
    :cite:`aanes2019electron`.

    Parameters
    ----------
    allow_download
        Whether to allow downloading the dataset from the internet to
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

    Notes
    -----
    The dataset is hosted in the GitHub repository
    https://github.com/pyxem/kikuchipy-data.

    The dataset carries a CC BY 4.0 license.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.nickel_ebsd_large(allow_download=True)
    >>> s
    <EBSD, title: patterns Scan 1, dimensions: (75, 55|60, 60)>
    >>> s.plot()
    """
    NiEBSDLarge = Dataset("nickel_ebsd_large/patterns.h5")
    file_path = NiEBSDLarge.fetch_file_path(allow_download, show_progressbar)
    return load(file_path, **kwargs)


def ni_gain(
    number: int = 1,
    allow_download: bool = False,
    show_progressbar: Optional[bool] = None,
    **kwargs,
) -> EBSD:
    """EBSD dataset of (149, 200) patterns of (60, 60) pixels from
    polycrystalline recrystallized Ni, acquired on a NORDIF UF-1100
    detector :cite:`aanes2019electron`.

    Ten datasets are available from the same region of interest,
    acquired with increasing gain on the detector, from no gain to
    maximum gain.

    Parameters
    ----------
    number
        Dataset number 1-10. The camera gains in dB are 0, 3,
        6, 9, 12, 15, 17, 20, 22 and 24. Default is dataset number 1,
        acquired without detector gain.
    allow_download
        Whether to allow downloading the dataset from the internet to
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
    ni_gain_calibration, nickel_ebsd_small, nickel_ebsd_large

    Notes
    -----
    The datasets are hosted in the Zenodo repository
    https://doi.org/10.5281/zenodo.7497682 and comprise about 100 MB
    each as zipped files and about 116 MB when unzipped. Each zipped
    file is deleted after it is unzipped.

    The datasets carry a CC BY 4.0 license.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.ni_gain(allow_download=True, lazy=True)  # doctest: +SKIP
    >>> s  # doctest: +SKIP
    <LazyEBSD, title: Pattern, dimensions: (200, 149|60, 60)>
    """
    gain = [0, 3, 6, 9, 12, 15, 17, 20, 22, 24]
    collection_name = f"scan{number}_gain{gain[number - 1]}db.zip"

    NiGain = Dataset(f"ni_gain/{number}/Pattern.dat", collection_name=collection_name)
    file_path = NiGain.fetch_file_path(allow_download, show_progressbar)

    return load(file_path, **kwargs)  # pragma: no cover


def ni_gain_calibration(
    number: int = 1,
    allow_download: bool = False,
    show_progressbar: Optional[bool] = None,
    **kwargs,
) -> EBSD:
    """A few EBSD calibration patterns of (480, 480) pixels from
    polycrystalline recrystallized Ni, acquired on a NORDIF UF-1100
    detector :cite:`aanes2019electron`.

    The patterns are used to calibrate the detector-sample geometry
    of the datasets in :func:`~kikuchipy.data.ni_gain`. The calibration
    patterns were acquired with no gain on the detector.

    Parameters
    ----------
    number
        Dataset number 1-10. Default is dataset number 1, i.e. the
        calibration patterns used to calibrate the detector-sample
        geometry for the dataset acquired without detector gain.
    allow_download
        Whether to allow downloading the dataset from the internet to
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
    ni_gain, nickel_ebsd_small, nickel_ebsd_large

    Notes
    -----
    The datasets are hosted in the Zenodo repository
    https://doi.org/10.5281/zenodo.7497682 and comprise about 100 MB
    each as zipped files and about 116 MB when unzipped. Each zipped
    file is deleted after it is unzipped.

    The datasets carry a CC BY 4.0 license.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.ni_gain_calibration(allow_download=True, lazy=True)  # doctest: +SKIP
    >>> s  # doctest: +SKIP
    <LazyEBSD, title: Calibration patterns, dimensions: (9|480, 480)>
    """
    gain = [0, 3, 6, 9, 12, 15, 17, 20, 22, 24]
    collection_name = f"scan{number}_gain{gain[number - 1]}db.zip"

    NiGainCalibration = Dataset(
        f"ni_gain/{number}/Setting.txt", collection_name=collection_name
    )
    file_path = NiGainCalibration.fetch_file_path(allow_download, show_progressbar)

    return load(file_path, **kwargs)  # pragma: no cover


def si_ebsd_moving_screen(
    distance: int = 0,
    allow_download: bool = False,
    show_progressbar: Optional[bool] = None,
    **kwargs,
) -> EBSD:
    """One EBSD pattern of (480, 480) pixels from a single crystal
    silicon sample, acquired on a NORDIF UF-420 detector
    :cite:`aanes2022electron3`.

    Three patterns are available from the same sample position but with
    different sample-screen distances: normal position (``0``) and 5 mm
    and 10 mm greater than the normal position (``5`` and ``10``).

    They were acquired to test the moving-screen projection center
    estimation technique :cite:`hjelen1991electron`.

    Parameters
    ----------
    distance
        Sample-screen distance away from the normal position, either 0,
        5 or 10.
    allow_download
        Whether to allow downloading the dataset from the internet to
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
    si_wafer

    Notes
    -----
    The datasets are hosted in the GitHub repository
    https://github.com/pyxem/kikuchipy-data.

    The datasets carry a CC BY 4.0 license.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.si_ebsd_moving_screen(allow_download=True)
    >>> s
    <EBSD, title: si_in Scan 1, dimensions: (|480, 480)>
    >>> s.plot()
    """
    datasets = {0: "in", 5: "out5mm", 10: "out10mm"}
    suffix = datasets[int(distance)]

    SiEBSDMovingScreen = Dataset(f"silicon_ebsd_moving_screen/si_{suffix}.h5")
    file_path = SiEBSDMovingScreen.fetch_file_path(allow_download, show_progressbar)

    return load(file_path, **kwargs)


@deprecated("0.8", "si_ebsd_moving_screen", removal="0.9")
def silicon_ebsd_moving_screen_in(
    allow_download: bool = False, show_progressbar: Optional[bool] = None, **kwargs
) -> EBSD:
    """One EBSD pattern of (480, 480) pixels from a single crystal
    silicon sample, acquired on a NORDIF UF-420 detector
    :cite:`aanes2022electron3`.

    This pattern and two other patterns from the same sample position
    but with 5 mm and 10 mm greater sample-screen-distances were
    acquired to test the moving-screen projection center estimation
    technique :cite:`hjelen1991electron`.

    Parameters
    ----------
    allow_download
        Whether to allow downloading the dataset from the internet to
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

    Notes
    -----
    The dataset is hosted in the GitHub repository
    https://github.com/pyxem/kikuchipy-data.

    The dataset carries a CC BY 4.0 license.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.si_ebsd_moving_screen(0, allow_download=True)
    >>> s
    <EBSD, title: si_in Scan 1, dimensions: (|480, 480)>
    >>> s.plot()
    """
    SiEBSDMovingScreenIn = Dataset("silicon_ebsd_moving_screen/si_in.h5")
    file_path = SiEBSDMovingScreenIn.fetch_file_path(allow_download, show_progressbar)
    return load(file_path, **kwargs)


@deprecated("0.8", "si_ebsd_moving_screen", removal="0.9")
def silicon_ebsd_moving_screen_out5mm(
    allow_download: bool = False, show_progressbar: Optional[bool] = None, **kwargs
) -> EBSD:
    """One EBSD pattern of (480, 480) pixels from a single crystal
    silicon sample, acquired on a NORDIF UF-420 detector
    :cite:`aanes2022electron3`.

    This pattern and two other patterns from the same sample position
    but with sample-screen-distances 5 mm shorter
    (:func:`silicon_ebsd_moving_screen_in`) and 5 mm greater
    (:func:`silicon_ebsd_moving_screen_out10mm`) were acquired to test
    the moving-screen projection center estimation technique
    :cite:`hjelen1991electron`.

    Parameters
    ----------
    allow_download
        Whether to allow downloading the dataset from the internet to
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

    Notes
    -----
    The dataset is hosted in the GitHub repository
    https://github.com/pyxem/kikuchipy-data.

    The dataset carries a CC BY 4.0 license.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.si_ebsd_moving_screen(5, allow_download=True)
    >>> s
    <EBSD, title: si_out5mm Scan 1, dimensions: (|480, 480)>
    >>> s.plot()
    """
    SiEBSDMovingScreenOut5mm = Dataset("silicon_ebsd_moving_screen/si_out5mm.h5")
    file_path = SiEBSDMovingScreenOut5mm.fetch_file_path(
        allow_download, show_progressbar
    )
    return load(file_path, **kwargs)


@deprecated("0.8", "si_ebsd_moving_screen", removal="0.9")
def silicon_ebsd_moving_screen_out10mm(
    allow_download: bool = False, show_progressbar: Optional[bool] = None, **kwargs
) -> EBSD:
    """One EBSD pattern of (480, 480) pixels from a single crystal
    silicon sample, acquired on a NORDIF UF-420 detector
    :cite:`aanes2022electron3`.

    This pattern and two other patterns from the same sample position
    but with sample-screen-distances 10 mm shorter
    (:func:`silicon_ebsd_moving_screen_in`) and 5 mm shorter
    (:func:`silicon_ebsd_moving_screen_out5mm`) were acquired to test
    the moving-screen projection center estimation technique
    :cite:`hjelen1991electron`.

    Parameters
    ----------
    allow_download
        Whether to allow downloading the dataset from the internet to
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

    Notes
    -----
    The dataset is hosted in the GitHub repository
    https://github.com/pyxem/kikuchipy-data.

    The dataset carries a CC BY 4.0 license.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.si_ebsd_moving_screen(10, allow_download=True)
    >>> s
    <EBSD, title: si_out10mm Scan 1, dimensions: (|480, 480)>
    >>> s.plot()
    """
    SiEBSDMovingScreenOut10mm = Dataset("silicon_ebsd_moving_screen/si_out10mm.h5")
    file_path = SiEBSDMovingScreenOut10mm.fetch_file_path(
        allow_download, show_progressbar
    )
    return load(file_path, **kwargs)


def si_wafer(
    allow_download: bool = False, show_progressbar: Optional[bool] = None, **kwargs
) -> EBSD:
    """EBSD dataset of (50, 50) patterns of (480, 480) pixels from a
    single crystal silicon wafer, acquired on a NORDIF UF-420 detector
    :cite:`aanes2022electron3`.

    The dataset was acquired in order to test various ways to calibrate
    projection centers (PCs), e.g. the moving-screen PC estimation
    technique :cite:`hjelen1991electron`. The EBSD pattern in
    :func:`silicon_ebsd_moving_screen_in` is from this dataset.

    Parameters
    ----------
    allow_download
        Whether to allow downloading the dataset from the internet to
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
    silicon_ebsd_moving_screen_in, silicon_ebsd_moving_screen_out5mm,
    silicon_ebsd_moving_screen_out10mm

    Notes
    -----
    The dataset is hosted in the Zenodo repository
    https://doi.org/10.5281/zenodo.7491388 and comprises 311 MB as a
    zipped file and about 581 MB when unzipped. The zipped file is
    deleted after it is unzipped.

    The dataset carries a CC BY 4.0 license.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.si_wafer(allow_download=True, lazy=True)  # doctest: +SKIP
    >>> s  # doctest: +SKIP
    <EBSD, title: Pattern, dimensions: (50, 50|480, 480)>
    """
    SiWafer = Dataset("si_wafer/Pattern.dat", collection_name="ebsd_si_wafer.zip")
    file_path = SiWafer.fetch_file_path(allow_download, show_progressbar)
    return load(file_path, **kwargs)  # pragma: no cover


# ---------------------------- Simulations --------------------------- #


def nickel_ebsd_master_pattern_small(**kwargs) -> EBSDMasterPattern:
    """(401, 401) ``uint8`` square Lambert or stereographic projection
    of the northern and southern hemisphere of a nickel master pattern
    at 20 keV accelerating voltage.

    The master pattern was simulated with *EMsoft*
    :cite:`callahan2013dynamical`.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to :func:`~kikuchipy.load`.

    Returns
    -------
    ebsd_master_pattern_signal
        EBSD master pattern signal.

    See Also
    --------
    ebsd_master_pattern

    Notes
    -----
    The dataset carries a CC BY 4.0 license.

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
    NiEBSDMasterPatternSmall = Dataset(
        "emsoft_ebsd_master_pattern/ni_mc_mp_20kv_uint8_gzip_opts9.h5"
    )
    file_path = NiEBSDMasterPatternSmall.fetch_file_path()
    return load(file_path, **kwargs)


def ebsd_master_pattern(
    phase: str,
    allow_download: bool = False,
    show_progressbar: Optional[bool] = None,
    **kwargs,
) -> EBSDMasterPattern:
    r"""EBSD master pattern of an available phase of (1001, 1001) pixel
    resolution in both the square Lambert or stereographic projection.

    Master patterns were simulated with *EMsoft*
    :cite:`callahan2013dynamical`.

    Parameters
    ----------
    phase
        Name of available phase. Options are (see *Notes* for details):
        ni, al, si, austenite, ferrite, steel_chi, steel_sigma.
    allow_download
        Whether to allow downloading the dataset from the internet to
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
    ebsd_master_pattern_signal
        EBSD master pattern signal.

    See Also
    --------
    nickel_ebsd_master_pattern_small, si_ebsd_master_pattern

    Notes
    -----
    The master patterns are downloaded in HDF5 files (carrying a CC BY
    4.0 license) from Zenodo.

    ===========  =========  =================  ============  ==============  ======================
    phase        Name       Symbol             Energy [keV]  File size [GB]  Zenodo DOI
    ===========  =========  =================  ============  ==============  ======================
    ni           nickel     Ni                 5-20          0.3             10.5281/zenodo.7628443
    al           aluminium  Al                 10-20         0.2             10.5281/zenodo.7628365
    si           silicon    Si                 5-20          0.3             10.5281/zenodo.7498729
    austenite    austenite  :math:`\gamma`-Fe  10-20         0.3             10.5281/zenodo.7628387
    ferrite      ferrite    :math:`\alpha`-Fe  5-20          0.3             10.5281/zenodo.7628394
    steel_chi    chi        Fe36Cr15Mo7        10-20         0.6             10.5281/zenodo.7628417
    steel_sigma  sigma      FeCr               5-20          1.5             10.5281/zenodo.7628443
    ===========  =========  =================  ============  ==============  ======================

    Examples
    --------
    Import master pattern in the stereographic projection

    >>> import kikuchipy as kp
    >>> s = kp.data.ebsd_master_pattern("ni", hemisphere="both")  # doctest: +SKIP
    >>> s  # doctest: +SKIP
    <EBSDMasterPattern, title: ni_mc_mp_20kv, dimensions: (16, 2|1001, 1001)>
    >>> s.projection  # doctest: +SKIP
    'stereographic'
    """
    datasets = {
        "ni": "ni_mc_mp_20kv.h5",
        "al": "al_mc_mp_20kv.h5",
        "si": "si_mc_mp_20kv.h5",
        "austenite": "austenite_mc_mp_20kv.h5",
        "ferrite": "ferrite_mc_mp_20kv.h5",
        "steel_chi": "steel_chi_mc_mp_20kv.h5",
        "steel_sigma": "steel_sigma_mc_mp_20kv.h5",
    }

    dset = Dataset("ebsd_master_pattern/" + datasets[phase.lower()])
    file_path = dset.fetch_file_path(allow_download, show_progressbar)

    return load(file_path, **kwargs)  # pragma: no cover


class Dataset:
    file_relpath: Path
    file_package_path: Path
    file_cache_path: Path
    expected_md5_hash: str = ""
    collection_name: Optional[str] = None

    def __init__(
        self,
        file_relpath: Union[Path, str],
        collection_name: Optional[str] = None,
    ) -> None:
        if isinstance(file_relpath, str):
            file_relpath = Path(file_relpath)
        self.file_package_path = Path(os.path.dirname(__file__)) / file_relpath

        file_relpath = "data" / file_relpath
        self.file_relpath = file_relpath
        self.file_cache_path = Path(marshall.path) / self.file_relpath

        self.expected_md5_hash = registry_hashes[self.file_relpath_str]

        self.collection_name = collection_name

    @property
    def file_relpath_str(self) -> str:
        return self.file_relpath.as_posix()

    @property
    def is_in_collection(self) -> bool:
        return self.collection_name is not None

    @property
    def is_in_package(self) -> bool:
        return self.file_package_path.exists()

    @property
    def is_in_cache(self) -> bool:
        return self.file_cache_path.exists()

    @property
    def file_directory(self) -> Path:
        return Path(os.path.join(*self.file_relpath.parts[1:-1]))

    @property
    def file_path(self) -> Path:
        if self.is_in_package:
            return self.file_package_path
        else:
            return self.file_cache_path

    @property
    def file_path_str(self) -> str:
        return self.file_path.as_posix()

    @property
    def md5_hash(self) -> Union[str, None]:
        if self.file_path.exists():
            return pooch.file_hash(self.file_path_str, alg="md5")
        else:
            return None

    @property
    def has_correct_hash(self) -> bool:
        return self.md5_hash == self.expected_md5_hash.split(":")[1]

    @property
    def url(self) -> Union[str, None]:
        if self.file_relpath_str in registry_urls:
            return registry_urls[self.file_relpath_str]
        elif self.is_in_collection and "data/" + self.collection_name in registry_urls:
            return registry_urls["data/" + self.collection_name]
        else:
            return None

    def fetch_file_path_from_collection(
        self, downloader: pooch.HTTPDownloader
    ) -> file_path:  # pragma: no cover
        file_paths = marshall.fetch(
            os.path.join("data", self.collection_name),
            downloader=downloader,
            processor=pooch.Unzip(extract_dir=self.file_directory),
        )

        os.remove(os.path.join(marshall.path, "data", self.collection_name))

        # Ensure the file is in the collection
        desired_name = self.file_relpath.name
        for fpath in map(Path, file_paths):
            if desired_name == fpath.name:
                break
        else:
            raise ValueError(
                f"File {self.file_relpath.name} not found in the collection "
                f"{self.collection_name} at {self.url}. This is surprising. Please "
                "report it to the developers at "
                "https://github.com/pyxem/kikuchipy/issues/new."
            )

        return self.file_relpath_str

    def fetch_file_path(
        self, allow_download: bool = False, show_progressbar: Optional[bool] = None
    ) -> str:
        if show_progressbar is None:
            show_progressbar = hs.preferences.General.show_progressbar
        downloader = pooch.HTTPDownloader(progressbar=show_progressbar)

        if self.is_in_package:
            if self.has_correct_hash:
                # Bypass pooch since the file is not in the cache
                return self.file_path_str
            else:  # pragma: no cover
                raise AttributeError(
                    f"File {self.file_path_str} has incorrect MD5 hash {self.md5_hash}"
                    f", while {self.expected_md5_hash.split(':')[1]} was expected. This"
                    " is surprising. Please report it to the developers at "
                    "https://github.com/pyxem/kikuchipy/issues/new."
                )
        elif self.is_in_cache:
            if self.has_correct_hash:
                file_path = self.file_relpath_str
            elif allow_download:  # pragma: no cover
                if self.is_in_collection:
                    file_path = self.fetch_file_path_from_collection(downloader)
                else:
                    file_path = self.file_relpath_str
            else:  # pragma: no cover
                raise ValueError(
                    f"File {self.file_path_str} must be re-downloaded from the "
                    f"repository file {self.url} to your local cache {marshall.path}. "
                    "Pass `allow_download=True` to allow this re-download."
                )
        else:
            if allow_download:  # pragma: no cover
                if self.is_in_collection:
                    file_path = self.fetch_file_path_from_collection(downloader)
                else:
                    file_path = self.file_relpath_str
            else:
                raise ValueError(
                    f"File {self.file_relpath_str} must be downloaded from the "
                    f"repository file {self.url} to your local cache {marshall.path}. "
                    "Pass `allow_download=True` to allow this download."
                )

        return marshall.fetch(file_path, downloader=downloader)
