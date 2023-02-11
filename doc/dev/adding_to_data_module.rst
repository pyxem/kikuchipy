.. _adding-data-to-data-module:

Adding data to the data module
==============================

Example datasets used in the documentation and tests are included in the
:mod:`kikuchipy.data` module via the `pooch <https://www.fatiando.org/pooch/latest/>`__
Python library. These are listed in a file registry (``kikuchipy.data._registry.py``)
with their file verification string (hash, MD5, obtain with e.g. ``md5sum <file>``) and
location, the latter potentially not within the package but from the `kikuchipy-data
<https://github.com/pyxem/kikuchipy-data>`__ repository or elsewhere, since some files
are considered too large to include in the package.

If a required dataset isn't in the package, but is in the registry, it can be downloaded
from the repository when the user passes ``allow_download=True`` to e.g.
:func:`~kikuchipy.data.nickel_ebsd_large`. The dataset is then downloaded to a local
cache, in the location returned from ``pooch.os_cache("kikuchipy")``. The location can
be set with a global `KIKUCHIPY_DATA_DIR` variable locally, e.g. by setting
``export KIKUCHIPY_DATA_DIR=~/kikuchipy_data`` in ``~/.bashrc``. Pooch handles
downloading, caching, version control, file verification (against hash) etc. of files
not included in the package. If we have updated the file hash, pooch will re-download
it. If the file is available in the cache, it can be loaded as the other files in the
data module.

With every new version of kikuchipy, a new directory of datasets with the version name
is added to the cache directory. Any old directories are not deleted automatically, and
should then be deleted manually if desired.