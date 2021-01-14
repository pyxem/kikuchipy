=============
Open datasets
=============

.. note::

    See the :mod:`~kikuchipy.data` module for small test data sets.

Zipped data sets available via `Zenodo <https://zenodo.org>`_, in this example
with `record-number` named `data.zip`, can be downloaded:

.. code-block::

    >>> from urllib.request import urlretrieve
    >>> files = urlretrieve(
    ...     url='https://zenodo.org/record/<record-number>/files/data.zip',
    ...     filename='./downloaded_data.zip'
    ... )

- :cite:`aanes2019electron`
