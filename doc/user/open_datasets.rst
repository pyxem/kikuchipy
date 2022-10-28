=============
Open datasets
=============

.. note::

    See the :mod:`~kikuchipy.data` module for small test data sets.

Zipped data sets available via `Zenodo <https://zenodo.org>`_, in this example
with `record-number` named `data.zip`, can be downloaded like this:

.. code-block::

    >>> from urllib.request import urlretrieve
    >>> files = urlretrieve(
    ...     url='https://zenodo.org/record/<record-number>/files/data.zip',
    ...     filename='./downloaded_data.zip'
    ... )

This is a non-exhaustive list of EBSD datasets openly available on the internet which
can be read by kikuchipy:

- :cite:`shi2021high`
- :cite:`shi2022high`
- :cite:`wilkinson2018small`
- :cite:`aanes2019electron`
- :cite:`aanes2022electron`
