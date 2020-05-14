=============
Open datasets
=============

Zipped data sets available via `Zenodo <https://zenodo.org>`_, in this example
with `record-number` named `data.zip`, can be downloaded:

.. code-block:: python

    >>> from urllib.request import urlretrieve
    >>> files = urlretrieve(
            url='https://zenodo.org/record/<record-number>/files/data.zip',
            filename='./downloaded_data.zip')

.. [Anes2019]
    H. W. Ånes, J. Hjelen, A. T. J. van Helvoort, & K. Marthinsen, "Electron
    backscatter patterns from Nickel acquired with varying camera gain," [Data
    set], (2015), Zenodo: http://doi.org/10.5281/zenodo.3265037.
