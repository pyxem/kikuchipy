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
    Håkon Wiik Ånes, Jarle Hjelen, Antonius T. J. van Helvoort, & Knut
    Marthinsen. (2019). Electron backscatter patterns from Nickel acquired with
    varying camera gain [Data set]. Zenodo.
    `http://doi.org/10.5281/zenodo.3265037
    <http://doi.org/10.5281/zenodo.3265037>`_
