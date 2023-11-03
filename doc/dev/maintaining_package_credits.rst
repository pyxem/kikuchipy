Maintaining package credits
===========================

Whenever we get a new contributor, they should be added to the package credits.
Unless they do not want to, of course.
We maintain three separate sources for the list of contributors:

* ``kikuchipy/release.py``: Package metadata used by PyPI and other places
* ``.zenodo.json``: Zenodo entry
* All-contributors table in the README

In the package metadata and the Zenodo entry, the initial commiter is listed first, with
the others sorted by line contributions.

The All-contributors table in the README is updated locally using their command line
interface (see their `web page <https://allcontributors.org/>`_ for the docs).