============
Contributing
============

kikuchipy is meant to be a community maintained project. We welcome contributions in the
form of bug reports, documentation, code, feature requests, and more. These guidelines
provide resources on how best to contribute.

For new users, checking out the `GitHub guides <https://guides.github.com>`_ are
recommended.

This project follows the `all-contributors
<https://github.com/all-contributors/all-contributors>`_ specification.

Questions, comments, and feedback
=================================

Have any questions, comments, suggestions for improvements, or any other inquiries
regarding the project? Feel free to `ask a question
<https://github.com/pyxem/kikuchipy/discussions>`_, `open an issue
<https://github.com/pyxem/kikuchipy/issues>`_ or `make a pull request
<https://github.com/pyxem/kikuchipy/pulls>`_ in our GitHub
repository. We also have a `Gitter chat <https://gitter.im/pyxem/kikuchipy>`_.

Code of Conduct
===============

kikuchipy has a :doc:`Code of Conduct <code_of_conduct>` that should be honoured by
everyone who participates in the kikuchipy community.

.. _setting-up-a-development-installation:

Setting up a development installation
=====================================

You need a `fork <https://guides.github.com/activities/forking/#fork>`_ of the
`repository <https://github.com/pyxem/kikuchipy>`_ in order to make changes to
kikuchipy.

Make a local copy of your forked repository and change directories::

    $ git clone https://github.com/your-username/kikuchipy.git
    $ cd kikuchipy

Set the ``upstream`` remote to the main kikuchipy repository::

    $ git remote add upstream https://github.com/pyxem/kikuchipy.git

We recommend installing in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
with the `Miniconda distribution <https://docs.conda.io/en/latest/miniconda.html>`_::

   $ conda create --name kp-dev
   $ conda activate kp-dev

Then, install the required dependencies while making the development version available
globally (in the ``conda`` environment)::

   $ pip install --editable .[dev]

This installs all necessary development dependencies, including those for running tests
and building documentation.

Code style
==========

The code making up kikuchipy is formatted closely following the `Style Guide for Python
Code <https://www.python.org/dev/peps/pep-0008/>`_ with `The Black Code style
<https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>`_. We
use `pre-commit <https://pre-commit.com>`_ to run ``black`` automatically prior to each
local commit. Please install it in your environment::

    $ pre-commit install

Next time you commit some code, your code will be formatted inplace according to
``black``.

Note that ``black`` won't format `docstrings
<https://www.python.org/dev/peps/pep-0257/>`_. We follow the `numpydoc
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ standard.

Comment lines should preferably be limited to 72 characters.

Package imports should be structured into three blocks with blank lines between them
(descending order): standard library (like ``os`` and ``typing``), third party packages
(like ``numpy`` and ``hyperspy``) and finally kikuchipy imports.

Making changes
==============

If you want to add a new feature, branch off of the ``develop`` branch, and when you
want to fix a bug, branch off of ``main`` instead.

To create a new feature branch that tracks the upstream development branch::

    $ git checkout develop -b your-awesome-feature-name upstream/develop

When you've made some changes you can view them with::

    $ git status

Add and commit your created, modified or deleted files::

   $ git add my-file-or-directory
   $ git commit -s -m "An explanatory commit message"

The ``-s`` makes sure that you sign your commit with your `GitHub-registered email
<https://github.com/settings/emails>`_ as the author. You can set this up following
`this GitHub guide
<https://help.github.com/en/github/setting-up-and-managing-your-github-user-account/setting-your-commit-email-address>`_.

Keeping your branch up-to-date
==============================

If you are adding a new feature, make sure to merge ``develop`` into your feature
branch. If you are fixing a bug, merge ``main`` into your bug fix branch instead.

To update a feature branch, switch to the ``develop`` branch::

   $ git checkout develop

Fetch changes from the upstream branch and update ``develop``::

   $ git pull upstream develop --tags

Update your feature branch::

   $ git checkout your-awesome-feature-name
   $ git merge develop

Sharing your changes
====================

Update your remote branch::

   $ git push -u origin your-awesome-feature-name

You can then make a `pull request
<https://guides.github.com/activities/forking/#making-a-pull-request>`_ to kikuchipy's
``develop`` branch for new features and ``main`` branch for bug fixes. Good job!

Building and writing documentation
==================================

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ for documenting functionality.
Install necessary dependencies to build the documentation::

   $ pip install --editable .[doc]

Then, build the documentation from the ``doc`` directory::

   $ cd doc
   $ make html

The documentation's HTML pages are built in the ``doc/build/html`` directory from files
in the `reStructuredText (reST)
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ plaintext
markup language. They should be accessible in the browser by typing
``file:///your/absolute/path/to/kikuchipy/doc/build/html/index.html`` in the address
bar.

Tips for writing Jupyter Notebooks that are meant to be converted to reST text files by
`nbsphinx <https://nbsphinx.readthedocs.io/en/latest/>`_:

- All notebooks should have a Markdown (MD) cell with this message at the top, "This
  notebook is part of the `kikuchipy` documentation https://kikuchipy.org. Links to the
  documentation won't work from the notebook.", and have ``"nbsphinx": "hidden"`` in the
  cell metadata so that the message is not visible when displayed in the documentation.
- Use ``_ = ax[0].imshow(...)`` to disable Matplotlib output if a Matplotlib command is
  the last line in a cell.
- Refer to our API reference with this general MD
  ``[fft_filter()](../reference.rst#kikuchipy.signals.EBSD.fft_filter)``. Remember to
  add the parentheses ``()`` for functions and methods.
- Reference external APIs via standard MD like
  ``[Signal2D](http://hyperspy.org/hyperspy-doc/current/api/hyperspy._signals.signal2d.html)``.
- The Sphinx gallery thumbnail used for a notebook is set by adding the
  ``nbsphinx-thumbnail`` tag to a code cell with an image output. The notebook must be
  added to the gallery in the README.rst to be included in the documentation pages.
- The Furo Sphinx theme displays the documentation in a light or dark theme, depending
  on the browser/OS setting. It is important to make sure the documentation is readable
  with both themes. This means explicitly printing the signal axes manager, like
  ``print(s.axes_manager)``, and displaying all figures with a white background for axes
  labels and ticks and figure titles etc. to be readable.

In general, we run all notebooks every time the documentation is built with Sphinx, to
ensure that all notebooks are compatible with the current API at all times. This is
important! For computationally expensive notebooks however, we store the cell outputs so
the documentation doesn't take too long to build, either by us locally or the Read The
Docs GitHub action. To check that the notebooks with cell outputs stored are compatible
with the current API as well, we run a scheduled GitHub Action every Monday morning
which checks that the notebooks run OK and that they produced the same output now as
when they were last executed. We use `nbval <https://nbval.readthedocs.io/en/latest/>`_
for this.

Running and writing tests
=========================

All functionality in kikuchipy is tested via the `pytest <https://docs.pytest.org>`_
framework. The tests reside in a ``test`` directory within each module. Tests are short
methods that call functions in kikuchipy and compare resulting output values with known
answers. Install necessary dependencies to run the tests::

   $ pip install --editable .[tests]

Some useful `fixtures <https://docs.pytest.org/en/latest/fixture.html>`_, like a dummy
scan and corresponding background pattern, are available in the ``conftest.py`` file.

.. note::

    Some :mod:`kikuchipy.data` module tests check that data not part of the package
    distribution can be downloaded from the `kikuchipy-data GitHub repository
    <https://github.com/pyxem/kikuchipy-data>`_, thus downloading some datasets of ~15
    MB to your local cache.

To run the tests::

   $ pytest --cov --pyargs kikuchipy

The ``--cov`` flag makes `coverage.py <https://coverage.readthedocs.io/en/latest/>`_
print a nice report in the terminal. For an even nicer presentation, you can use
``coverage.py`` directly::

   $ coverage html

Then, you can open the created ``htmlcov/index.html`` in the browser and inspect the
coverage in more detail.

To run only a specific test function or class, .e.g the ``TestEBSD`` class::

    $ pytest -k TestEBSD

This is useful when you only want to run a specific test and not the full test suite,
e.g. when you're creating or updating a test. But remember to run the full test suite
before pushing!

Docstring examples are tested `with pytest
<https://docs.pytest.org/en/stable/doctest.html>`_ as well::

   $ pytest --doctest-modules --ignore-glob=kikuchipy/*/tests

Adding data to the data module
==============================

Test data for user guides and tests are included in the :mod:`kikuchipy.data` module via
the `pooch <https://www.fatiando.org/pooch/latest/>`_ Python library. These are listed
in a file registry (`kikuchipy.data._registry.py`) with their file verification string
(hash, SHA256, obtain with e.g. `sha256sum <file>`) and location, the latter potentially
not within the package but from the `kikuchipy-data
<https://github.com/pyxem/kikuchipy-data>`_ repository, since some files are considered
too large to include in the package.

If a required dataset isn't in the package, but is in the registry, it can be downloaded
from the repository when the user passes `allow_download=True` to e.g.
:func:`~kikuchipy.data.nickel_ebsd_large`. The dataset is then downloaded to a local
cache, e.g. `/home/user/.cache/kikuchipy/`. Pooch handles downloading, caching, version
control, file verification (against hash) etc. If we have updated the file hash, pooch
will re-download it. If the file is available in the cache, it can be loaded as the
other files in the data module.

The desired data cache directory used by pooch can be set with a global
`KIKUCHIPY_DATA_DIR` variable locally, e.g. by setting
`export KIKUCHIPY_DATA_DIR=~/kikuchipy_data` in `~/.bashrc`.

Improving performance
=====================
When we write code, it's important that we (1) get the correct result, (2) don't fill up
memory, and (3) that the computation doesn't take too long. To keep memory in check, we
should use `Dask <https://docs.dask.org/en/latest/>`_ wherever possible. To speed up
computations, we should use `Numba <https://numba.pydata.org/numba-doc/dev/>`_ wherever
possible.

Continuous integration (CI)
===========================

We use `GitHub Actions <https://github.com/pyxem/kikuchipy/actions>`_ to ensure that
kikuchipy can be installed on Windows, macOS and Linux (Ubuntu). After a successful
installation of the package, the CI server runs the tests. After the tests return no
errors, code coverage is reported to `Coveralls
<https://coveralls.io/github/pyxem/kikuchipy?branch=develop>`_. Add "[skip ci]" or to a
commit message to skip this workflow on any commit to a pull request, as explained
