============
Contributing
============

kikuchipy is meant to be a community maintained project. We welcome
contributions in the form of bug reports, documentation, code, feature requests,
and more. These guidelines provide resources on how best to contribute.

For new users, checking out the `GitHub guides <https://guides.github.com>`_ are
recommended.

Questions, comments, and feedback
=================================

Have any questions, comments, suggestions for improvements, or any other
inquiries regarding the project? Feel free to
`ask a question <https://github.com/pyxem/kikuchipy/discussions>`_,
`open an issue <https://github.com/pyxem/kikuchipy/issues>`_ or
`make a pull request <https://github.com/pyxem/kikuchipy/pulls>`_ in our GitHub
repository.

Code of Conduct
===============

kikuchipy has a :doc:`Code of Conduct <code_of_conduct>` that should be honoured
by everyone who participates in the kikuchipy community.

.. _setting-up-a-development-installation:

Setting up a development installation
=====================================

You need a `fork <https://guides.github.com/activities/forking/#fork>`_ of the
`repository <https://github.com/pyxem/kikuchipy>`_ in order to make changes
to kikuchipy.

Make a local copy of your forked repository and change directories::

    $ git clone https://github.com/your-username/kikuchipy.git
    $ cd kikuchipy

Set the ``upstream`` remote to the main kikuchipy repository::

    $ git remote add upstream https://github.com/pyxem/kikuchipy.git

We recommend installing in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
with the `Miniconda distribution
<https://docs.conda.io/en/latest/miniconda.html>`_::

   $ conda create --name kikuchipy
   $ conda activate kikuchipy

Then, install the required dependencies while making the development version
available globally (in the ``conda`` environment)::

   $ pip install --editable .[dev]

This installs all necessary development dependencies, including those for
running tests and building documentation.

Code style
==========

The code making up kikuchipy is formatted closely following the `Style Guide for
Python Code <https://www.python.org/dev/peps/pep-0008/>`_ with `The Black Code
style <https://black.readthedocs.io/en/stable/the_black_code_style.html>`_. We
use `pre-commit <https://pre-commit.com>`_ to run ``black`` automatically prior
to each local commit. Please install it in your environment::

    $ pre-commit install

Next time you commit some code, your code will be formatted inplace according
to our `black configuration
<https://github.com/pyxem/kikuchipy/blob/master/pyproject.toml>`_.

Note that ``black`` won't format `docstrings
<https://www.python.org/dev/peps/pep-0257/>`_. We follow the `numpydoc
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_
standard.

Comment lines should preferably be limited to 72 characters.

Package imports should be structured into three blocks with blank lines between
them (descending order): standard library (like ``os`` and ``typing``), third
party packages (like ``numpy`` and ``hyperspy``) and finally kikuchipy imports.

Making changes
==============

Create a new feature branch::

    $ git checkout master -b your-awesome-feature-name

When you've made some changes you can view them with::

    $ git status

Add and commit your created, modified or deleted files::

   $ git add my-file-or-directory
   $ git commit -s -m "An explanatory commit message"

The ``-s`` makes sure that you sign your commit with your `GitHub-registered
email <https://github.com/settings/emails>`_ as the author. You can set this up
following `this GitHub guide
<https://help.github.com/en/github/setting-up-and-managing-your-github-user-account/setting-your-commit-email-address>`_.

Keeping your branch up-to-date
==============================

Switch to the ``master`` branch::

   $ git checkout master

Fetch changes and update ``master``::

   $ git pull upstream master --tags

Update your feature branch::

   $ git checkout your-awesome-feature-name
   $ git merge master

Sharing your changes
====================

Update your remote branch::

   $ git push -u origin your-awesome-feature-name

You can then make a `pull request
<https://guides.github.com/activities/forking/#making-a-pull-request>`_ to
kikuchipy's ``master`` branch. Good job!

Building and writing documentation
==================================

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ for documenting
functionality. Install necessary dependencies to build the documentation::

   $ pip install --editable .[doc]

Then, build the documentation from the ``doc`` directory::

   $ cd doc
   $ make html

The documentation's HTML pages are built in the ``doc/build/html`` directory
from files in the `reStructuredText (reST)
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
plaintext markup language. They should be accessible in the browser by typing
``file:///your-absolute/path/to/kikuchipy/doc/build/html/index.html`` in the
address bar.

Tips for writing Jupyter Notebooks that are meant to be converted to reST text
files by `nbsphinx <https://nbsphinx.readthedocs.io/en/latest/>`_:

- All notebooks should have a Markdown (MD) cell with this message at the top,
  "This notebook is part of the `kikuchipy` documentation https://kikuchipy.org.
  Links to the documentation won't work from the notebook.", and have
  ``"nbsphinx": "hidden"`` in the cell metadata so that the message is not
  visible when displayed in the documentation.
- Use ``_ = ax[0].imshow(...)`` to disable Matplotlib output if a Matplotlib
  command is the last line in a cell.
- Refer to our API reference with this general MD
  ``[fft_filter()](reference.rst#kikuchipy.signals.EBSD.fft_filter)``. Remember
  to add the parentheses ``()``.
- Reference external APIs via standard MD like
  ``[Signal2D](http://hyperspy.org/hyperspy-doc/current/api/hyperspy._signals.signal2d.html)``.
- The Sphinx gallery thumbnail used for a notebook is set by adding the
  ``nbsphinx-thumbnail`` tag to a code cell with an image output. The notebook
  must be added to the gallery in the README.rst to be included in the
  documentation pages.

Running and writing tests
=========================

All functionality in kikuchipy is tested via the `pytest
<https://docs.pytest.org>`_ framework. The tests reside in a ``test`` directory
within each module. Tests are short methods that call functions in kikuchipy
and compare resulting output values with known answers. Install necessary
dependencies to run the tests::

   $ pip install --editable .[tests]

Some useful `fixtures <https://docs.pytest.org/en/latest/fixture.html>`_, like a
dummy scan and corresponding background pattern, are available in the
``conftest.py`` file.

To run the tests::

   $ pytest --cov --pyargs kikuchipy

The ``--cov`` flag makes `coverage.py
<https://coverage.readthedocs.io/en/latest/>`_ print a nice report in the
terminal. For an even nicer presentation, you can use ``coverage.py`` directly::

   $ coverage html

Then, you can open the created ``htmlcov/index.html`` in the browser and inspect
the coverage in more detail.

.. note::

   Some :mod:`kikuchipy.data` module tests check that data not part of the
   package distribution can be downloaded from the `kikuchipy-data GitHub
   repository <https://github.com/pyxem/kikuchipy-data>`_, thus downloading some
   datasets of ~15 MB to your local cache.

Continuous integration (CI)
===========================

We use `GitHub Actions <https://github.com/pyxem/kikuchipy/actions>`_ to ensure
that kikuchipy can be installed on Windows, macOS and Linux (Ubuntu). After a
successful installation, the CI server runs the tests. After the tests return no
errors, code coverage is reported to `Coveralls
<https://coveralls.io/github/pyxem/kikuchipy?branch=master>`_.
