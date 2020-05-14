============
Contributing
============

kikuchipy is meant to be a community maintained project. We welcome
contributions in the form of bug reports, documentation, code, feature requests,
and more. These guidelines provide resources on how best to contribute.

For new users, checking out the `GitHub guides <https://guides.github.com>`_ are
recommended.

.. Many of these steps follow napari's contributor guide:
   https://github.com/napari/napari/blob/master/docs/CONTRIBUTING.md

.. _setting-up-a-development-installation:

Setting up a development installation
=====================================

You need a `fork <https://guides.github.com/activities/forking/#fork>`_ of the
`repository <https://github.com/kikuchipy/kikuchipy>`_ in order to make changes
to kikuchipy.

Make a local copy of your forked repository and change directories::

    $ git clone https://github.com/your-username/kikuchipy.git
    $ cd kikuchipy

Set the ``upstream`` remote to the main kikuchipy repository::

    $ git remote add upstream https://github.com/kikuchipy/kikuchipy.git

We recommend installing in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
with the `Miniconda distribution
<https://docs.conda.io/en/latest/miniconda.html>`_::

   $ conda create --name kikuchipy python=3.8
   $ conda activate kikuchipy

Then, install the required dependencies while making the development version
available globally (in the ``conda`` environment)::

   $ pip install --editable .[dev]

This installs all necessary development dependencies, including those for
running tests and building documentation.

The code making up kikuchipy is formatted closely following the `Style Guide for
Python Code <https://www.python.org/dev/peps/pep-0008/>`_ with `The Black Code
style <https://black.readthedocs.io/en/stable/the_black_code_style.html>`_. We
use `pre-commit <https://pre-commit.com>`_ to run ``black`` automatically prior
to each local commit. Please install it in your environment::

    $ pre-commit install

Next time you commit some code, your code will be formatted inplace according
to our `black configuration
<https://github.com/kikuchipy/kikuchipy/blob/master/pyproject.toml>`_.

Note that ``black`` won't format `docstrings
<https://www.python.org/dev/peps/pep-0257/>`_. We follow the `numpydoc
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_
standard.

.. _making-changes:

Making changes
==============

Create a new feature branch::

    $ git checkout master -b your-awesome-feature-name

When you've made some changes you can view them with::

    $ git status

Add and commit your created, modified or deleted files::

   $ git add my-file-or-directory
   $ git commit -s -m "An explanatory commit message"

The ``-s`` makes sure that you sign your commit with your `GitHub-registered
email <https://github.com/settings/emails>`_ as the author. You can set this up
following `this GitHub guide
<https://help.github.com/en/github/setting-up-and-managing-your-github-user-account/setting-your-commit-email-address>`_.

.. _keeping-your-branch-up-to-date:

Keeping your branch up-to-date
==============================

Switch to the ``master`` branch::

   $ git checkout master

Fetch changes and update ``master``::

   $ git pull upstream master --tags

Update your feature branch::

   $ git checkout your-awesome-feature-name
   $ git merge master

.. _sharing-your-changes:

Sharing your changes
====================

Update your remote branch::

   $ git push -u origin your-awesome-feature-name

You can then make a `pull request
<https://guides.github.com/activities/forking/#making-a-pull-request>`_ to
kikuchipy's ``master`` branch. Good job!

.. _building-the-documentation:

Building the documentation
==========================

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ for documenting
functionality. Install necessary dependencies to build the documentation::

   $ pip install --editable .[doc]

Then, build the documentation from the ``doc`` directory::

   $ cd doc
   $ make html

The documentation's ``html`` pages are built in the ``doc/build/html`` directory
from files in the `reStructuredText
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
plaintext markup language. They should be accessible in the browser by typing
``file:///your-absolute/path/to/kikuchipy/doc/build/html/index.html`` in the
address bar.

.. _running-and-writing-tests:

Running and writing tests
=========================

All functionality in kikuchipy is tested via the `pytest
<https://docs.pytest.org>`_ framework. The tests reside in a ``test`` directory
within each module. Tests are short methods that call functions in kikuchipy
and compare resulting output values with known answers. Install necessary
dependencies to run the tests::

   $ pip install --editable .[tests]

Some useful `fixtures <https://docs.pytest.org/en/latest/fixture.html>`_, like a
dummy scan and corresponding background pattern, are available in the
``conftest.py`` file.

To run the tests::

   $ pytest --cov --pyargs kikuchipy

The ``--cov`` flag makes `coverage.py
<https://coverage.readthedocs.io/en/latest/>`_ print a nice report in the
terminal. For an even nicer presentation, you can use ``coverage.py`` directly::

   $ coverage html

Then, you can open the created ``htmlcov/index.html`` in the browser and inspect
the coverage in more detail.

.. _code-of-conduct:

Code of Conduct
===============

kikuchipy has a :doc:`Code of Conduct <code_of_conduct>` that should be honoured
by everyone who participates in the kikuchipy community.

.. _questions-comments-and-feedback:

Questions, comments, and feedback
=================================

Have any questions, comments, suggestions for improvements, or any other
inquiries regarding the project? Feel free to open an issue in our `GitHub Issue
Tracker <https://github.com/kikuchipy/kikuchipy/issues>`_.

.. _continuous-integration:

Continuous integration (CI)
===========================

We use `Travis CI <https://travis-ci.com/kikuchipy/kikuchipy>`_ to ensure that
kikuchipy can be installed on Windows, macOS and Linux (Ubuntu). After a
successful installation, the CI server runs the tests. After the tests return no
errors, code coverage is reported to `Coveralls
<https://coveralls.io/github/kikuchipy/kikuchipy?branch=master>`_.
