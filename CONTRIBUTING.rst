============
Contributing
============

KikuchiPy is meant to be a community maintained project. We welcome
contributions in the form of bug reports, documentation, code, feature requests,
and more. These guidelines provide resources on how best to contribute.

.. _issues:

Issues
======

Known bugs and potential features are discussed in the `GitHub Issue Tracker
<https://github.com/kikuchipy/kikuchipy/issues>`_.

.. _tests:

Tests
=====

All functionality in KikuchiPy is tested via the `pytest
<https://docs.pytest.org>`_ framework.

.. _docstrings:

Docstrings
==========

Functions and methods to be used by the user roughly follow the `numpydoc
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_
standard.

.. _code-formatting:

Code formatting
===============

The code making up KikuchiPy is formatted closely following the `Style Guide for
Python Code <https://www.python.org/dev/peps/pep-0008/>`_ with `The *Black* Code
style <https://black.readthedocs.io/en/stable/the_black_code_style.html>`_.

.. _writing-documentation:

Writing documentation
=====================

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ for documenting
functionality.

.. _continuous-integration:

Continuous integration (CI)
===========================

We use `Travis CI <https://travis-ci.org/kikuchipy/kikuchipy>`_ to ensure that
KikuchiPy can be installed on Windows, macOS and Linux (Ubuntu) with both
``conda`` and ``pip``. After a successful installation, the CI server runs the
tests. After the tests return no errors, code coverage is reported to
`Coveralls <https://coveralls.io/github/kikuchipy/kikuchipy?branch=master>`_.
