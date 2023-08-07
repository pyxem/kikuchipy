Running and writing tests
=========================

All functionality in kikuchipy is tested via the :doc:`pytest <pytest:index>` framework.
The tests reside in a ``tests/`` directory within each module.
Tests are short methods that call functions in kikuchipy and compare resulting output
values with known answers.
Install necessary dependencies to run the tests::

    pip install --editable .[tests]

Some useful :doc:`fixtures <pytest:explanation/fixtures>`, like a dummy scan and
corresponding background pattern, are available in the ``conftest.py`` file.

.. note::

    Some :mod:`kikuchipy.data` module tests check that data not part of the package
    distribution can be downloaded from the https://github.com/pyxem/kikuchipy-data
    GitHub repository, thus downloading some datasets of ~15 MB to your local cache.

    This also means that running tests require an internet connection.

To run the tests::

    pytest --cov --pyargs kikuchipy

The ``--cov`` flag makes :doc:`coverage.py <coverage:index>` print a nice report in the
terminal.
For an even nicer presentation, you can use ``coverage.py`` directly::

    coverage html

Then, you can open the created ``htmlcov/index.html`` in the browser and inspect the
coverage in more detail.

To run only a specific test function or class, .e.g the ``TestEBSD`` class::

    pytest -k TestEBSD

This is useful when you only want to run a specific test and not the full test suite,
e.g. when you're creating or updating a test.
But remember to run the full test suite before pushing!

Docstring examples are tested :doc:`with pytest <pytest:how-to/doctest>` as well.
If you're in the top directory you can run::

    pytest --doctest-modules --ignore-glob=kikuchipy/*/tests kikuchipy/*.py

Functionality using Numba
-------------------------

Tips for writing tests of Numba decorated functions:

- A Numba decorated function ``numba_func()`` is only covered if it is called in the
  test as ``numba_func.py_func()``.
- Always test a Numba decorated function calling ``numba_func()`` directly, in addition
  to ``numba_func.py_func()``, because the machine code function might give different
  results on different OS with the same Python code.
  See this issue https://github.com/pyxem/kikuchipy/issues/496 for a case where this
  happened.

Functionality using multiprocessing
-----------------------------------

Some functionality may run in parallel using :mod:`multiprocessing`, such as
:func:`pyebsdindex.pcopt.optimize_pso` which is used in
:meth:`~kikuchipy.signals.ebsd.hough_indexing_optimize_pc`.
A test of this functionality may hang when run in a parallel test run using
:mod:`pytest-xdist`.
To ensure the multiprocessing-part only runs when pytest-xdist is not used, we can
ensure that the value of the ``worker_id`` fixture provided by pytest-xdist is
``"master"``.