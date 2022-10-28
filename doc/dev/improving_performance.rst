Improving performance
=====================

When we write code, it's important that we (1) get the correct result, (2) don't fill up
memory, and (3) that the computation doesn't take too long. To keep memory in check, we
should use `Dask <https://docs.dask.org/en/latest/>`__ wherever possible. To speed up
computations, we should use `Numba <https://numba.pydata.org/numba-doc/dev/>`__ wherever
possible.

To check whether a change is an improvement or a regression, a benchmark should be
written. These are stored in the top directory ``kikuchipy/benchmarks``. Benchmarks are
run using `pytest-benchmark
<https://pytest-benchmark.readthedocs.io/en/stable/index.html>`__::

    pytest --benchmark-only