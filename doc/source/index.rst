KikuchiPy
=========

*KikuchiPy is an open-source Python library for processing of electron
backscatter diffraction (EBSD) patterns.*

Processing builds upon the tools for multi-dimensional data analysis provided
by the `HyperSpy`_ library. This means that the EBSD class, which has several
common methods for processing of EBSD patterns, also inherits all methods from
HyperSpy's `Signal2D`_ and `Signal`_ classes.

.. _`HyperSpy`: https://hyperspy.org
.. _`Signal2D`: https://hyperspy.org/hyperspy-doc/current/user_guide/signal2d.html
.. _`Signal`: https://hyperspy.org/hyperspy-doc/current/user_guide/tools.html

.. toctree::
    :maxdepth: 2
    :hidden:

    install.rst
    load_save_data.rst
    reference.rst
