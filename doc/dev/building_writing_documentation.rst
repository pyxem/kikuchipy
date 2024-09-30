Building and writing documentation
==================================

The documentation contains three categories of documents: ``examples``, ``tutorials``,
and the ``reference``.
The documentation strategy is based on the `Diátaxis Framework
<https://diataxis.fr/>`__.
New documents should fit into one of these categories.

We use :doc:`Sphinx <sphinx:index>` for documenting functionality.
Install necessary dependencies to build the documentation::

    pip install --editable ".[doc]"

.. note::

    The tutorials and examples require some small datasets to be downloaded via the
    :mod:`kikuchipy.data` module upon building the documentation.
    See the section on the :ref:`data module <adding-data-to-data-module>` for more
    details.

Then, build the documentation from the ``doc`` directory::

    cd doc
    make html

The documentation's HTML pages are built in the ``doc/_build/html`` directory from files
in the :doc:`reStructuredText (reST) <sphinx:usage/restructuredtext/basics>` plaintext
markup language.
They should be accessible in the browser by typing
``file:///your/absolute/path/to/kikuchipy/doc/_build/html/index.html`` in the address
bar.

We can link to other documentation in reStructuredText files using
:doc:`Intersphinx <sphinx:usage/extensions/intersphinx>`.
Which links are available from a package's documentation can be obtained like so::

    python -m sphinx.ext.intersphinx https://hyperspy.org/hyperspy-doc/current/objects.inv

We use :doc:`Sphinx-Gallery <sphinx-gallery:index>` to build the :ref:`examples`.
The examples are located in the top source directory ``examples/``, and a new directory
``doc/examples/`` is created when the docs are built.

We use :doc:`nbsphinx <nbsphinx:index>` for converting notebooks into tutorials
displayed in the documentation.
Code lines in notebooks should be :ref:`formatted with black <code-style>`.

Writing tutorial notebooks
--------------------------

Here are some tips for writing tutorial notebooks:

- All notebooks should have a Markdown cell with this message at the top, "This
  notebook is part of the kikuchipy documentation https://kikuchipy.org.
  Links to the documentation won't work from the notebook.", and have
  ``"nbsphinx": "hidden"`` in the cell metadata so that the message is not visible when
  displayed in the documentation.
- Use ``_ = ax[0].imshow(...)`` to silence ``matplotlib`` output if a ``matplotlib``
  command is the last line in a cell.
- Refer to our API reference with this general Markdown
  ``[fft_filter()](../reference/generated/kikuchipy.signals.EBSD.fft_filter.rst)``.
  Remember to add the parentheses ``()`` to functions and methods.
- Reference sections in other tutorial notebooks using this general Markdown
  ``[image quality](feature_maps.ipynb#image-quality)``.
- Reference external APIs via standard Markdown like
  ``[Signal2D](https://hyperspy.org/hyperspy-doc/current/api/hyperspy._signals.signal2d.html)``.
- The Sphinx gallery thumbnail used for a notebook is set by adding the
  ``nbsphinx-thumbnail`` tag to a code cell with an image output.
  The notebook must be added to the appropriate topic in ``doc/tutorials/index.rst`` to
  be included in the documentation pages.
- ``pydata_sphinx_theme`` displays the documentation in a light or dark theme, depending
  on the browser/OS setting.
  It is important to make sure the documentation is readable with both themes.
  This means explicitly printing the signal axes manager, like
  ``print(s.axes_manager)``, and displaying all figures with a white background for axes
  labels and ticks and figure titles etc. to be readable.
- Whenever the documentation is built (locally or on the Read the Docs server),
  ``nbsphinx`` only runs the notebooks *without* any cell output stored.
  It is recommended that notebooks are stored without cell output, so that functionality
  within them are run and tested to ensure continued compatibility with code changes.
  Cell output should only be stored in notebooks which are too computationally intensive
  for the Read the Docs server to handle, which has a limit of 15 minutes and 3 GB of
  memory per :doc:`documentation build <readthedocs:builds>`.
- We also use ``black`` to format notebooks cells, see the page on :ref:`code-style` for
  details.
  To prevent ``black`` from automatically formatting regions of your code, please wrap
  these code blocks with the following::

      # fmt: off
      python_code_block = not_to_be_formatted
      # fmt: on

  Please see the :doc:`black documentation <black:index>` for more details.
- Displaying interactive 3D plots with :doc:`PyVista <pyvista:index>` requires a Jupyter
  backend.
  We previously used ``panel``, which PyVista does not support anymore.
  Instead, they recommend using ``trame``, but this does not work with ``nbsphinx`` yet.
  Thus, the previously interactive 3D plots are now static.
  The Jupyter backend used by PyVista can be set in a notebook cell at the top of the
  notebook via ``pyvista.set_jupyter_backend("static")``.

In general, we run all notebooks every time the documentation is built with Sphinx, to
ensure that all notebooks are compatible with the current API at all times.
This is important!
For computationally expensive notebooks however, we store the cell outputs so the
documentation doesn't take too long to build, either by us locally or the Read The Docs
GitHub action.
To check that the notebooks with stored cell outputs are compatible with the current
API, we run a scheduled GitHub Action every Monday morning which checks that the
notebooks run OK and that they produce the same output now as when they were last
executed.
We use :doc:`nbval <nbval:index>` for this.

The tutorial notebooks can be run interactively in the browser with the help of Binder.
When creating a server from the kikuchipy source code, Binder installs the packages
listed in the environment.yml configuration file, which must include all doc
dependencies listed in setup.py necessary to run the notebooks.

Writing API reference
---------------------

Inherited attributes and methods are not listed in the API reference unless they are
explicitly coded in the inheriting class.
To see an example of this behavior, see the source code of
:class:`~kikuchipy.signals.EBSDMasterPattern`, which inherits attributes and methods
from a private class ``KikuchiMasterPattern``.
