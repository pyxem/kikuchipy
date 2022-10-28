.. _code-style:

Code style
==========

The code making up kikuchipy is formatted closely following the `Style Guide for Python
Code <https://peps.python.org/pep-0008/>`__ with `The Black Code style
<https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>`__. We
use `pre-commit <https://pre-commit.com>`__ to run ``black`` automatically prior to each
local commit. Please install it in your environment::

    pre-commit install

Next time you commit some code, your code will be formatted inplace according to
``black``.

``black`` can format Jupyter notebooks as well. Code lines in tutorial notebooks should
be limited to 80 characters to display nicely in the documentation::

    black -l 80 <your_nice_notebook>.ipynb

Note that ``black`` won't format `docstrings
<https://peps.python.org/pep-0257/>`__. We follow the `numpydoc
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`__ standard
(with some exceptions), and the docstrings are checked against this standard when
building the documentation.

Comment lines should preferably be limited to 72 characters.

Package imports should be structured into three blocks with blank lines between them
(descending order): standard library (like ``os`` and ``typing``), third party packages
(like ``numpy`` and ``hyperspy``) and finally kikuchipy imports.

We use type hints in the function definition without type duplication in the function
docstring, for example::

    def my_function(a: int, b: Optional[bool] = None) -> Tuple[float, np.ndarray]:
        """This is a new function.

        Parameters
        ----------
        a
            Explanation of ``a``.
        b
            Explanation of flag ``b``. Default is ``None``.

        Returns
        -------
        values
            Explanation of returned values.
        """

We import modules lazily using the specification in `PEP 562
<https://peps.python.org/pep-0562/>`__.