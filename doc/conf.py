# Configuration file for the Sphinx documentation app.
# See the documentation for a full list of configuration options:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime
import inspect
import os
from os.path import relpath, dirname
import re
import sys

import pyvista
from numpydoc.docscrape_sphinx import SphinxDocString

import kikuchipy as kp


# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.append("../")

# Project information
project = "kikuchipy"
copyright = f"2019-{datetime.now().year}, {kp.release.author}"
author = kp.release.author
version = kp.release.version
release = kp.release.version

master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

# Create links to references within kikuchipy's documentation to these
# packages
intersphinx_mapping = {
    # Package
    "dask": ("https://docs.dask.org/en/stable", None),
    "diffpy.structure": ("https://www.diffpy.org/diffpy.structure", None),
    "diffsims": ("https://diffsims.readthedocs.io/en/latest", None),
    "hyperspy": ("https://hyperspy.org/hyperspy-doc/current", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "imageio": ("https://imageio.readthedocs.io/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numba": ("https://numba.pydata.org/numba-doc/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "orix": ("https://orix.readthedocs.io/en/stable", None),
    "pooch": ("https://www.fatiando.org/pooch/latest", None),
    "pyebsdindex": ("https://pyebsdindex.readthedocs.io/en/stable", None),
    "python": ("https://docs.python.org/3", None),
    "pyvista": ("https://docs.pyvista.org", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "skimage": ("https://scikit-image.org/docs/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    # Docs
    "black": ("https://black.readthedocs.io/en/stable", None),
    "conda": ("https://conda.io/projects/conda/en/latest", None),
    "defdap": ("https://defdap.readthedocs.io/en/latest", None),
    "nbsphinx": ("https://nbsphinx.readthedocs.io/en/latest", None),
    "nbval": ("https://nbval.readthedocs.io/en/latest", None),
    "numpydoc": ("https://numpydoc.readthedocs.io/en/latest", None),
    "pythreejs": ("https://pythreejs.readthedocs.io/en/stable", None),
    "pyxem": ("https://pyxem.readthedocs.io/en/latest", None),
    "readthedocs": ("https://docs.readthedocs.io/en/stable", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "sphinx-gallery": ("https://sphinx-gallery.github.io/stable", None),
    "xcdskd": ("https://xcdskd.readthedocs.io/en/latest", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files. This image also
# affects html_static_path and html_extra_path.
exclude_patterns = ["build", "_static/logo/*.ipynb"]

# HTML theming: pydata-sphinx-theme
# https://pydata-sphinx-theme.readthedocs.io
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/pyxem/kikuchipy",
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "Gitter",
            "url": "https://gitter.im/pyxem/kikuchipy",
            "icon": "fab fa-gitter",
        },
    ],
    "logo": {
        "alt_text": "kikuchipy",
        "image_dark": "logo/plasma_banner_dark.png",
    },
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "pyxem",
    "github_repo": "kikuchipy",
    "github_version": "develop",
    "doc_path": "doc",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Logo
html_logo = "_static/logo/plasma_banner.png"
html_favicon = "_static/logo/plasma_favicon.png"

# nbsphinx
# --------
# https://nbsphinx.readthedocs.io
nbsphinx_execute = "auto"  # always, auto, never
nbsphinx_allow_errors = True
nbsphinx_execute_arguments = [
    "--InlineBackend.rc=figure.facecolor='w'",
]
# Taken from nbsphinx' own nbsphinx configuration file, with slight
# modifications to point nbviewer and Binder to the GitHub develop
# branch links when the documentation is launched from a kikuchipy
# version with "dev" in the version
if "dev" in version:
    release_version = "develop"
else:
    release_version = "v" + version
nbsphinx_prolog = (
    r"""
{% set docname = 'doc/' + env.doc2path(env.docname, base=None)[:-6] + '.ipynb' %}

.. admonition:: Live notebook

    You can run this notebook in a `live session <https://mybinder.org/v2/gh/pyxem/kikuchipy/"""
    + release_version
    + r"""?filepath={{ docname|e }}>`__ |Binder| or view it `on Github <https://github.com/pyxem/kikuchipy/blob/"""
    + release_version
    + r"""/{{ docname|e }}>`__.

.. |Binder| image:: https://static.mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pyxem/kikuchipy/"""
    + release_version
    + r"""?filepath={{ docname|e }}
"""
)

# Whether to show all warnings when building the documentation
nitpicky = False

# sphinxcontrib-bibtex
# --------------------
# https://sphinxcontrib-bibtex.readthedocs.io
bibtex_bibfiles = ["user/bibliography.bib"]
bibtex_reference_style = "author_year"

# sphinx-codeautolink
# -------------------
codeautolink_custom_blocks = {
    "python3": None,
    "pycon3": "sphinx_codeautolink.clean_pycon",
}

# Relevant for the PDF build with LaTeX
latex_elements = {
    # pdflatex doesn't like some Unicode characters, so a replacement
    # for one of them is made here
    "preamble": r"\DeclareUnicodeCharacter{2588}{-}"
}


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object.
    This is taken from SciPy's conf.py:
    https://github.com/scipy/scipy/blob/develop/doc/source/conf.py.
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    startdir = os.path.abspath(os.path.join(dirname(kp.__file__), ".."))
    fn = relpath(fn, start=startdir).replace(os.path.sep, "/")

    if fn.startswith("kikuchipy/"):
        m = re.match(r"^.*dev0\+([a-f0-9]+)$", version)
        pre_link = "https://github.com/pyxem/kikuchipy/blob/"
        if m:
            return pre_link + "%s/%s%s" % (m.group(1), fn, linespec)
        elif "dev" in version:
            return pre_link + "develop/%s%s" % (fn, linespec)
        else:
            return pre_link + "v%s/%s%s" % (version, fn, linespec)
    else:
        return None


# PyVista
# -------
# https://docs.pyvista.org
pyvista.global_theme.window_size = [600, 600]
pyvista.set_jupyter_backend("pythreejs")

# -- Copy button customization (taken from PyVista)
# Exclude traditional Python prompts from the copied code
copybutton_prompt_text = r">>> ?|\.\.\. "
copybutton_prompt_is_regexp = True

# sphinx.ext.autodoc
# ------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autosummary_ignore_module_all = False
autosummary_imported_members = True
autodoc_typehints_format = "short"
autodoc_default_options = {
    "show-inheritance": True,
}

# numpydoc
# --------
# https://numpydoc.readthedocs.io
numpydoc_show_class_members = False
numpydoc_use_plots = True
numpydoc_xref_param_type = True
# fmt: off
numpydoc_validation_checks = {
    "all",   # All but the following:
    "ES01",  # Not all docstrings need an extend summary
    "EX01",  # Examples: Will eventually enforce
    "GL01",  # Contradicts numpydoc examples
    "GL02",  # Appears to be broken?
    "GL07",  # Appears to be broken?
    "GL08",  # Methods can be documented in super class
    "PR01",  # Parameters can be documented in super class
    "PR02",  # Properties with setters might have docstrings w/"Returns"
    "PR04",  # Doesn't seem to work with type hints?
    "RT01",  # Abstract classes might not have return sections
    "SA01",  # Not all docstrings need a "See Also"
    "SA04",  # "See Also" section does not need descriptions
    "SS06",  # Not possible to make all summaries one line
    "YD01",  # Yields: No plan to enforce
}
# fmt: on

# matplotlib.sphinxext.plot_directive
# -----------------------------------
# https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html
plot_formats = ["png"]
plot_html_show_source_link = False
plot_html_show_formats = False
plot_include_source = True


def _str_examples(self):
    examples_str = "\n".join(self["Examples"])
    if (
        self.use_plots
        and (
            re.search(r"\b(.plot)\b", examples_str)
            or re.search(r"\b(.imshow)\b", examples_str)
        )
        and "plot::" not in examples_str
    ):
        out = []
        out += self._str_header("Examples")
        out += [".. plot::", ""]
        out += self._str_indent(self["Examples"])
        out += [""]
        return out
    else:
        return self._str_section("Examples")


SphinxDocString._str_examples = _str_examples

# Sphinx-Gallery
# --------------
# https://sphinx-gallery.github.io
sphinx_gallery_conf = {
    "backreferences_dir": "reference/generated",
    "doc_module": ("kikuchipy",),
    "examples_dirs": "../examples",
    "filename_pattern": "^((?!sgskip).)*$",
    "gallery_dirs": "examples",
    "reference_url": {"kikuchipy": None},
    "run_stale_examples": False,
    "show_memory": True,
}
autosummary_generate = True

# Download example datasets prior to building the docs
print("[kikuchipy] Downloading example datasets (if not found in cache)")
_ = kp.data.nickel_ebsd_large(allow_download=True)
_ = kp.data.silicon_ebsd_moving_screen_in(allow_download=True)
