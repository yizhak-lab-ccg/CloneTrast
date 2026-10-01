# Sphinx configuration for CloneTrast.

from __future__ import annotations

import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE.parent / "src"))

# -- Project information -----------------------------------------------------
GITHUB_USER = "yizhak-lab-ccg"
GITHUB_REPO = "CloneTrast"

def _read_version_from_pyproject() -> str:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    pyproject = HERE.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        return str(tomllib.load(f)["project"]["version"])


try:
    info = metadata("clonetrast")
    project_name = info.get("Name") or "CloneTrast"
    author = info.get("Author") or "CloneTrast authors"
    release = info.get("Version") or _read_version_from_pyproject()
except PackageNotFoundError:
    project_name = "CloneTrast"
    author = "CloneTrast authors"
    release = _read_version_from_pyproject()

project = str(project_name)
version = str(release)
release = str(release)
project_copyright = f"{datetime.now().year}, {author}"

repository_url = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}"

html_context = {
    "display_github": True,
    "github_user": GITHUB_USER,
    "github_repo": GITHUB_REPO,
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
if os.environ.get("READTHEDOCS") == "True":
    html_context["READTHEDOCS"] = True

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
]

autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_custom_sections = [("Params", "Parameters")]

autodoc_default_options = {
    "members": True,
    "member-order": "groupwise",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

napoleon_preprocess_types = True
napoleon_type_aliases = {
    "AnnData": ":class:`anndata.AnnData`",
    "Path": ":class:`pathlib.Path`",
    "NDArray": ":class:`numpy.typing.NDArray`",
    "np.ndarray": ":class:`numpy.ndarray`",
    "torch.Tensor": ":class:`torch.Tensor`",
}

myst_heading_anchors = 6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")

# myst-nb: render notebooks with existing outputs (do not re-execute on build/RTD)
nb_execution_show_tb = True
nb_execution_raise_on_error = False
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
nb_execution_timeout = 60

if os.environ.get("READTHEDOCS") == "True":
    nb_execution_mode = "off"
    nb_execution_timeout = 120
    nb_execution_allow_errors = True

suppress_warnings = [
    "mystnb.unknown_mime_type",
    "duplicate_object",
]

typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".ipynb": "myst-nb",
}

intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Open Graph (canonical URL on Read the Docs when set)
ogp_site_url = html_baseurl or "https://clonetrast.readthedocs.io/en/latest/"

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/images/CloneTrast_logo_for_git.png"

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
    "logo": {
        "image_light": "_static/images/CloneTrast_logo_for_git.png",
        "image_dark": "_static/images/CloneTrast_logo_for_git.png",
    },
    "show_navbar_depth": 1,
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
}
