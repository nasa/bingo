# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
os.environ["PYTHONPATH"] = os.path.abspath('../..')

import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'Bingo'
copyright = '2022, United States Government as represented by the ' \
            'Administrator of the National Aeronautics and Space Administration'
author = 'Geoffrey Bomarito'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_panels',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'nbsphinx',
    'nbsphinx_link'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = '_static/transparent_logo.png'
# html_favicon = '_static/transparent_logo.png'

html_context = {
    'default_mode': 'light'
}

html_theme_options = {
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/nasa/bingo',
            'icon': 'fab fa-github-square',
            'type': 'fontawesome'
        }
    ],
    'navbar_end': ['navbar-icon-links'],
    'footer_items': ['copyright', 'sphinx-version']
}

autosummary_generate = True

# prevent sphinx-panels from loading boostrap since pydata-sphinx-theme
# will do it
panels_add_bootstrap_css = False
