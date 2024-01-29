# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TEflow'
copyright = '2023-2024 Jianbo ZHU'
author = 'Jianbo ZHU'

# sphinx-apidoc -e --tocfile index -o api_doc/ ../../src/teflow/

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'zh_CN'
locale_dirs = ['locale/']
gettext_compact = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# alabaster, classic, sphinxdoc, scrolls, agogo, traditional, nature, haiku, pyramid, bizstyle
html_theme = 'sphinxdoc'
html_static_path = ['_static']

autoclass_content = 'both'

latex_elements = {
    'extraclassoptions': 'openany',
    'preamble': r'''
       \usepackage{indentfirst}
       \setlength{\parindent}{2em}
    ''',
    'printindex': r'\def\twocolumn[#1]{#1}\printindex',
}
