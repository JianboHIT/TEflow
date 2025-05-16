# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from teflow._version import __version__

project = 'TEflow'
copyright = '2023-2024 Jianbo ZHU'
author = 'Jianbo ZHU'
version = '.'.join(__version__.split('.')[:2])
release = __version__

# sphinx-apidoc -e --tocfile index -o api_doc/ ../../src/teflow/

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'matplotlib.sphinxext.plot_directive',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'zh_CN'
locale_dirs = ['locale/']
gettext_compact = False

# -- Options for HTML and Latex output ----------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_static_path = ['_static']

# # alabaster, classic, sphinxdoc, scrolls, agogo, traditional, nature, haiku, pyramid, bizstyle
# html_theme = 'sphinxdoc'
# autoclass_content = 'both'

extensions.append("sphinx_wagtail_theme")
html_theme = 'sphinx_wagtail_theme'
html_theme_options = {
    "project_name": f"Welcome to {project}'s documentation!",
    "github_url": "https://github.com/JianboHIT/TEflow/blob/master/docs/",
    "logo_height": 10,
    "logo_width": 10,
    "header_links": ", ".join([
        "Github|https://github.com/JianboHIT/TEflow",
        "Gitee|https://gitee.com/joulehit/TEflow",
        "PyPI|https://pypi.org/project/TEflow",
    ]),
    "footer_links": "/",
}
html_show_sourcelink = False

latex_elements = {
    'extraclassoptions': 'openany',
    'preamble': r'''
       \usepackage{indentfirst}
       \setlength{\parindent}{2em}
    ''',
    'printindex': r'\def\twocolumn[#1]{#1}\printindex',
}

# -- Options for Matplotlib plot ---------------------------------------------
plot_formats = [('png', 300), ('hires.png', 600), ('pdf', 300)]
plot_html_show_formats = False
plot_html_show_source_link = False
plot_rcparams = {
    # Figure settings
    'figure.figsize': (4, 3),

    # Subplot parameters
    'figure.subplot.left': 0.15,
    'figure.subplot.right': 0.85,
    'figure.subplot.bottom': 0.15,
    'figure.subplot.top': 0.90,
    'figure.subplot.wspace': 0.4,
    'figure.subplot.hspace': 0.4,

    # Font settings
    'font.family': 'Arial',
    'font.size': 9,

    # Math text settings
    'mathtext.fontset': 'cm',

    # Tick settings
    'xtick.direction': 'in',
    'xtick.labelsize': 'medium',
    'xtick.major.width': 1,
    'xtick.major.size': 4,
    'xtick.minor.width': 1,
    'xtick.minor.size': 2.5,
    'xtick.minor.visible': True,
    'xtick.minor.ndivs': 2,

    'ytick.direction': 'in',
    'ytick.labelsize': 'medium',
    'ytick.major.width': 1,
    'ytick.major.size': 4,
    'ytick.minor.width': 1,
    'ytick.minor.size': 2.5,
    'ytick.minor.visible': True,
    'ytick.minor.ndivs': 2,

    # Axes settings
    'axes.linewidth': 1.5,
    'axes.labelsize': 'large',

    # Legend settings
    'legend.fancybox': False,
    'legend.fontsize': 'medium',
    'legend.borderpad': 0.2,
}
