.. Toolkit documentation master file, created by
   sphinx-quickstart on Thu Nov 23 22:04:00 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================================
Welcome to Toolkit's documentation!
===================================

This is the documentation for my data science Toolkit (`github repo <https://github.com/clementcome/toolkit>`_).

Quickstart
==========

Install the package with pip:

.. code-block:: bash

    pip install clementcome-toolkit

Import the package and use it:

.. code-block:: python

    import cc_tk

.. note::

    This project is still in development and only a few functionalities are available.
    Do not hesitate to contact me if you have any questions or suggestions.

.. toctree::
    :maxdepth: 2
    :caption: How-to guides

    how-to/demo_relationship
    how-to/demo_correlation
    how-to/plot

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/cc_tk

.. toctree::
    :maxdepth: 2
    :caption: Tutorial

    tutorial/classification

Reason behind this project
==========================

Its first pupose is to reduce boilerplate code and make my life easier when switching between projects.

On the side benefit, it also serves as a learning experience to develop a python package and document it on my own aligned with the best practices.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
