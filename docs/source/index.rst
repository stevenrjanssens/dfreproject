.. AstroReproject documentation master file

Welcome to reprojection's documentation!
========================================

Reprojection is a Python package for reprojecting astronomical images. The code runs using torch in order to speed up calculations.

The idea behind this package was to make a stripped down version of the `reproject` package affiliated with astropy in order to reduce computational time.
We achieve approximately 20X faster computations with this package using the GPU and 10X using the CPU for images taken by the Dragonfly Telephoto Array. Take a look at the demos for an example.
We note that the only projection we currently support is the Tangential Gnomonic projection which is the most popular in optical astronomy.
If you have need for this package to work with another projection geometry, please open a GitHub ticket.

For the sake of transparency of the calculations `dfreproject` performs, we've written out the computations at each step in the `methodology section <methodology.rst>`_ file.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   methodology
   examples


Note: Claude.ai was used in preparing the docstrings for the functions.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
