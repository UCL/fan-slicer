susi
===============================

.. image:: https://weisslab.cs.ucl.ac.uk/susi/susi/raw/master/project-icon.png
   :height: 128px
   :width: 128px
   :target: https://weisslab.cs.ucl.ac.uk/susi/susi
   :alt: Logo

.. image:: https://weisslab.cs.ucl.ac.uk/susi/susi/badges/master/build.svg
   :target: https://weisslab.cs.ucl.ac.uk/susi/susi/pipelines
   :alt: GitLab-CI test status

.. image:: https://weisslab.cs.ucl.ac.uk/susi/susi/badges/master/coverage.svg
    :target: https://weisslab.cs.ucl.ac.uk/susi/susi/commits/master
    :alt: Test coverage

.. image:: https://readthedocs.org/projects/susi/badge/?version=latest
    :target: http://susi.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status



Author: Ester Bonmati

susi is part of the `scikit-surgery`_ software project, developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_, part of `University College London (UCL)`_.

susi supports Python 2.7 and Python 3.6.

susi is currently a demo project, which will add/multiply two numbers. Example usage:

::

    python susi.py 5 8
    python susi.py 3 6 --multiply

Please explore the project structure, and implement your own functionality.

Developing
----------

Cloning
^^^^^^^

You can clone the repository using the following command:

::

    git clone https://weisslab.cs.ucl.ac.uk/susi/susi


Running tests
^^^^^^^^^^^^^
Pytest is used for running unit tests:
::

    pip install pytest
    python -m pytest


Linting
^^^^^^^

This code conforms to the PEP8 standard. Pylint can be used to analyse the code:

::

    pip install pylint
    pylint --rcfile=tests/pylintrc susi


Installing
----------

You can pip install directly from the repository as follows:

::

    pip install git+https://weisslab.cs.ucl.ac.uk/susi/susi



Contributing
^^^^^^^^^^^^

Please see the `contributing guidelines`_.


Useful links
^^^^^^^^^^^^

* `Source code repository`_
* `Documentation`_


Licensing and copyright
-----------------------

Copyright 2020 University College London.
susi is released under the BSD-3 license. Please see the `license file`_ for details.


Acknowledgements
----------------

Supported by `Wellcome`_ and `EPSRC`_.


.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
.. _`source code repository`: https://weisslab.cs.ucl.ac.uk/susi/susi
.. _`Documentation`: https://susi.readthedocs.io
.. _`scikit-surgery`: https://github.com/UCL/scikit-surgery/wiki
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://weisslab.cs.ucl.ac.uk/susi/susi/blob/master/CONTRIBUTING.rst
.. _`license file`: https://weisslab.cs.ucl.ac.uk/susi/susi/blob/master/LICENSE

