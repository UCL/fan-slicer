Fan-slicer
===============================

.. image:: https://github.com/jramalhinho/fan-slicer/raw/main/project-icon.jpg
   :height: 128px
   :target: https://github.com/UCL/fan-slicer
   :alt: Logo

.. image:: https://github.com/jramalhinho/fan-slicer/workflows/.github/workflows/ci.yml/badge.svg
   :target: https://github.com/UCL/fan-slicer/actions/
   :alt: GitLab-CI test status

.. image:: https://github.com/jramalhinho/fan-slicer/badges/main/coverage.svg
    :target: https://github.com/UCL/fan-slicer/commits/main
    :alt: Test coverage

Author: João Ramalhinho

Fan-slicer is a python and Pycuda package that enables the sampling of arbitrarily positioned 2D Ultrasound-shaped (fan)
planes from a 3D volume.
CUDA kernels are used with Pycuda to enable the fast sampling of multiple images in one run.
Fan-slicer samples images from both binary and non-binary volumes.
Additionally, slicing of rectangular planes (linear probe case) is also supported.
Fan-slicer is developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_ on top of the
`Python Template from Scikit-Surgery`_.

Using
^^^^^

A complete use case example is provided in *simulation_demo.py*.
This script contains code for the sampling of 10 evenly spaced fan-shaped planes from both 2 volumes:

* An abdominal contrast enhanced CT centered at the liver (a non-binary intensity volume).

* A vessel segmented volume of the same liver (binary volume) that is extracted from vessel tree vtk files.
Further details on the parameterisation of the planes, pose formulation, and usage of functions
are provided in the following `guide`_.

Cloning
^^^^^^^

You can clone the repository using the following command:

::

    git clone https://github.com/UCL/fan-slicer


Adding dependencies
^^^^^^^^^^^^^^^^^^^

Dependencies must be specified in requirements.txt, as this is used
by tox to automatically install the dependencies in a clean virtual
env in ```fanslicer/.tox```.


Running tests
^^^^^^^^^^^^^
Pytest is used for running unit tests, but it is run via tox:
::

    git clone https://github.com/UCL/fan-slicer.git
    cd fanslicer
    tox -e pycuda

and tox will install all dependencies then run pytest and pylint.


Linting
^^^^^^^
This code conforms to the PEP8 standard. Pylint is used to analyse the code,
but again, it is run via tox:

::

    tox -e lint


Documentation
^^^^^^^^^^^^^
Documentation is generated via Sphinx, but again ... you guessed it,
you run it via tox.

::

    tox -e docs


Installing
----------

You can pip install directly from the repository as follows:

::

    pip install git+https://github.com/jramalhinho/fan-slicer



Contributing
^^^^^^^^^^^^

Please see the `contributing guidelines`_.


Useful links
^^^^^^^^^^^^

* `Source code repository`_


Licensing and copyright
-----------------------

Copyright 2020 University College London.
Fan-slicer is released under the BSD-3 license. Please see the `license file`_ for details.


Acknowledgements
----------------

Supported by `Wellcome`_ and `EPSRC`_.


.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
.. _`Python Template from Scikit-Surgery`: https://github.com/SciKit-Surgery/PythonTemplate
.. _`source code repository`: https://github.com/UCL/fan-slicer
.. _`scikit-surgery`: https://github.com/UCL/scikit-surgery/wiki
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://github.com/UCL/fan-slicer/blob/master/CONTRIBUTING.rst
.. _`license file`: https://github.com/UCL/fan-slicer/blob/master/LICENSE
.. _`guide`: https://github.com/UCL/fan-slicer/blob/master/USING.rst
