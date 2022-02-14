Fan-slicer
===============================

.. image:: https://github.com/jramalhinho/fan-slicer/raw/main/project-icon.jpg
   :height: 128px
   :target: https://github.com/jramalhinho/fan-slicer
   :alt: Logo

.. image:: https://github.com/jramalhinho/fan-slicer/workflows/.github/workflows/ci.yml/badge.svg
   :target: https://github.com/jramalhinho/fan-slicer/actions/
   :alt: GitLab-CI test status

.. image:: https://github.com/jramalhinho/fan-slicer/badges/main/coverage.svg
    :target: https://github.com/jramalhinho/fan-slicer/commits/main
    :alt: Test coverage

.. image:: https://readthedocs.org/projects/fan-slicer/badge/?version=latest
    :target: http://fan-slicer.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status



Author: João Ramalhinho

SliceSampler is a python and CUDA package used for the sampling of 2D fan-shaped slices from a 3D volume.
SliceSampler is developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_.

Developing
----------

Cloning
^^^^^^^

You can clone the repository using the following command:

::

    git clone https://github.com/jramalhinho/fan-slicer


Adding dependencies
^^^^^^^^^^^^^^^^^^^

Dependencies must be specified in requirements.txt, as this is used
by tox to automatically install the dependencies in a clean virtual
env in ```slicesampler/.tox```.


Running tests
^^^^^^^^^^^^^
Pytest is used for running unit tests, but it is run via tox:
::

    git clone https://github.com/jramalhinho/fan-slicer.git
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
.. _`source code repository`: https://github.com/jramalhinho/fan-slicer
.. _`scikit-surgery`: https://github.com/UCL/scikit-surgery/wiki
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://github.com/jramalhinho/fan-slicer/blob/master/CONTRIBUTING.rst
.. _`license file`: https://github.com/jramalhinho/fan-slicer/blob/master/LICENSE

