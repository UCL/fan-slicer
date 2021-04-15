SliceSampler
===============================

.. image:: https://weisslab.cs.ucl.ac.uk/UltrasoundNoTracker/slicesampler/raw/master/project-icon.png
   :height: 128px
   :width: 128px
   :target: https://weisslab.cs.ucl.ac.uk/UltrasoundNoTracker/slicesampler
   :alt: Logo

.. image:: https://weisslab.cs.ucl.ac.uk/UltrasoundNoTracker/slicesampler/badges/master/build.svg
   :target: https://weisslab.cs.ucl.ac.uk/UltrasoundNoTracker/slicesampler/pipelines
   :alt: GitLab-CI test status

.. image:: https://weisslab.cs.ucl.ac.uk/UltrasoundNoTracker/slicesampler/badges/master/coverage.svg
    :target: https://weisslab.cs.ucl.ac.uk/UltrasoundNoTracker/slicesampler/commits/master
    :alt: Test coverage

.. image:: https://readthedocs.org/projects/slicesampler/badge/?version=latest
    :target: http://slicesampler.readthedocs.io/en/latest/?badge=latest
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

    git clone https://weisslab.cs.ucl.ac.uk/UltrasoundNoTracker/slicesampler


Adding dependencies
^^^^^^^^^^^^^^^^^^^

Dependencies must be specified in requirements.txt, as this is used
by tox to automatically install the dependencies in a clean virtual
env in ```slicesampler/.tox```.


Running tests
^^^^^^^^^^^^^
Pytest is used for running unit tests, but it is run via tox:
::

    git clone https://weisslab.cs.ucl.ac.uk/UltrasoundNoTracker/slicesampler.git
    cd slicesampler
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

    pip install git+https://weisslab.cs.ucl.ac.uk/UltrasoundNoTracker/slicesampler



Contributing
^^^^^^^^^^^^

Please see the `contributing guidelines`_.


Useful links
^^^^^^^^^^^^

* `Source code repository`_


Licensing and copyright
-----------------------

Copyright 2020 University College London.
susi is released under the BSD-3 license. Please see the `license file`_ for details.


Acknowledgements
----------------

Supported by `Wellcome`_ and `EPSRC`_.


.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
.. _`source code repository`: https://weisslab.cs.ucl.ac.uk/UltrasoundNoTracker/slicesampler
.. _`scikit-surgery`: https://github.com/UCL/scikit-surgery/wiki
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://weisslab.cs.ucl.ac.uk/susi/susi/blob/master/CONTRIBUTING.rst
.. _`license file`: https://weisslab.cs.ucl.ac.uk/susi/susi/blob/master/LICENSE

