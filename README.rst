Fan-slicer
===============================

.. image:: https://github.com/jramalhinho/fan-slicer/raw/main/project-icon.jpg
   :height: 128px
   :target: https://github.com/UCL/fan-slicer
   :alt: Logo

.. image:: https://github.com/jramalhinho/fan-slicer/workflows/.github/workflows/ci.yml/badge.svg
   :target: https://github.com/UCL/fan-slicer/actions/
   :alt: GitLab-CI test status

.. image:: https://img.shields.io/badge/DOI-10.5334%2Fjors.422-blue
    :target: http://doi.org/10.5334/jors.422
    :alt: The Jors Paper


Author: João Ramalhinho

Fan-slicer is a python and Pycuda package that enables the sampling of arbitrarily positioned 2D Ultrasound-shaped (fan)
planes from a 3D volume.
CUDA kernels are used with Pycuda to enable the fast sampling of multiple images in one run.
Fan-slicer samples images from both binary and non-binary volumes.
Additionally, slicing of rectangular planes (linear probe case) is also supported.
Fan-slicer is developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_ on top of the
`Python Template from Scikit-Surgery`_.

Installing from Github
^^^^^^^^^^^^^^^^^^^^^^

You can clone the repository using the following command:

::

    git clone https://github.com/UCL/fan-slicer.git

Install dependencies preferably in a clean virtual environment by using the following commands:

::

    cd fan-slicer
    pip install -r requirements-pycuda.txt

To run tests, use the following command:

::

     python -m pytest -v -s ./tests-pycuda


Installing with pip
^^^^^^^^^^

Alternatively, you can also pip install directly from the repository:

::

    pip install git+https://github.com/UCL/fan-slicer


Tested environments
^^^^^^^^^^^^^^^^^^^

Operating systems

* Ubuntu 18.04.5 LTS , with `CUDA Toolkit 10.1`_, gcc 7.5.0 as C++ compiler.

* Windows 10, with `CUDA Toolkit 11.3`_ and `Visual Studio 2019`_ for C++ compiler.

* Windows 10/11 with Windows Subsystem Linux (WSL2), using the following commands:

    1. Install `CUDA on WSL`_
    2. Install required libraries:

::

    sudo apt-get install build-essential gcc libboost-all.dev

Tested on python 3.6, 3.7, 3.8.

Using
^^^^^

A complete use case example is provided in *simulation_demo.py*.
This script contains code for the sampling of 10 evenly spaced fan-shaped planes from two volumes:

* An abdominal contrast enhanced CT centered at the liver (a non-binary intensity volume).

* A vessel segmented volume of the same liver (binary volume) that is extracted from vessel tree vtk files.
Further details on the parameterisation of the planes, pose formulation, and usage of functions
are provided in the following `guide`_.

Citing
^^^^^^

If you use this software in your research, please cite:

Ramalhinho, J., Dowrick, T., Bonmati, E., Clarkson, M. J., 2023.
Fan-Slicer: A Pycuda Package for Fast Reslicing of Ultrasound Shaped Planes "
Journal of Open Research Software 11(1), p.3 DOI: http://doi.org/10.5334/jors.422

Contributing
^^^^^^^^^^^^

Please see the `contributing guidelines`_.




Licensing and copyright
-----------------------

Copyright 2022 University College London.
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
.. _`Visual Studio 2019`: https://learn.microsoft.com/en-us/visualstudio/releases/2019/release-notes
.. _`CUDA Toolkit 11.3`: https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
.. _`CUDA Toolkit 10.1`: https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal
.. _`CUDA on WSL`: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
