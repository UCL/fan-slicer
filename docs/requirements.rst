.. highlight:: shell

.. _requirements:

===============================================
Requirements for fanslicer
===============================================

This is the software requirements file for fanslicer. The requirements listed below should define
what fanslicer does. Each requirement can be matched to a unit test that
checks whether the requirement is met.

Requirements
~~~~~~~~~~~~
+------------+--------------------------------------------------------+-------------------------------------+
|    ID      |  Description                                           |  Test                               |
+============+========================================================+=====================================+
|    0000    |  Module has a help page                                |  pylint, see                        |
|            |                                                        |  tests/pylint.rc and tox.ini        |
+------------+--------------------------------------------------------+-------------------------------------+
|    0001    |  Functions are documented                              |  pylint, see                        |
|            |                                                        |  tests/pylint.rc and tox.ini        |
+------------+--------------------------------------------------------+-------------------------------------+
|    0002    |  Package has a version number                          |  No test yet, handled by git.       |
+------------+--------------------------------------------------------+-------------------------------------+
|    0003    |  Creates binary volume from vtk surfaces               | test_voxelisation                   |
+------------+--------------------------------------------------------+-------------------------------------+
|    0004    |  Generates fan-shaped images from 3D binary models     | test_binary_image_sim               |
+------------+--------------------------------------------------------+-------------------------------------+
|    0005    |  Generates fan-shaped images from 3D volumes           | test_intensity_image_sim            |
+------------+--------------------------------------------------------+-------------------------------------+
|    0006    |  Generates rectangular images from 3D binary models    | test_linear_probe_sim               |
+------------+--------------------------------------------------------+-------------------------------------+
|    0007    |  Generates rectangular images from 3D volumes          | test_linear_probe_sim               |
+------------+--------------------------------------------------------+-------------------------------------+

