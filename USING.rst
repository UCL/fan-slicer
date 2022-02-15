USER GUIDE
===============================
Sections below describe how to use and parameterise the plane sampling.

Binary and Non-binary volumes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For **non-binary volumes**, instantiate an object of the class ``IntensityVolume`` in *pycuda_simulation/intensity_volume.py*

::

    IntensityVolume(config_dir, vol_dir, image_num, downsampling, file_type, npy_config)

* ``config_dir`` is a *.json* file with image parameters (described in the next section).
* ``vol_dir`` is the directory with the volume data to be sliced. Currently three different formats are supported and defined  by ``file_type``: *npy* for .npy, *nii* for .nii, *dicom* for DICOM.  An additional config file ``npy_config`` is needed for the ``npy`` option.
* ``image_num`` is the number of simulated images per run. For improved speed, should be kept the same across runs.
* ``downsampling`` is an integer for downsampling of the resulting 2D images.

For **binary volumes**, instantiate an object of the class ``SegmentedVolume`` in *pycuda_simulations/segmented_volume.py*

::

    SegmentedVolume(mesh_dir, config_dir, image_num, downsampling, voxel_size)

* ``config_dir``, ``image_num`` and ``downsampling`` are the same as for ``IntensityVolume``. However, ``config dir`` holds specific information on the surfaces to be intersected.
* ``mesh_dir`` is the directory with the surfaces from which a binary volume will be generated and sampled. Currently, only *.vtk* is supported. The user can adapt any other file as long as a mesh structure as the one defined in *pycuda_simulation/mesh.py* is generated.
* ``voxel_size`` is the isotropic voxel size of the generated binary volume. If *.npy* files with the same name as the *.vtk* surfaces do not exist on ``mesh_dir``, the method ``voxelise_mesh`` is used to generate the respective isotropic binary volume.

For **both volume objects**, (defined as ``volume``) image sampling is done by calling

::

        volume.simulate_image(pose, image_num, out_points)

* ``pose`` is a 4x4 matrix holding the plane poses (rotation and translation) that are to be sampled. Pose parameterisation is described below.
* ``image_num`` is the number of poses/images to be generated.
* ``out_points`` is a flag to decide whether the plane sampling 3D coordinates are extracted or not.
* For ``IntensityVolume``, an array of ``image_num`` grayscale images are returned.
* For ``SegmentedVolume``, two arrays are output: one with RGB images where colors are given to each of the intersected surface (colors defined in ``config_dir``), and one with grayscale images where each intersected surface has a different integer value.



Curvilinear (Fan) Plane parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: fan_shapes.png
   :width: 600

Fig. 1 - Left: Fan-shape parameters. Right: Plane pose parameters.

To control the fan shape images, three sets of parameters must be defined in ``config_dir``:

* ``fan_geometry``, set of 6 parameters defining the fan shape and sampling in 2D.

    * *dtheta*: resolution of angular sampling of the fan, in **degrees**.
    * *da*: the depth line resolution, in **millimeters**
    * *theta*: the total fan aperture, in **degrees**.
    * *a*: the total depth line length starting from the curvature centre *O*, in **pixels**.
    * *c*: the depth of the fan, i.e the length of the fan line, in **pixels**.
    * *b*: the offset between the image rectangle and the fan line origin (the transducer contact point), in **pixels**.

* ``image_dimensions``, 2 integers, the size of the rectangle where the fan image is displayed (*w* and *h*).
* ``pixel_dimensions``, 2 floats, the 2D resolution of the final image.

To define a new configuration, the user must calculate these parameters and adjust ``image_dimensions`` so that the fan lies neatly inside the simulated image.

4x4 matrices should be in agreement with the frame of reference in the right side of Fig. 1, as in the matrix below:

.. image:: pose_matrix.png
   :width: 150


Linear (Rectangle) Plane parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To simulate rectangular images, change the ``transducer_type`` from *curvilinear* to *linear*. In this case, only 2 sets of parameters are needed:

* ``image_dimensions``, 2 integers, the size of the rectangular image.
* ``pixel_dimensions``, 2 floats, the 2D resolution of the image.

The reference of these planes is similar to the curvilinear case, but *P* is **located at the top left corner of the plane**.


Additional examples
^^^^^^^^^^^^^^^^^^^

Examples of multiple configurations with curvilinear and linear arrays are used in the tests of *tests-pycuda/test_pycuda_simulations.py*


Further Usage
^^^^^^^^^^^^^

Current code version is compatible with Pytorch and Tensorflow CUDA allocation as long as the GPU Compute Exclusive Mode is disabled.
