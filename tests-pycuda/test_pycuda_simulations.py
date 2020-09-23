# coding=utf-8

"""Tests on pycuda simulation of 2D fan shaped images"""

import os
import numpy as np
import matplotlib.pyplot as plt
import susi.pycuda_simulation.mesh as mesh
import susi.pycuda_simulation.segmented_volume as svol
import susi.pycuda_simulation.intensity_volume as ivol

# These tests access the data in data_LUS and data_EUS
# used in the demo scripts


def test_voxelisation():
    """
    Test to check binary model generation from
    mesh
    """
    voxel_size = 0.5
    mesh_dir = "data/data_LUS/hepatic veins.vtk"

    # Load the test mesh (hepatic vein)
    test_mesh = mesh.load_mesh_from_vtk(mesh_dir)

    # Remove saved binary model, in case it exists
    if os.path.isfile("tests-pycuda/data/binary_map.npy"):
        os.remove("tests-pycuda/data/binary_map.npy")

    # Voxelise mesh
    volume = svol.voxelise_mesh(test_mesh,
                                voxel_size,
                                margin=[20, 20, 20],
                                save_dir="tests-pycuda/data/")

    test_volume = np.load("tests-pycuda/data/binary_map_hepatic_veins.npy")
    # Check if binary model is the same
    np.testing.assert_array_equal(test_volume, volume)


def test_binary_image_sim():
    """
    Test binary image simulation
    THIS TEST NEEDS PYCUDA
    """
    # Set config
    config_dir = "config/models_binary_LUS_config.json"
    mesh_dir = "data/data_LUS/"
    # Load the segmented volume
    liver_volume = svol.SegmentedVolume(config_dir=config_dir,
                                        mesh_dir=mesh_dir,
                                        voxel_size=0.5)
    # Define test poses
    pose1 = np.array([[-0.46,	0.29, -0.84	, -41.77],
                      [0.28, 0.94, 0.18, -47.72],
                      [0.84, -0.15, -0.52, -253.44],
                      [0, 0, 0, 1]])
    pose2 = np.array([[-0.46,	0.29, -0.84	, -41.77],
                      [0.28, 0.94, 0.18, -47.72],
                      [0.84, -0.15, -0.52, -273.24],
                      [0, 0, 0, 1]])
    test_poses = np.hstack((pose1, pose2))

    # Load expected image (an extra channel is added, be careful to remove)
    image1 = plt.imread('tests-pycuda/data/binary_liver_image_0.png', format='png')
    image2 = plt.imread('tests-pycuda/data/binary_liver_image_1.png', format='png')

    # Simulate images
    _, _, colored_map = \
        liver_volume.simulate_image(poses=test_poses,
                                    downsampling=2,
                                    image_num=2)

    # Assert the two images
    # Due to numerical error in saving images,
    # accept tolerance of 10^-7
    test_image1 = colored_map[:, :, :, 0]
    np.testing.assert_array_almost_equal(test_image1,
                                         image1[:, :, 0:3],
                                         decimal=7)
    test_image2 = colored_map[:, :, :, 1]
    np.testing.assert_array_almost_equal(test_image2,
                                         image2[:, :, 0:3],
                                         decimal=7)


def test_intensity_image_sim():
    """
    Test intensity image simulation
    THIS TEST NEEDS PYCUDA
    """
    # Set config
    config_dir = "config/models_intensity_LUS_config.json"
    vol_dir = "data/data_LUS/CT_Dicom/000/"
    # Load the segmented volume
    ct_volume = ivol.IntensityVolume(config_dir=config_dir,
                                     vol_dir=vol_dir,
                                     file_type="dicom")

    # Define test poses
    pose1 = np.array([[-0.46,	0.29, -0.84	, -41.77],
                      [0.28, 0.94, 0.18, -47.72],
                      [0.84, -0.15, -0.52, -253.44],
                      [0, 0, 0, 1]])
    pose2 = np.array([[-0.46,	0.29, -0.84	, -41.77],
                      [0.28, 0.94, 0.18, -47.72],
                      [0.84, -0.15, -0.52, -273.24],
                      [0, 0, 0, 1]])
    test_poses = np.hstack((pose1, pose2))

    # Load expected image (as array, as the bounds are not
    # between 0 and 255 or 0 and 1)
    image1 = np.load('tests-pycuda/data/intensity_liver_image_0.npy')
    image2 = np.load('tests-pycuda/data/intensity_liver_image_1.npy')

    # Simulate images
    _, ct_map = \
        ct_volume.simulate_image(poses=test_poses,
                                 downsampling=2,
                                 image_num=2)

    # Assert the two images
    # Due to numerical error in saving images,
    # accept tolerance of 10^-3
    test_image1 = ct_map[:, :, 0]
    np.testing.assert_array_almost_equal(test_image1,
                                         image1,
                                         decimal=3)
    test_image2 = ct_map[:, :, 1]
    np.testing.assert_array_almost_equal(test_image2,
                                         image2,
                                         decimal=3)



