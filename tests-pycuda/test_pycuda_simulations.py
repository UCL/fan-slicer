# coding=utf-8

"""Tests on pycuda simulation of 2D fan shaped images"""

import os
import numpy as np
import matplotlib.pyplot as plt
import fanslicer.pycuda_simulation.segmented_volume as svol
import fanslicer.pycuda_simulation.intensity_volume as ivol
from fanslicer.pycuda_simulation import mesh

# These tests access the data used in the demo script

def test_voxelisation():
    """
    Test to check binary model generation from
    mesh
    """
    voxel_size = 0.5
    mesh_dir = "data/hepatic veins.vtk"

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
    config_dir1 = "tests-pycuda/config/binary_config_1.json"
    config_dir2 = "tests-pycuda/config/binary_config_2.json"
    mesh_dir = "data/"

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

    # Load expected images (an extra channel is added, be careful to remove)
    image1 = plt.imread('tests-pycuda/data/binary_liver_image_0.png', format='png')
    image2 = plt.imread('tests-pycuda/data/binary_liver_image_1.png', format='png')
    image3 = plt.imread('tests-pycuda/data/binary_liver_image_2.png', format='png')
    image4 = plt.imread('tests-pycuda/data/binary_liver_image_3.png', format='png')

    # Load the segmented volume
    liver_volume = svol.SegmentedVolume(config_dir=config_dir1,
                                        mesh_dir=mesh_dir,
                                        voxel_size=0.5,
                                        downsampling=2,
                                        image_num=2)
    # Simulate images
    _, _, colored_map1 = \
        liver_volume.simulate_image(poses=test_poses,
                                    image_num=2)

    # Load the segmented volume
    liver_volume = svol.SegmentedVolume(config_dir=config_dir2,
                                        mesh_dir=mesh_dir,
                                        voxel_size=0.5,
                                        downsampling=2,
                                        image_num=2)
    # Simulate images
    _, _, colored_map2 = \
        liver_volume.simulate_image(poses=test_poses,
                                    image_num=2)

    # Assert the four images
    # Due to numerical error in saving images,
    # accept tolerance of 10^-7
    test_image1 = colored_map1[:, :, :, 0]
    assert_curvilinear_image(test_image1,
                             image1[:, :, 0:3],
                             ratio_tolerance=0.01,
                             decimal=7)
    test_image2 = colored_map1[:, :, :, 1]
    assert_curvilinear_image(test_image2,
                             image2[:, :, 0:3],
                             ratio_tolerance=0.01,
                             decimal=7)
    test_image3 = colored_map2[:, :, :, 0]
    assert_curvilinear_image(test_image3,
                             image3[:, :, 0:3],
                             ratio_tolerance=0.01,
                             decimal=7)
    test_image4 = colored_map2[:, :, :, 1]
    assert_curvilinear_image(test_image4,
                             image4[:, :, 0:3],
                             ratio_tolerance=0.01,
                             decimal=7)


def test_intensity_image_sim():
    """
    Test intensity image simulation
    THIS TEST NEEDS PYCUDA
    """
    # Set two possible configs
    config_dir1 = "tests-pycuda/config/intensity_config_1.json"
    config_dir2 = "tests-pycuda/config/intensity_config_2.json"
    vol_dir = "data/CT_test_volume.npy"

    # Define test poses
    pose1 = np.array([[-0.46, 0.29, -0.84, -41.77],
                      [0.28, 0.94, 0.18, -47.72],
                      [0.84, -0.15, -0.52, -253.44],
                      [0, 0, 0, 1]])
    pose2 = np.array([[-0.46, 0.29, -0.84, -41.77],
                      [0.28, 0.94, 0.18, -47.72],
                      [0.84, -0.15, -0.52, -273.24],
                      [0, 0, 0, 1]])
    test_poses = np.hstack((pose1, pose2))

    # Load expected images (as array, as the bounds are not
    # between 0 and 255 or 0 and 1)
    image1 = np.load('tests-pycuda/data/intensity_liver_image_0.npy')
    image2 = np.load('tests-pycuda/data/intensity_liver_image_1.npy')
    image3 = np.load('tests-pycuda/data/intensity_liver_image_2.npy')
    image4 = np.load('tests-pycuda/data/intensity_liver_image_3.npy')

    # Test first config, load the volume
    ct_volume = ivol.IntensityVolume(config_dir=config_dir1,
                                     vol_dir=vol_dir,
                                     file_type="npy",
                                     npy_config="tests-pycuda/config/CT_npy_test_config.json",
                                     downsampling=2)
    # Simulate images with first config
    _, ct_map = \
        ct_volume.simulate_image(poses=test_poses,
                                 image_num=2)
    # Repeat with second config
    ct_volume = ivol.IntensityVolume(config_dir=config_dir2,
                                     vol_dir=vol_dir,
                                     file_type="npy",
                                     npy_config="tests-pycuda/config/CT_npy_test_config.json",
                                     downsampling=2)
    # Simulate images with first config
    _, ct_map2 = \
        ct_volume.simulate_image(poses=test_poses,
                                 image_num=2)

    # Assert the four resulting images
    # Due to numerical error in saving images,
    # accept tolerance of 10^-3
    test_image1 = ct_map[:, :, 0]
    assert_curvilinear_image(test_image1,
                             image1,
                             ratio_tolerance=0.01,
                             decimal=3)
    test_image2 = ct_map[:, :, 1]
    assert_curvilinear_image(test_image2,
                             image2,
                             ratio_tolerance=0.01,
                             decimal=3)
    test_image3 = ct_map2[:, :, 0]
    assert_curvilinear_image(test_image3,
                             image3,
                             ratio_tolerance=0.01,
                             decimal=3)
    test_image4 = ct_map2[:, :, 1]
    assert_curvilinear_image(test_image4,
                             image4,
                             ratio_tolerance=0.01,
                             decimal=3)


def test_linear_probe_sim():
    """
    Test linear image simulation, with both intensity and
    segmented models
    """
    # Set two possible configs
    config_dir = "tests-pycuda/config/linear_config.json"
    vol_dir = "data/CT_test_volume.npy"
    mesh_dir = "data/"

    # Define test poses
    pose1 = np.array([[-0.46, 0.29, -0.84, -41.77],
                      [0.28, 0.94, 0.18, -47.72],
                      [0.84, -0.15, -0.52, -253.44],
                      [0, 0, 0, 1]])
    pose2 = np.array([[-0.46, 0.29, -0.84, -41.77],
                      [0.28, 0.94, 0.18, -47.72],
                      [0.84, -0.15, -0.52, -273.24],
                      [0, 0, 0, 1]])
    test_poses = np.hstack((pose1, pose2))

    # Load expected images (as array, as the bounds are not
    # between 0 and 255 or 0 and 1)
    image1 = np.load('tests-pycuda/data/intensity_linear_liver_image_0.npy')
    image2 = np.load('tests-pycuda/data/intensity_linear_liver_image_1.npy')
    bin_image1 = plt.imread('tests-pycuda/data/'
                            'binary_linear_liver_image_0.png', format='png')
    bin_image2 = plt.imread('tests-pycuda/data/'
                            'binary_linear_liver_image_1.png', format='png')

    # Test intensity images
    ct_volume = ivol.IntensityVolume(config_dir=config_dir,
                                     vol_dir=vol_dir,
                                     file_type="npy",
                                     npy_config="tests-pycuda/config/CT_npy_test_config.json",
                                     downsampling=2,
                                     image_num=2)
    # Simulate images with first config
    _, ct_map = \
        ct_volume.simulate_image(poses=test_poses,
                                 image_num=2)

    # Test binary/segmented images
    liver_volume = svol.SegmentedVolume(config_dir=config_dir,
                                        mesh_dir=mesh_dir,
                                        voxel_size=0.5,
                                        downsampling=2,
                                        image_num=2)
    # Simulate images
    _, _, colored_map = \
        liver_volume.simulate_image(poses=test_poses,
                                    image_num=2)

    # Assert the four resulting images
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
    test_image3 = colored_map[:, :, :, 0]
    np.testing.assert_array_almost_equal(test_image3,
                                         bin_image1,
                                         decimal=7)
    test_image4 = colored_map[:, :, :, 1]
    np.testing.assert_array_almost_equal(test_image4,
                                         bin_image2,
                                         decimal=7)


def assert_curvilinear_image(gen_image,
                             test_image,
                             ratio_tolerance,
                             decimal):
    """
    A function to perform assert_array_almost_equal between
    two images gen_image and test_image. A ratio_tolerance is used
    to allow some values to not be up to the decimal accuracy
    """
    # First, compare the two arrays
    error = abs(gen_image - test_image)
    tolerance = 10 ** -decimal
    # Check how many errors are above the decimal
    mismatch_ratio = np.sum(error > tolerance) / (np.prod(error.shape))
    # Now assert that the ratio is smaller than the ratio tolerance
    np.testing.assert_array_less(mismatch_ratio,
                                 ratio_tolerance,
                                 err_msg="Generated images have a ratio of " + str(mismatch_ratio)
                                         + " mismatched elements in the image")
