# coding=utf-8

"""Tests on pycuda simulation of 2D fan shaped images"""

import os
import numpy as np
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
    mesh_dir = "data_LUS/hepatic veins.vtk"

    # Load the test mesh (hepatic vein)
    test_mesh = mesh.load_mesh_from_vtk(mesh_dir)

    # Remove saved binary model, in case it exists
    if os.path.isfile("data_tests/binary_map.npy"):
        os.remove("data_tests/binary_map.npy")

    # Voxelise mesh
    volume = svol.voxelise_mesh(test_mesh,
                                voxel_size,
                                margin=[20, 20, 20],
                                save_dir="data_tests/")

    test_volume = np.load("data_tests/binary_map_hepatic_veins.npy")
    # Check if binary model is the same
    np.testing.assert_array_equal(test_volume, volume)


def test_binary_image_sim():
    """
    Test binary image simulation
    """
    # Set config
    config_dir = "config/models_binary_LUS_config.json"
    mesh_dir = "data_LUS/"
    # Load the segmented volume
    liver_volume = svol.SegmentedVolume(config_dir=config_dir,
                                        mesh_dir=mesh_dir,
                                        voxel_size=0.5)
    assert 0==0




