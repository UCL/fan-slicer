# coding=utf-8

"""
A demo script to illustrate simulation of segmented and intensity images with
ultrasound image shapes.
"""


import numpy as np
import matplotlib.pyplot as plt
import fanslicer.pycuda_simulation.segmented_volume as svol
import fanslicer.pycuda_simulation.intensity_volume as ivol


# Load the segmented volume
liver_volume = svol.SegmentedVolume(config_dir="config/models_binary_LUS_config.json",
                                    mesh_dir="data/",
                                    voxel_size=0.5,
                                    downsampling=2,
                                    image_num=10)
# Load the intensity volume
ct_volume = ivol.IntensityVolume(config_dir="config/models_intensity_LUS_config.json",
                                 vol_dir="data/CT_test_volume.npy",
                                 file_type="npy",
                                 npy_config="config/CT_npy_volume_config.json",
                                 downsampling=2,
                                 image_num=10)

# Define a set of 10 poses, and slice generated volumes
pose = np.array([[-0.46,	0.29, -0.84	, -41.77],
                 [0.28, 0.94, 0.18, -47.72],
                 [0.84, -0.15, -0.52, -253.44],
                 [0, 0, 0, 1]])

poses = np.tile(pose, (1, 10))
poses[2, 3:40:4] = poses[2, 3:40:4] - np.arange(0, 20, 2)

# Simulate binary maps, always matching image_num with the number of poses
points, binary_map, colored_map = \
    liver_volume.simulate_image(poses=poses, image_num=10)

points, ct_map = \
    ct_volume.simulate_image(poses=poses, image_num=10)


# Show results image by image
for i in range(10):
    fig = plt.figure()
    plt.clf()
    ax1 = fig.add_subplot(121)
    ax1.imshow(colored_map[:, :, :, i])
    ax2 = fig.add_subplot(122)
    ax2.imshow(ct_map[:, :, i], cmap='gray', vmin=-200)
    plt.tight_layout()
    plt.show()
