# coding=utf-8

"""
A demo script that shows simulation of Laparoscopic
Ultrasound planes in both segmented and original CT
data.
"""


import numpy as np
import matplotlib.pyplot as plt
import slicesampler.pycuda_simulation.segmented_volume as svol
import slicesampler.pycuda_simulation.intensity_volume as ivol


# Load the segmented volume
liver_volume = svol.SegmentedVolume(config_dir="config/models_binary_LUS_config.json",
                                    mesh_dir="data/data_LUS/",
                                    voxel_size=0.5,
                                    downsampling=2,
                                    image_num=100)
# Load the intensity volume
ct_volume = ivol.IntensityVolume(config_dir="config/models_intensity_LUS_config.json",
                                 vol_dir="data/data_LUS/CT_Dicom/000/",
                                 file_type="dicom",
                                 downsampling=2,
                                 image_num=100)

# Define a set of 100 poses, and slice generated volumes
pose = np.array([[-0.46,	0.29, -0.84	, -41.77],
                 [0.28, 0.94, 0.18, -47.72],
                 [0.84, -0.15, -0.52, -253.44],
                 [0, 0, 0, 1]])

poses = np.tile(pose, (1, 100))
poses[2, 3:400:4] = poses[2, 3:400:4] - np.arange(0, 20, 0.2)

# Simulate binary maps, always matching image_num with the number of poses
# And using downsampling to reduce image dimensions
points, binary_map, colored_map = \
    liver_volume.simulate_image(poses=poses, image_num=100)

points, ct_map = \
    ct_volume.simulate_image(poses=poses, image_num=100)


# Show results image by image
for i in range(100):
    fig = plt.figure()
    plt.cla()
    ax = fig.add_subplot(121)
    ax.imshow(colored_map[:, :, :, i])
    ax = fig.add_subplot(122)
    ax.imshow(ct_map[:, :, i], cmap='gray', vmin=-200)
    plt.show()
