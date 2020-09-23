# coding=utf-8

"""
A demo script that shows simulation of Endoscopic
Ultrasound planes in the original CT
"""


import numpy as np
import matplotlib.pyplot as plt
import susi.pycuda_simulation.intensity_volume as ivol


ct_volume = ivol.IntensityVolume(vol_dir="data/data_EUS/Case0001_img.nii",
                                 config_dir="config/models_intensity_EUS_config.json",
                                 file_type='nii')

# Define the two ground truth poses shown in Bonmati et al.
# IPCAI 2018
vertex1 = np.array([160.5, 136.2, 92]) + ct_volume.bound_box[0, :]
pose1 = np.array([[-0.5, -0.45,	-0.74, vertex1[0]],
                  [-0.26, 0.89, -0.36, vertex1[1]],
                  [0.82, 0.01, -0.56, vertex1[2]],
                  [0, 0, 0, 1]])

vertex2 = np.array([121.9, 165.7, 85.5]) + ct_volume.bound_box[0, :]
pose2 = np.array([[0.1, 0.97, 0.19, vertex2[0]],
                  [-0.99, 0.08, 0.1, vertex2[1]],
                  [0.082, -0.2, 0.97, vertex2[2]],
                  [0, 0, 0, 1]])
poses = np.hstack((pose1, pose2))

# Slice volume with the two poses (attention to image_num which must be 2)
points, image = ct_volume.simulate_image(poses=poses, downsampling=1, image_num=2)

# Display the images
fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(image[:, :, 0], cmap='gray', vmin=0)
ax = fig.add_subplot(122)
ax.imshow(image[:, :, 1], cmap='gray', vmin=0)
plt.show()

