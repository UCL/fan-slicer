# coding=utf-8

# pylint:disable=too-many-locals,unsupported-assignment-operation

"""
Module with intensity volume class, to be used
for simulation of 2D intensity maps from a 3D
intensity volume
"""

import json
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import nibabel as nib
import pycuda.driver as drv
import susi.pycuda_simulation.cuda_reslicing as cres


class IntensityVolume:
    """
    Class that contains a 3D intensity volume image
    and tools for reslicing it
    """
    def __init__(self,
                 config_dir='',
                 vol_dir='',
                 file_type='dicom'):
        """
        Construct the volume class
        """
        self.planar_resolution = None
        self.ct_volume = None
        self.voxel_size = None
        self.bound_box = None
        self.xdim = None
        self.ydim = None
        self.zdim = None

        if os.path.isfile(config_dir):
            config_file = open(config_dir)
            self.config = json.load(config_file)
        else:
            print("No valid config file")

        # Check whether a nii or dicom is to be
        # loaded

        if file_type == 'dicom':
            self.load_volume_from_dicom(vol_dir)

        if file_type == 'nii':
            self.load_volume_from_nii(vol_dir)

    def load_volume_from_dicom(self, dicom_dir):
        """
        Loads volume from Dicom
        """
        if not os.path.isdir(dicom_dir):
            warnings.warn("No valid file directory")
            return 0

        image_list = os.listdir(dicom_dir)
        image_list.sort()

        # Get the parameters of the volume by checking the first image
        first_image = dicom.dcmread(dicom_dir + image_list[0])
        # Get planar resolution
        self.planar_resolution = first_image.PixelSpacing
        # Get z stepping
        z_step = first_image.SpacingBetweenSlices
        # Define voxel size
        self.voxel_size = np.hstack((self.planar_resolution,
                                     abs(z_step)))

        # Get x y z dimensions
        self.xdim = first_image.pixel_array.shape[0]
        self.ydim = first_image.pixel_array.shape[1]
        self.zdim = len(image_list)
        self.ct_volume = np.zeros([self.xdim, self.ydim, self.zdim])

        # Get intensity scales
        for dicom_key in first_image.keys():
            if first_image[dicom_key].keyword == 'RescaleIntercept':
                intensity_bias = first_image[dicom_key].value
            if first_image[dicom_key].keyword == 'RescaleSlope':
                intensity_slope = first_image[dicom_key].value

        # Go through every image
        for i in range(self.zdim):
            # Get image
            current_image = dicom.dcmread(dicom_dir + image_list[i]).pixel_array
            # Add to volume, taking into account z direction
            if z_step > 0:
                self.ct_volume[:, :, i] = current_image \
                    * intensity_slope + intensity_bias
            else:
                self.ct_volume[:, :, self.zdim - i - 1] \
                    = current_image * intensity_slope \
                    + intensity_bias

        # Define bounding box
        min_x = first_image.ImagePositionPatient[0]
        max_x = min_x + self.planar_resolution[0] * (self.xdim - 1)
        min_y = first_image.ImagePositionPatient[1]
        max_y = min_y + self.planar_resolution[1] * (self.xdim - 1)

        if z_step < 0:
            max_z = first_image.ImagePositionPatient[2]
            min_z = max_z + z_step * (self.zdim - 1)
        else:
            min_z = first_image.ImagePositionPatient[2]
            max_z = min_z + z_step * (self.zdim - 1)

        self.bound_box = np.array([[min_x, min_y, min_z],
                                   [max_x, max_y, max_z]])

        return 0

    def load_volume_from_nii(self, nii_dir):
        """
        Loads volume from nii
        """
        nii_file = nib.load(nii_dir)
        volume = nii_file.get_fdata()
        volume = np.flip(volume, axis=0)
        volume = np.flip(volume, axis=1)
        self.ct_volume = np.asarray(volume)
        self.xdim = volume.shape[0]
        self.ydim = volume.shape[1]
        self.zdim = volume.shape[2]
        # Get resolution parameters
        affine = nii_file.affine
        self.planar_resolution = abs(np.array([affine[0, 0],
                                               affine[1, 1]]))

        self.voxel_size = abs(np.array([affine[0, 0],
                                        affine[1, 1],
                                        affine[2, 2]]))

        # Get bounding box, checking orientations
        if affine[2, 2] > 0:
            max_z = affine[2, 3] + affine[2, 2] * (self.zdim-1)
            min_z = affine[2, 3]
        else:
            min_z = affine[2, 3] + affine[2, 2] * (self.zdim-1)
            max_z = affine[2, 3]

        if affine[1, 1] > 0:
            max_y = affine[1, 3] + affine[1, 1] * (self.ydim-1)
            min_y = affine[1, 3]
        else:
            min_y = affine[1, 3] + affine[1, 1] * (self.ydim-1)
            max_y = affine[1, 3]

        if affine[0, 0] > 0:
            max_x = affine[0, 3] + affine[0, 0] * (self.xdim-1)
            min_x = affine[0, 3]
        else:
            min_x = affine[0, 3] + affine[0, 0] * (self.xdim-1)
            max_x = affine[0, 3]

        self.bound_box = np.array([[min_x, min_y, min_z],
                                   [max_x, max_y, max_z]])

    def scroll_volume(self):
        """
        Show volume
        """
        for z_ind in range(self.zdim):
            plt.cla()
            plt.imshow(self.ct_volume[:, :, z_ind], cmap='gray')
            plt.pause(0.01)

    def simulate_image(self,
                       poses=np.eye(4),
                       image_num=1,
                       downsampling=1):
        """
        Function that generates a 2D CT projection from a set of poses
        """
        # Get config parameters for the simulation
        image_dimensions = np.array(self.config["simulation"]
                                    ["image_dimensions"])
        pixel_size = np.array(self.config["simulation"]
                              ["pixel_size"])

        # Simulate images
        if self.config["simulation"]["transducer"] == "curvilinear":
            # Get curvilinear array parameters
            fan_parameters = self.config["simulation"]["fan_geometry"]
            points, images = intensity_slice_volume(
                             self.ct_volume,
                             self.bound_box,
                             image_dimensions,
                             fan_parameters,
                             pixel_size,
                             self.voxel_size,
                             poses=poses,
                             image_num=image_num,
                             downsampling=downsampling)
        else:
            points, images = linear_intensity_slice_volume(
                             self.ct_volume,
                             self.bound_box,
                             image_dimensions,
                             pixel_size,
                             self.voxel_size,
                             poses=poses,
                             image_num=image_num,
                             downsampling=downsampling)

        return points, images


def intensity_slice_volume(intensity_volume,
                           bound_box,
                           image_dim,
                           fan_parameters,
                           pixel_size,
                           voxel_size,
                           poses=np.eye(4),
                           image_num=1,
                           downsampling=1):
    """
    Function that slices the volume with parameters in config
    """
    # Get geometrical parameters of fan shape as a float:
    # 0-Angular ray resolution, 1-ray depth resolution, 2-angular aperture
    # 3-ray depth, 4-ray offset to origin, 5-ray offset to image top
    fan_parameters = np.array(fan_parameters)
    fan_parameters[0] = np.deg2rad(fan_parameters[0])
    fan_parameters[2] = np.deg2rad(fan_parameters[2])
    fan_parameters[3:6] = fan_parameters[3:6] * fan_parameters[1]
    fan_parameters = fan_parameters.astype(np.float32)

    # Convert inputs to appropriate float and int for cuda kernels,
    # first pixel size of output images
    pixel_size = (downsampling * pixel_size).astype(np.float32)
    # Dimensions, as width, height, and number of images
    image_dim = np.append(image_dim / downsampling, image_num).astype(np.int32)
    im_size = image_dim[0] * image_dim[1]
    # Voxel size of volume to reslice
    voxel_size = voxel_size.astype(np.float32)

    # Calculate 2D dimensions of the planar fan point cloud using
    # fan params
    coord_w = len(np.arange((-fan_parameters[2] / 2).astype(np.float32),
                            (fan_parameters[2] / 2).astype(np.float32),
                            fan_parameters[0]))
    coord_h = len(np.arange(fan_parameters[4],
                            fan_parameters[4] + fan_parameters[3],
                            fan_parameters[1]))

    # Store these dimensions as int
    slice_dim = np.array([coord_w, coord_h, image_num]).astype(np.int32)

    # Convert poses to 1D array to be input in a kernel
    pose_array = np.zeros((1, 9 * image_num)).astype(np.float32)
    # And an array to offset fan position per image plane
    offset_array = np.zeros((1, 3 * image_num)).astype(np.float32)
    for p_ind in range(image_num):
        pose = poses[:, 4 * p_ind:4 * (p_ind + 1)]
        # Allocate the pose
        pose_array[0, 9 * p_ind:9 * (p_ind + 1)] = \
            np.hstack((pose[0, 0:2], pose[0, 3],
                       pose[1, 0:2], pose[1, 3],
                       pose[2, 0:2], pose[2, 3]))
        # Allocate the offset
        offset_array[0, 3 * p_ind:3 * (p_ind + 1)] = pose[0:3, 1]

    # 1-Run position computation kernel, first assign position output,
    # with float, both in 2D and 3D
    positions_2d = np.zeros((1, coord_w * coord_h * image_num * 3))\
        .astype(np.float32)
    positions_3d_linear = np.zeros((1, coord_w * coord_h * image_num * 3))\
        .astype(np.float32)
    # Get kernel
    transform_kernel = cres.reslicing_kernels.get_function("transform")
    # Then run it
    transform_kernel(drv.Out(positions_3d_linear),
                     drv.Out(positions_2d),
                     drv.In(pose_array),
                     drv.In(offset_array),
                     drv.In(fan_parameters),
                     np.int32(image_num),
                     block=(1, 1, 3),
                     grid=(coord_w, coord_h, image_num))

    # Collect 1D Output, and convert to Nx3 output
    positions_3d = positions_3d_linear.\
        reshape([3, coord_w * coord_h * image_num]).T

    # 2-Next step, run slicing kernel, where intensity values are
    # placed in the positions. First assign the output with float
    intensity_maps = np.zeros((1, coord_w * coord_h * image_num))\
        .astype(np.float32)
    intensity_volume_dims = np.hstack((bound_box[0, :],
                                       intensity_volume.shape[0],
                                       intensity_volume.shape[1],
                                       intensity_volume.shape[2]))\
        .astype(np.float32)

    # Put the 3D volume in one 1D array
    intensity_volume_linear = intensity_volume.astype(np.float32)
    intensity_volume_linear = intensity_volume_linear\
        .reshape([1, np.prod(intensity_volume.shape)], order="F")

    # Call kernel from file
    slice_kernel = cres.reslicing_kernels.get_function('weighted_slice')
    slice_kernel(drv.Out(intensity_maps),
                 drv.In(positions_3d_linear),
                 drv.In(intensity_volume_linear),
                 drv.In(intensity_volume_dims),
                 drv.In(voxel_size),
                 drv.In(slice_dim),
                 block=(1, 1, 1),
                 grid=(coord_w, coord_h, image_num))

    # This line is to see the unwarped intersection for debugging purposes
    # intensity_maps_output = intensity_maps.reshape(coord_h, coord_w)

    # 3-Map pixels to fan like image
    # Define bounds of image output in 2d coordinates as float
    image_bounding_box = np.array([-image_dim[0] * pixel_size[0]/2*1000,
                                   0, image_dim[0],
                                   image_dim[1]]).astype(np.float32)

    # Allocate output images, the intensity image as a float, and the
    # fan mask as an int
    intensity_images = np.zeros((1, np.prod(image_dim))).astype(np.float32)
    mask = np.zeros((1, np.prod(image_dim))).astype(np.int32)
    # Call kernel from file
    map_kernel = cres.reslicing_kernels.get_function('intensity_map_back')
    # Then run it, multiplying coordinates value by a 1000, in order
    # to avoid sampling errors
    map_kernel(drv.Out(intensity_images),
               drv.Out(mask),
               drv.In(intensity_maps),
               drv.In(positions_2d*1000),
               drv.In(slice_dim),
               drv.In(image_bounding_box),
               drv.In(pixel_size*1000),
               block=(1, 1, 1),
               grid=(coord_w, coord_h, image_num))

    # Create a volume with generated images
    intensity_image_array = np.zeros((image_dim[1],
                                      image_dim[0],
                                      image_dim[2])).astype(np.float32)
    for plane in range(image_num):
        # Get image and reshape it
        current_image = intensity_images[0, im_size*plane:
                                         im_size*(plane+1)]
        # Get masks that weight values
        current_mask = mask[0, im_size*plane:
                            im_size*(plane + 1)]
        # Normalise by amount of points added to image output, using the
        # the occurrences output by mask, ignoring divide error
        with np.errstate(divide='ignore'):
            current_image = np.divide(current_image, current_mask)

        current_image = current_image.reshape(image_dim[0], image_dim[1]).T
        # Scale intensities, by setting nan values to minimum
        nan_indexes = np.where(np.isnan(current_image))
        current_image[nan_indexes] = np.nanmin(current_image)
        # Allocate to output
        intensity_image_array[:, :, plane] = current_image

    # Output a stack of images, where each z-slice has a plane,
    # and the corresponding 3D positions
    return positions_3d, intensity_image_array


def linear_intensity_slice_volume(intensity_volume,
                                  bound_box,
                                  image_dim,
                                  pixel_size,
                                  voxel_size,
                                  poses=np.eye(4),
                                  image_num=1,
                                  downsampling=1):
    """
    Function that slices a volume with a linear
    section defined by image_dim
    """

    # Convert inputs to appropriate float and int for cuda kernels,
    # first pixel size of output images
    pixel_size = downsampling * pixel_size.astype(np.float32)
    # Dimensions, as width, height, and number of images
    image_dim = np.append(image_dim / downsampling, image_num).astype(np.int32)
    # Voxel size of volume to reslice
    voxel_size = voxel_size.astype(np.float32)

    # Get image dims
    coord_w = int(image_dim[0])
    coord_h = int(image_dim[1])
    im_size = coord_w * coord_h

    # Convert poses to 1D array to be input in a kernel
    pose_array = np.zeros((1, 9 * image_num)).astype(np.float32)
    for p_ind in range(image_num):
        pose = poses[:, 4*p_ind:4*(p_ind+1)]
        # Allocate the pose
        pose_array[0, 9*p_ind:9*(p_ind+1)] = \
            np.hstack((pose[0, 0:2], pose[0, 3],
                       pose[1, 0:2], pose[1, 3],
                       pose[2, 0:2], pose[2, 3]))

    # 1-Run position computation kernel, first assign position output,
    # with float
    positions_3d_linear = np.zeros((1, np.prod(image_dim)*3))\
        .astype(np.float32)
    # Get kernel
    transform_kernel = cres.reslicing_kernels.get_function("linear_transform")
    # Then run it
    transform_kernel(drv.Out(positions_3d_linear),
                     drv.In(pose_array),
                     drv.In(pixel_size),
                     drv.In(image_dim),
                     block=(1, 1, 3),
                     grid=(coord_w, coord_h, image_num))
    # Collect 1D Output, and convert to Nx3 output
    positions_3d = positions_3d_linear.reshape([3, im_size*image_num]).T

    # 2-Next step, run slicing kernel, where pixels are placed in the positions
    intensity_maps = np.zeros((1, coord_w*coord_h*image_num))\
        .astype(np.float32)
    intensity_volume_dims = np.hstack((bound_box[0, :],
                                       intensity_volume.shape[0],
                                       intensity_volume.shape[1],
                                       intensity_volume.shape[2]))\
        .astype(np.float32)

    # Allocate the intensities volume in a 1D array
    intensity_volume_linear = intensity_volume.astype(np.float32)
    intensity_volume_linear = intensity_volume_linear \
        .reshape([1, np.prod(intensity_volume.shape)], order="F")

    # Call kernel from file
    slice_kernel = cres.reslicing_kernels.get_function('weighted_slice')
    slice_kernel(drv.Out(intensity_maps),
                 drv.In(positions_3d_linear),
                 drv.In(intensity_volume_linear),
                 drv.In(intensity_volume_dims),
                 drv.In(voxel_size),
                 drv.In(image_dim),
                 block=(1, 1, 1),
                 grid=(coord_w, coord_h, image_num))

    # Create a volume with generated images
    intensity_image_array = np.zeros((coord_h,
                                      coord_w,
                                      image_num)).astype(np.float32)

    for plane in range(image_num):
        # Get each image and reshape it
        current_image = intensity_maps[0, im_size*plane:
                                       im_size*(plane+1)]
        current_image = current_image.reshape(coord_h, coord_w)
        # Allocate to output
        intensity_image_array[:, :, plane] = current_image

    # Output a stack of images, where each z-slice has a plane,
    # and the corresponding 3D positions
    return positions_3d, intensity_image_array
