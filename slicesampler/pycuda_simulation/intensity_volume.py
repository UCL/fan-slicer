# coding=utf-8

# pylint:disable=too-many-locals,unsupported-assignment-operation,too-many-instance-attributes

"""
Module with intensity volume class, to be used
for simulation of 2D intensity maps from a 3D
intensity volume
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import nibabel as nib
import pycuda.driver as drv
import pycuda.gpuarray as gpua
from pycuda.compiler import SourceModule
import slicesampler.pycuda_simulation.cuda_reslicing as cres


class IntensityVolume:
    """
    Class that holds a 3D intensity volume image
    and tools for reslicing it
    """
    def __init__(self,
                 config_dir,
                 vol_dir,
                 image_num=1,
                 downsampling=1,
                 file_type='npy',
                 npy_config=None):
        """
        Create intensity volume object

        :param config_dir: json file with reslicing parameters
        :param vol_dir: file with 3D volume
        :param file_type: type of 3D volume to be loaded,
        currently nii or dicom
        :param image_num: number of images to consider for preallocation
        :param downsampling: downsampling factor on image dimensions
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
            raise ValueError("No valid config file!")

        # Check whether a nii or dicom is to be
        # loaded

        if file_type == 'dicom':
            self.load_volume_from_dicom(vol_dir)

        if file_type == 'nii':
            self.load_volume_from_nii(vol_dir)

        if file_type == 'npy':
            self.load_volume_from_npy(vol_dir, npy_config)

        # In order to speed up slicing, preallocate variables
        # Call function to preallocate relevant variables
        # to an existing list, first the GPU ones
        self.g_variables = []
        # Image dimensioning parameters
        self.image_variables = []
        # Kernel dimensioning
        self.blockdim = np.array([1, 1])
        # Initialise image num and downsample
        self.image_num = None
        self.downsampling = None
        # Now run allocation to set these vars
        self.preallocate_gpu_var(image_num=image_num,
                                 downsampling=downsampling)

        # Read kernel source code in C++
        self.kernel_code = cres.RESLICING_KERNELS

    def load_volume_from_dicom(self, dicom_dir):
        """
        Loads volume from Dicom

        :param dicom_dir: dicom file
        """
        if not os.path.isdir(dicom_dir):
            raise ValueError("No valid file directory for dicom!")

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

        :param nii_dir: nii file
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

    def load_volume_from_npy(self, npy_dir, npy_config):
        """
        Loads volume from npy file

        :param npy_dir: nii file
        :param npy_config: volume resolution for the npy volume
        """
        # Add volume data
        self.ct_volume = np.load(npy_dir)

        # Add resolution parameters, first get config
        if os.path.isfile(npy_config):
            npy_config_file = open(npy_config)
            npy_config = json.load(npy_config_file)
        else:
            raise ValueError("No valid config for npy file!")

        # Now load the parameters
        self.planar_resolution = np.array(npy_config["planar resolution"])
        self.voxel_size = np.array(npy_config["voxel size"])
        self.bound_box = np.array(npy_config["bounding box"])

        return 0

    def scroll_volume(self):
        """
        Shows volume stored in intensity volume object
        """
        for z_ind in range(self.zdim):
            plt.cla()
            plt.imshow(self.ct_volume[:, :, z_ind], cmap='gray')
            plt.pause(0.01)

    def preallocate_gpu_var(self,
                            image_num,
                            downsampling):
        """
        Function to generate local gpu variables that will
        be used for simulation. Variable sizes depend on the
        config parameters. g_ prefix indicates gpu variables

        :param image_num: maximum number of images to be simulated
        :param downsampling: downsampling value on image dimensions
        per call
        """
        # First check if current image variables are empty or not,
        # (if they have been set before). If they are not, reset
        if self.g_variables:
            self.g_variables = []

        if self.image_variables:
            self.image_variables = []

        # Check if downsampling is at least 1
        if downsampling < 1:
            raise ValueError("Downsampling must be greater than 1")

        # Check if maximum number of images is valid
        if not isinstance(image_num, int) or image_num <= 0:
            raise ValueError('image_num must be positive integer')

        self.image_num = image_num
        self.downsampling = downsampling

        # Now, choose between curvilinear and linear array
        transducer_type = self.config["simulation"]["transducer"]
        if transducer_type == "curvilinear":
            # For the curvilinear case, get
            # geometrical parameters of fan shape as a float:
            # 0-Angular ray resolution, 1-ray depth resolution, 2-angle aperture
            # 3-ray depth, 4-ray offset to origin, 5-ray offset to image top
            fan_parameters = np.array(self.config["simulation"]["fan_geometry"])
            fan_parameters[0] = np.deg2rad(fan_parameters[0])
            fan_parameters[2] = np.deg2rad(fan_parameters[2])
            fan_parameters[3:6] = fan_parameters[3:6] * fan_parameters[1]
            fan_parameters = fan_parameters.astype(np.float32)
            # Append them to image variables (becomes index 0)
            self.image_variables.append(fan_parameters)

            # Get point cloud dimensions from fan parameters, necessary to
            # know how many points will be sampled and used for intersection
            coord_w = len(np.arange((-fan_parameters[2] / 2).astype(np.float32),
                                    (fan_parameters[2] / 2).astype(np.float32),
                                    fan_parameters[0]))
            coord_h = len(np.arange(fan_parameters[4],
                                    fan_parameters[4] + fan_parameters[3],
                                    fan_parameters[1]))

            # Append to image variables (becomes index 1)
            slice_dim = np.array([coord_w, coord_h, image_num]).astype(np.int32)
            self.image_variables.append(slice_dim)

            # Through downsampling, obtain the output image dimensions
            # and append (becomes index 2)
            image_dim_2d = np.array(self.config["simulation"]
                                    ["image_dimensions"])
            image_dim = np.append(image_dim_2d / downsampling, image_num) \
                .astype(np.int32)
            self.image_variables.append(image_dim)

            # Do the same for the image pixel size (becomes index 3)
            pixel_size = np.array(self.config["simulation"]["pixel_size"])
            pixel_size = (downsampling * pixel_size).astype(np.float32)
            self.image_variables.append(pixel_size)

            # Knowing these dimensions, now append preallocate all
            # GPU variables. First, 2D and 3D positions of the fans
            # (become index 0 and 1, respectively)
            self.g_variables. \
                append(gpua.GPUArray((1, np.prod(slice_dim) * 3),
                                     dtype=np.float32))
            # The 3D positions, with the same size (becomes index 1)
            self.g_variables.\
                append(gpua.GPUArray((1, np.prod(slice_dim) * 3),
                                     dtype=np.float32))

            # The fan intersection with the volume (becomes index 2)
            self.g_variables. \
                append(gpua.GPUArray((1, np.prod(slice_dim)),
                                     dtype=np.float32))

            # The volume to be slice, in a 1D array. The only non-empty
            # array (becomes index 3)
            volume = self.ct_volume.copy()
            volume = volume.reshape([1, np.prod(volume.shape)], order="F")
            self.g_variables.append(gpua.to_gpu(volume.astype(np.float32)))

            # Now, the outputs, with image_dim as dimension, both images
            # and fan shape outline used for interpolation (become
            # index 4 and 5, respectively)
            self.g_variables. \
                append(gpua.GPUArray((1, np.prod(image_dim)),
                                     dtype=np.float32))
            self.g_variables. \
                append(gpua.GPUArray((1, np.prod(image_dim)),
                                     dtype=np.int32))

            # Determine optimal blocksize for kernels
            blockdim_x, blockdim_y = cres.get_block_size(coord_w, coord_h)
            self.blockdim = np.array([blockdim_x, blockdim_y])

        elif transducer_type == "linear":
            # For the linear case, variable definition is simpler
            # Get rectangular plane dimensions first, and append
            # to image variables (becomes index 0)
            image_dim_2d = np.array(self.config["simulation"]
                                    ["image_dimensions"])
            image_dim = np.append(image_dim_2d / downsampling, image_num) \
                .astype(np.int32)
            self.image_variables.append(image_dim)

            # Do the same for the image pixel size (becomes index 1)
            pixel_size = np.array(self.config["simulation"]["pixel_size"])
            pixel_size = (downsampling * pixel_size).astype(np.float32)
            self.image_variables.append(pixel_size)

            # Now preallocate gpu variables, first the positions
            # (becomes index 0)
            self.g_variables. \
                append(gpua.GPUArray((1, np.prod(image_dim) * 3),
                                     dtype=np.float32))

            # Secondly, volume intersections that do not
            # need to be warped in this case (becomes index 1)
            self.g_variables. \
                append(gpua.GPUArray((1, np.prod(image_dim)),
                                     dtype=np.float32))

            # The volume to be intersected (becomes
            # index 2)
            volume = self.ct_volume.copy()
            volume = volume.reshape([1, np.prod(volume.shape)], order="F")
            self.g_variables.append(gpua.to_gpu(volume.astype(np.float32)))

            # Determine optimal blocksize for kernels
            blockdim_x, blockdim_y = cres.get_block_size(image_dim[0],
                                                         image_dim[1])
            self.blockdim = np.array([blockdim_x, blockdim_y])
        else:
            # In case the transducer is another option
            raise ValueError("No valid transducer type!")

    def simulate_image(self,
                       poses=np.eye(4),
                       image_num=1,
                       out_points=False):
        """
        Function that generates a set of 2D CT images from
        intensity volume. Uses the function
        intensity_slice_volume or linear_intensity_slice_volume

        :param poses: array with probe poses
        :param image_num: number of images to slice
        :param out_points: bool to get sampling positions or not
        :return: positions in 3D, stack of resulting images
        """

        # Check if number of images matches number of poses
        if poses.shape[1]/4 != image_num:
            raise ValueError("Input poses do not match image number!")

        # In order to not fix the number of images to be used, check
        # if image num is the same as the one considered by the object
        # If they differ, preallocate again
        current_image_num = self.image_num

        if image_num != current_image_num:
            self.preallocate_gpu_var(image_num=image_num,
                                     downsampling=self.downsampling)
            print("Number of images was changed from " +
                  str(current_image_num) + " to " + str(image_num))

        # Simulate images
        volume_dim = self.ct_volume.shape
        if self.config["simulation"]["transducer"] == "curvilinear":
            points, images = intensity_slice_volume(
                             self.kernel_code,
                             self.image_variables,
                             self.g_variables,
                             self.blockdim,
                             self.bound_box,
                             volume_dim,
                             self.voxel_size,
                             poses=poses,
                             out_points=out_points)
        else:
            points, images = linear_intensity_slice_volume(
                             self.kernel_code,
                             self.image_variables,
                             self.g_variables,
                             self.blockdim,
                             self.bound_box,
                             volume_dim,
                             self.voxel_size,
                             poses=poses,
                             out_points=out_points)
        return points, images


def intensity_slice_volume(kernel_code,
                           image_variables,
                           g_variables,
                           blockdim,
                           bound_box,
                           vol_dim,
                           voxel_size,
                           poses,
                           out_points=False):

    """
    Function that slices an intensity volume with fan shaped sections
    section defined by poses of a curvilinear array

    :param kernel_code: CUDA C++ kernel code to compile
    :param image_variables: image dimensioning variable list
    :param g_variables: All preallocated GPU variables
    as described in the preallocation function. A list with
    the following indexes:
    0 - fan positions in 2D
    1 - fan positions in 3D
    2 - intensities mapped in fan positions
    3 - the target intensity volume
    4 - the output images in image space
    5 - the 2D fan mask outline
    :param blockdim: block dimensions for CUDA kernels
    :param bound_box: bounding box of target volume
    :param vol_dim: 3D intensity volume dimensions
    :param voxel_size: voxel_size of the volume
    :param poses: input set of poses
    :param out_points: bool to get fan positions or not
    :return: positions in 3D, stack of resulting images
    """

    # First, compile kernel code with SourceModule
    cuda_modules = SourceModule(kernel_code)

    # Get image variables from input
    fan_parameters = image_variables[0]
    slice_dim = image_variables[1]
    image_dim = image_variables[2]
    pixel_size = image_variables[3]

    # Define voxel size for intersection of intensity volume
    voxel_size = voxel_size.astype(np.float32)

    # Get size of one image, useful to get array of images
    im_size = image_dim[0] * image_dim[1]

    # Get block and grid dimensions as int
    blockdim_x = int(blockdim[0])
    blockdim_y = int(blockdim[1])
    griddim_x = int(slice_dim[0] / blockdim_x)
    griddim_y = int(slice_dim[1] / blockdim_y)
    image_num = int(slice_dim[2])

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

    # 1-Run position computation kernel, acts on index 0 and 1 of
    # the gpu variables, get kernel
    transform_kernel = cuda_modules.get_function("transform")
    # Then run it
    transform_kernel(g_variables[1],
                     g_variables[0],
                     drv.In(pose_array),
                     drv.In(offset_array),
                     drv.In(fan_parameters),
                     np.int32(image_num),
                     block=(blockdim_x, blockdim_y, 3),
                     grid=(griddim_x, griddim_y, image_num))

    # Collect the output to a CPU array
    positions_3d = np.empty((1, np.prod(slice_dim) * 3), dtype=np.float32)
    # In case points are to be used or visualised (with out_points as True)
    if out_points is True:
        g_variables[1].get(positions_3d)
        positions_3d = positions_3d.reshape([3, np.prod(slice_dim)]).T

    # 2-Next step, run slicing kernel, where intensity values are
    # placed in the positions. Define volume dimensions
    intensity_volume_dims = np.hstack((bound_box[0, :],
                                       vol_dim[0],
                                       vol_dim[1],
                                       vol_dim[2])).astype(np.float32)

    # Call kernel from file
    slice_kernel = cuda_modules.get_function('weighted_slice')
    slice_kernel(g_variables[2],
                 g_variables[1],
                 g_variables[3],
                 drv.In(intensity_volume_dims),
                 drv.In(voxel_size),
                 drv.In(slice_dim),
                 block=(blockdim_x, blockdim_y, 1),
                 grid=(griddim_x, griddim_y, image_num))

    # 3-Map pixels to fan like image
    # Define bounds of image output in 2d coordinates as float
    image_bounding_box = np.array([-image_dim[0] * pixel_size[0]/2*1000,
                                   0, image_dim[0],
                                   image_dim[1]]).astype(np.float32)

    # Allocate output images, the intensity image as a float, and the
    # fan outline as an int. These must be in CPU.
    intensity_images = np.empty((1, np.prod(image_dim)), dtype=np.float32)
    masks = np.empty((1, np.prod(image_dim)), dtype=np.int32)
    # Call kernel from file
    map_kernel = cuda_modules.get_function('intensity_map_back')

    # Then run it, multiplying coordinates value by a 1000, in order
    # to avoid sampling errors
    map_kernel(g_variables[4],
               g_variables[5],
               g_variables[2],
               g_variables[0]*1000,
               drv.In(slice_dim),
               drv.In(image_bounding_box),
               drv.In(pixel_size*1000),
               block=(blockdim_x, blockdim_y, 1),
               grid=(griddim_x, griddim_y, image_num))

    # Create a volume with generated images
    intensity_image_array = np.zeros((image_dim[1],
                                      image_dim[0],
                                      image_dim[2])).astype(np.float32)

    # Gather the results
    g_variables[4].get(intensity_images)
    g_variables[4].fill(0)
    g_variables[5].get(masks)
    g_variables[5].fill(0)

    for plane in range(image_num):
        # Get image and reshape it
        current_image = intensity_images[0, im_size*plane:
                                         im_size*(plane+1)]
        # Get masks that weight values
        current_mask = masks[0, im_size*plane:
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


def linear_intensity_slice_volume(kernel_code,
                                  image_variables,
                                  g_variables,
                                  blockdim,
                                  bound_box,
                                  vol_dim,
                                  voxel_size,
                                  poses,
                                  out_points=False):
    """
    Function that slices an intensity volume with rectangular sections
    defined by poses of a linear array

    :param kernel_code: CUDA C++ kernel code to compile
    :param image_variables: image dimensioning variable list
    :param g_variables: All preallocated GPU variables
    as described in the preallocation function. A list with
    the following indexes:
    0 - rectangle positions in 3D
    1 - rectangular intensity images
    2 - the target intensity volume
    :param blockdim: block dimensions for CUDA kernels
    :param bound_box: bounding box of target volume
    :param vol_dim: 3D intensity volume dimensions
    :param voxel_size: voxel_size of the volume
    :param poses: input set of poses
    :param out_points: bool to get rectangular positions or not
    :return: positions in 3D, stack of resulting images
    """

    # First, compile kernel code with SourceModule
    cuda_modules = SourceModule(kernel_code)

    # Get image variables from input
    image_dim = image_variables[0]
    pixel_size = image_variables[1]

    # Define voxel size for intersection of intensity volume
    voxel_size = voxel_size.astype(np.float32)

    # Get size of one image, useful to get array of images
    im_size = image_dim[0] * image_dim[1]

    # Get block and grid dimensions as int
    blockdim_x = int(blockdim[0])
    blockdim_y = int(blockdim[1])
    griddim_x = int(image_dim[0] / blockdim_x)
    griddim_y = int(image_dim[1] / blockdim_y)
    image_num = int(image_dim[2])

    # Convert poses to 1D array to be input in a kernel
    pose_array = np.zeros((1, 9 * image_num)).astype(np.float32)
    for p_ind in range(image_num):
        pose = poses[:, 4*p_ind:4*(p_ind+1)]
        # Allocate the pose
        pose_array[0, 9*p_ind:9*(p_ind+1)] = \
            np.hstack((pose[0, 0:2], pose[0, 3],
                       pose[1, 0:2], pose[1, 3],
                       pose[2, 0:2], pose[2, 3]))

    # 1-Run position computation kernel, acts on index 0
    # the gpu variables, get kernel
    transform_kernel = cuda_modules.get_function("linear_transform")
    # Then run it
    transform_kernel(g_variables[0],
                     drv.In(pose_array),
                     drv.In(pixel_size),
                     drv.In(image_dim),
                     block=(blockdim_x, blockdim_y, 3),
                     grid=(griddim_x, griddim_y, image_num))

    # Collect the output to a CPU array
    positions_3d = np.empty((1, np.prod(image_dim) * 3), dtype=np.float32)
    # In case points are to be used or visualised (with out_points as True)
    if out_points is True:
        g_variables[0].get(positions_3d)
        positions_3d = positions_3d.reshape([3, np.prod(image_dim)]).T

    # 2-Next step, run slicing kernel, where intensity values are
    # placed in the positions. Define volume dimensions
    intensity_volume_dims = np.hstack((bound_box[0, :],
                                       vol_dim[0],
                                       vol_dim[1],
                                       vol_dim[2])).astype(np.float32)

    # Allocate space for output images, in CPU
    intensity_images = np.empty((1, np.prod(image_dim)), dtype=np.float32)

    # Call kernel from file
    slice_kernel = cuda_modules.get_function('weighted_slice')
    slice_kernel(g_variables[1],
                 g_variables[0],
                 g_variables[2],
                 drv.In(intensity_volume_dims),
                 drv.In(voxel_size),
                 drv.In(image_dim),
                 block=(blockdim_x, blockdim_y, 1),
                 grid=(griddim_x, griddim_y, image_num))

    # Create a volume with generated images
    intensity_image_array = np.zeros((image_dim[1],
                                      image_dim[0],
                                      image_dim[2])).astype(np.float32)

    # Gather the results
    g_variables[1].get(intensity_images)

    for plane in range(image_num):
        # Get each image and reshape it
        current_image = intensity_images[0, im_size*plane:
                                         im_size*(plane+1)]
        current_image = current_image.reshape(image_dim[1], image_dim[0])
        # Allocate to output
        intensity_image_array[:, :, plane] = current_image

    # Output a stack of images, where each z-slice has a plane,
    # and the corresponding 3D positions
    return positions_3d, intensity_image_array
