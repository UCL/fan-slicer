# coding=utf-8

# pylint:disable=too-many-locals,too-many-branches

"""
Module segmented volume class, to be used for
simulation of 2D segmented maps of a binary volume
"""

import json
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pycuda.driver as drv
from scipy.ndimage.morphology import binary_fill_holes as fill
from scipy.ndimage.morphology import binary_erosion as erode
from scipy.ndimage.morphology import binary_dilation as dilate
import susi.pycuda_simulation.mesh as mesh
import susi.pycuda_simulation.cuda_reslicing as cres


class SegmentedVolume:
    """
    Class that holds a segmented volume, with both
    meshes and 3D binary volumes
    """
    def __init__(self,
                 mesh_dir,
                 config_dir,
                 voxel_size=1.0):
        """
        Create segmented volume object

        :param mesh_dir: directory with vtk models used in slicing
        :param config_dir: json file with reslicing parameters and
        model names to be used
        :param voxel_size: isotropic voxel size considered to
        generate the binary volumes for each vtk model

        """
        self.binary_volumes = dict()
        if voxel_size > 0:
            self.voxel_size = voxel_size
        else:
            raise ValueError("Voxel size must be positive!")

        # Load meshes if a directory is given
        self.config = None
        self.meshes = dict()
        if os.path.isfile(config_dir):
            config_file = open(config_dir)
            self.config = json.load(config_file)
        else:
            raise ValueError("No valid config file!")

        # First, load meshes to constructor
        self.load_vtk_from_dir(mesh_dir)

        # Then, load or generate simulation binary volumes
        self.load_binary_volumes(mesh_dir)

    def load_vtk_from_dir(self,
                          mesh_dir):
        """
        Loads vtk files into mesh3D objects, according
        to self.config

        :param mesh_dir: directory with vtk files
        """
        if self.config is None:
            raise ValueError("SegmentedVolume object has no config")

        if not os.path.isdir(mesh_dir):
            raise ValueError("No valid mesh directory")

        # Get relevant files from the config
        meshes_to_load = self.config["models"]["files"]
        mesh_dict = {}
        for file in meshes_to_load:
            mesh_file = os.path.join(mesh_dir, file + '.vtk')
            # Allocate mesh to mesh list if it exists
            if os.path.isfile(mesh_file):
                mesh_dict[file.replace(" ", "_")] =\
                    mesh.load_mesh_from_vtk(mesh_file)
            else:
                raise ValueError(file + '.vtk not found')

        self.meshes = mesh_dict
        return 0

    def load_binary_volumes(self,
                            data_dir):
        """
        Load or generate binary models from relevant meshes
        If binary volumes do not exist in data dir, a binary volume
        is generated for every relevant mesh defined in config

        :param data_dir: directory from where binary volumes
        is loaded/saved
        """
        if not os.path.isdir(data_dir):
            raise ValueError("No valid data directory")

        # Prepare dictionary that contains models
        volume_dict = dict()
        for model in range(len(self.config['simulation']
                           ['simulation_models'])):
            # Check if model is intended for simulation
            if self.config['simulation']['simulation_models'][model]:
                model_name = self.config['models']['files'][model]
                model_name = model_name.replace(" ", "_")

                # Get a bounding box and  define volume margin
                margin = np.array([20, 20, 20])
                bound_box = self.meshes[model_name].get_bounding_box()
                bound_box[0, :] = np.floor(bound_box[0, :]) - margin
                bound_box[1, :] = np.floor(bound_box[1, :]) + margin

                # Check if a binary map already exists
                binary_name = 'binary_' + model_name + '.npy'
                if os.path.isfile(data_dir + binary_name):
                    # Load a pre-saved model
                    volume = np.load(data_dir + binary_name)
                    print('Loaded ' + binary_name)
                else:
                    # Generate a model
                    volume = voxelise_mesh(self.meshes[model_name],
                                           self.voxel_size,
                                           margin,
                                           save_dir=data_dir,
                                           file_name=binary_name)

                # Allocate to dictionary with bounding box
                volume_dict[model_name] = [volume, bound_box]

        # Allocate final results
        self.binary_volumes = volume_dict
        return 0

    def simulate_image(self,
                       poses=np.eye(4),
                       image_num=1,
                       downsampling=1):
        """
        Function that generates a set of images from multiple
        segmented models stored in self.config. Uses the function
        slice_volume or linear_slice_volume

        :param poses: array with probe poses
        :param image_num: number of images to simulate
        :param downsampling: downsampling value on image dimensions
        :return: positions in 3D, stack of resulting images with
        multiple labels, and stack with colored images for
        visualisation
        """
        # Get config parameters for the simulation
        image_dimensions = np.array(self.config["simulation"]
                                    ["image_dimensions"])
        pixel_size = np.array(self.config["simulation"]
                              ["pixel_size"])
        voxel_size = np.array([self.voxel_size,
                               self.voxel_size,
                               self.voxel_size])

        # Check if number of images matches number of poses
        if poses.shape[1] / 4 != image_num:
            raise ValueError("Input poses do not match image number!")

        # Check if downsampling is at least 1
        if downsampling < 1:
            raise ValueError("Downsampling must be greater than 1")

        # Prepare outputs
        dim = (image_dimensions/downsampling).astype(int)
        visual_images = np.zeros((dim[1], dim[0], 3, image_num))
        simulation_images = np.zeros((dim[1], dim[0], image_num))

        # Go through the models that should be intersected
        for model in range(len(self.config["simulation"]["simulation_models"])):
            # Check if model index m is to be intersected
            if self.config["simulation"]["simulation_models"][model]:

                # Pick the model to reslice
                model_name = self.config["models"]["files"][model]
                model_name = model_name.replace(" ", "_")

                if self.config["simulation"]["transducer"] == "curvilinear":

                    fan_parameters = self.config["simulation"]["fan_geometry"]
                    points, images, mask = slice_volume(
                                    self.binary_volumes[model_name][0],
                                    self.binary_volumes[model_name][1],
                                    image_dimensions,
                                    fan_parameters,
                                    pixel_size,
                                    voxel_size,
                                    poses=poses,
                                    image_num=image_num,
                                    downsampling=downsampling)
                else:
                    points, images = linear_slice_volume(
                                     self.binary_volumes[model_name][0],
                                     self.binary_volumes[model_name][1],
                                     image_dimensions,
                                     pixel_size,
                                     voxel_size,
                                     poses=poses,
                                     image_num=image_num,
                                     downsampling=downsampling)
                # Add images to output
                simulation_images = simulation_images\
                    + images.astype(int)*(model+1)

                # Create colored images, just for visualisation
                model_color = np.array(self.config["simulation"]
                                       ["colors"][model])
                visual_images[:, :, 0, :] = visual_images[:, :, 0, :] + \
                    images * model_color[0] / 255
                visual_images[:, :, 1, :] = visual_images[:, :, 1, :] + \
                    images * model_color[1] / 255
                visual_images[:, :, 2, :] = visual_images[:, :, 2, :] + \
                    images * model_color[2] / 255

        # Add grey outline, in case the array is curvilinear
        if self.config["simulation"]["transducer"] == "curvilinear":
            outline = np.repeat(1 - mask[:, :, np.newaxis], 3, axis=2).\
                astype(int)*210/255
            outline = np.repeat(outline[:, :, :, np.newaxis],
                                image_num, axis=3)
            visual_images = visual_images + outline

        return points, simulation_images, visual_images

    def show_plane(self,
                   image_array,
                   image_indexes,
                   point_array):
        """
        Show intersection and plane geometry in 3D model
        No suitable way of showing meshes, so this method
        needs improvements

        :param image_array: stack of images to show
        :param image_indexes: stack index of image to be shown
        :param point_array: point cloud with stack of plane points
        """
        # Get number of points per plane
        points_per_plane = int(point_array.shape[0]/image_array.shape[3])
        # First, prepare figure
        fig = plt.figure()
        # Add 3D visualisation subplot
        ax_3d = fig.add_subplot(121, projection='3d')
        # Get the meshes to be plotted
        for model in range(len(self.meshes.keys())):
            # Add mesh to plot
            if self.config["simulation"]["simulation_models"][model]:
                model_name = self.config["models"]["files"][model]\
                    .replace(" ", "_")
                model = self.meshes[model_name]
                # Get color and opacity of models
                model_color = np.array([self.config["simulation"]
                                       ["colors"][model]])/255
                # model_opacity = np.array([self.config["simulation"]
                #                          ["opacity"][model]])
                ax_3d.scatter(model.vertices[0:-1:1, 0],
                              model.vertices[0:-1:1, 1],
                              model.vertices[0:-1:1, 2],
                              color=model_color,
                              alpha=0.5)

        # Add plane point cloud
        ax_3d.scatter(point_array[image_indexes*points_per_plane:
                                  points_per_plane*(image_indexes + 1):10, 0],
                      point_array[image_indexes*points_per_plane:
                                  points_per_plane*(image_indexes + 1):10, 1],
                      point_array[image_indexes*points_per_plane:
                                  points_per_plane*(image_indexes + 1):10, 2],
                      color=[0, 0, 0])

        # Add 2D visualisation subplot
        ax_2d = fig.add_subplot(122)
        ax_2d.imshow(image_array[:, :, :, image_indexes])

        plt.show()
        return 0


def voxelise_mesh(input_mesh,
                  voxel_size,
                  margin=None,
                  save_dir=None,
                  file_name=None):
    """
    Method that generates binary volume from an input mesh
    :param input_mesh: triangular mesh to be voxelised
    :param voxel_size: 3D voxel size
    :param margin: 3D vector with additional voxel margin
    around the bounding box of the input mesh
    :param save_dir: directory to save file
    :param file_name: name of file to save
    :return: 3D binary volume
    """
    if margin is None:
        margin = np.array([0, 0, 0])

    bound_box = input_mesh.get_bounding_box()
    # Add margins
    bound_box[0, :] = bound_box[0, :] - margin
    bound_box[1, :] = bound_box[1, :] + margin
    # Define output size (x, y, z)
    dimensions = (np.ceil(bound_box[1, :])
                  - np.floor(bound_box[0, :]))/voxel_size
    # Round and convert to integer
    bin_dimensions = np.ceil(dimensions).astype(int)
    # Create empty volume
    bin_volume = np.zeros(bin_dimensions, dtype=bool)

    # Get point coordinates and faces
    v_x = input_mesh.vertices[:, 0]
    v_y = input_mesh.vertices[:, 1]
    v_z = input_mesh.vertices[:, 2]
    t_x = v_x[input_mesh.faces]
    t_y = v_y[input_mesh.faces]
    t_z = v_z[input_mesh.faces]

    # Get face/triangles bounding box
    tx_min = np.amin(t_x, axis=1)
    ty_min = np.amin(t_y, axis=1)
    tz_min = np.amin(t_z, axis=1)
    tx_max = np.amax(t_x, axis=1)
    ty_max = np.amax(t_y, axis=1)
    tz_max = np.amax(t_z, axis=1)

    # 1-Intersecting XY plane
    xyplane_x = np.arange(np.floor(bound_box[0, 0]),
                          np.ceil(bound_box[1, 0]), voxel_size)
    xyplane_y = np.arange(np.floor(bound_box[0, 1]),
                          np.ceil(bound_box[1, 1]), voxel_size)

    # Loop through points with perpendicular ray and store them
    inter_xy = np.empty((0, 3), dtype=float)
    for x_ind in xyplane_x:
        for y_ind in xyplane_y:
            # Get intersectable triangles
            inter_t = np.asarray(np.where((tx_min <= x_ind)
                                          & (tx_max >= x_ind)
                                          & (ty_min <= y_ind)
                                          & (ty_max >= y_ind)))
            # Test each of these triangles for intersection
            for t_ind in inter_t[0, :]:
                # Define the ray
                origin = np.array([x_ind, y_ind, 0])
                direction = np.array([0, 0, 1])
                # Get triangle coordinates
                triangle_xyz = input_mesh.vertices[input_mesh.faces[t_ind, :]]
                # Test intersection
                flag, dist = ray_triangle_intersection(origin,
                                                       direction,
                                                       triangle_xyz)

                if flag:
                    intersection = origin + dist * direction
                    inter_xy = np.append(inter_xy, [intersection], axis=0)

    print('Intersected XY plane')

    # 2-Intersecting XZ plane
    xzplane_x = np.arange(np.floor(bound_box[0, 0]),
                          np.ceil(bound_box[1, 0]), voxel_size)
    xzplane_z = np.arange(np.floor(bound_box[0, 2]),
                          np.ceil(bound_box[1, 2]), voxel_size)

    # Loop through points with perpendicular ray and store them
    inter_xz = np.empty((0, 3), dtype=float)
    for x_ind in xzplane_x:
        for z_ind in xzplane_z:
            # Get intersectable triangles
            inter_t = np.asarray(np.where((tx_min <= x_ind)
                                          & (tx_max >= x_ind)
                                          & (tz_min <= z_ind)
                                          & (tz_max >= z_ind)))
            # Test each of these triangles for intersection
            for t_ind in inter_t[0, :]:
                # Define the ray
                origin = np.array([x_ind, 0, z_ind])
                direction = np.array([0, 1, 0])
                # Get triangle coordinates
                triangle_xyz = input_mesh.vertices[input_mesh.faces[t_ind, :]]
                # Test intersection
                flag, dist = ray_triangle_intersection(origin,
                                                       direction,
                                                       triangle_xyz)

                if flag:
                    intersection = origin + dist * direction
                    inter_xz = np.append(inter_xz, [intersection], axis=0)

    print('Intersected XZ plane')

    # 3-Intersecting YZ plane
    yzplane_y = np.arange(np.floor(bound_box[0, 1]),
                          np.ceil(bound_box[1, 1]), voxel_size)
    yzplane_z = np.arange(np.floor(bound_box[0, 2]),
                          np.ceil(bound_box[1, 2]), voxel_size)

    # Loop through points with perpendicular ray and store them
    inter_yz = np.empty((0, 3), dtype=float)
    for y_ind in yzplane_y:
        for z_ind in yzplane_z:
            # Get intersectable triangles
            inter_t = np.asarray(np.where((ty_min <= y_ind)
                                          & (ty_max >= y_ind)
                                          & (tz_min <= z_ind)
                                          & (tz_max >= z_ind)))
            # Test each of these triangles for intersection
            for t_ind in inter_t[0, :]:
                # Define the ray
                origin = np.array([0, y_ind, z_ind])
                direction = np.array([1, 0, 0])
                # Get triangle coordinates
                triangle_xyz = input_mesh.vertices[input_mesh.faces[t_ind, :]]
                # Test intersection
                flag, dist = ray_triangle_intersection(origin,
                                                       direction,
                                                       triangle_xyz)

                if flag:
                    intersection = origin + dist * direction
                    inter_yz = np.append(inter_yz, [intersection], axis=0)

    print('Intersected YZ plane')

    # Allocate indexes to binary image
    final_intersections = np.vstack((inter_xy, inter_xz, inter_yz))
    final_intersections = np.ceil((final_intersections -
                                   np.floor(bound_box[0, :]))/voxel_size) - 1

    # While there is no faster option
    for plane in range(final_intersections.shape[0]):
        x_ind = final_intersections[plane, 0].astype(int)
        y_ind = final_intersections[plane, 1].astype(int)
        z_ind = final_intersections[plane, 2].astype(int)
        bin_volume[x_ind, y_ind, z_ind] = True

    # Finally, go through z planes and fill vessels
    for plane in range(bin_volume.shape[2]):
        z_slice = bin_volume[:, :, plane].astype(int)
        closed_z_slice = fill(z_slice)
        bin_volume[:, :, plane] = closed_z_slice.astype(bool)

    if os.path.isdir(save_dir):
        if file_name is None:
            file_name = 'binary_map.npy'
        np.save(save_dir + file_name, bin_volume)

    return bin_volume


def ray_triangle_intersection(origin,
                              direction,
                              xyz):
    """
    Checks if ray defined by origin o and
    direction d intersects triangle with coordinates
    3 x 3 in xyz

    :param origin: origin of ray
    :param direction: direction of ray
    :param xyz: coordinates of triangle in 3 x 3 matrix
    :return: boolean with intersection
    """
    epsilon = 0.00001
    p_0 = xyz[0, :]
    p_1 = xyz[1, :]
    p_2 = xyz[2, :]

    e_1 = p_1 - p_0
    e_2 = p_2 - p_0
    q_value = np.cross(direction, e_2)
    a_value = np.dot(e_1, q_value)

    # Check if ray is parallel to face
    if np.abs(a_value) < epsilon:
        return 0, 0

    f_value = 1 / a_value
    s_value = origin - p_0
    u_value = f_value * np.dot(s_value, q_value)

    # Check if intersection is not within face
    if u_value < 0:
        return 0, 0

    r_value = np.cross(s_value, e_1)
    v_value = f_value * np.dot(direction, r_value)

    # Check again
    if (v_value < 0) | (v_value + u_value > 1):
        return 0, 0

    dist = f_value * np.dot(e_2, r_value)
    flag = 1

    return flag, dist


def slice_volume(binary_volume,
                 binary_bound_box,
                 image_dim,
                 fan_parameters,
                 pixel_size,
                 voxel_size,
                 poses=np.eye(4),
                 image_num=1,
                 downsampling=1):
    """
    Function that slices a binary volume with fan shaped sections
    section defined by poses of a curvilinear array

    :param binary_volume: binary volume to be intersected
    :param binary_bound_box: bounding box of volume
    :param image_dim: Number of pixels as [width height]
    :param fan_parameters: fan shape parameters
    :param pixel_size: pixel dimensions
    :param voxel_size: 3D voxel size of binary volume
    :param poses: array with probe poses
    :param image_num: number of images to generate
    :param downsampling: image dimensions downsampling value
    :return: positions in 3D, stack of resulting images, image
    with fan shape outline
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
        pose = poses[:, 4*p_ind:4*(p_ind+1)]
        # Allocate the pose
        pose_array[0, 9*p_ind:9*(p_ind+1)] = \
            np.hstack((pose[0, 0:2], pose[0, 3],
                       pose[1, 0:2], pose[1, 3],
                       pose[2, 0:2], pose[2, 3]))
        # Allocate the offset
        offset_array[0, 3*p_ind:3*(p_ind+1)] = pose[0:3, 1]

    # 1-Run position computation kernel, first assign position output,
    # with float, both in 2D and 3D
    positions_2d = np.zeros((1, coord_w*coord_h*image_num*3))\
        .astype(np.float32)
    positions_3d_linear = np.zeros((1, coord_w*coord_h*image_num*3))\
        .astype(np.float32)
    # Get kernel from file
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
    positions_3d = positions_3d_linear.reshape([3, coord_w*coord_h*image_num]).T

    # 2-Next step, run slicing kernel, where pixels are placed in the positions
    # First assign the output with int
    binary_maps = np.zeros((1, coord_w*coord_h*image_num))\
        .astype(np.int32)
    # An array of floats describing the binary volume dimensions
    binary_volume_dims = np.hstack((binary_bound_box[0, :],
                                    binary_volume.shape[0],
                                    binary_volume.shape[1],
                                    binary_volume.shape[2])).astype(np.float32)
    # Allocate the binary volume in a 1D array
    binary_volume_linear = np.swapaxes(binary_volume, 0, 1)
    binary_volume_linear = binary_volume_linear.\
        reshape([1, np.prod(binary_volume.shape)], order="F")

    # Call kernel from file
    slice_kernel = cres.reslicing_kernels.get_function('slice')
    # Then run it
    slice_kernel(drv.Out(binary_maps),
                 drv.In(positions_3d_linear),
                 drv.In(binary_volume_linear),
                 drv.In(binary_volume_dims),
                 drv.In(voxel_size),
                 drv.In(slice_dim),
                 block=(1, 1, 1),
                 grid=(coord_w, coord_h, image_num))

    # 3-Map pixels to fan like image
    # Define bounds of image output in 2d coordinates as float
    image_bounding_box = np.array([-image_dim[0] * pixel_size[0]/2 * 1000,
                                   0, image_dim[0],
                                   image_dim[1]]).astype(np.float32)
    # Allocate output images, the binary image as an int, and the
    # fan mask as a boolean
    binary_images = np.zeros((1, np.prod(image_dim))).astype(np.int32)
    mask = np.zeros((1, np.prod(image_dim))).astype(bool)
    # Call kernel from file
    map_kernel = cres.reslicing_kernels.get_function('map_back')
    # Then run it, multiplying coordinates value by a 1000, in order
    # to avoid sampling errors
    map_kernel(drv.Out(binary_images),
               drv.Out(mask),
               drv.In(binary_maps),
               drv.In(positions_2d*1000),
               drv.In(slice_dim),
               drv.In(image_bounding_box),
               drv.In(pixel_size*1000),
               block=(1, 1, 1),
               grid=(coord_w, coord_h, image_num))

    # Create a volume with generated images
    binary_image_array = np.zeros((image_dim[1],
                                   image_dim[0],
                                   image_dim[2])).astype(bool)

    for plane in range(image_num):
        # Get image and reshape it
        current_image = binary_images[0, im_size*plane:
                                      im_size*(plane+1)]
        current_image = current_image.reshape(image_dim[0], image_dim[1]).T
        # Morphological operations to clean image
        current_image = erode(current_image, iterations=2)
        current_image = dilate(current_image, iterations=2)
        # Allocate to output
        binary_image_array[:, :, plane] = current_image

    # Get the fan mask, mostly used for visualisation
    mask = mask[0, 0:im_size]
    mask = mask.reshape(image_dim[0], image_dim[1]).T

    # Output a stack of images, where each z-slice has a plane,
    # and the corresponding 3D positions, plus an outline of the fan
    return positions_3d, binary_image_array, mask


def linear_slice_volume(binary_volume,
                        binary_bound_box,
                        image_dim,
                        pixel_size,
                        voxel_size,
                        poses=np.eye(4),
                        image_num=1,
                        downsampling=1):
    """
    Function that slices a binary volume with rectangular sections
    defined by poses of a linear array

    :param binary_volume: binary volume to be intersected
    :param binary_bound_box: bounding box of volume
    :param image_dim: Number of pixels as [width height]
    :param pixel_size: pixel dimensions
    :param voxel_size: 3D voxel size of binary volume (3 values)
    :param poses: array with probe poses
    :param image_num: number of images to generate
    :param downsampling: image dimensions downsampling value
    :return: positions in 3D, stack of resulting images
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
    # Get kernel from file
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
    # First assign the output with int
    binary_maps = np.zeros((1, coord_w*coord_h*image_num))\
        .astype(np.int32)
    # An array of floats describing the binary volume dimensions
    binary_volume_dims = np.hstack((binary_bound_box[0, :],
                                    binary_volume.shape[0],
                                    binary_volume.shape[1],
                                    binary_volume.shape[2])).astype(np.float32)
    # Allocate the binary volume in a 1D array
    binary_volume_linear = np.swapaxes(binary_volume, 0, 1)
    binary_volume_linear = binary_volume_linear.\
        reshape([1, np.prod(binary_volume.shape)], order="F")

    # Call kernel from file
    slice_kernel = cres.reslicing_kernels.get_function('slice')
    # Then run it
    slice_kernel(drv.Out(binary_maps),
                 drv.In(positions_3d_linear),
                 drv.In(binary_volume_linear),
                 drv.In(binary_volume_dims),
                 drv.In(voxel_size),
                 drv.In(image_dim),
                 block=(1, 1, 1), grid=(coord_w, coord_h, image_num))

    # Create a volume with generated images
    binary_image_array = np.zeros((coord_h,
                                   coord_w,
                                   image_num)).astype(bool)

    for plane in range(image_num):
        # Get each image and reshape it
        current_image = binary_maps[0, im_size*plane:
                                    im_size*(plane+1)]
        current_image = current_image.reshape(coord_h, coord_w)
        # Morphological operations to clean image
        current_image = erode(current_image, iterations=2)
        current_image = dilate(current_image, iterations=2)
        # Allocate to output
        binary_image_array[:, :, plane] = current_image

    # Output a stack of images, where each z-slice has a plane,
    # and the corresponding 3D positions
    return positions_3d, binary_image_array


def show_volume(bin_volume):
    """
    Function that scrolls through volume in Z direction

    :param bin_volume: binary volume to show
    """
    if len(bin_volume.shape) != 3:
        warnings.warn("Not a valid volume")
    else:
        # Display z slices of volume
        for z_ind in range(bin_volume.shape[2]):
            plt.cla()
            z_slice = bin_volume[:, :, z_ind].astype(int)
            plt.title('Slice number ' + str(z_ind))
            plt.imshow(z_slice, cmap='gray')
            plt.pause(.001)
