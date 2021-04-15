# coding=utf-8

"""
File that holds C++ CUDA kernels used for simulation, and
other relevant functions
"""

# pylint:disable=unused-import,duplicate-code

import numpy as np
import pycuda.driver as drv
import pycuda.autoinit


# Relevant kernels for reslicing, store in one file
RESLICING_KERNELS = """
__global__ void transform(float *xyz_position, float *xy_position, const float *tracker_array, const float *offset_array, const float *curvilinear_params, const int image_number)
{   
    // This kernel calculates the position coordinates of multiple fan shaped planes in 3D
    // Works for both segmented and intensity reslices
    int width = floorf(curvilinear_params[2] / curvilinear_params[0]);
    int height = floorf(curvilinear_params[3] / curvilinear_params[1]);

    long int index = (blockIdx.x*blockDim.x + threadIdx.x) + (blockIdx.y * blockDim.y + threadIdx.y)*width + (blockIdx.z) * height * width + image_number * height * width * threadIdx.z;

    /* Multiply indexes by the tracker matrix*/
    int track_index = 9 * blockIdx.z + blockDim.z*threadIdx.z;
    /*Obtain another index to add transducer offset*/
    int offset_index = 3 * blockIdx.z + threadIdx.z;

    /*Multiply by tracker and appropriate polar coordinate*/
    int x_index = (blockIdx.x * blockDim.x + threadIdx.x);
    int y_index = (blockIdx.y * blockDim.y + threadIdx.y);
    float angle = -curvilinear_params[2] / 2 + x_index*curvilinear_params[0];
    float line_depth = curvilinear_params[4] + y_index*curvilinear_params[1];

    float image_y_offset = curvilinear_params[4] - curvilinear_params[5];

    float sum1 = tracker_array[track_index] * line_depth * sinf(angle);
    float sum2 = tracker_array[track_index + 1] * line_depth * cosf(angle);
    float sum3 = tracker_array[track_index + 2];
    float offset = offset_array[offset_index] * image_y_offset;
    xyz_position[index] = sum1 + sum2 + sum3 - offset;


    if (threadIdx.z == 0)
    {
        xy_position[index] = line_depth * sinf(angle);
    }
    else if (threadIdx.z == 1)
    {
        xy_position[index] = line_depth * cosf(angle) - image_y_offset;
    } 
    else
    {
        xy_position[index] =0;
    }
    
}

__global__ void linear_transform(float *xyz_position, const float *tracker_array, const float *pixel_size, const int *dimensions)
{
    // This kernel calculates the position coordinates of multiple rectangular shaped planes in 3D
    // Works for both segmented and intensity reslices
    int width = dimensions[0];
    int height = dimensions[1];
    int image_number = dimensions[2];

    long int index = (blockIdx.x * blockDim.x + threadIdx.x) + (blockIdx.y * blockDim.y + threadIdx.y) * width + (blockIdx.z) * height * width + image_number*height * width * threadIdx.z;

    /* Multiply indexes by the tracker matrix*/
    int track_index = 9 * blockIdx.z + blockDim.z * threadIdx.z;

    /*Multiply by tracker*/
    int x_index = (blockIdx.x * blockDim.x + threadIdx.x);
    int y_index = (blockIdx.y * blockDim.y + threadIdx.y);
    float x_coord = (-width/2 + x_index) * pixel_size[0];
    float y_coord = y_index * pixel_size[1];

    float sum1 = tracker_array[track_index] * x_coord;
    float sum2 = tracker_array[track_index + 1] * y_coord;
    float sum3 = tracker_array[track_index + 2];
    xyz_position[index] = sum1 + sum2 + sum3;    
}

__global__ void slice(int *binary_array, const float *plane_array, const bool *binary_surface, const float *bounding_box, const float *voxel_size, int *dimensions)
{   
    // This kernel calculates the intersection values between a set of 3D points and a segmented 3D volume
    int width = dimensions[0];
    int height = dimensions[1];
    int image_number = dimensions[2];
    int vol_size = height * width * image_number;
    unsigned long int index = (blockIdx.x * blockDim.x + threadIdx.x) + (blockIdx.y * blockDim.y + threadIdx.y) * width + (blockIdx.z) * height * width;
    float voxelised_point[3];

    // Define a local bounding box variable, the original input has lower limits and dimensions in a [1 6] vector
    float box[6];
    box[0] = bounding_box[0];
    box[1] = bounding_box[1];
    box[2] = bounding_box[2];
    box[3] = box[0] + bounding_box[3] * voxel_size[0];
    box[4] = box[1] + bounding_box[4] * voxel_size[1];
    box[5] = box[2] + bounding_box[5] * voxel_size[2];

    /*Voxelise current point in this thread with nearest neighbour*/

    /* First for X coordinate*/
    if (abs(plane_array[index] - floorf(plane_array[index] / voxel_size[0]) * voxel_size[0]) < voxel_size[0] / 2)
    {
        voxelised_point[0] = floorf(plane_array[index] / voxel_size[0])*voxel_size[0];
    }
    else
    {
        voxelised_point[0] = ceilf(plane_array[index] / voxel_size[0])*voxel_size[0];
    }

    /* First for Y coordinate*/
    if (abs(plane_array[index + vol_size] - floorf(plane_array[index + vol_size] / voxel_size[1]) * voxel_size[1]) < voxel_size[1] / 2)
    {
        voxelised_point[1] = floorf(plane_array[index + vol_size] / voxel_size[1]) * voxel_size[1];
    }
    else
    {
        voxelised_point[1] = ceilf(plane_array[index + vol_size] / voxel_size[1]) * voxel_size[1];
    }

    /* First for Z coordinate*/
    if (abs(plane_array[index + vol_size * 2] - floorf(plane_array[index + vol_size * 2] / voxel_size[2]) * voxel_size[2]) < voxel_size[2] / 2)
    {
        voxelised_point[2] = floorf(plane_array[index + vol_size * 2] / voxel_size[2]) * voxel_size[2];
    }
    else
    {
        voxelised_point[2] = ceilf(plane_array[index + vol_size * 2] / voxel_size[2]) * voxel_size[2];
    }

    int x_index, y_index, z_index;
    unsigned long int total_index;
    bool intersection_value;
    /* Check if the voxelised point is within bounding box boundaries*/
    if ((voxelised_point[0] > box[0] && voxelised_point[0] < box[3]) && (voxelised_point[1] > box[1] && voxelised_point[1] < box[4]) && (voxelised_point[2] > box[2] && voxelised_point[2] < box[5]))
    {

        /* Find where the voxels are located in the binary surface voxelised grid */
        x_index = (int)((voxelised_point[0] - box[0]) / voxel_size[0]);
        y_index = (int)((voxelised_point[1] - box[1]) / voxel_size[1]);
        z_index = (int)((voxelised_point[2] - box[2]) / voxel_size[2]);

        /* Get the boolean value of surface in this index */

        total_index = x_index * (int)bounding_box[4] + y_index + z_index * (int)bounding_box[3] * (int)bounding_box[4];
        intersection_value = binary_surface[total_index];

        /* Allocate this result to the output*/
        binary_array[index] = intersection_value;

    }
    else /* Out of boundaries */
    {
        binary_array[index] = 0;
    }
}

__global__ void map_back(int *binary_images, bool *binary_masks, const int *curvilinear_binary, const float *plane_array, const int *output_image_dimensions, const float *image_bounding_box, const float *pixel_size)
{	
    // This kernel calculates segmented volume values in polar coordinates (fan shape) to a 2D image
    int width = output_image_dimensions[0];
    int height = output_image_dimensions[1];
    int image_number = output_image_dimensions[2];
    int vol_size = width * height * image_number;
    long int index = (blockIdx.x * blockDim.x + threadIdx.x) + (blockIdx.y * blockDim.y + threadIdx.y) * width + (blockIdx.z) * height * width;

    /* Define a local bounding box variable, the original input has lower limits and dimensions in a [1 4] vector*/
    float box[4];
    box[0] = image_bounding_box[0];
    box[1] = image_bounding_box[1];
    box[2] = box[0] + image_bounding_box[2]*pixel_size[0];
    box[3] = box[1] + image_bounding_box[3]*pixel_size[1];
    
    /* Get current point in 2D space */
    float current_point[2];
    current_point[0] = plane_array[index];
    current_point[1] = plane_array[index + vol_size]; 
    
    /* Calculate the reference pixel in the square that will interpolate the result*/
    float ref_pixelised_point[2];
    ref_pixelised_point[0] = floorf(current_point[0] / pixel_size[0]) * pixel_size[0];
    ref_pixelised_point[1] = floorf(current_point[1] / pixel_size[1]) * pixel_size[1];
    
    for (int i = 0; i < 4; i++){
        /* Calculate indexes (0,1;0,1) that define square around point*/
        int aux_index[2];
        aux_index[0] = (int)floorf(double(i / 2));
        aux_index[1] = (int)floorf(double(i % 2));
        
        /* Get coords of this voxel */
        float contributor_pixel[2];
        contributor_pixel[0] = ref_pixelised_point[0] + aux_index[0] * pixel_size[0];
        contributor_pixel[1] = ref_pixelised_point[1] + aux_index[1] * pixel_size[1];
        
        /* Now, add intensity value */
        int intersection_value;
        int x_index, y_index;
        unsigned int total_index;
        
        if (contributor_pixel[0] > box[0] && contributor_pixel[0] < box[2] && contributor_pixel[1] > box[1] && contributor_pixel[1] < box[3])
        {
            /*Pixel exists, intensity computed by finding appropriate index, including an offset adjustment of bounding box*/
            x_index = (int)((contributor_pixel[0] - box[0]) / pixel_size[0]);
            y_index = (int)((contributor_pixel[1] - box[1]) / pixel_size[1]);

            /* Total index in 1D array */
            total_index = x_index*(int)image_bounding_box[3] + y_index + (blockIdx.z) * image_bounding_box[3] * image_bounding_box[2];
            intersection_value = curvilinear_binary[index];
            /* Allocate this result to the output*/
            atomicAdd(&binary_images[total_index], intersection_value);
            binary_masks[total_index] = 1;
            
        }
        else
        {/* pixel does not exist, no intensity*/

        }
        }
}

__global__ void weighted_slice(float *intersections_array, const float *plane_array, const float *intensities_array, const float *bounding_box, const float *voxel_size, int *dimensions)
{
    // This kernel calculates the intersection values between a set of 3D points and a 3D intensity volume
    int width = dimensions[0];
    int height = dimensions[1];
    int image_number = dimensions[2];
    int vol_size = height * width * image_number;
    unsigned int index = (blockIdx.x * blockDim.x + threadIdx.x) * height + (blockIdx.y * blockDim.y + threadIdx.y) + (blockIdx.z) * height * width;

    /* Calculate the current point in 3D space*/
    float current_point[3];
    current_point[0] = plane_array[index]; /* X coordinate*/
    current_point[1] = plane_array[index + vol_size]; /* Y coordinate*/
    current_point[2] = plane_array[index + vol_size * 2]; /* Z coordinate*/

    /* Define cubic size of voxel */
    const float voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2];
        
    /* Define bounding box boundaries*/
    float box[6];
    box[0] = bounding_box[0];
    box[1] = bounding_box[1];
    box[2] = bounding_box[2];
    box[3] = box[0] + bounding_box[3] * voxel_size[0];
    box[4] = box[1] + bounding_box[4] * voxel_size[1];
    box[5] = box[2] + bounding_box[5] * voxel_size[2];

    /* Calculate the reference voxel in the cube that will interpolate the result*/
    float ref_voxelised_point[3];
    ref_voxelised_point[0] = floorf(current_point[0] / voxel_size[0])*voxel_size[0];
    ref_voxelised_point[1] = floorf(current_point[1] / voxel_size[1])*voxel_size[1];
    ref_voxelised_point[2] = floorf(current_point[2] / voxel_size[2])*voxel_size[2];

    
    /* Define sum variable, in case a for loop is used*/
    float sum = 0.0;
    for (int i = 0; i < 8; i++){
    /* Calculate indexes (0,1;0,1;0,1) that define cube around the point*/
    int aux_index[3];
    aux_index[0] = (int)floorf(double(i / 4));
    aux_index[1] = (int)floorf(double(i % 4) / 2);
    aux_index[2] = (int)((i % 4) % 2);

    /* Get coords of this voxel */
    float contributor_voxel[3];
    contributor_voxel[0] = ref_voxelised_point[0] + aux_index[0] * voxel_size[0];
    contributor_voxel[1] = ref_voxelised_point[1] + aux_index[1] * voxel_size[1];
    contributor_voxel[2] = ref_voxelised_point[2] + aux_index[2] * voxel_size[2];

    /* Calculate trilinear interpolation weight coefficient*/
    float distance_factor, dx, dy, dz;
    dx = (voxel_size[0] - abs(current_point[0] - contributor_voxel[0]));
    dy = (voxel_size[1] - abs(current_point[1] - contributor_voxel[1]));
    dz = (voxel_size[2] - abs(current_point[2] - contributor_voxel[2]));
    distance_factor = abs(dx)*abs(dy)*abs(dz) / voxel_volume;


    /* Attribute intensity value of voxel*/
    float intensity;
    int x_index, y_index, z_index;
    unsigned int total_index;
    /* Check if voxel is inside the boundaries */
    if (contributor_voxel[0] > box[0] && contributor_voxel[0] < box[3] && contributor_voxel[1] > box[1] && contributor_voxel[1] < box[4] && contributor_voxel[2] > box[2] && contributor_voxel[2] < box[5])
    {
        /*Voxel exists, intensity computed by finding appropriate index, including an offset adjustment of bounding box*/
        x_index = (int)((contributor_voxel[0] - box[0]) / voxel_size[0]);
        y_index = (int)((contributor_voxel[1] - box[1]) / voxel_size[1]);
        z_index = (int)((contributor_voxel[2] - box[2]) / voxel_size[2]);

        /* Total index in 1D array */
        total_index = x_index * (int)bounding_box[4] + y_index + z_index * (int)bounding_box[3] * (int)bounding_box[4];
        intensity = intensities_array[total_index];

    }
    else
    {/* Voxel does not exist, no intensity*/
        intensity = 0.0;
    }

    /*Increment the value in the intersection array*/
    sum = sum + distance_factor*intensity;
    }
intersections_array[index] = sum;
}

__global__ void intensity_map_back(float *intensity_images, int *binary_masks, const float *curvilinear_intensities, const float *plane_array, const int *output_image_dimensions, const float *image_bounding_box, const float *pixel_size)
{	
    // This kernel calculates intensity values in polar coordinates (fan shape) to a 2D image
    int width = output_image_dimensions[0];
    int height = output_image_dimensions[1];
    int image_number = output_image_dimensions[2];
    int vol_size = width * height * image_number;
    long int index = (blockIdx.x * blockDim.x + threadIdx.x) + (blockIdx.y * blockDim.y + threadIdx.y) * width + (blockIdx.z) * height * width;

    /* Define a local bounding box variable, the original input has lower limits and dimensions in a [1 4] vector*/
    float box[4];
    box[0] = image_bounding_box[0];
    box[1] = image_bounding_box[1];
    box[2] = box[0] + image_bounding_box[2]*pixel_size[0];
    box[3] = box[1] + image_bounding_box[3]*pixel_size[1];
    
    /* Get current point in 2D space */
    float current_point[2];
    current_point[0] = plane_array[index];
    current_point[1] = plane_array[index + vol_size]; 
    
    /* Calculate the reference pixel in the square that will interpolate the result*/
    float ref_pixelised_point[2];
    ref_pixelised_point[0] = floorf(current_point[0] / pixel_size[0]) * pixel_size[0];
    ref_pixelised_point[1] = floorf(current_point[1] / pixel_size[1]) * pixel_size[1];

    for (int i = 0; i < 4; i++){
        /* Calculate indexes (0,1;0,1) that define square around point*/
        int aux_index[2];
        aux_index[0] = (int)floorf(double(i / 2));
        aux_index[1] = (int)floorf(double(i % 2));
        
        /* Get coords of this voxel */
        float contributor_pixel[2];
        contributor_pixel[0] = ref_pixelised_point[0] + aux_index[0] * pixel_size[0];
        contributor_pixel[1] = ref_pixelised_point[1] + aux_index[1] * pixel_size[1];
        
        /* Now, add intensity value */
        float intensity;
        int x_index, y_index;
        unsigned int total_index;
        
        if (contributor_pixel[0] > box[0] && contributor_pixel[0] < box[2] && contributor_pixel[1] > box[1] && contributor_pixel[1] < box[3])
        {
            /*Pixel exists, intensity computed by finding appropriate index, including an offset adjustment of bounding box*/
            x_index = (int)((contributor_pixel[0] - box[0]) / pixel_size[0]);
            y_index = (int)((contributor_pixel[1] - box[1]) / pixel_size[1]);

            /* Total index in 1D array */
            total_index = x_index * (int)image_bounding_box[3] + y_index + (blockIdx.z) * image_bounding_box[3] * image_bounding_box[2];
            intensity = curvilinear_intensities[index];
            /* Allocate this result to the output*/
            atomicAdd(&intensity_images[total_index], intensity);
            atomicAdd(&binary_masks[total_index], 1);
            
        }
        else
        {/* pixel does not exist, no intensity*/

        }
    }
}
"""


# Function to estimate blocksize efficiently
def get_block_size(dim_x,
                   dim_y):
    """
    Function to calculate block size according to x and y
    dimensions of images to be sampled

    :param dim_x: x dimension
    :param dim_y: y dimension
    :return: 2 block size parameters
    """
    # Consider a limited set of prime numbers
    prime_list = np.array([2, 3, 5, 7])
    # Get GPU size limits, max in x, max in y, and block max
    max_x = pycuda.autoinit.device.get_attribute(
                drv.device_attribute.MAX_BLOCK_DIM_Y)
    max_y = pycuda.autoinit.device.get_attribute(
                drv.device_attribute.MAX_BLOCK_DIM_Y)
    max_b = pycuda.autoinit.device.get_attribute(
                drv.device_attribute.MAX_THREADS_PER_BLOCK)/3

    # Set initial values for block sizes
    b_x = 1
    b_y = 1
    # Now divide dims by primes
    for prime in prime_list:
        # First for x
        while np.mod(dim_x, prime) == 0 and \
                b_x * prime < max_x and \
                b_x * prime * b_y < max_b:
            # Divid dim
            dim_x = dim_x / prime
            # Increase blocksize
            b_x = b_x * prime

        # Then for y
        while np.mod(dim_y, prime) == 0 and \
                b_y * prime < max_y and \
                b_x * prime * b_y < max_b:
            # Divid dim
            dim_y = dim_y / prime
            # Increase blocksize
            b_y = b_y * prime

    return b_x, b_y
