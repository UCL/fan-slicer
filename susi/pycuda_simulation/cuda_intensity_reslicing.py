# coding=utf-8

"""
File that holds C++ CUDA kernels used for simulation of
fan shaped CT intensity maps with bilinear interpolation
"""

# pylint:disable=unused-import,duplicate-code

import pycuda.autoinit
from pycuda.compiler import SourceModule

# Relevant kernels for intensity reslicing, store in one file


int_reslicing_kernels = SourceModule("""
__global__ void transform(float *xyz_position, float *xy_position, const float *tracker_array, const float *offset_array, const float *curvilinear_params, const int image_number)
{
    int width = floorf(curvilinear_params[2] / curvilinear_params[0]);
    int height = floorf(curvilinear_params[3] / curvilinear_params[1]);

    long int index = (blockIdx.x*blockDim.x + threadIdx.x) + (blockIdx.y*blockDim.y + threadIdx.y)*width + (blockIdx.z)*height*width + image_number*height*width*threadIdx.z;

    /* Multiply indexes by the tracker matrix*/
    int track_index = 9 * blockIdx.z + blockDim.z*threadIdx.z;
    /*Obtain another index to add transducer offset*/
    int offset_index = 3 * blockIdx.z + threadIdx.z;

    /*Multiply by tracker and appropriate polar coordinate*/
    int x_index = (blockIdx.x*blockDim.x + threadIdx.x);
    int y_index = (blockIdx.y*blockDim.y + threadIdx.y);
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



__global__ void weighted_slice(float *intersections_array, const float *plane_array, const float *intensities_array, const float *bounding_box, const float *voxel_size, int width, int height, int depth)
{
    int vol_size = height*width*depth;
    unsigned int index = (blockIdx.x*blockDim.x + threadIdx.x)*height + (blockIdx.y*blockDim.y + threadIdx.y) + (blockIdx.z)*height*width;

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
    if (contributor_voxel[0] > box[0] && contributor_voxel[0]<box[3] && contributor_voxel[1]>box[1] && contributor_voxel[1]<box[4] && contributor_voxel[2]>box[2] && contributor_voxel[2] < box[5])
    {
        /*Voxel exists, intensity computed by finding appropriate index, including an offset adjustment of bounding box*/
        x_index = (int)((contributor_voxel[0] - box[0]) / voxel_size[0]);
        y_index = (int)((contributor_voxel[1] - box[1]) / voxel_size[1]);
        z_index = (int)((contributor_voxel[2] - box[2]) / voxel_size[2]);

        /* Total index in 1D array */
        total_index = x_index*(int)bounding_box[4] + y_index + z_index*(int)bounding_box[3] * (int)bounding_box[4];
        intensity = intensities_array[total_index];

    }
    else
    {/* Voxel does not exist, no intensity*/
        intensity = 0.0;
    }

    /* round precision
    float precision=0.1;
    sum = ceil(sum/precision)*precision;*/
    sum = sum + distance_factor*intensity;
    /*Increment the value in the intersection array*/
    }
intersections_array[index] = sum;
}

__global__ void intensity_map_back(float *intensity_images, int *binary_masks, const float *curvilinear_intensities, const float *plane_array, const int *output_image_dimensions, const float *image_bounding_box, const float *pixel_size)
{	
    /*Size of each array of coordinates*/
    int width = output_image_dimensions[0];
    int height = output_image_dimensions[1];
    int image_number = output_image_dimensions[2];
    int vol_size = width * height * image_number;
    long int index = (blockIdx.x*blockDim.x + threadIdx.x) + (blockIdx.y*blockDim.y + threadIdx.y)*width + (blockIdx.z)*height*width;

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
        
        if (contributor_pixel[0] > box[0] && contributor_pixel[0]<box[2] && contributor_pixel[1]>box[1] && contributor_pixel[1]<box[3])
        {
            /*Pixel exists, intensity computed by finding appropriate index, including an offset adjustment of bounding box*/
            x_index = (int)((contributor_pixel[0] - box[0]) / pixel_size[0]);
            y_index = (int)((contributor_pixel[1] - box[1]) / pixel_size[1]);

            /* Total index in 1D array */
            total_index = x_index*(int)image_bounding_box[3] + y_index + (blockIdx.z)*image_bounding_box[3] * image_bounding_box[2];
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
""")