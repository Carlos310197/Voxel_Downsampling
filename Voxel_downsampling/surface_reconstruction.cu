#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>  

__global__
void replace_idx(int* d_surface, int idx_change, int idx_replace)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (d_surface[tid * 3 + 0] == idx_change) d_surface[tid * 3 + 0] = idx_replace;
	if (d_surface[tid * 3 + 1] == idx_change) d_surface[tid * 3 + 1] = idx_replace;
	if (d_surface[tid * 3 + 1] == idx_change) d_surface[tid * 3 + 2] = idx_replace;
}

//----FIRST PASS----
//Group connections from all points in a voxel
__global__
void first_pass(float* input_cloud, int* surface, int* idx_points, int* pos_out, int* repeat, int num_points_out, int num_points)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int block_size = 1024, grid_size = num_points / block_size + 1;//for replacement
	int idx, idx_in_voxel;
	float centroid[3] = {};

	//take all valid points and with repeatitions greater than 1
	if (tid < num_points_out && repeat[tid] > 1)
	{
		//get the first index of the voxel
		idx = idx_points[pos_out[tid]];
		centroid[0] = 0; centroid[1] = 0; centroid[2] = 0;
		for (int i = 1; i < repeat[tid]; i++)
		{
			idx_in_voxel = idx_points[pos_out[tid] + i];
			replace_idx <<< grid_size, block_size >>> (surface, idx_in_voxel, idx);
			centroid[0] += input_cloud[idx_in_voxel * 3 + 0];
			centroid[1] += input_cloud[idx_in_voxel * 3 + 1];
			centroid[2] += input_cloud[idx_in_voxel * 3 + 2];
		}
		input_cloud[idx * 3 + 0] = centroid[0] / repeat[tid];
		input_cloud[idx * 3 + 1] = centroid[1] / repeat[tid];
		input_cloud[idx * 3 + 2] = centroid[2] / repeat[tid];
	}
	//__syncthreads();
}

__global__
void second_pass()
{

}

int generate_surface_reconstruction(int* d_surface, float* d_input_cloud, int* h_idx_points, int* h_pos_out, int* h_repeat, int num_points_out)
{
	// allocate memory on GPU
	int* d_pos_out, * d_repeat;
	size_t bytes_out_int = (size_t)num_points_out * sizeof(int);
	checkCudaErrors(cudaMalloc(&d_pos_out, bytes_out_int));
	checkCudaErrors(cudaMalloc(&d_repeat, bytes_out_int));

	// transfer data from CPU to GPU
	checkCudaErrors(cudaMemcpy(d_pos_out, h_pos_out, bytes_out_int, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_repeat, h_repeat, bytes_out_int, cudaMemcpyHostToDevice));

	// number of threads
	int block_size = 1024, grid_size = num_points_out / block_size + 1;
	printf("\nBlock size: %d\n", block_size);
	printf("Grid size: %d\n", grid_size);
	printf("Total number of threads: %d\n", block_size * grid_size);
	cudaError_t err;

	first_pass <<< grid_size, block_size >>> ();
}