/*
Surface Reconstruction for any OS1 LiDAR based on voxel downsampling of a point cloud
Author: Carlos Huapaya
*/

//Input: Non-overlapped point cloud and its resulting surface reconstruction arrays
//Output: Surface reconstruction of the given point cloud based on the voxel downsampling algorithm (DXF format)

///NOTES:
//The size of the voxels has to be a user input (for now).
//There is a correspondence between the size of the voxels and the form of the scanned scene.
//The greater the voxels, the more information loss of the scene and so the less number of triangles (lighter DXF file)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include "utlis.h"

#define LEAF_SIZE 300.0f

float* read_point_cloud(const char* name, int* num_points);
void generate_voxel_structure(float* input_cloud, float* leaf_size, int num_points);

int main()
{
	int n_donuts = 6;//number of donuts to process
	const char sphere_name[] = "dataXYZ1.csv";//name of the input cloud
	int num_points = 0;//initialize the number of points in the cloud
	cudaError_t err, cudaStatus;

	//------------------------------------------
	//-----------Read the point cloud-----------
	//------------------------------------------
	float* h_sphere_pc = read_point_cloud(sphere_name, &num_points);
	if (h_sphere_pc == NULL) return -1;//check if there was any errors
	printf("Number of points read: %d\n", num_points);

	//allocate memory for the point cloud in the GPU
	float* d_sphere_pc;
	size_t bytes_sphere = 3 * num_points * sizeof(float);
	cudaStatus = cudaMalloc((void**)&d_sphere_pc, bytes_sphere);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc RANGE failed!");
		return -1;
	}

	//transfer the point cloud from the CPU to GPU
	cudaStatus = cudaMemcpy(d_sphere_pc, h_sphere_pc, bytes_sphere, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy ECNT failed!");
		return -1;
	}

	//------------------------------------------
	//-----Create the voxel grid structure------
	//------------------------------------------
	
	//allocate memory for the size of the voxels on CPU
	float h_leaf_size[] = { LEAF_SIZE, LEAF_SIZE, LEAF_SIZE };// size of voxel
	float d_leaf_size;
	cudaStatus = cudaMalloc((void**)&d_leaf_size, 3 * sizeof(float*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc RANGE failed!");
		return -1;
	}
	
	//allocate memory for voxel structure on CPU
	size_t size_out_f = (size_t)num_points * sizeof(float);
	size_t size_out_i = (size_t)num_points * sizeof(int);
	float* h_idx_voxels = (float*)malloc(size_out_f);//see if used later
	int* h_pos_out = (int*)malloc(size_out_i);//see if used later
	int* h_repeat = (int*)malloc(size_out_i);
	for (int i = 0; i < num_points; i++) h_repeat[i] = 0;//fill the repeat array with zeros

	//allocate memory for voxel structure on GPU
	float* d_idx_voxels;
	int* d_pos_out, * d_repeat;
	cudaStatus = cudaMalloc((void**)&d_idx_voxels, size_out_f);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc RANGE failed!");
		return -1;
	}
	cudaStatus = cudaMalloc((void**)&d_pos_out, size_out_i);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc RANGE failed!");
		return -1;
	}
	cudaStatus = cudaMalloc((void**)&d_repeat, size_out_i);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc RANGE failed!");
		return -1;
	}
	
	//transfer the data of the repeat array from CPU to GPU
	cudaStatus = cudaMemcpy(d_repeat, h_repeat, size_out_i, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy ECNT failed!");
		return -1;
	}











	//setting threads: For now, to make it easy
	int threadsPerBlock = 1024;
	int numBlocks = num_points/threadsPerBlock;
	printf("Number of threads per block: %d\n", threadsPerBlock);
	printf("Number of blocks: %d\n", numBlocks);

	//compute the voxel grid structure
	generate_voxel_grid << < numBlocks, threadsPerBlock >> > (d_sphere_pc, d_leaf_size, d_idx_voxels, d_pos_out, d_repeat);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error in generate_voxel_grid kernel: %s\n", cudaGetErrorString(err));



	return 0;
}
