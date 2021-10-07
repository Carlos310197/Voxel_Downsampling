#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <helper_cuda.h>
#include <helper_functions.h>    

float min(int lenght, float* x, int inc)
{
	float minimum = x[0];//initialize minimum
	for (int i = inc; i < lenght; i += inc)
	{
		if (x[i] < minimum)
			minimum = x[i];
	}
	return minimum;
}

float max(int lenght, float* x, int inc)
{
	float maximum = x[0];//initialize maximum
	for (int i = inc; i < lenght; i += inc)
	{
		if (x[i] > maximum)
			maximum = x[i];
	}
	return maximum;
}

int get_max(int* a, int n)
{
	int max = a[0];
	for (int i = 1; i < n; i++)
		if (a[i] > max)
			max = a[i];
	return max;
}

__device__
float my_floor(float num)
{
	if (num < 0)
		return (int)num - 1;
	else
		return (int)num;
}

//----FIRST PASS----
//Go over all points and insert them into the index vector with calculated indices
//Points with the same index value (#voxel) will contribute to the same points of resulting cloud point
__global__
void first_pass(float* point_cloud, int num_points, float* min_b, float* div_mul, float* inv_leaf_size, int* idx_voxels)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;//number of threads = number of points
	idx_voxels[tid] = (int)((my_floor(point_cloud[tid * 3 + 0] * inv_leaf_size[0]) - min_b[0]) * div_mul[0] +
							(my_floor(point_cloud[tid * 3 + 1] * inv_leaf_size[1]) - min_b[1]) * div_mul[1] +
							(my_floor(point_cloud[tid * 3 + 2] * inv_leaf_size[2]) - min_b[2]) * div_mul[2]);
	if (tid < 19460 && tid > 19450) printf("%d: %d\n", tid + 1, idx_voxels[tid]);
	/*if (tid == 0) printf("inv_leaf_size: [%.4f,%.4f,%.4f]\n",inv_leaf_size[0],inv_leaf_size[1],inv_leaf_size[2]);
	if (tid == 0) printf("div_mul: [%.3f,%.3f,%.3f]\n", div_mul[0], div_mul[1], div_mul[2]);
	if (tid == 0) printf("min_b: [%.3f,%.3f,%.3f]\n", min_b[0], min_b[1], min_b[2]);*/
}

//----SECOND PASS----
//Sort the index vector using value representing target cell as index (according to the #voxel)
//In effect, all points belonging to the same output will be next to each other
//Radix Sort
void second_pass(int* array, int* idx, int n)
{
	int bucket_cnt[10] = {};
	size_t bucket_size = (size_t)10 * (size_t)n * sizeof(int);
	int* bucket = (int*)malloc(bucket_size);
	int* idx_bucket = (int*)malloc(bucket_size);;
	int i, j, k, r, NOP = 0, divisor = 1, lar;

	// set the indices in order (0,1,2,...,n)
	for (int i = 0; i < n; i++) idx[i] = i;

	//count the number of passes
	lar = get_max(array, n);
	while (lar > 0)
	{
		NOP++;     // No of passes
		lar /= 10; // largest number
	}

	//go trough each pass (ones, tens, hundrends, ...)
	for (int pass = 0; pass < NOP; pass++)
	{
		//initialize the bucket counter with zeros
		for (i = 0; i < 10; i++)
		{
			bucket_cnt[i] = 0;
		}

		//compute the current digit for each number
		for (i = 0; i < n; i++)
		{
			r = (array[i] / divisor) % 10;//digit
			bucket[r + 10 * bucket_cnt[r]] = array[i];//fill the bucket with the correspondent number
			idx_bucket[r + 10 * bucket_cnt[r]] = idx[i];//and the other with the index
			bucket_cnt[r] += 1;//update the counter
		}

		//sort taking numbers in order
		i = 0;
		for (k = 0; k < 10; k++)
		{
			for (j = 0; j < bucket_cnt[k]; j++)
			{
				array[i] = bucket[k + 10 * j];
				idx[i] = idx_bucket[k + 10 * j];
				i++;
			}
		}

		//update the divisor
		divisor *= 10;
	}
}

//----THIRD PASS----
//Count output cells. We need to skip all the same adjacent idx values
int third_pass(int* idx_points, int* idx_voxels, int n, int* pos_out, int* repeat)
{
	int counter = 0;
	pos_out[counter] = 0;
	repeat[counter]++;
	for (int i = 1; i < n; i++)
	{
		if (idx_voxels[i] != idx_voxels[i - 1]) //change
		{
			counter++;
			pos_out[counter] = i;
		}
		repeat[counter]++;
	}
	return (counter + 1);
}

int generate_voxel_structure(float* h_input_cloud, float* d_input_cloud, int num_points, float* h_leaf_size, int* h_idx_points, int* h_idx_voxels, int* h_pos_out, int* h_repeat)
{
	//-------SET THE PARAMETERS-------
	float h_inv_leaf_size[3];
	for (int i = 0; i < 3; i++) h_inv_leaf_size[i] = 1.0f / h_leaf_size[i];

	//get the minimum and maximum dimensions
	// cblas_isamin and cblas_isamax compute the minimum and maximum ABSOLUTE values
	float min_p[3] = {}, max_p[3] = {};
	min_p[0] = min(3 * num_points, h_input_cloud + 0, 3);
	min_p[1] = min(3 * num_points, h_input_cloud + 1, 3);
	min_p[2] = min(3 * num_points, h_input_cloud + 2, 3);
	max_p[0] = max(3 * num_points, h_input_cloud + 0, 3);
	max_p[1] = max(3 * num_points, h_input_cloud + 1, 3);
	max_p[2] = max(3 * num_points, h_input_cloud + 2, 3);
	printf("min dimensions\nx: %.3f\ny: %.3f\nz: %.3f\n", min_p[0], min_p[1], min_p[2]);
	printf("max dimensions\nx: %.3f\ny: %.3f\nz: %.3f\n", max_p[0], max_p[1], max_p[2]);

	// compute the minimum and maximum bounding box values
	// and compute the number of divisions along all axis
	float h_min_b[3] = {}, max_b[3] = {};//min and max values along all axis
	float h_div_b[3] = {};//number of divisions along all axis
	for (int i = 0; i < 3; i++)
	{
		h_min_b[i] = floor((float)(min_p[i] * h_inv_leaf_size[i]));
		max_b[i] = floor((float)(max_p[i] * h_inv_leaf_size[i]));
		h_div_b[i] = max_b[i] - h_min_b[i] + 1.0f;
	}
	int n_voxels = (int)(h_div_b[2], h_div_b[0] * h_div_b[1] * h_div_b[2]);
	printf("\nVoxel in x axis: %.0f\nVoxel in y axis: %.0f\nVoxel in z axis: %.0f\nTotal number of voxels: %d\n",
		h_div_b[0], h_div_b[1], h_div_b[2], n_voxels);

	// set up the division multiplier
	float h_div_mul[3] = { 1.0f, h_div_b[0], h_div_b[0] * h_div_b[1] };//this is also called the leading dimensions for each axis

	// declare variables GPU
	float* d_min_b, * d_div_mul, * d_inv_leaf_size;
	int* d_idx_voxels;
	size_t size_input_cloud = (size_t)num_points * (size_t)3 * sizeof(float);
	size_t size_3f = (size_t)3 * sizeof(float);
	size_t size_idx = (size_t)num_points * (size_t)3 * sizeof(int);

	// allocate memory on GPU
	checkCudaErrors(cudaMalloc(&d_min_b, size_3f));
	checkCudaErrors(cudaMalloc(&d_div_mul, size_3f));
	checkCudaErrors(cudaMalloc(&d_inv_leaf_size, size_3f));
	checkCudaErrors(cudaMalloc(&d_idx_voxels, size_idx));

	// transfer the data from CPU to GPU
	checkCudaErrors(cudaMemcpy(d_inv_leaf_size, h_inv_leaf_size, size_3f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_min_b, h_min_b, size_3f, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_div_mul, h_div_mul, size_3f, cudaMemcpyHostToDevice));

	// number of threads
	int block_size = 1024, grid_size = num_points / block_size;
	cudaError_t err;

	//----FIRST PASS----
	first_pass << < grid_size, block_size >> > (d_input_cloud, num_points, d_min_b, d_div_mul, d_inv_leaf_size, d_idx_voxels);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) printf("Error in first_pass kernel: %s\n", cudaGetErrorString(err));

	// copy out the idx_voxels to the CPU
	h_idx_voxels = (int*)malloc(size_idx);
	checkCudaErrors(cudaMemcpy(h_idx_voxels, d_idx_voxels, size_3f, cudaMemcpyDeviceToHost));
	/*printf("h_idx_voxels:\n");
	for (int i = 0; i < num_points; i++) printf("%d: %d\n", i + 1, h_idx_voxels);*/

	//----SECOND PASS----
	h_idx_points = (int*)malloc(size_idx);
	second_pass(h_idx_voxels, h_idx_points, num_points);
	printf("h_idx_voxels:\n");
	for (int i = 0; i < 10; i++) printf("%d: %d\n", i + 1, h_idx_voxels);

	//----THIRD PASS----
	h_pos_out = (int*)malloc(size_idx);
	h_repeat = (int*)malloc(size_idx);
	int num_points_out = third_pass(h_idx_points, h_idx_voxels, num_points, h_pos_out, h_repeat);

	printf("Voxel structure done!\n");

	// free memory allocated
	cudaFree(d_idx_voxels);
	cudaFree(d_min_b), cudaFree(d_div_mul), cudaFree(d_inv_leaf_size);

	return num_points_out;
}