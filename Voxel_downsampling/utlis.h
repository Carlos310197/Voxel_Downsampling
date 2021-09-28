#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

float* read_point_cloud(const char* name, int* num_points)
{
	//initialize memory for the point cloud with 3 points
	size_t pc_bytes = (size_t)(3) * 3 * sizeof(float);
	size_t new_pc_bytes;
	float* point_cloud = (float*)malloc(pc_bytes);
	if (!point_cloud) { printf("Error allocating memory for point cloud\n"); return NULL; }

	//read from the file
	const int N_LINE = 2048;
	FILE* document;
	fopen_s(&document, name, "r");//open the CSV document
	if (!document) { printf("File opening failed\n"); return NULL; }
	char line[N_LINE]; //pointer to the string in each line
	char* token = NULL;
	char sep[] = ",\n"; //space separation
	char* next_token = NULL;
	char* next_ptr = NULL;

	fgets(line, N_LINE, document);//read header

	//the cloud is stored using column-major format
	int i = 0;
	*num_points = 0;
	while (fgets(line, N_LINE, document) != NULL)
	{
		new_pc_bytes = (size_t)(3) * ((size_t)i + 1) * sizeof(float);
		if (i > 0) point_cloud = (float*)realloc(point_cloud, new_pc_bytes);//reallocate memory
		if (!point_cloud) { printf("Error allocating memory for point cloud\n"); return NULL; }
		token = strtok_s(line, sep, &next_token);
		while (token != NULL)//on the line
		{
			point_cloud[i] = strtof(token, &next_ptr);//convert from string to float
			token = strtok_s(NULL, sep, &next_token);//read next string
			i++;
		}
		(*num_points)++;
	}

	fclose(document);//close the document
	return point_cloud;
}

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

void generate_voxel_structure(float* input_cloud, int num_points, float* leaf_size, float* idx_points)
{
	//-------SET THE PARAMETERS-------
	float inv_leaf_size[3];
	for (int i = 0; i < 3; i++) inv_leaf_size[i] = 1.0f / leaf_size[i];

	//get the minimum and maximum dimensions
	// cblas_isamin and cblas_isamax compute the minimum and maximum ABSOLUTE values
	float min_p[3] = {}, max_p[3] = {};
	min_p[0] = min(3 * num_points, input_cloud + 0, 3);
	min_p[1] = min(3 * num_points, input_cloud + 1, 3);
	min_p[2] = min(3 * num_points, input_cloud + 2, 3);
	max_p[0] = max(3 * num_points, input_cloud + 0, 3);
	max_p[1] = max(3 * num_points, input_cloud + 1, 3);
	max_p[2] = max(3 * num_points, input_cloud + 2, 3);
	printf("min dimensions\nx: %.3f\ny: %.3f\nz: %.3f\n", min_p[0], min_p[1], min_p[2]);
	printf("max dimensions\nx: %.3f\ny: %.3f\nz: %.3f\n", max_p[0], max_p[1], max_p[2]);

	// compute the minimum and maximum bounding box values
	// and compute the number of divisions along all axis
	float min_b[3] = {}, max_b[3] = {};//min and max values along all axis
	float div_b[3] = {};//number of divisions along all axis
	for (int i = 0; i < 3; i++)
	{
		min_b[i] = floor((float)(min_p[i] * inv_leaf_size[i]));
		max_b[i] = floor((float)(max_p[i] * inv_leaf_size[i]));
		div_b[i] = max_b[i] - min_b[i] + 1.0f;
	}
	int n_voxels = (int)(div_b[2], div_b[0] * div_b[1] * div_b[2]);
	printf("\nVoxel in x axis: %.0f\nVoxel in y axis: %.0f\nVoxel in z axis: %.0f\nTotal number of voxels: %d\n",
		div_b[0], div_b[1], div_b[2], n_voxels);

	// set up the division multiplier
	float div_mul[3] = { 1.0f, div_b[0], div_b[0] * div_b[1] };//this is also called the leading dimensions for each axis
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
}

//----SECOND PASS----
//Sort the index vector using value representing target cell as index (according to the #voxel)
//In effect, all points belonging to the same output will be next to each other
void second_pass()