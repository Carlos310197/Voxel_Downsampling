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
#include <helper_cuda.h>
#include <helper_functions.h>
#include <time.h>

#define LEAF_SIZE 800.0f

float* read_point_cloud(const char* name, int* num_points);
int* read_surface(const char* name, int* num_points);
float* voxel_downsampling(float* h_input_cloud, float* d_input_cloud, int num_points, float* h_leaf_size, int** h_idx_points, int** h_idx_voxels, int** h_pos_out, int** h_repeat, int* num_points_out);
int* generate_surface_reconstruction(int* h_surface, int* h_idx_points, int* h_pos_out, int* h_repeat, int num_points_out, int num_points, int num_triangles, int* num_triangles_out);

int main()
{
	int n_donuts = 6;//number of donuts to process
	const char sphere_name[] = "point_cloud_mine.csv";//name of the input cloud
	const char surface_name[] = "surface.csv";//name of the input cloud
	int num_points = 0, num_triangles = 0;//initialize the number of points and triangles
	int num_points_out = 0, num_triangles_out = 0;//reduced number of points and triangles
	//cudaError_t err, cudaStatus;
	clock_t start, end;
	long double time;

	//------------------------------------------
	//-----------Read the point cloud-----------
	//------------------------------------------
	float* h_sphere_pc = read_point_cloud(sphere_name, &num_points);
	if (h_sphere_pc == NULL) return -1;//check if there was any errors
	printf("Number of points read: %d\n", num_points);

	//allocate memory for the point cloud in the GPU
	float* d_sphere_pc;
	size_t bytes_sphere = (size_t)3 * (size_t)num_points * sizeof(float);
	checkCudaErrors(cudaMalloc(&d_sphere_pc, bytes_sphere));

	//transfer the point cloud from the CPU to GPU
	checkCudaErrors(cudaMemcpy(d_sphere_pc, h_sphere_pc, bytes_sphere, cudaMemcpyHostToDevice));

	//------------------------------------------
	//-------------Read the surface-------------
	//------------------------------------------
	int* h_surface = read_surface(surface_name, &num_triangles);
	if (h_surface == NULL) return -1;//check if there was any errors
	printf("Number of triangles read: %d\n", num_triangles);

	int* h_surface_new = (int*)malloc((size_t)num_triangles * (size_t)3 * sizeof(int));
	printf("Size of h_surface: %d\n", sizeof(h_surface) / sizeof(int));

	//------------------------------------------
	//-----Create the voxel grid structure------
	//------------------------------------------
	printf("\n-----------Voxel Structure-----------\n");
	float h_leaf_size[3] = { LEAF_SIZE, LEAF_SIZE, LEAF_SIZE };// size of voxel
	int* h_idx_points = nullptr, * h_idx_voxels = nullptr, * h_pos_out = nullptr, * h_repeat = nullptr;

	// generate the voxel grid structure
	start = clock();
	float* downsampled_cloud = voxel_downsampling(h_sphere_pc, d_sphere_pc, num_points, h_leaf_size, &h_idx_points, &h_idx_voxels, &h_pos_out, &h_repeat, &num_points_out);
	end = clock();
	time = (double)(end - start) / (double)(CLOCKS_PER_SEC) * 1e3;
	printf("Elapsed time voxel structure: %.4lf ms\n", time);

	//------------------------------------------
	//----Compute the surface reconstruction----
	//------------------------------------------
	printf("\n---------Surface Reconstruction---------\n");
	start = clock();
	int* downsampled_surface = generate_surface_reconstruction(h_surface, h_idx_points, h_pos_out, h_repeat, num_points_out, num_points, num_triangles, &num_triangles_out);
	end = clock();
	time = (double)(end - start) / (double)(CLOCKS_PER_SEC) * 1e3;
	printf("Elapsed time surface reconstruction: %.4lf ms\n", time);

	//------------------------------------------
	//----------Generate the DXF file-----------
	//------------------------------------------
	//copy this from "Reconstruccion_tuberia"


	free(h_idx_points), free(h_idx_voxels), free(h_pos_out), free(h_repeat);
	free(h_sphere_pc), free(h_surface), free(downsampled_surface);

	return 0;
}

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

int* read_surface(const char* name, int* num_points)
{
	//initialize memory for the point cloud with 3 points
	size_t pc_bytes = (size_t)(3) * 3 * sizeof(int);
	size_t new_pc_bytes;
	int* surface = (int*)malloc(pc_bytes);
	if (!surface) { printf("Error allocating memory for point cloud\n"); return NULL; }

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
		new_pc_bytes = (size_t)(3) * ((size_t)i + 1) * sizeof(int);
		if (i > 0) surface = (int*)realloc(surface, new_pc_bytes);//reallocate memory
		if (!surface) { printf("Error allocating memory for surface\n"); return NULL; }
		token = strtok_s(line, sep, &next_token);
		while (token != NULL)//on the line
		{
			surface[i] = (int)strtof(token, &next_ptr);//convert from string to float
			token = strtok_s(NULL, sep, &next_token);//read next string
			i++;
		}
		(*num_points)++;
	}

	fclose(document);//close the document
	return surface;
}