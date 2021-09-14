#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mkl.h"

#define SYNTHETHIC_NUM_POINTS 10
#define MIN_LIMIT -5.0f
#define MAX_LIMIT 5.0f
#define LEAF_SIZE 300.0f

float* get_synthetic_cloud(int num_points);
float RandomFloat(float min, float max);
int readData(const char* name, float* point_cloud);
float* get_real_cloud(const char* name, int* num_points);
float* voxel_down_sampling(float* input_cloud, float* leaf_size, int num_points);
float min(int lenght, float* x, int inc);
float max(int lenght, float* x, int inc);
void bubble_sort(float* voxels, float* points, int n);
int voxel_out_cells(float* idx_points, float* idx_voxels, int n, int* pos_out, int* repeat);
void centroid(float* idx_points, int* pos_out, int* repeat, int num_points_out, float* input_cloud, float* out_cloud);

int main()
{
	//--------------------------------------
	//-----Load or Create Point Cloud-------
	//--------------------------------------
	const char mode_point_cloud[] = "real";//this could be synthetic or real data
	float* point_cloud, *downsampled_cloud;
	int num_points = 0;

	//choose wheter working with synthetic or real data
	if (!(strcmp(mode_point_cloud, "synthetic")))
	{
		point_cloud = get_synthetic_cloud(SYNTHETHIC_NUM_POINTS);
		num_points = SYNTHETHIC_NUM_POINTS;
	}
	else if (!(strcmp(mode_point_cloud, "real")))
	{
		point_cloud = get_real_cloud("dataXYZ1.csv",&num_points);
		printf("Number of points read: %d\n", num_points);
	}
	else
	{
		printf("Error introducing the load point cloud mode\n");
		return -1;
	}

	//check if there was any erros
	if (point_cloud == NULL) return -1;

	//------------------------------------------
	//-----Compute the voxel downsampling-------
	//------------------------------------------
	float leaf_size[] = { LEAF_SIZE, LEAF_SIZE, LEAF_SIZE };
	downsampled_cloud = voxel_down_sampling(point_cloud, leaf_size, num_points);

	free(point_cloud);
	return 0;
}

float* get_synthetic_cloud(int num_points)
{
	//allocate memory for the point cloud
	size_t pc_bytes = (size_t)(3) * num_points * sizeof(float);
	float* point_cloud = (float*)malloc(pc_bytes);
	if (!point_cloud){printf("Error allocating memory for point cloud\n");return NULL;}

	//limits in the 3 axes
	float x_min = MIN_LIMIT, x_max = MAX_LIMIT;
	float y_min = MIN_LIMIT, y_max = MAX_LIMIT;
	float z_min = MIN_LIMIT, z_max = MAX_LIMIT;

	//generate random points (stored using column-major format) and print
	printf("x\ty\tz\n");
	for (int i = 0; i < num_points; i++)
	{
		point_cloud[i * 3 + 0] = RandomFloat(x_min, x_max);
		point_cloud[i * 3 + 1] = RandomFloat(y_min, y_max);
		point_cloud[i * 3 + 2] = RandomFloat(z_min, z_max);
		printf("%d: %.3f, %.3f, %.3f\n", i+1, point_cloud[i * 3 + 0], point_cloud[i * 3 + 1], point_cloud[i * 3 + 2]);
	}

	return point_cloud;
}

float RandomFloat(float min, float max) 
{
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = max - min;
	float r = random * diff;
	return min + r;
}

float* get_real_cloud(const char* name, int* num_points)
{
	//initialize memory for the point cloud with 3 points
	size_t pc_bytes = (size_t)(3) * 3 * sizeof(float);
	size_t new_pc_bytes;
	float* point_cloud = (float*)malloc(pc_bytes);
	if (!point_cloud){printf("Error allocating memory for point cloud\n");return NULL;}

	//read from the file
	const int N_LINE = 2048;
	FILE* document;
	fopen_s(&document, name, "r");//open the CSV document
	if (!document){printf("File opening failed\n");return NULL;}
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
		if (i > 0) point_cloud = (float*)realloc(point_cloud, new_pc_bytes);
		if (!point_cloud) { printf("Error allocating memory for point cloud\n"); return NULL; }
		token = strtok_s(line, sep, &next_token);
		while (token != NULL)//in the line
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

float* voxel_down_sampling(float* input_cloud, float* leaf_size, int num_points)
{
	float start = second();
	//-------SET THE PARAMETERS-------
	float inv_leaf_size[3];
	for (int i = 0; i < 3; i++) inv_leaf_size[i] = 1.0f / leaf_size[i];

	//find x, y and z arrays of the point cloud
	size_t coord_size = (size_t)num_points * sizeof(float);
	float* x = (float*)malloc(coord_size);
	float* y = (float*)malloc(coord_size);
	float* z = (float*)malloc(coord_size);
	if(!x || !y || !z) { printf("Error allocating memory x, y or z arrays\n"); return NULL; }
	for (int i = 0; i < num_points; i++)
	{
		x[i] = input_cloud[i * 3 + 0];
		y[i] = input_cloud[i * 3 + 1];
		z[i] = input_cloud[i * 3 + 2];
	}

	//get the minimum and maximum dimensions
	// cblas_isamin and cblas_isamax compute the minimum and maximum ABSOLUTE values
	float min_p[3] = {}, max_p[3] = {};
	min_p[0] = min(num_points, x, 1);
	min_p[1] = min(num_points, y, 1);
	min_p[2] = min(num_points, z, 1);
	max_p[0] = max(num_points, x, 1);
	max_p[1] = max(num_points, y, 1);
	max_p[2] = max(num_points, z, 1);
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

	//----FIRST PASS----
	//Go over all points and insert them into the index vector with calculated indices
	//Points with the same index value (#voxel) will contribute to the same points of resulting cloud point
	size_t ijk_size = (size_t)num_points * 3 * sizeof(float);
	float* ijk = (float*)malloc(ijk_size);
	float* min_b_mat = (float*)malloc(ijk_size);
	cblas_scopy(3 * num_points, input_cloud, 1, ijk, 1);//copy out the point cloud to the ijk matrix
	for (int i = 0; i < 3; i++) cblas_sscal(num_points, inv_leaf_size[i], ijk + i, 3);//scale each axis
	vsFloor(3 * num_points, ijk, ijk);//floor all coordinates
	/*for (int i = 0; i < num_points; i++) printf("x[%d]: %0.f\ny[%d]: %0.f\nz[%d]: %0.f\n",
		i + 1, ijk[i * 3 + 0], i + 1, ijk[i * 3 + 1], i + 1, ijk[i * 3 + 2]);*/
	for (int i = 0; i < num_points; i++)//copy the min bounding box coord to a matrix
	{
		min_b_mat[i * 3 + 0] = -min_b[0];
		min_b_mat[i * 3 + 1] = -min_b[1];
		min_b_mat[i * 3 + 2] = -min_b[2];
	}
	//floor(...)-min_b[xyz]
	float a = 1.0f; int n = 3 * num_points, inc = 1;
	saxpy(&n, &a, ijk, &inc, min_b_mat, &inc);
	//compute the indices according to the division multipliers
	size_t idx_size = (size_t)num_points * sizeof(float);
	float* idx_voxels = (float*)malloc(idx_size);
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, num_points, 3, 1, div_mul, 1, min_b_mat, 3, 0, idx_voxels, 1);
	/*printf("\nPASS 1\npoint voxel\n");
	for (int i = 0; i < num_points; i++) printf("%d\t%0.f\n", i + 1, idx_voxels[i]);*/

	//----SECOND PASS----
	//Sort the index vector using value representing target cell as index (according to the #voxel)
	//In effect, all points belonging to the same output will be next to each other

	float* idx_points = (float*)malloc(idx_size);
	bubble_sort(idx_voxels, idx_points, num_points);
	/*printf("\nPASS 2\npoint voxel\n");
	for (int i = 0; i < 20; i++) printf("%0.f\t%0.f\n", idx_points[i] + 1.0f, idx_voxels[i]);*/

	//----THIRD PASS----
	//Count output cells. We need to skip all the same adjacent idx values
	size_t out_size = num_points * sizeof(int);
	int* pos_out = (int*)malloc(out_size);
	int* repeat = (int*)malloc(out_size);
	for (int i = 0; i < num_points; i++) repeat[i] = 0;
	int num_points_out = voxel_out_cells(idx_points, idx_voxels, num_points, pos_out, repeat);
	/*printf("\nrepeat:\n");
	for (int i = 0; i < 100; i++) printf("%d: %d\n", i + 1, repeat[i]);
	printf("\nPASS 3\npoint  voxel  pos_out  repeat\n");
	int j = 0;
	for (int i = 0; i < num_points; i++)
	{
		printf("%0.f\t%0.f\t", idx_points[i] + 1.0f, idx_voxels[i]);
		if (idx_points[pos_out[j]] == idx_points[i]) { printf("%d\t%d\n", j, repeat[j]); j++; }
		else printf("\n");
	}*/
	
	//----FOURTH PASS----
	//Compute the centroids and insert them into their final position
	float* output_cloud = (float*)malloc(3 * num_points_out * sizeof(float));
	centroid(idx_points, pos_out, repeat, num_points_out, input_cloud, output_cloud);
	/*printf("\nPASS 4\npoint x   y   z\n");
	for (int i = 0; i < num_points_out; i++) printf("%d\t%.3f\t%.3f\t%.3f\n", i + 1, 
											output_cloud[i * 3 + 0], output_cloud[i * 3 + 1], output_cloud[i * 3 + 2]);*/

	float end = second();
	printf("Elapsed time: %.3f ms\n", (end - start) * 1000.0f);

	//write out the resulting point cloud
	FILE* document;
	fopen_s(&document, "dataXYZ1_downsampled.csv", "w");//open the CSV document
	if (!document) { printf("File opening failed\n"); return NULL; }
	fprintf(document, "x,y,z\n");//header
	for (int i = 0; i < num_points_out; i++) fprintf(document, "%.5f,%.5f,%.5f\n", 
										output_cloud[i * 3 + 0], output_cloud[i * 3 + 1], output_cloud[i * 3 + 2]);
	fclose(document);//close the document

	//write out the voxels with the number of points inside each
	fopen_s(&document, "log_voxels.csv", "w");//open the CSV document
	if (!document) { printf("File opening failed\n"); return NULL; }
	fprintf(document, "voxels,num_points\n");//header
	for (int i = 0; i < num_points_out; i++) fprintf(document, "%d\n", (int)idx_voxels[i]);
	fclose(document);//close the document

	free(x), free(y), free(z);
	free(ijk), free(min_b_mat);
	free(idx_voxels), free(idx_points);
	free(pos_out), free(repeat);
	free(output_cloud);
	return NULL;
}

float min(int lenght, float* x, int inc)
{
	float minimum = x[0];//initialize minimum
	for (int i = 1; i < lenght; i += inc) 
	{ 
		if (x[i] < minimum) 
			minimum = x[i]; 
	}
	return minimum;
}

float max(int lenght, float* x, int inc)
{
	float maximum = x[0];//initialize maximum
	for (int i = 1; i < lenght; i += inc) 
	{ 
		if (x[i] > maximum) 
			maximum = x[i]; 
	}
	return maximum;
}

void bubble_sort(float* voxels, float* points, int n)
{
	// set the point indices in order (0,1,2,...,n)
	for (int i = 0; i < n; i++) points[i] = (float)i;

	// sort the voxel indices array
	float temp;
	for (int k = 0; k < n - 1; k++) {
		// (n-k-1) is for ignoring comparisons of elements which have already been compared in earlier iterations
		for (int i = 0; i < n - k - 1; i++) {
			if (voxels[i] > voxels[i + 1]) {
				// here swapping of positions is being done.
				temp = voxels[i];
				voxels[i] = voxels[i + 1];
				voxels[i + 1] = temp;

				temp = points[i];
				points[i] = points[i + 1];
				points[i + 1] = temp;
			}
		}
	}
}

int voxel_out_cells(float* idx_points, float* idx_voxels, int n, int* pos_out, int* repeat)
{
	int counter = 0;
	pos_out[counter] = 0;
	repeat[counter]++;
	for (int i = 1; i < n; i++)
	{
		if ((int)idx_voxels[i] != (int)idx_voxels[i - 1]) //change
		{
			counter++;
			pos_out[counter] = i;
		}
		repeat[counter]++;
	}
	return (counter + 1);
}

void centroid(float* idx_points, int* pos_out, int* repeat, int num_points_out, float* input_cloud, float* out_cloud)
{
	float sum[3] = {};
	for (int i = 0; i < num_points_out; i++)
	{
		int rep = repeat[i];//current number of repetitions
		sum[0] = 0.0f; sum[1] = 0.0f; sum[2] = 0.0f;
		for (int j = 0; j < rep; j++)
		{
			sum[0] += input_cloud[(int)idx_points[pos_out[i] + j] * 3 + 0];
			sum[1] += input_cloud[(int)idx_points[pos_out[i] + j] * 3 + 1];
			sum[2] += input_cloud[(int)idx_points[pos_out[i] + j] * 3 + 2];
		}
		out_cloud[i * 3 + 0] = sum[0] / (float)rep;
		out_cloud[i * 3 + 1] = sum[1] / (float)rep;
		out_cloud[i * 3 + 2] = sum[2] / (float)rep;
	}
}