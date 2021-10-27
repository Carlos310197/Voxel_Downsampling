#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void first_pass(int* surface, int num_triangles, int* voxel_list, int* new_surface)
{
	// Go trhough all surface triangles and replace them using the voxel list
	for (int i = 0; i < num_triangles; i++)
	{
		new_surface[i * 3 + 0] = voxel_list[surface[i * 3 + 0]];
		new_surface[i * 3 + 1] = voxel_list[surface[i * 3 + 1]];
		new_surface[i * 3 + 2] = voxel_list[surface[i * 3 + 2]];
	}
}

int second_pass(int* surface, int num_triangles)
{
	int* temp_surface = (int*)malloc((size_t)num_triangles * (size_t)3 * sizeof(int));
	int s[3], counter = 0;

	// Go trough all the triangles
	for (int i = 0; i < num_triangles; i++)
	{
		s[0] = surface[i * 3 + 0]; s[1] = surface[i * 3 + 1]; s[2] = surface[i * 3 + 2];
		if (s[0] != s[1] && s[1] != s[2] && s[0] != s[2])
		{
			temp_surface[i * 3 + 0] = s[0];
			temp_surface[i * 3 + 1] = s[1];
			temp_surface[i * 3 + 2] = s[2];
			counter++;
		}
	}

	// Copy out the result
	memcpy(surface, temp_surface, (size_t)num_triangles * (size_t)3 * sizeof(int));

	free(temp_surface);

	return counter;
}

int* generate_surface_reconstruction(int* h_surface, int* h_idx_points, int* h_pos_out, int* h_repeat, int num_points_out, int num_points, int num_triangles, int* num_triangles_out)
{
	int* h_new_surface = (int*)malloc((size_t)num_triangles * (size_t)3 * sizeof(int));

	// Genearte the voxel list
	int* voxel_list = (int*)malloc((size_t)num_points * sizeof(int));
	for (int i = 0; i < num_points_out; i++)
	{
		for (int j = 0; j < h_repeat[i]; j++) voxel_list[h_idx_points[h_pos_out[i] + j]] = i;
	}

	// Call the first-pass function
	first_pass(h_surface, num_triangles, voxel_list, h_new_surface);

	// Call the second-pass function
	*num_triangles_out = second_pass(h_new_surface, num_triangles);

	free(voxel_list);

	printf("\nNumber of triangles: %d\n", *num_triangles_out);

	return h_new_surface;
}