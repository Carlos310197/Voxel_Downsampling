#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

