/** @author Beniel Thileepan
  * 
  */
#pragma once

#define EPSILON 0.00001

struct GridParameter
{
	unsigned int logical_size_x;
	unsigned int logical_size_y;
    unsigned int logical_size_z;

	unsigned int act_size_x;
	unsigned int act_size_y;
    unsigned int act_size_z;

	unsigned int grid_size_x;
	unsigned int grid_size_y;
    unsigned int grid_size_z;

	unsigned int batch;
	unsigned int num_iter;
};

struct heat3DParameter
{
    float h;
    float alpha; //diffusivity
    float delta_t;
    float K;
};
