
#include <math.h>
#include <poisson_cpu.h>

int stencil_computation(float* current, float* next, ops::hls::GridPropertyCore data_g, unsigned int batch)
{
	for(unsigned int bat = 0; bat < batch; bat++){
		int offset = bat * data_g.grid_size[0] * data_g.grid_size[1];

		for(unsigned int i = 0; i < data_g.actual_size[1]; i++){
			for(unsigned int j = 0; j < data_g.actual_size[0]; j++){

				if(i == 0 || j == 0 || i == data_g.actual_size[0] -1  || j==data_g.actual_size[1]-1){
					next[i*data_g.grid_size[0] + j + offset] = current[i*data_g.grid_size[0] + j + offset] ;
				} else {
					next[i*data_g.grid_size[0] + j + offset] = current[i*data_g.grid_size[0] + j + offset] * 0.5f + \
						   (current[(i-1)*data_g.grid_size[0] + j+ offset] + current[(i+1)*data_g.grid_size[0] + j+offset]) * 0.125f + \
						   (current[i*data_g.grid_size[0] + j+1+offset] + current[i*data_g.grid_size[0] + j-1+offset]) * 0.125f;
				}
			}
		}
	}
    return 0;
}

double square_error(float* current, float* next, ops::hls::GridPropertyCore data_g, unsigned int batch)
{
	double sum = 0;

	for(unsigned int bat = 0; bat < batch; bat++){
		int offset = bat * data_g.grid_size[0] * data_g.grid_size[1];

		for(unsigned int i = 0; i < data_g.actual_size[1]; i++){
			for(unsigned int j = 0; j < data_g.actual_size[0]; j++){
				int index = i*data_g.grid_size[0] + j+offset;
				float v1 = (next[index]);
				float v2 = (current[index]);

				if(fabs(v1-v2)/(fabs(v1) + fabs(v2)) >= 0.000001 && (fabs(v1) + fabs(v2)) > 0.000001 ){ //TODO: This epsilon can be parameterized
					printf("i:%d j:%d v1:%f v2:%f\n", i, j, v1, v2);
				}

				sum += next[index]*next[index] - current[index]*current[index];
			}
		}
	}
    return sum;
}

int copy_grid(float* grid_s, float* grid_d, ops::hls::GridPropertyCore data_g, unsigned int batch)
{
    for(unsigned int bat = 0; bat < batch; bat++){
    	int offset = bat * data_g.grid_size[0] * data_g.grid_size[1];

    	for(unsigned int i = 0; i < data_g.actual_size[1]; i++){
    		for(unsigned int j = 0; j < data_g.actual_size[0]; j++){
    			grid_d[i * data_g.grid_size[0] + j + offset] = grid_s[i * data_g.grid_size[0] + j + offset];
    		}
		}
    }
    return 0;
}

int initialise_grid(float* grid, ops::hls::GridPropertyCore data_g, unsigned int batch)
{
	for(unsigned int bat = 0; bat < batch; bat++){
		int offset = bat * data_g.grid_size[0] * data_g.grid_size[1];

		for(unsigned int i = 0; i < data_g.actual_size[1]; i++){
			for(unsigned int j = 0; j < data_g.actual_size[0]; j++){

				if(i == 0 || j == 0 || i == data_g.actual_size[0] - 1  || j==data_g.actual_size[1] - 1){
					float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
					grid[i * data_g.grid_size[0] + j + offset] = r;
				} else {
					grid[i * data_g.grid_size[0] + j + offset] = 0;
				}
			}
		}
	}
	return 0;
}