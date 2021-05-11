#include "range_libc/CudaRangeLib.h"
// #include "range_libc/RangeUtils.h"


#include <cuda.h>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdio.h>

#define DIST_THRESHOLD 0.0
#define STEP_COEFF 0.999

__device__ float distance(int x, int y, float *distMap, int width, int height) {
	return distMap[x * height + y];
}

__global__ void cuda_ray_marching(float * ins, float * outs, float * distMap, int width, int height, float max_range, int num_casts) {
	int ind = blockIdx.x*blockDim.x + threadIdx.x;
	if (ind >= num_casts) return;
	float x0 = ins[ind*3];
	float y0 = ins[ind*3+1];
	float theta = ins[ind*3+2];

	float ray_direction_x = cosf(theta);
	float ray_direction_y = sinf(theta);

	int px = 0;
	int py = 0;

	float t = 0.0;
	float out = max_range;
	// int iters = 0;
	while (t < max_range) {
		px = x0 + ray_direction_x * t;
		py = y0 + ray_direction_y * t;

		if (px >= width || px < 0 || py < 0 || py >= height) {
			out = max_range;
			break;
		}

		float d = distance(px,py, distMap, width, height);

		if (d <= DIST_THRESHOLD) {
			float xd = px - x0;
			float yd = py - y0;
			out =  sqrtf(xd*xd + yd*yd);
			break;
		}

		t += fmaxf(d * STEP_COEFF, 1.0);
		// iters ++;
	}
	outs[ind] = out;
}

__global__ void cuda_ray_marching_world_to_grid(float * ins, float * outs, float * distMap, int width, int height, float max_range, int num_casts, float x_origin_world, float y_origin_world, float scale_world, float inv_scale_world, float sin_th_world, float cos_th_world, float rotation_const) {
	int ind = blockIdx.x*blockDim.x + threadIdx.x;
	if (ind >= num_casts) return;

	float x_world = ins[ind*3];
	float y_world = ins[ind*3+1];
	float theta_world = ins[ind*3+2];

	// convert x0,y0,theta from world to grid space coordinates
	float x0 = (x_world - x_origin_world) * inv_scale_world;
	float y0 = (y_world - y_origin_world) * inv_scale_world;

  // Careful - th_world is the angle of the map in the world frame, thus it converts from map to world
  // To convert from world to map, we need to use the inverse rotation matrix here
	float temp = x0;
	x0 =   cos_th_world * x0   + sin_th_world * y0;
	y0 = - sin_th_world * temp + cos_th_world * y0;
	float theta = -theta_world + rotation_const;

	// swap components
	temp = x0;
	x0 = y0;
	y0 = temp;

	// do ray casting
	float ray_direction_x = cosf(theta);
	float ray_direction_y = sinf(theta);

	int px = 0;
	int py = 0;

	float t = 0.0;
	float out = max_range;
	while (t < max_range) {
		px = x0 + ray_direction_x * t;
		py = y0 + ray_direction_y * t;

		if (px >= width || px < 0 || py < 0 || py >= height) {
			out = max_range;
			break;
		}

		float d = distance(px,py, distMap, width, height);

		if (d <= DIST_THRESHOLD) {
			float xd = px - x0;
			float yd = py - y0;
			out =  sqrtf(xd*xd + yd*yd);
			break;
		}

		t += fmaxf(d * STEP_COEFF, 1.0);
	}
	outs[ind] = out * scale_world;
}

__global__ void cuda_ray_marching_angles_world_to_grid(float * ins, float * outs, float * distMap, int width, int height, float max_range, int num_particles, int num_angles, float x_origin_world, float y_origin_world, float scale_world, float inv_scale_world, float sin_th_world, float cos_th_world, float rotation_const) {
	int ind = blockIdx.x*blockDim.x + threadIdx.x;
	if (ind >= num_angles*num_particles) return;

	int angle_ind = fmodf( ind, num_angles );
	int particle_ind = (float) ind / (float) num_angles;

	float x_world = ins[particle_ind*3];
	float y_world = ins[particle_ind*3+1];
	float theta_world = ins[particle_ind*3+2];

	// convert x0,y0,theta from world to grid space coordinates
	float x0 = (x_world - x_origin_world) * inv_scale_world;
	float y0 = (y_world - y_origin_world) * inv_scale_world;

  // Careful - th_world is the angle of the map in the world frame, thus it converts from map to world
  // To convert from world to map, we need to use the inverse rotation matrix here
	float temp = x0;
	x0 =   cos_th_world * x0   + sin_th_world * y0;
	y0 = - sin_th_world * temp + cos_th_world * y0;
	float theta = -theta_world + rotation_const - ins[num_particles * 3 + angle_ind];

	// swap components
	temp = x0;
	x0 = y0;
	y0 = temp;

	// do ray casting
	float ray_direction_x = cosf(theta);
	float ray_direction_y = sinf(theta);

	int px = 0;
	int py = 0;

	float t = 0.0;
	float out = max_range;
	while (t < max_range) {
		px = x0 + ray_direction_x * t;
		py = y0 + ray_direction_y * t;

		if (px >= width || px < 0 || py < 0 || py >= height) {
			out = max_range;
			break;
		}

		float d = distance(px,py, distMap, width, height);

		if (d <= DIST_THRESHOLD) {
			float xd = px - x0;
			float yd = py - y0;
			out =  sqrtf(xd*xd + yd*yd);
			break;
		}
		t += fmaxf(d * STEP_COEFF, 1.0);
	}
	outs[ind] = out * scale_world;
}

__device__ int clamp(float val, float min, float max) {
	val = val>max?max:val;
	val = val<min?min:val;
	return (int)val;
}

// this should be optimized to use shared memory, otherwise the random read performance is not great
__global__ void cuda_eval_sensor_table(float * obs, float * ranges, double * outs, double * sensorTable, int rays_per_particle, int particles, float inv_scale_world, int max_range) {
	int ind = blockIdx.x*blockDim.x + threadIdx.x;
	if (ind >= rays_per_particle) return;

	int r = clamp(obs[ind] * inv_scale_world,0,max_range-1.0);
	for (int i = 0; i < particles; ++i)
	{
		int d = clamp(ranges[ind + ind * i],0,max_range-1.0);
		outs[ind+i*rays_per_particle] = sensorTable[r*max_range+d];
	}
}

// this should be optimized to use shared memory, otherwise the random read performance is not great
// __global__ void cuda_accumulate_weights(float * obs, float * ranges, double * outs, double * sensorTable, int rays_per_particle, int particles, float inv_scale_world, int max_range) {
// 	int ind = blockIdx.x*blockDim.x + threadIdx.x;
// 	if (ind >= particles) return;

// 	double weight = 1.0;
// 	for (int i = 0; i < rays_per_particle; ++i)
// 	{
// 		weight *= outs[ind*rays_per_particle+i];

// 	}
// }

void err_check() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
    	printf("Error: %s\n", cudaGetErrorString(err));
}

RayMarchingCUDA::RayMarchingCUDA(std::vector<std::vector<float> > grid, int w, int h, float mr)
	: width(w), height(h), max_range(mr) {
	cudaMalloc((void **)&d_ins, sizeof(float) * CHUNK_SIZE * 3);
	cudaMalloc((void **)&d_outs, sizeof(float) * CHUNK_SIZE);
	cudaMalloc((void **)&d_distMap, sizeof(float) * width * height);

	// convert vector format to raw float array, y axis is quickly changing dimension
	// float raw_grid[width*height];
	float *raw_grid = new float[width*height];
	for (int i = 0; i < width; ++i) std::copy(grid[i].begin(), grid[i].end(), &raw_grid[i*height]);

	cudaMemcpy(d_distMap, raw_grid, width*height*sizeof(float), cudaMemcpyHostToDevice);
	free(raw_grid);
}

RayMarchingCUDA::~RayMarchingCUDA() {
	cudaFree(d_ins); cudaFree(d_outs); cudaFree(d_distMap);
}

void RayMarchingCUDA::set_sensor_table(double *table, int t_w) {
	table_width = t_w;
	int table_size = sizeof(double) * table_width * table_width;
	cudaMalloc((void **)&d_sensorTable, table_size);
	cudaMemcpy(d_sensorTable, table, table_size, cudaMemcpyHostToDevice);
}

// num_casts must be less than or equal to chunk size
void RayMarchingCUDA::calc_range_many(float *ins, float *outs, int num_casts) {
	// copy queries to GPU buffer
	cudaMemcpy(d_ins, ins, sizeof(float) * num_casts * 3,cudaMemcpyHostToDevice);
	// execute queries on the GPU
	cuda_ray_marching<<< CHUNK_SIZE / NUM_THREADS, NUM_THREADS >>>(d_ins,d_outs, d_distMap, width, height, max_range, num_casts);
	err_check();

	// copy results back to CPU
	cudaMemcpy(outs,d_outs,sizeof(float)*num_casts,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}

// num_casts must be less than or equal to chunk size
void RayMarchingCUDA::numpy_calc_range(float *ins, float *outs, int num_casts) {
	#if WORLD_TO_GRID_CONVERSION == 1
	// copy queries to GPU buffer
	cudaMemcpy(d_ins, ins, sizeof(float) * num_casts * 3,cudaMemcpyHostToDevice);
	// execute queries on the GPU, have to pass coordinate space conversion constants
	cuda_ray_marching_world_to_grid<<< CHUNK_SIZE / NUM_THREADS, NUM_THREADS >>>(d_ins,d_outs, d_distMap, width, height, max_range,
		num_casts, x_origin_world, y_origin_world, scale_world, inv_scale_world, sin_th_world, cos_th_world, rotation_const);
	err_check();
	// copy results back to CPU
	cudaMemcpy(outs,d_outs,sizeof(float)*num_casts,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	#else
	std::cout << "GPU numpy_calc_range only works with ROS world to grid conversion enabled" << std::endl;
	#endif
}


// num_casts must be less than or equal to chunk size
void RayMarchingCUDA::numpy_calc_range_angles(float * ins, float * angles, float * outs, int num_particles, int num_angles) {
	#if WORLD_TO_GRID_CONVERSION == 1
	// copy queries to GPU buffer
	cudaMemcpy(d_ins, ins, sizeof(float) * num_particles * 3,cudaMemcpyHostToDevice);
	// also copy angles to end of GPU buffer, this assumes there is enough space (which there should be)
	cudaMemcpy(&d_ins[num_particles * 3], angles, sizeof(float) * num_angles,cudaMemcpyHostToDevice);
	// execute queries on the GPU, have to pass coordinate space conversion constants
	cuda_ray_marching_angles_world_to_grid<<< CHUNK_SIZE / NUM_THREADS, NUM_THREADS >>>(d_ins,d_outs, d_distMap, width, height, max_range,
		num_particles, num_angles, x_origin_world, y_origin_world, scale_world, inv_scale_world, sin_th_world, cos_th_world, rotation_const);
	err_check();
	// copy results back to CPU
	cudaMemcpy(outs,d_outs,sizeof(float)*num_particles*num_angles,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	#else
	std::cout << "GPU numpy_calc_range_angles only works with ROS world to grid conversion enabled" << std::endl;
	#endif
}

void RayMarchingCUDA::calc_range_repeat_angles_eval_sensor_model(float * ins, float * angles, float * obs, double * weights, int num_particles, int num_angles) {
	#if WORLD_TO_GRID_CONVERSION == 1
	std::cout << "Do not use calc_range_repeat_angles_eval_sensor_model for GPU, unimplemented" << std::endl;
	std::cout << "Instead use numpy_calc_range_angles followed by a standard sensor evaluation method." << std::endl;
	// if (!allocated_weights) {
	// 	cudaMalloc((void **)&d_weights, num_particles*num_angles*sizeof(double));
	// 	allocated_weights = true;
	// }
	// // copy queries to GPU buffer
	// cudaMemcpy(d_ins, ins, sizeof(float) * num_particles * 3,cudaMemcpyHostToDevice);
	// // also copy angles to end of GPU buffer, this assumes there is enough space (which there should be)
	// cudaMemcpy(&d_ins[num_particles * 3], angles, sizeof(float) * num_angles,cudaMemcpyHostToDevice);

	// // execute queries on the GPU, have to pass coordinate space conversion constants
	// cuda_ray_marching_angles_world_to_grid<<< CHUNK_SIZE / NUM_THREADS, NUM_THREADS >>>(d_ins,d_outs, d_distMap,
	// 	width, height, max_range, num_particles, num_angles, x_origin_world, y_origin_world,
	// 	scale_world, inv_scale_world, sin_th_world, cos_th_world, rotation_const);

	// cudaMemcpy(d_ins, obs, sizeof(float) * num_angles,cudaMemcpyHostToDevice);

	// // read from sensor table
	// cuda_eval_sensor_table<<< CHUNK_SIZE / NUM_THREADS, NUM_THREADS >>>(d_ins, d_outs, d_weights, d_sensorTable, num_angles, num_particles, inv_scale_world, table_width);

	// cuda_eval_sensor_table<<< CHUNK_SIZE / NUM_THREADS, NUM_THREADS >>>(d_ins, d_outs, d_weights, d_sensorTable, num_angles, num_particles, inv_scale_world, table_width);
	// // cuda_eval_sensor_table(angles, d_sensorTable)
	// // multiplicatively accumulate weights on the GPU


	// // copy weights back to CPU
	// cudaMemcpy(weights,d_weights,sizeof(double)*num_particles,cudaMemcpyDeviceToHost);
	// cudaDeviceSynchronize();
	#else
	std::cout << "GPU numpy_calc_range_angles only works with ROS world to grid conversion enabled" << std::endl;
	#endif
}
