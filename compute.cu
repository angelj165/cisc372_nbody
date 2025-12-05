//Angel Jose and Samita Bomasamudram
#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "config.h"

__global__ void nbody_kernel(vector3* d_pos, vector3* d_vel, double* d_mass, int numEntities) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numEntities) {
        return;
    }

    double pos_x = d_pos[i][0];
    double pos_y = d_pos[i][1];
    double pos_z = d_pos[i][2];

    double vel_x = d_vel[i][0];
    double vel_y = d_vel[i][1];
    double vel_z = d_vel[i][2];

    double accel_x_sum = 0.0;
    double accel_y_sum = 0.0;
    double accel_z_sum = 0.0;

    for (int j = 0; j < numEntities; j++) {
        if (i == j) {
            continue;
        }

        double delta_x = pos_x - d_pos[j][0];
        double delta_y = pos_y - d_pos[j][1];
        double delta_z = pos_z - d_pos[j][2];

        double distance_squared = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
        double distance = sqrt(distance_squared);
        
        if (distance_squared > 1e-10) {
            double acceleration_magnitude = -1.0 * GRAV_CONSTANT * d_mass[j] / distance_squared;
            
            accel_x_sum += acceleration_magnitude * delta_x / distance;
            accel_y_sum += acceleration_magnitude * delta_y / distance;
            accel_z_sum += acceleration_magnitude * delta_z / distance;
        }
    }

    vel_x += accel_x_sum * INTERVAL;
    vel_y += accel_y_sum * INTERVAL;
    vel_z += accel_z_sum * INTERVAL;

    pos_x += vel_x * INTERVAL;
    pos_y += vel_y * INTERVAL;
    pos_z += vel_z * INTERVAL;

    d_vel[i][0] = vel_x;
    d_vel[i][1] = vel_y;
    d_vel[i][2] = vel_z;

    d_pos[i][0] = pos_x;
    d_pos[i][1] = pos_y;
    d_pos[i][2] = pos_z;
}

extern "C" void compute() {
    vector3 *dev_pos, *dev_vel;
    double *dev_mass;
    size_t size_vec = NUMENTITIES * sizeof(vector3);
    size_t size_mass = NUMENTITIES * sizeof(double);
    cudaError_t err;

    err = cudaMalloc((void**)&dev_pos, size_vec);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Unable to allocate GPU memory for positions (dev_pos): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMalloc((void**)&dev_vel, size_vec);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Unable to allocate GPU memory for velocities (dev_vel): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMalloc((void**)&dev_mass, size_mass);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Unable to allocate GPU memory for masses (dev_mass): %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(dev_pos, hPos, size_vec, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vel, hVel, size_vec, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mass, mass, size_mass, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (NUMENTITIES + threads_per_block - 1) / threads_per_block;

    nbody_kernel<<<blocks_per_grid, threads_per_block>>>(dev_pos, dev_vel, dev_mass, NUMENTITIES);
    

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    cudaDeviceSynchronize();

    cudaMemcpy(hPos, dev_pos, size_vec, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dev_vel, size_vec, cudaMemcpyDeviceToHost);

    cudaFree(dev_pos);
    cudaFree(dev_vel);
    cudaFree(dev_mass);
}