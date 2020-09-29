#ifndef DISTANCE_H
#define DISTANCE_H

#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include <stdio.h>

#define WARP_SIZE           32
#define MAX_BLOCK_THREADS   512
#define MAX_WARPS_BLOCK_Y   16
#define MIN_WARPS_BLOCK_Y   1


typedef struct{
    dim3 block;
    dim3 grid;
    int s_mem;
}BestSplit;


BestSplit findBestSplit(int n, int d);

void parallelReduce(float *distances, float *data, int n, int d);

__global__ void cudaReduce(float *temp, float *distances, int n, int d, int r);
__global__ void cudaDotProduct(float *dataset, float *point, float *product, int n, int d, int r);



#endif  // DISTANCE_H