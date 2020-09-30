#ifndef DISTANCE_H
#define DISTANCE_H

#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include <iostream>
using namespace std;

#define WARP_SIZE           32
#define MAX_BLOCK_THREADS   512
#define MAX_WARPS_BLOCK_Y   16
#define MIN_WARPS_BLOCK_Y   1

//#define GLOBAL_SYNCHRONIZATION

typedef struct{
    dim3 block;
    dim3 grid;
    int s_mem;
}BestSplit;


BestSplit findBestSplit(int n, int d);

void parallelDistance(float *distances, float *data, float *index_map, int n, int d, int iter);

__global__ void cudaReduce(float *temp, float *distances, int n, int d, int r);
__global__ void cudaDotProduct(float *dataset, float *index_map, float *product, int n, int d, int r, int iter);



#endif  // DISTANCE_H