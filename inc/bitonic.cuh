#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include <iostream>
#include <chrono>
using namespace std;

#define GLOBAL_SYNCHRONIZATION


#define CEIL_DIV(x, y)      ((x / y) + (x % y != 0))

#define ASCENDING           1
#define DESCENDING          0

#define BLOCK_SIZE          512


long bitonic(float *data, unsigned int n, unsigned int dir);
void bitonic(float *data, float *index_map, unsigned int n, unsigned int dir,unsigned int iter); // overloaded function based hwich also sorts the index map

__global__ void cudaGridBitonic(float *d, unsigned int n, unsigned int step_iter, unsigned int curr_step, unsigned int dir);
__global__ void cudaGridBitonic(float *d, float *index_map, unsigned int n, unsigned int step_iter, unsigned int curr_step, unsigned int dir,unsigned int iter); // overloaded function

__global__ void cudaBlockBitonic(float *d, int n, int dir);
__device__ void cudaCompAndSwap(float *d, int i, int j, int dir);
__device__ void cudaSwap(float *a, float *b);


