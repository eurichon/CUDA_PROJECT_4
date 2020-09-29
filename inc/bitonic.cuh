#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"

#define CEIL_DIV(x, y)          ((x / y) + (x % y != 0))


__global__ void cudaGridBitonic(float *d, int n, int step_iter, int curr_step, int dir);

__global__ void cudaBlockBitonicReduce(float *d, int n, int rate, int dir);

__global__ void cudaBlockBitonic(float *d, int n, int dir);

__device__ void cudaCompAndSwap(float *d, int i, int j, int dir);

__device__ void cudaSwap(float *a, float *b);


