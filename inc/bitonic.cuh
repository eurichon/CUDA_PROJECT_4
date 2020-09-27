#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"



__global__ void cudaBlockBitonic(float *d, int n, int dir, int force, int value);
__global__ void cudaGridBitonic(float *d, int n, int step, int dir, int force);
__device__ void cudaSwap(float *a, float *b);
__device__ void cudaCompAndSwap(float *d, int i, int j, int dir);
__device__ void cudaBlockBitonicMerge(float *data, int thread_id, int block_id, int step, int dir, int force);