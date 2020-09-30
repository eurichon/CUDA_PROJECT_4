#ifndef VPTREE_H
#define VPTREE_H

#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include <iostream>
#include <stdio.h>
#include <chrono>

#include "bitonic.cuh"
#include "distance.cuh"
using namespace std;

#define ASYNC_CUDA_ERROR            -3
#define SYNCH_CUDA_ERROR            -4

void createVPTree(float *dataset, int n, int d);
void initIndexes(float *d, int n);
void copyIndexes(float *dest, float *source, int n);

__global__ void cudaInitIndexes(float *d, int n);
__global__ void cudaCopyIndexes(float *dest, float *source, int n);


#endif  // VPTREE_H