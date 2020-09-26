#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include <iostream>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <math.h>

using namespace std;

#define MEMORY_ERROR        -1
#define MIN_VALUE          -100.0
#define MAX_VALUE           100.0


__global__ void cudaBlockBitonic(float *d, int n);
__device__ void cudaSwap(float *a, float *b);
__device__ void cudaCompAndSwap(float *d, int i, int j, int dir);



int cmpfunc (const void * a, const void * b) {
    return (int)( *(float*)a - *(float*)b );
 }



int main(int argc, char *argv[]){

    int n = atoi(argv[1]);

    float *h_data; 
    float *d_data;
    
    //create and populate host dataset
    h_data = (float *)malloc(n * sizeof(float));
    if(h_data == NULL){
        cout << "Not enough memory. Aborting ..." << endl;
        free(h_data);
        return MEMORY_ERROR;
    }else{
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator(seed);
        uniform_real_distribution<float> distribution(MIN_VALUE, MAX_VALUE);

        for(int i = 0; i < n; i++){
            h_data[i] = distribution(generator);
        }
    }

   
    // // print data
    // for(int i = 0;i < SIZE; i++){
    //     cout << h_data[i] << endl;
    // }

    // create and populate device dataset
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);


    // the good stuff
    auto start = std::chrono::high_resolution_clock::now();

    dim3 block(n/2, 1);
    int shared_mem = n *  sizeof(float);

    cudaBlockBitonic<<<1, block, shared_mem>>>(d_data, n); 
    

    
  

    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();

    auto finish = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
    cout << " GPU time: " << gpu_time << endl;

    if (errSync != cudaSuccess)
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    if (errSync == cudaSuccess && errAsync == cudaSuccess)
        printf("Kernals succesfully finished without any errors!\n");

    start = std::chrono::high_resolution_clock::now();

    qsort(h_data, n, sizeof(float), cmpfunc);

    finish = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
    cout << "Quicksort CPU time: " << cpu_time << endl;


    cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // print data
    bool success = 1;
    for(int i = 1;i < n; i++){
        if(h_data[i - 1] > h_data[i]){
            success = 0;
            cout << "error is i" << i << endl;
            break;
        }
    }

    cout << "The result is: " << success << endl;
    cout << "Speed up: " << (float)cpu_time / gpu_time << endl;


    // free memeory
    //free(h_data);
    cudaFree(d_data);

    return 0;
}





__global__ void cudaBlockBitonic(float *d, int n){
    extern __shared__ float s_mem[];
    
    // copy data
    s_mem[threadIdx.x] = d[threadIdx.x];
    s_mem[blockDim.x + threadIdx.x] = d[blockDim.x + threadIdx.x];

    __syncthreads();


    for(int i = 1; i <= n/2; i <<= 1){
        int step = i;
        int power_step = step << 1;
        int a = threadIdx.x / step;
        int b = threadIdx.x % step;
        int pos = a * power_step + b;
        int dir = (a % 2 == 0);

        cudaCompAndSwap(s_mem, pos, pos + step, dir);

        while(step >= 2){
            step = step >> 1;
            power_step = power_step >> 1;
            a = threadIdx.x / step;
            b = threadIdx.x % step;
            pos = a * power_step + b;
            
            __syncthreads();
            cudaCompAndSwap(s_mem, pos, pos + step, dir);            
        }

        __syncthreads();
    }

    d[threadIdx.x] = s_mem[threadIdx.x];
    d[blockDim.x + threadIdx.x] = s_mem[blockDim.x + threadIdx.x];
}



__device__ void cudaCompAndSwap(float *d, int i, int j, int dir){
    if((d[i] > d[j]) == dir){
        cudaSwap((d + i), (d + j));
    }
}


__device__ void cudaSwap(float *a, float *b){
    float temp = *a;
    *a = *b;
    *b = temp;
}




__device__ void bitonicSort(float *d ){

}



__device__ void bitonicMerge(){

}


