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

int bitonic(float *h_data, int n, int dir);

__global__ void cudaBlockBitonic(float *d, int n, int dir, int force);
__global__ void cudaGridBitonic(float *d, int n, int dir);

__device__ void cudaSwap(float *a, float *b);
__device__ void cudaCompAndSwap(float *d, int i, int j, int dir);



int cmpfunc (const void * a, const void * b) {
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return (fa > fb) - (fa < fb);
 }



int main(int argc, char *argv[]){

    int n = atoi(argv[1]);
    int dir = atoi(argv[2]);

    float *h_data; 

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

   
    
    int gpu_time = bitonic(h_data, n, dir);


    // check
    bool success = 1;
    for(int i = 1; i < n; i++){
        if(dir == 1){
            if(h_data[i - 1] >  h_data[i]){
                success = 0;
                cout << "error in i: " << i << ", " << h_data[i - 1] - h_data[i] << endl;
                break;
            }
        }else{
            if(h_data[i - 1] <  h_data[i]){
                success = 0;
                cout << "error in i: " << i << ", " << h_data[i - 1] - h_data[i] << endl;
                break;
            }
        }
        
    }

    for(int i = 2000; i < 2100; i++){
        cout << i << ") " << h_data[i] << endl;
    }



    auto start = std::chrono::high_resolution_clock::now();
    qsort(h_data, n, sizeof(float), cmpfunc);
    auto finish = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
    cout << "Quicksort CPU time: " << cpu_time << endl;


    cout << "The result is: " << success << endl;
    cout << "Speed up: " << (float)cpu_time / gpu_time << endl;
    

    // free memeory
    free(h_data);
    
    return 0;
}




__global__ void cudaGridBitonic(float *d, int n, int dir){
    extern __shared__ float s_mem[];


    int offset_a = blockIdx.x * blockDim.x;
    int offset_b = n / 2;

    // copy data
    s_mem[threadIdx.x] = d[threadIdx.x + offset_a];
    s_mem[blockDim.x + threadIdx.x] = d[threadIdx.x + offset_a + offset_b];

    __syncthreads();
    int pos = threadIdx.x;
    int step = blockDim.x;
    cudaCompAndSwap(s_mem, pos, pos + step, dir);

    __syncthreads();

    d[threadIdx.x + offset_a] = s_mem[threadIdx.x];
    d[threadIdx.x + offset_a + offset_b] = s_mem[blockDim.x + threadIdx.x];
}


__global__ void cudaBlockBitonic(float *d, int n, int dir, int force){
    extern __shared__ float s_mem[];

    int offset = blockIdx.x * (2 * blockDim.x);
    
    // copy data
    s_mem[threadIdx.x] = d[threadIdx.x + offset];
    s_mem[blockDim.x + threadIdx.x] = d[blockDim.x + threadIdx.x + offset];

    __syncthreads();
    if(force == -1){
        for(int i = 1; i <= n/2; i <<= 1){
            int step = i;
            int power_step = step << 1;
            int a = threadIdx.x / step;
            int b = threadIdx.x % step;
            int pos = a * power_step + b;
            int c_dir;

            if(dir == 1){
                c_dir = (a % 2 == (blockIdx.x % 2));
            }else{
                c_dir = (a % 2 != (blockIdx.x % 2));
            }

            cudaCompAndSwap(s_mem, pos, pos + step, c_dir);

            while(step >= 2){
                step = step >> 1;
                power_step = power_step >> 1;
                a = threadIdx.x / step;
                b = threadIdx.x % step;
                pos = a * power_step + b;
                
                __syncthreads();
                cudaCompAndSwap(s_mem, pos, pos + step, c_dir);            
            }

            __syncthreads();
        }
    }else{
        int step = n/2;
        int power_step = step << 1;
        int a = threadIdx.x / step;
        int b = threadIdx.x % step;
        int pos = a * power_step + b;
        int c_dir = force;

        cudaCompAndSwap(s_mem, pos, pos + step, c_dir);

        while(step >= 2){
            step = step >> 1;
            power_step = power_step >> 1;
            a = threadIdx.x / step;
            b = threadIdx.x % step;
            pos = a * power_step + b;
            
            __syncthreads();
            cudaCompAndSwap(s_mem, pos, pos + step, c_dir);            
        }

        __syncthreads();
    }

    d[threadIdx.x + offset] = s_mem[threadIdx.x];
    d[blockDim.x + threadIdx.x + offset] = s_mem[blockDim.x + threadIdx.x];
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



int bitonic(float *h_data, int n, int dir){
    float *d_data = NULL;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    if(n <= 2048){
        dim3 grid(1, 1);
        dim3 block((n / 2), 1);
        int shared_mem = n * sizeof(float);
        cudaBlockBitonic<<<grid, block, shared_mem>>>(d_data, n, dir, -1);
    }else{
        dim3 grid(2, 1);
        dim3 block(1024, 1);
        int shared_mem = 2048 * sizeof(float);

        cudaBlockBitonic<<<grid, block, shared_mem>>>(d_data, 2048, dir, -1);
        cudaGridBitonic<<<grid, block, shared_mem>>>(d_data, n, dir);
        cudaBlockBitonic<<<grid, block, shared_mem>>>(d_data, 2048, dir, dir);


    }



   
    
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();

    auto finish = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
    cout << " GPU time: " << gpu_time << endl;

    cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

    if (errSync != cudaSuccess)
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

    cudaFree(d_data);
    return gpu_time;
}

