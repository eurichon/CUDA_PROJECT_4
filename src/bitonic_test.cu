#include <iostream>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <math.h>
#include "bitonic.cuh"

using namespace std;

#define MEMORY_ERROR        -1
#define MIN_VALUE          -100.0
#define MAX_VALUE           100.0

#define FREE               -1
#define FORCED              0

#define ASCENDING           1
#define DESCENDING          0

int bitonic(float *h_data, int n, int dir);

// testing
void serialBitonicGrid(int thread,float *d, int n, int step, int dir, int force);
void serialBlockBitonic(int thread, int block,float *d, int n, int dir, int force, int value);
void serialBlockBitonicMerge(float *data, int thread_id, int block_id, int step, int dir, int force);

int cmpfunc (const void * a, const void * b) {
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return (fa > fb) - (fa < fb);
}



int main(int argc, char *argv[]){

    int n = atoi(argv[1]);
    int dir = atoi(argv[2]);
    int step = atoi(argv[3]);

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

    auto start = std::chrono::high_resolution_clock::now();
    
    qsort(h_data, n, sizeof(float), cmpfunc);

    auto finish = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
    cout << "Quicksort CPU time: " << cpu_time << endl;

    cout << "The result is: " << success << endl;
    cout << "Speed up: " << (float)cpu_time / gpu_time << endl;
    
    

    //test serially the logic 
    // cout << "Free" << endl;
    // for(int i = 0; i < n/2; i++)
    //     serialBitonicGrid(i, NULL, n, step, dir, FREE);
    // cout << endl;

    // cout << "Forced" << endl;
    // for(int i = 0; i < n/2; i++)
    //     serialBitonicGrid(i, NULL, n, step, dir, FORCED);
    //     cout << endl;

    // cout << "Free" << endl;
    // for(int i = 0; i < n/2; i++)
    //     serialBlockBitonic(1, i, NULL, 2, dir, FORCED, step);
    // cout << endl;




    // free memeory
    free(h_data);
    
    return 0;
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
        cudaBlockBitonic<<<grid, block, shared_mem>>>(d_data, n, dir, FREE, 0);
    }else{
        dim3 grid(n / 2048, 1);
        dim3 block(1024, 1);
        int shared_mem = 2048 * sizeof(float);

        // cudaBlockBitonic<<<grid, block, shared_mem>>>(d_data, 2048, dir, FREE, 0);

        // cudaGridBitonic<<<grid, block, shared_mem>>>(d_data, n, 2048, dir, FREE);
        // cudaBlockBitonic<<<grid, block, shared_mem>>>(d_data, 2048, dir, FORCED, 2);

        // cudaGridBitonic<<<grid, block, shared_mem>>>(d_data, n, 4096, dir, FORCED);
        // cudaBlockBitonic<<<grid, block, shared_mem>>>(d_data, 2048, dir, FORCED, 4);

        cudaBlockBitonic<<<grid, block, shared_mem>>>(d_data, 2048, dir, FREE, 0);

        for(int s = 2048; s < n/2; s <<= 1){
            cudaGridBitonic<<<grid, block, shared_mem>>>(d_data, n, s, dir, FREE);
            cudaBlockBitonic<<<grid, block, shared_mem>>>(d_data, 2048, dir, FORCED, s / 1024);
        }

        cudaGridBitonic<<<grid, block, shared_mem>>>(d_data, n, n/2, dir, FORCED);
        cudaBlockBitonic<<<grid, block, shared_mem>>>(d_data, 2048, dir, FORCED, n / 2048);        
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










void serialBitonicGrid(int thread,float *d, int n, int step, int dir, int force){
    //extern __shared__ float s_mem[];

    int thread_id = thread;
    bool flag = 1;
    int c_dir;

    for(int s = step; s >= 2; s >>= 1){ 
        int power_step = (s << 1);
        int a = thread_id / s;
        int b = thread_id % s;
        int pos = a * power_step + b;
        if(flag){
            c_dir = dir;

            if(force == -1)
                c_dir = (a % 2 != dir); // to check

            flag = 0;
        }
        

        // copy data
        cout << "Thread: " << thread_id << " [" << pos << ", " << pos + s << "]  dir= " << c_dir << endl;

        // s_mem[threadIdx.x] = d[pos];
        // s_mem[threadIdx.x + blockDim.x] = d[pos + s];

        // __syncthreads();

        // // relative call to gshared memory
        // cudaCompAndSwap(s_mem, (threadIdx.x), (threadIdx.x + blockDim.x), c_dir);

        // __syncthreads();

        // // copy data
        // d[pos] = s_mem[threadIdx.x];
        // d[pos + s] = s_mem[threadIdx.x + blockDim.x];
    }
}







void serialBlockBitonic(int thread, int block, float *d, int n, int dir, int force, int value){
    //extern __shared__ float s_mem[];

    // int offset = blockIdx.x * (2 * blockDim.x);
    
    // // copy data from global memory to the shared memory
    // s_mem[threadIdx.x] = d[threadIdx.x + offset];
    // s_mem[blockDim.x + threadIdx.x] = d[blockDim.x + threadIdx.x + offset];

    
    if(force == -1){
        // used for normal sorting for 1 block up to 2048 or n blocks of 2048 elements
        for(int step = 1; step <= n/2; step <<= 1){
            serialBlockBitonicMerge(NULL, thread, block, step, dir, force);
        }
    }else{
        // used as a complement to merge the blocks when elements are less than 2048 and can fit into a block
        // calculate current direction
        int n_dir;
        if(dir){
            n_dir = ((block & value) > 0)?(0):(1);
        }else{
            n_dir = ((block & value) > 0)?(1):(0);
        }
        serialBlockBitonicMerge(NULL, thread, block, n/2, n_dir, force);
    }
}

void serialBlockBitonicMerge(float *data, int thread_id, int block_id, int step, int dir, int force){
    int power_step = step << 1;
    int a = thread_id / step;
    int b = thread_id % step;
    int pos = a * power_step + b;
    int c_dir = dir;

    if(force == -1){
        if(dir == 1){
            c_dir = (a % 2 == (block_id % 2));
        }else{
            c_dir = (a % 2 != (block_id % 2));
        }
    }
    cout << "Thread: " << block_id << " [" << pos << ", " << pos + step << "]  dir= " << c_dir << endl;

    //cudaCompAndSwap(data, pos, pos + step, c_dir);

    while(step >= 2){
        step = step >> 1;
        power_step = power_step >> 1;
        a = thread_id / step;
        b = thread_id % step;
        pos = a * power_step + b;
        
        //__syncthreads();
        //udaCompAndSwap(data, pos, pos + step, c_dir);          
        cout << "Thread: " << thread_id << " [" << pos << ", " << pos + step << "]  dir= " << c_dir << endl;
  
    }
}