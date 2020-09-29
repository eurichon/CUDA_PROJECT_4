#include <iostream>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <math.h>
#include "bitonic.cuh"

using namespace std;

#define MEMORY_ERROR        -1
#define MIN_VALUE          -1000.0
#define MAX_VALUE           1000.0

#define ASCENDING           1
#define DESCENDING          0

long bitonic(float *h_data, int n, int dir);

// testing
void serialBitonicGrid(int block, float *d, int step_iter, int curr_step, int dir);
void serialBlockBitonic(int thread, int block, int dir, int rate);


int cmpfunc (const void * a, const void * b) {
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return (fa > fb) - (fa < fb);
}



void initArray(float *data, int n){
    if(data == NULL){
        cout << "Not enough memory. Aborting ..." << endl;
        free(data);
        exit(MEMORY_ERROR);
    }else{
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator(seed);
        uniform_real_distribution<float> distribution(MIN_VALUE, MAX_VALUE);

        for(int i = 0; i < n; i++){
            data[i] = distribution(generator);
        }
    }
}




int main(int argc, char *argv[]){

    int n = atoi(argv[1]);
    int dir = atoi(argv[2]);
    int step = atoi(argv[3]);

    float *h_data; 
    h_data = (float *)malloc(n * sizeof(float));
    initArray(h_data, n);

   
    
    long gpu_time = bitonic(h_data, n, dir);


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

    // for(int i = 2000; i < 2100;  i++){
    //     cout << i << ")" << h_data[i] << endl;
    // }





    initArray(h_data, n);


    auto start = std::chrono::high_resolution_clock::now();
    
    qsort(h_data, n, sizeof(float), cmpfunc);

    auto finish = std::chrono::high_resolution_clock::now();
    long cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
    cout << "Quicksort CPU time: " << cpu_time << endl;

    cout << "The result is: " << success << endl;
    cout << "Speed up: " << (double)cpu_time / gpu_time << endl;
    
    
    

    // //test serially the logic 
    // cout << "Free" << endl;
    // for(int i = 0; i < n/2; i++)
    //     serialBlockBitonic(1, i, dir, step);
    // cout << endl;

    // cout << "Forced" << endl;

    // for(int j = 1; j <= n/2; j <<= 1){
    //     for(int k = j; k >= 1; k >>= 1){
    //         for(int i = 0; i < n/2; i++)
    //             serialBitonicGrid(i, NULL, j, k, dir);

    //         cout << endl;
    //     }
    //     cout << "========" << endl;
    // }

    

        

    // cout << "Free" << endl;
    // for(int i = 0; i < n/2; i++)
    //     serialBlockBitonic(1, i, NULL, 2, dir, FORCED, step);
    // cout << endl;




    // free memeory
    free(h_data);
    
    return 0;
}




long bitonic(float *h_data, int n, int dir){
    float *d_data = NULL;


    

    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);



    auto start = std::chrono::high_resolution_clock::now();
    int lanuch = 0;

    if(n <= 2048){
        dim3 grid(1, 1);
        dim3 block((n / 2), 1);
        int shared_mem = n * sizeof(float);
        cudaBlockBitonic<<<grid, block, shared_mem>>>(d_data, n, dir);
    }else{
        dim3 grid(n / 2048, 1);
        dim3 block(1024, 1);
        int shared_mem = 2048 * sizeof(float);

       
        for(int s = 4; s <= n; s <<= 1){
            for(int s2 = s; s2 > 0; s2 >>= 1){
                cudaGridBitonic<<<grid, block>>>(d_data, n, s, s2, dir);
            
                lanuch++;
            }
            
        }
        
    }

    cout << "launches " << lanuch << endl;

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










void serialBitonicGrid(int block, float *d, int step_iter, int curr_step, int dir){

    // calculate thread id in the grid
    // int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int thread = block;

    int offset = (curr_step << 1);
    int a = thread / curr_step;
    int b = thread % curr_step;
    int pos = a * offset + b;
    
    int c_dir = ((thread / step_iter) % 2) != dir;

    cout << "Thread: " << thread << " [" << pos << ", " << pos + curr_step << "]  dir= " << c_dir << endl;

    // cudaCompAndSwap(d, pos, (pos + curr_step), c_dir);  // to do it manually
}







void serialBlockBitonic(int thread, int block, int dir, int rate){
    // extern __shared__ float s_mem[];

    // int offset = blockIdx.x * (2 * blockDim.x);

    // // copy data from global memory to the shared memory
    // s_mem[threadIdx.x] = d[threadIdx.x + offset];
    // s_mem[blockDim.x + threadIdx.x] = d[blockDim.x + threadIdx.x + offset];
    thread = block;
    int c_dir = ((block & rate) == 0)?(dir):(!dir);
    
    for(unsigned int s = 1; s >= 1; s >>= 1){
        

        int step = s;
        int a = thread / s;
        int b = thread % s;
        int pos = a * (step << 1) + b;

        cout << "Thread: " << thread << " [" << pos << ", " << pos + s << "]  dir= " << c_dir << endl;
        
        // __syncthreads();
        // cudaCompAndSwap(s_mem, pos, pos + step, c_dir);
    }     
}

