#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include <iostream>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <math.h>
using namespace std;

// Define error values
#define MEMORY_ERROR        -2
#define INPUT_ERROR         -1

// Define Constant values
#define MIN_VALUE          -10000.0
#define MAX_VALUE           10000.0


#define ERROR_THRESHOLD     10e-1

const char Result[2][10] = {"SUCCEDED", "FAILED"};


void printData(float *dataset, int n, int d);
void calculateDistances(float *distances, float *dataset, float *point, int n, int d);

__global__ void deviceCalculatedDistances(float *dist, float *data, float *point, int n, int d);


int main(int argc, char *argv[]){
    int d;      // number of dimensions
    int n;      // number of points

    // host variables
    float *h_dataset = NULL;
    float *h_distances = NULL;
    float *h_test_distances = NULL;

    // device variables
    float *d_point = NULL;
    float *d_dataset = NULL;
    float *d_distances = NULL;

    if(argc != 3){
        cout << "Wrong number of arguments. Aborting ..." << endl;
        return INPUT_ERROR;
    }else{
        // assign values
        n = atoi(argv[1]);
        d = atoi(argv[2]);

        cout << "Initializing " << n << " random data points of " << d << " dimensions" << endl;
    }

    // creating the Random dataset
    h_dataset = (float *)malloc(n * d * sizeof(float));

    if(h_dataset == NULL){
        cout << "Not enough memory. Aborting ..." << endl;
        free(h_dataset);
        return MEMORY_ERROR;
    }else{
        // initialize random generator engine
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator(seed);
        uniform_real_distribution<float> distribution(MIN_VALUE, MAX_VALUE);

        int length = n * d;

        for(int i = 0; i < length; i++){
            h_dataset[i] = distribution(generator);
        }
    }

    // serial stuff
    h_test_distances = (float *)malloc(n * sizeof(float));
    h_distances = (float *)malloc(n * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();

    calculateDistances(h_test_distances, h_dataset, &h_dataset[0], n, d);

    auto finish = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
    //std::cout << "CPU Time: "<<std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << " ns\n";

    // cuda stuff
    cudaMalloc(&d_dataset, n * d * sizeof(float)); 
    cudaMalloc(&d_point, d * sizeof(float)); 
    cudaMalloc(&d_distances, n * sizeof(float)); 


    start = std::chrono::high_resolution_clock::now();

    // copy the dataset to the device
    cudaMemcpy(d_dataset, h_dataset, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_point, &d_dataset[0], d * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 16;
    dim3 num_of_blocks(n/block_size + (n % block_size != 0), 1);
    dim3 threads_per_block(block_size, 8);
    cout << "Dimensions of blocks: " << num_of_blocks.x << ", " << num_of_blocks.y << endl;

    deviceCalculatedDistances<<<num_of_blocks, threads_per_block>>>(d_distances,  d_dataset, d_point, n, d);

    // error checking
    cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess)
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
        

    // copy the calculated distances back to the host
    cudaMemcpy(h_distances, d_distances, n * sizeof(float), cudaMemcpyDeviceToHost);

    finish = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
    //std::cout << "GPU Time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << " ns\n";


    cout << "Total Speed up: " << (float)cpu_time / gpu_time << endl;

    // validation
    bool fail = 0;
    for(int i = 0; i < n; i++){
        if(abs(h_test_distances[i] - h_distances[i]) > ERROR_THRESHOLD){
            cout << "Error exeeded threshold("<< ERROR_THRESHOLD << "): " << abs(h_test_distances[i] - h_distances[i]) << endl;
            fail = 1;
            break;
        }
    }

    cout << "Program: " << Result[fail] << endl;

    // printData(h_distances, n ,1);



    cudaFree(d_dataset);
    cudaFree(d_distances);
    

    free(h_distances);
    free(h_dataset);

    return 0;
}


void calculateDistances(float *distances, float *dataset, float *point, int n, int d){
    float temp;

    for(int i = 0; i < n; i++){
        temp = 0.0;

        for(int j = 0; j < d; j++){
            temp += pow(dataset[i * d + j] - point[j], 2);
        }
        distances[i] = pow(temp, 0.5);
    }
}


void printData(float *dataset, int n, int d){
    cout << "============ Dataset ============" << endl;

    for(int i = 0; i < n; i++){
        cout << i + 1 << ") ";
        for(int j = 0; j < d - 1; j++){
            cout << dataset[i * d + j] << ", ";
        }
        cout << dataset[(i + 1) * d - 1] << endl;
    }
}


__global__ void deviceCalculatedDistances(float *dist, float *data, float *point, int n, int d){
    __shared__ float sum[256];

    int thread_id = threadIdx.x * blockDim.y + threadIdx.y;
    int block_id = blockIdx.x * gridDim.y + blockIdx.y;

    int elements_per_block = blockDim.x * d;
    int elements_offset = block_id * elements_per_block;

    int iterations = d / blockDim.y + (d % blockDim.y != 0);        // add +1 if the division is not perfect
    
    sum[thread_id] = 0;

    int line_pos = threadIdx.x * d;

    if(line_pos + elements_offset < n * d){
        for(int i = 0; i < iterations; i++){
            int row_pos = i * blockDim.y + threadIdx.y;
            
            if(row_pos < d){
                float val = data[elements_offset + line_pos + row_pos] - point[row_pos];
                sum[thread_id] = sum[thread_id] + pow(val, 2);
            }
        }

        __syncthreads();

        // reduction of sum
        for(int i = blockDim.y/2; i > 0; i >>= 1){
            if(threadIdx.y < i){
                sum[thread_id] += sum[thread_id + i];
            }     
            __syncthreads();
        }

        if(thread_id % 8 == 0){
            dist[block_id * blockDim.x + thread_id/8] = sqrt((double)sum[thread_id]);
        }
    }
}

