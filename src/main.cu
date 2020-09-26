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
#define MIN_VALUE          -100.0
#define MAX_VALUE           100.0

#define WARP_SIZE           32
#define MAX_BLOCK_THREADS   512
#define MAX_WARPS_BLOCK_Y   16
#define MIN_WARPS_BLOCK_Y   1

#define VERBOSE


#define ERROR_THRESHOLD     100

const char Result[2][10] = {"SUCCEDED", "FAILED"};

typedef struct{
    dim3 block;
    dim3 grid;
    int s_mem;
}BestSplit;



BestSplit findBestSplit(int n, int d);


void cudaGPUDetails();
void printData(float *dataset, int n, int d);
void calculateDistances(float *distances, float *dataset, float *point, int n, int d);


void parallelReduce(float *distances, float *data, int n, int d);
void serialReduce(float *sum, float *data, int n, int d);

__global__ void deviceCalculatedDistances(float *dist, float *data, float *point, int n, int d);


__global__ void cudaReduce(float *temp, float *distances, int n, int d, int r);
__global__ void cudaDotProduct(float *dataset, float *point, float *product, int n, int d, int r);


int main(int argc, char *argv[]){
    // show cuda gpu details
    cudaGPUDetails();

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

    // cout << "Dataset" << endl;
    // printData(h_dataset, n, d);
    // cout << "Distances by CPU" << endl;
    //printData(h_test_distances, n, 1);


    // cuda stuff
    cudaMalloc(&d_dataset, n * d * sizeof(float)); 
    cudaMalloc(&d_point, d * sizeof(float)); 
    cudaMalloc(&d_distances, n * sizeof(float)); 

    // copy the dataset to the device
    cudaMemcpy(d_dataset, h_dataset, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_point, &d_dataset[0], d * sizeof(float), cudaMemcpyHostToDevice);


    start = std::chrono::high_resolution_clock::now();

    parallelReduce(d_distances ,d_dataset, n , d);

    finish = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();



    // copy the calculated distances back to the host
    cudaMemcpy(h_distances, d_distances, n * sizeof(float), cudaMemcpyDeviceToHost);

    

    #ifdef VERBOSE
    cout << "CPU Time: " << cpu_time << endl;
    cout << "GPU Time: " << gpu_time << endl;
    cout << "Total Speed up: " << (float)cpu_time / gpu_time << endl;
    #endif


    // validation
    bool fail = 0;
    for(int i = 0; i < n; i++){
        float error = abs(pow(h_test_distances[i],2) - h_distances[i]);
        if(error > ERROR_THRESHOLD){
            cout << "Error exeeded threshold("<< ERROR_THRESHOLD << "): " << error << endl;
            fail = 1;
            break;
        }
    }

    cout << "Program: " << Result[fail] << endl;

    cudaFree(d_dataset);
    cudaFree(d_distances);

    free(h_distances);
    free(h_dataset);

    return 0;
}



/*
 *  Function that reduces a sum in parallel efficiently by managing kernel assignments
*/

BestSplit findBestSplit(int n, int d){
    BestSplit split;

    int req_warps_y = d / WARP_SIZE  + (d % WARP_SIZE != 0);

    float min_error = 100000.0;
    int best_dividor = -1;
    int num_of_warps = -1;

    for(int split = 1; split < 1000; split++){
        float warps_in_block_y = (float)req_warps_y / split;
        
        if(MIN_WARPS_BLOCK_Y <= warps_in_block_y && warps_in_block_y <= MAX_WARPS_BLOCK_Y){
            float split_error = warps_in_block_y - (int)(req_warps_y / split);
            
            if(min_error > split_error){
                min_error = split_error;
                best_dividor = split;     
                num_of_warps = warps_in_block_y;
            }
        }else if(warps_in_block_y < 1){
            break;
        }
    }

    
    split.block.y = num_of_warps * WARP_SIZE;
    split.block.x = MAX_BLOCK_THREADS / split.block.y;
    split.grid.x = n / split.block.x + (n % split.block.x != 0);
    split.grid.y = best_dividor + (min_error > 0);

    split.s_mem = (split.block.x + 1) * split.block.y * sizeof(float);


    #ifdef VERBOSE
    //cout << "Block: " << split.block.x << ", " << split.block.y << ", " << (split.block.x + 1) * split.block.y << endl;
    //cout << "Grid: " << split.grid.x << ", " << split.grid.y << endl;
    //cout << "Occupancy: " << (float)n / (split.block.x * split.grid.x) << ", " << (float)d / (split.block.y * split.grid.y) << endl;
    #endif
    
    return split;
}





void parallelReduce(float *distances, float *data, int n, int d){
    float *d_product_temp = NULL;

    BestSplit b_split = findBestSplit(n, d);
    

    
    // calculate temp block & grid sizes
    dim3 temp_block;
    dim3 temp_grid;

    int temp_size_x = b_split.block.x * b_split.grid.x;
    int temp_size_y = b_split.grid.y;
    

    temp_block.y = (temp_size_y / 8 + (temp_size_y % 8 != 0)) * 8; 
    temp_block.x = MAX_BLOCK_THREADS / temp_block.y;
    temp_grid.x = n / temp_block.x + (n % temp_block.x != 0);
    temp_grid.y = 1;

    int shared_mem = temp_block.x * temp_block.y * sizeof(float);


    #ifdef VERBOSE
    //cout << "Temp Block: " << temp_block.x << ", " << temp_block.y << endl;
    //cout << "Temp Grid: " << temp_grid.x << ", " << temp_grid.y << endl;
    #endif
    
    

    int temp_product_size = temp_size_x * temp_size_y * sizeof(float);
    cudaMalloc(&d_product_temp, temp_product_size); 

   
    int r = pow(2,ceil(log2(b_split.block.y)));
    //cout << r << endl;
    cudaDotProduct<<<b_split.grid, b_split.block, b_split.s_mem>>>(data, &data[0], d_product_temp, n, d, r);

    

    float *h_temp = (float *)malloc(temp_size_x * temp_size_y * sizeof(float));
    cudaMemcpy(h_temp, d_product_temp, temp_size_x * temp_size_y * sizeof(float),  cudaMemcpyDeviceToHost);

    // //cout << "Printing temp" << endl;
    // // printData(h_temp, temp_size_x, temp_size_y);
    // for(int i = 0; i < temp_size_x; i++){
    //     float sum = 0.0;
    //     for(int j = 0; j < temp_size_y; j++){
    //         sum = sum + h_temp[i * temp_size_y + j];
    //     }
    //     distances[i] = sqrt(abs(sum));
    //     //cout << i + 1 << ") " << sqrt(abs(sum)) << endl;
    // }


    r = pow(2,ceil(log2(temp_block.y)));
    cout << r << endl;
    cudaReduce<<<temp_grid, temp_block, shared_mem>>>(d_product_temp, distances, n, b_split.grid.y, r);
    

    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();

    #ifdef VERBOSE
    if (errSync != cudaSuccess)
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    if (errSync == cudaSuccess && errAsync == cudaSuccess)
        printf("Kernals succesfully finished without any errors!\n");
    #endif
}


__global__ void cudaDotProduct(float *data, float *point, float *product, int n, int d, int r){
    // copy dataset and point to shared memory
    extern __shared__ float s_mem[];

    int thread_id = threadIdx.x * blockDim.y + threadIdx.y;
    int point_offset = blockDim.x * blockDim.y;

    float *s_point = &s_mem[point_offset];
    float *s_data = &s_mem[0];

    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    // IF the thread belongs to the first row and is inside the block dimensions copy the data else if it outfside just zero initialize the array
    if(thread_id < blockDim.y && pos_y < d){
        s_point[threadIdx.y] = point[pos_y]; 
    }else if(thread_id < blockDim.y){
        s_point[threadIdx.y] = 0.0;
    }

    // copy data
    if(pos_x < n && pos_y < d){
        s_data[thread_id] = data[pos_x * d + pos_y];
    }else{
        s_data[thread_id] = 0.0;
    }
    
    __syncthreads();

    // calculate dot product
    s_data[thread_id] = s_data[thread_id] * s_data[thread_id] - 2 * s_data[thread_id] * s_point[threadIdx.y] + s_point[threadIdx.y] * s_point[threadIdx.y];
    
    __syncthreads();

    //reduce sum in parallel
    for(int s = r / 2; s > 0; s >>= 1){
        if(threadIdx.y < s && (threadIdx.y + s) < d){
            s_data[thread_id] += s_data[thread_id + s];
        }else{
            s_data[thread_id] += 0.0;
        }
        __syncthreads();
    }

    // copy data back
    pos_x = blockIdx.x * (blockDim.x * gridDim.y);
    pos_y = blockIdx.y;
    if(threadIdx.y == 0){
        product[pos_x + pos_y + threadIdx.x * gridDim.y] = s_data[thread_id];
    }
}


// complete unroll these iterations
__global__ void cudaReduce(float *temp, float *distances, int n, int d, int r){
    extern __shared__ float s_data[];

    int thread_id = threadIdx.x * blockDim.y + threadIdx.y;
    
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = threadIdx.y;

    // copy data to shared memory
    if(pos_x < n && pos_y < d){
        s_data[thread_id] = temp[pos_x * d + pos_y];
    }else{
        s_data[thread_id] = 0.0;
    }

    __syncthreads();

    for(int s = r / 2; s > 0; s >>= 1){
        if(threadIdx.y < s && (threadIdx.y + s) < d){
            s_data[thread_id] += s_data[thread_id + s];
        }
        __syncthreads();
    }

    // copy back the data
    if(threadIdx.y == 0 && pos_x < n){
        distances[pos_x] = s_data[thread_id];
    }
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
                sum[thread_id] = sum[thread_id] + val * val;
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




void cudaGPUDetails(){
    int n_devices;
    cudaGetDeviceCount(&n_devices);

    cout << "******** Getting GPU information ********" << endl;

    for (int dev = 0; dev < n_devices; dev++) {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, dev);

        if(n_devices == 0){
            if(device_prop.major == 9999 && device_prop.minor == 9999){
                cout << "No Cuda GPU has been detected" << endl;
                exit(-1);
            }else if(n_devices == 1){
                cout << "Found 1 device supporting CUDA" << endl;
            }else{
                cout << "There are " << n_devices << " supporting CUDA" << endl;
            }
        }

        cout << "Device " << dev << " name: " << device_prop.name << endl;
        cout << "Computatial Capabilities " << device_prop.major << "." << device_prop.minor << endl;
        cout << "Maximum global memory size: " << device_prop.totalGlobalMem << endl;
        cout << "Maximum Constant memory size: " << device_prop.totalConstMem << endl;
        cout << "Maximun shared memory size per block: " << device_prop.sharedMemPerBlock << endl;
        cout << "Maximum block dimensions: " << device_prop.maxThreadsDim[0] << " x " << device_prop.maxThreadsDim[1] << " x " << device_prop.maxThreadsDim[2] << endl;
        cout << "Maximum grid dimensions: " << device_prop.maxGridSize[0] << " x " << device_prop.maxGridSize[1] << " x " << device_prop.maxGridSize[2] << endl;
        cout << "Warp size: " << device_prop.warpSize << endl;
    } 

    cout << "*****************************************" << endl << endl << endl;
}
