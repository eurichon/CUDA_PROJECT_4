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

#define WARP_SIZE           32
#define MAX_BLOCK_THREADS   512
#define MAX_WARPS_BLOCK_Y   16
#define MIN_WARPS_BLOCK_Y   1


#define ERROR_THRESHOLD     10e-1

const char Result[2][10] = {"SUCCEDED", "FAILED"};

void cudaGPUDetails();
void printData(float *dataset, int n, int d);
void calculateDistances(float *distances, float *dataset, float *point, int n, int d);


void parallelReduce(float *data, int n, int d);
void serialReduce(float *sum, float *data, int n, int d);

__global__ void deviceCalculatedDistances(float *dist, float *data, float *point, int n, int d);



__global__ void cudaDotProduct(float *dataset, float *point, float *product, int n, int d);


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
    cout << "CPU TIME: " << cpu_time << endl;
    //std::cout << "CPU Time: "<<std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << " ns\n";


    // cuda stuff
    cudaMalloc(&d_dataset, n * d * sizeof(float)); 
    cudaMalloc(&d_point, d * sizeof(float)); 
    cudaMalloc(&d_distances, n * sizeof(float)); 



    start = std::chrono::high_resolution_clock::now();

    // copy the dataset to the device
    cudaMemcpy(d_dataset, h_dataset, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_point, &d_dataset[0], d * sizeof(float), cudaMemcpyHostToDevice);


    printData(h_dataset, n, d);
    parallelReduce(d_dataset, n , d);



    int block_size = 16;
    dim3 num_of_blocks(n/block_size + (n % block_size != 0), 1);
    dim3 threads_per_block(block_size, 8);
    
    //cout << "Dimensions of blocks: " << num_of_blocks.x << ", " << num_of_blocks.y << endl;

    

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



/*
 *  Function that reduces a sum in parallel efficiently by managing kernel assignments
*/

void parallelReduce(float *data, int n, int d){
    int grid_size_x, grid_size_y;
    int block_size_x, block_size_y;
    
    int req_warps_y = d / WARP_SIZE + (d % WARP_SIZE != 0);      
    
    float min_error = 100000.0;
    int num_of_warps = -1;
    int best_split = -1;

    for(int split = 1; split < 100; split++){
        float warps_in_block_y = (float)req_warps_y / split;
        
        if(MIN_WARPS_BLOCK_Y <= warps_in_block_y && warps_in_block_y <= MAX_WARPS_BLOCK_Y){
            float split_error = warps_in_block_y - (int)(req_warps_y / split);
            
            if(min_error > split_error){
                min_error = split_error;
                best_split = split;     
                num_of_warps = warps_in_block_y;
            }
        }else if(warps_in_block_y < 1){
            break;
        }
    }

    block_size_y = num_of_warps * WARP_SIZE;
    block_size_x = MAX_BLOCK_THREADS / block_size_y;
    grid_size_y = best_split + (min_error > 0);
    grid_size_x = n / block_size_x + (n % block_size_x != 0);

    
    cout << "Block: " << block_size_x << ", " << block_size_y << endl;
    cout << "Grid: " << grid_size_x << ", " << grid_size_y << endl;
    cout <<  "Occupancy: " << (float)n / (block_size_x * grid_size_x) << ", " << (float)d / (block_size_y * grid_size_y) << endl;


    int shared_mem_size = (block_size_x + 1) * block_size_y * sizeof(float);

    dim3 block(block_size_x, block_size_y);
    dim3 grid(grid_size_x, grid_size_y);

    
    int temp_product_size = block_size_x * grid_size_x * grid_size_y * sizeof(float);
    float *d_product_temp = NULL;
    cudaMalloc(&d_product_temp, temp_product_size); 
    float *h_product_temp = NULL;
    h_product_temp = (float *)malloc(temp_product_size);


    
    auto start = std::chrono::high_resolution_clock::now();

    cudaDotProduct<<<grid, block, shared_mem_size>>>(data, &data[0], d_product_temp, n, d);
    cudaMemcpy(h_product_temp, d_product_temp, temp_product_size, cudaMemcpyDeviceToHost);
    printData(h_product_temp, block_size_x * grid_size_x, grid_size_y);

    // reduce sub_product
    
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    

    auto finish = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
    cout << "Kernal time: " << cpu_time << endl << endl;
}



__global__ void cudaDotProduct(float *data, float *point, float *product, int n, int d){
    // copy dataset and point to shared memory
    extern __shared__ float s_mem[];
    
    int thread_id = threadIdx.x * blockDim.y + threadIdx.y;
    int point_offset = blockDim.x * blockDim.y;

    float *s_point = &s_mem[point_offset];
    float *s_data = &s_mem[0];

    // copy point
    if(thread_id < blockDim.y){
        s_point[threadIdx.y] =  point[threadIdx.y];
    }
    __syncthreads();


    // copy data
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(pos_x < n && pos_y < d){
        s_data[thread_id] = data[pos_x * d + pos_y];
    }else{
        s_data[thread_id] = 0.0;
    }
    

    // calculate dot product
    s_data[thread_id] = s_data[thread_id] * s_point[threadIdx.y];


    //reduce sum in parallel
    for(int s = blockDim.y / 2; s > 0; s >>= 1){
        if(threadIdx.y < s){
            s_data[thread_id] += s_data[thread_id + s];
        }
        __syncthreads();
    }

    // copy data back
    pos_x = (blockIdx.x * blockDim.x) * gridDim.y;
    pos_y = blockIdx.y;
    if(threadIdx.y == 0){
        product[pos_x + pos_y + threadIdx.x * gridDim.y] = s_data[thread_id];
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
