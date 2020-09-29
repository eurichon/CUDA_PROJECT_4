#include "vptree.cuh"


void createVPTree(float *dataset, int n, int d){
    float *d_dataset = NULL;
    float *d_distances = NULL;
    float *d_indexes = NULL;

    unsigned long length = n * d;

    // transfer dataset to device
    cudaMalloc(&d_dataset, length * sizeof(float));         // keeps the dataset
    cudaMalloc(&d_distances, n * sizeof(float));            // keeps the current distance results in each level
    cudaMalloc(&d_indexes, n * sizeof(float));              // index map so we dont need to swap the whole element - only their indexes

    cudaMemcpy(d_dataset, dataset, length * sizeof(float),  cudaMemcpyHostToDevice);

    initIndexes(d_indexes, n);


    auto start = std::chrono::high_resolution_clock::now();
    parallelReduce(d_distances, d_dataset, d_indexes, n, d, 2);
    auto finish = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
    cout << "Gpu: " << gpu_time << endl; 


    cudaFree(d_dataset);
    cudaFree(d_distances);
    cudaFree(d_indexes);
}


void initIndexes(float *d, int n){
    dim3 block(512, 1);
    dim3 grid(CEIL_DIV(n, 512), 1);
    cudaInitIndexes<<<block, grid>>>(d, n);
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess){
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        exit(SYNCH_CUDA_ERROR);
    }
		
	if (errAsync != cudaSuccess){
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
        exit(ASYNC_CUDA_ERROR);
    }
}


__global__ void cudaInitIndexes(float *d, int n){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  
    if(thread_id < n){
        d[thread_id] = thread_id;
    }
}


// void calculateDIstances(float *dataset, float *product, float *indexes, float *distances, int n, int d, int iter){
//     printf("Calculating Distances...");
//     // calculate distance products
//     dim3 block(16 , 32);
//     dim3 grid(CEIL_DIV(n,16), CEIL_DIV(d,32));

//     auto start = std::chrono::high_resolution_clock::now();
//     cudaCalculateProducts<<<grid, block>>>(dataset, product, indexes, n, d, iter);
//     cudaError_t errSync = cudaGetLastError();
//     cudaError_t errAsync = cudaDeviceSynchronize();
//     auto finish = std::chrono::high_resolution_clock::now();
//     auto result = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
//     cout << "Products in: " << result << endl;
    

//     // start = std::chrono::high_resolution_clock::now();
//     // // reduce products to the results grid and block & reduction
//     // int first_step = pow(2, ceil(log2(d/2)));
//     // for(int s = first_step; s >= 1024; s >>= 1){
//     //     cudaReduceGrid<<<grid, block>>>(product, n, d, s);
//     // }

//     // // reduce inside the block using shared memory
//     // block.x = 1; block.y = 512;
//     // grid.x = n;  grid.y = 1;
//     // int shared_mem = 2 * block.y * sizeof(float);
    
//     // cudaReduceBlock<<<grid, block, shared_mem>>>(product, distances, n, d);

//     // cudaError_t errSync = cudaGetLastError();
//     // cudaError_t errAsync = cudaDeviceSynchronize();
    
//     // finish = std::chrono::high_resolution_clock::now();
//     // result = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
//     // cout << "Reduction in: " << result << endl;

    
//     if (errSync != cudaSuccess){
//         printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
//         exit(SYNCH_CUDA_ERROR);
//     }
		
// 	if (errAsync != cudaSuccess){
//         printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
//         exit(ASYNC_CUDA_ERROR);
//     }
    
//     printf(" Finished\n");
// }


// // calculate products
// __global__ void cudaCalculateProducts(float *dataset, float *products, float *indexes, int n, int d, int iter){
//     // find threads id's
//     int thread_pos_x = threadIdx.x + blockIdx.x * blockDim.x;
//     int thread_pos_y = threadIdx.y + blockIdx.y * blockDim.y;

//     // if its position is within the dataset's limits
//     if(thread_pos_x < n && thread_pos_y < d){
//         //read (a) value of the dataset through the index map
//         int pos_a_x = indexes[thread_pos_x];
//         int pos_a_y = thread_pos_y;
//         float a = dataset[pos_a_x * d + pos_a_y];

//         // read (b) value of the dataset through the index map
//         int step = n / iter;
//         int index_pos = (thread_pos_x / step) * step; 

//         int pos_b_x = indexes[index_pos];  
//         int pos_b_y = thread_pos_y;
//         float b = dataset[pos_b_x * d + pos_b_y];

//         products[pos_a_x * d + pos_a_y] = a*a - 2*a*b + b*b;
//     }
// }


// // reduce products to distances
// __global__ void cudaReduceGrid(float *products, int n, int d, int step){
//     int thread_pos_x = threadIdx.x + blockIdx.x * blockDim.x;
//     int thread_pos_y = threadIdx.y + blockIdx.y * blockDim.y;

//     if(thread_pos_x < n && thread_pos_y < step && (thread_pos_y + step) < d){
//         int pos = thread_pos_x * d + thread_pos_y;
//         products[pos] += products[pos + step];
//     }
// }   


// __global__ void cudaReduceBlock(float *products, float *distances, int n, int d){
//     extern __shared__ float s_mem[];   // shared memory of 2048 size
    
//     // copy data to shared memory
//     s_mem[threadIdx.y] = products[blockIdx.x * d + threadIdx.y];
//     s_mem[threadIdx.y + blockDim.y] = products[blockIdx.x * d + threadIdx.y + blockDim.y];

//     __syncthreads();

//     // reduce  the result
//     for(unsigned int s = blockDim.y; s > 0; s >>= 1){
//         if(threadIdx.y < s){
//             s_mem[threadIdx.y] += s_mem[threadIdx.y + s];
//         }
//         __syncthreads();
//     }

//     // copy the result back to global
//     if(threadIdx.y == 0)
//         distances[blockIdx.x] = s_mem[0]; 

// }   