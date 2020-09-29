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
