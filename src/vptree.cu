#include "vptree.cuh"


void createVPTree(float *dataset, int n, int d){
    float *d_dataset = NULL;
    float *d_distances = NULL;
    float *d_indexes = NULL;

    unsigned long length = n * d;

    // to do create tree array to store indexes

    // transfer dataset to device
    cudaMalloc(&d_dataset, length * sizeof(float));         // keeps the dataset
    cudaMalloc(&d_distances, n * sizeof(float));            // keeps the current distance results in each level
    cudaMalloc(&d_indexes, n * sizeof(float));              // index map so we dont need to swap the whole element - only their indexes

    cudaMemcpy(d_dataset, dataset, length * sizeof(float),  cudaMemcpyHostToDevice);

    initIndexes(d_indexes, n);

    
    for(int i = 1; i <= n/2; i <<= 1){
        parallelDistance(d_distances, d_dataset, d_indexes, n, d, i);
        // store indexes & medieans on each level
        bitonic(d_distances, d_indexes, n, ASCENDING, i);
        cout << ".";
    }
    
    #ifdef GLOBAL_SYNCHRONIZATION
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();

    if (errSync != cudaSuccess)
        cout << "Sync kernel error: " << cudaGetErrorString(errSync) << " in vp tree" << endl;
	if (errAsync != cudaSuccess)
        cout << "Sync kernel error: " << cudaGetErrorString(errAsync) << " in vp tree" << endl;
    #endif

    cudaFree(d_dataset);
    cudaFree(d_distances);
    cudaFree(d_indexes);
}


void initIndexes(float *d, int n){
    dim3 block(512, 1);
    dim3 grid(CEIL_DIV(n, 512), 1);
    cudaInitIndexes<<<grid, block>>>(d, n);
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess){
        cout << "Sync kernel error: " << cudaGetErrorString(errSync) << " in init indexing reduce: " << endl;
        exit(SYNCH_CUDA_ERROR);
    }
		
	if (errAsync != cudaSuccess){
        cout << "Sync kernel error: " << cudaGetErrorString(errSync) << " in init indexing: " << endl;
        exit(ASYNC_CUDA_ERROR);
    }
}


__global__ void cudaInitIndexes(float *d, int n){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  
    if(thread_id < n){
        d[thread_id] = thread_id;
    }
}
