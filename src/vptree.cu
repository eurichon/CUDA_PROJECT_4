#include "vptree.cuh"


void createVPTree(float *dataset, int n, int d){
    float *d_dataset = NULL;
    float *d_distances = NULL;
    float *d_indexes = NULL;
    float *d_tree = NULL;
    float *d_vp_points = NULL;

    unsigned long length = n * d;
    unsigned long tree_depth = log2(n);
    unsigned long tree_size = n * tree_depth;

    unsigned long vp_points_size = n - 1;



    // transfer dataset to device
    cudaMalloc(&d_tree, tree_size * sizeof(float));
    cudaMalloc(&d_dataset, length * sizeof(float));         // keeps the dataset
    cudaMalloc(&d_distances, n * sizeof(float));            // keeps the current distance results in each level
    cudaMalloc(&d_indexes, n * sizeof(float));              // index map so we dont need to swap the whole element - only their indexes
    cudaMalloc(&d_vp_points, vp_points_size * sizeof(float));

    cudaMemcpy(d_dataset, dataset, length * sizeof(float),  cudaMemcpyHostToDevice);


    initIndexes(d_indexes, n);
    
    for(int i = 1; i <= n/2; i <<= 1){
        parallelDistance(d_distances, d_dataset, d_indexes, &d_vp_points[i-1], n, d, i);    // calculate distances
        copyIndexes(&d_tree[(unsigned int)log2(i) * n], d_indexes, n);                      // store indexes in tree level
        bitonic(d_distances, d_indexes, n, DESCENDING, i);                                  // sorted in descending order so as to suffle the vantage point 
    }
    
    
    #ifdef GLOBAL_SYNCHRONIZATION
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();



    if (errSync != cudaSuccess)
        cout << "Sync kernel error: " << cudaGetErrorString(errSync) << " in vp tree" << endl;
	if (errAsync != cudaSuccess)
        cout << "Sync kernel error: " << cudaGetErrorString(errAsync) << " in vp tree" << endl;
    #endif


    cudaFree(d_tree);
    cudaFree(d_dataset);
    cudaFree(d_distances);
    cudaFree(d_indexes);
    cudaFree(d_vp_points);
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


void copyIndexes(float *dest, float *source, int n){
    dim3 block(512, 1);
    dim3 grid(CEIL_DIV(n, 512), 1);
    cudaCopyIndexes<<<grid, block>>>(dest, source, n);
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


__global__ void cudaCopyIndexes(float *dest, float *source, int n){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  
    if(thread_id < n){
        dest[thread_id] = source[thread_id];
    }
}
