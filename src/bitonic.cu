#include "bitonic.cuh"



long bitonic(float *data, unsigned int n, unsigned int dir){
    auto start = std::chrono::high_resolution_clock::now();

    if(n <= 2048){
        dim3 grid(1, 1);
        dim3 block((n / 2), 1);
        int shared_mem = n * sizeof(float);
        cudaBlockBitonic<<<grid, block, shared_mem>>>(data, n, dir);
    }else{
        dim3 grid((n / BLOCK_SIZE), 1);
        dim3 block(BLOCK_SIZE, 1);
       
        for(int s = 1; s <= n/2; s <<= 1){
            for(int s2 = s; s2 > 0; s2 >>= 1){
                cudaGridBitonic<<<grid, block>>>(data, n, s, s2, dir);
            }
        }
    }

    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();

    if (errSync != cudaSuccess)
        cout << "Sync kernel error: " << cudaGetErrorString(errSync) << endl;
    if (errAsync != cudaSuccess)
        cout << "Sync kernel error: " << cudaGetErrorString(errAsync) << endl;

    auto finish = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();

    return (long)gpu_time;
}


void bitonic(float *data, float *index_map, unsigned int n, unsigned int dir,unsigned int iter){
    if(n <= 2048){
        // not implemented on overloaded function
        dim3 grid(1, 1);
        dim3 block((n / 2), 1);
        int shared_mem = n * sizeof(float);
        cudaBlockBitonic<<<grid, block, shared_mem>>>(data, n, dir);
    }else{
        dim3 grid((n / BLOCK_SIZE), 1);
        dim3 block(BLOCK_SIZE, 1);
       
        for(int s = 1; s <= n/(2 * iter); s <<= 1){
            for(int s2 = s; s2 > 0; s2 >>= 1){
                cudaGridBitonic<<<grid, block>>>(data, index_map, n, s, s2, dir, iter);
            }
        }
    }

    #ifndef GLOBAL_SYNCHRONIZATION
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();

    if (errSync != cudaSuccess)
        cout << "Sync kernel error: " << cudaGetErrorString(errSync) << " in bitonic at iter: "<< iter << endl;
    if (errAsync != cudaSuccess)
        cout << "Sync kernel error: " << cudaGetErrorString(errAsync) << " in bitonic at iter: "<< iter << endl;

    #endif
}


__global__ void cudaGridBitonic(float *d, unsigned int n, unsigned int step_iter, unsigned int curr_step, unsigned int dir){
    // calculate thread id in the grid
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int a = thread_id / curr_step;
    unsigned int b = thread_id % curr_step;
    unsigned int pos = a * (curr_step << 1) + b;
    
    unsigned int c_dir = ((thread_id / step_iter) % 2) != dir;

    // manually handle swapping to avoid accessing two 3 times global memory
    if((pos + curr_step) < n){
        float val_a = d[pos];
        float val_b = d[pos + curr_step];


        if((val_a > val_b) == c_dir){
            float temp = val_a;
            d[pos] = val_b;
            d[pos + curr_step] = temp;
        }
    }
}

// you have to also manually reduce the currsteps & step_iter accordingly
__global__ void cudaGridBitonic(float *d, float *index_map, unsigned int n, unsigned int step_iter, unsigned int curr_step, unsigned int dir,unsigned int iter){
    // calculate thread id in the grid
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // depending on the iteration we split the bitonic processes into multiple ones with smaller size
    unsigned int subset = (n >> 1) / iter;                          // because threads are half of n
    unsigned int subset_offset = (thread_id / subset) * subset;
    thread_id = thread_id % subset;                                 // split threads into subsets
    

    unsigned int a = thread_id / curr_step;
    unsigned int b = thread_id % curr_step;
    unsigned int pos = a * (curr_step << 1) + b;
    
    unsigned int c_dir = ((thread_id / step_iter) % 2) != dir;

    // manually handle swapping to avoid accessing two 3 times global memory
    if((pos + curr_step + subset_offset) < n){
        float val_a = d[pos + subset_offset];
        float val_b = d[pos + curr_step + subset_offset];


        if((val_a > val_b) == c_dir){
            float temp = val_a;
            d[pos + subset_offset] = val_b;
            d[pos + curr_step + subset_offset] = temp;

            // sort index map with respect to the distances
            temp = index_map[pos + subset_offset];
            index_map[pos] = index_map[pos + curr_step + subset_offset];
            index_map[pos + curr_step + subset_offset] = temp;
        }
    }
}


__global__ void cudaBlockBitonic(float *d, int n, int dir){
    extern __shared__ float s_mem[];

    int offset = blockIdx.x * (2 * blockDim.x);

    // copy data from global memory to the shared memory
    s_mem[threadIdx.x] = d[threadIdx.x + offset];
    s_mem[blockDim.x + threadIdx.x] = d[blockDim.x + threadIdx.x + offset];

    // single block or repeated multiple block

    dir = (blockIdx.x % 2 == 0)?(dir):(!dir);

    for(unsigned int i = 1; i <= n/2; i <<= 1){
        int c_dir = ((threadIdx.x / i) % 2) != dir;

        for(unsigned int s = i; s >= 1; s >>= 1){
            int step = s;
            int a = threadIdx.x / s;
            int b = threadIdx.x % s;
            int pos = a * (step << 1) + b;
            
            __syncthreads();
            cudaCompAndSwap(s_mem, pos, pos + step, c_dir);
        }     
    }

    __syncthreads();

    // copy results from shared memory back to global
    d[threadIdx.x + offset] = s_mem[threadIdx.x];
    d[blockDim.x + threadIdx.x + offset] = s_mem[blockDim.x + threadIdx.x];
}


/*
 * Device function with swaps elements if they dont agree with the given direction 
*/
__device__ void cudaCompAndSwap(float *d, int i, int j, int dir){
    if((d[i] > d[j]) == dir){
        cudaSwap((d + i), (d + j));
    }
}


/*
 * Device function which allows threads to swaps two float elements
*/
__device__ void cudaSwap(float *a, float *b){
    float temp = *a;
    *a = *b;
    *b = temp;
}

