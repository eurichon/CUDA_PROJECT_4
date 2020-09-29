#include "bitonic.cuh"


// CONSIDER REDUCING THE BLOCK SIZE 512
// CLEAR FLAG AND CALCULATIONS
// IMPROVED SWAP FOR GLOBAL MEMORY


__global__ void cudaGridBitonic(float *d, int n, int step_iter, int curr_step, int dir){
    //extern __shared__ float s_mem[];

    // calculate thread id in the grid
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int offset = (curr_step << 1);
    int a = thread_id / curr_step;
    int b = thread_id % curr_step;
    int pos = a * offset + b;
    
    int c_dir = ((thread_id / step_iter) % 2) != dir;

    // manually handle swapping to avoid accessing two 3 times global memory
    float val_a = d[pos];
    float val_b = d[pos + curr_step];

    if((val_a > val_b) == c_dir){
        float temp = val_a;
        d[pos] = val_b;
        d[pos + curr_step] = temp;
    }
}


__global__ void cudaBlockBitonicReduce(float *d, int n, int rate, int dir){
    extern __shared__ float s_mem[];

    int offset = blockIdx.x * (2 * blockDim.x);

    // copy data from global memory to the shared memory
    s_mem[threadIdx.x] = d[threadIdx.x + offset];
    s_mem[blockDim.x + threadIdx.x] = d[blockDim.x + threadIdx.x + offset];

    int c_dir = ((blockIdx.x & rate) == 0)?(dir):(!dir);

    for(unsigned int s = blockDim.x; s >= 1; s >>= 1){
        int step = s;
        int a = threadIdx.x / s;
        int b = threadIdx.x % s;
        int pos = a * (step << 1) + b;
        
        __syncthreads();
        cudaCompAndSwap(s_mem, pos, pos + step, c_dir);
    }     

    __syncthreads();

    // copy results from shared memory back to global
    d[threadIdx.x + offset] = s_mem[threadIdx.x];
    d[blockDim.x + threadIdx.x + offset] = s_mem[blockDim.x + threadIdx.x];
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

