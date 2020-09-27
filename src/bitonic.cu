#include "bitonic.cuh"





// SYCHRONIZATION ERROR - NEED TO BREAK IN KERNEL LANUCHES - CHANGE IN LOOP BITONIC
// CONSIDER REDUCING THE BLOCK SIZE 256
// CLEAR FLAG AND CALCULATIONS
// IMPROVED SWAP FOR GLOBAL MEMORY

__global__ void cudaGridBitonic(float *d, int n, int step, int dir, int force){
    extern __shared__ float s_mem[];

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int c_dir;
    int flag = 1;

    for(int s = step; s >= 2048; s >>= 1){ 
        int power_step = (s << 1);
        int a = thread_id / s;
        int b = thread_id % s;
        int pos = a * power_step + b;
        
        if(flag){
            c_dir = dir;

            if(force == -1)
                c_dir = (a % 2 != dir); // to check

            flag = 0;
        }
        // copy data
        // s_mem[threadIdx.x] = d[pos];
        // s_mem[threadIdx.x + blockDim.x] = d[pos + s];

        // __syncthreads();

        // // relative call to gshared memory
        // cudaCompAndSwap(s_mem, threadIdx.x, (threadIdx.x + blockDim.x), c_dir);

        // __syncthreads();

        // // copy data
        // d[pos] = s_mem[threadIdx.x];
        // d[pos + s] = s_mem[threadIdx.x + blockDim.x];

        cudaCompAndSwap(d, pos, (pos + s), c_dir);
    }
}


/*
 *  Global function that optimize bitonic sort's steps for operation that is included to up 2048 elements
 *  force parameter used to determine if it is a normal bitonic sort on a block or a recursive handle of a array with size > 2048 elements
 */
__global__ void cudaBlockBitonic(float *d, int n, int dir, int force, int value){
    extern __shared__ float s_mem[];

    int offset = blockIdx.x * (2 * blockDim.x);
    
    // copy data from global memory to the shared memory
    s_mem[threadIdx.x] = d[threadIdx.x + offset];
    s_mem[blockDim.x + threadIdx.x] = d[blockDim.x + threadIdx.x + offset];

    __syncthreads();

    if(force == -1){
        // used for normal sorting for 1 block up to 2048 or n blocks of 2048 elements
        for(int step = 1; step <= n/2; step <<= 1){
            cudaBlockBitonicMerge(s_mem, threadIdx.x, blockIdx.x, step, dir, force);
        }
    }else{
        // used as a complement to merge the blocks when elements are less than 2048 and can fit into a block
        // calculate current direction
        int n_dir;
        if(dir){
            n_dir = ((blockIdx.x & value) > 0)?(0):(1);
        }else{
            n_dir = ((blockIdx.x & value) > 0)?(1):(0);
        }
        
        cudaBlockBitonicMerge(s_mem, threadIdx.x, blockIdx.x, n/2, n_dir, force);
    }

    __syncthreads();

    // copy results from shared memory back to global
    d[threadIdx.x + offset] = s_mem[threadIdx.x];
    d[blockDim.x + threadIdx.x + offset] = s_mem[blockDim.x + threadIdx.x];
}


__device__ void cudaBlockBitonicMerge(float *data, int thread_id, int block_id, int step, int dir, int force){
    int power_step = step << 1;
    int a = thread_id / step;
    int b = thread_id % step;
    int pos = a * power_step + b;
    int c_dir = dir;

    if(force == -1){
        if(dir == 1){
            c_dir = (a % 2 == (block_id % 2));
        }else{
            c_dir = (a % 2 != (block_id % 2));
        }
    }
    
    cudaCompAndSwap(data, pos, pos + step, c_dir);

    while(step >= 2){
        step = step >> 1;
        power_step = power_step >> 1;
        a = threadIdx.x / step;
        b = threadIdx.x % step;
        pos = a * power_step + b;
        
        __syncthreads();
        cudaCompAndSwap(data, pos, pos + step, c_dir);            
    }
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

