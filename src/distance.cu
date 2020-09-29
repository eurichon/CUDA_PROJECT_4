#include "distance.cuh"

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
    
    return split;
}





void parallelReduce(float *distances, float *data, float *index_map, int n, int d, int iter){
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

    int temp_product_size = temp_size_x * temp_size_y * sizeof(float);
    cudaMalloc(&d_product_temp, temp_product_size); 

    int r = pow(2,ceil(log2(b_split.block.y)));
    cudaDotProduct<<<b_split.grid, b_split.block, b_split.s_mem>>>(data, index_map, d_product_temp, n, d, r, iter);

    r = pow(2,ceil(log2(temp_block.y)));
    
    cudaReduce<<<temp_grid, temp_block, shared_mem>>>(d_product_temp, distances, n, b_split.grid.y, r);
    
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();

    
    if (errSync != cudaSuccess)
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}


__global__ void cudaDotProduct(float *data, float *index_map, float *product, int n, int d, int r, int iter){
    // copy dataset and point to shared memory
    extern __shared__ float s_mem[];

    int thread_id = threadIdx.x * blockDim.y + threadIdx.y;
    int point_offset = blockDim.x * blockDim.y;

    float *s_point = &s_mem[point_offset];
    float *s_data = &s_mem[0];

    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(pos_x < n && pos_y < d){
        // IF the thread belongs to the first row and is inside the block dimensions copy the data else if it outfside just zero initialize the array
        int true_pos_x = index_map[pos_x];

        int step = n / iter;
        int point_pos = (pos_x / step) * step; 

        if(thread_id < blockDim.y){
            // add here
            int true_point_pos = index_map[point_pos];
            s_point[threadIdx.y] = data[true_point_pos * d + pos_y]; 
        }else if(thread_id < blockDim.y){
            s_point[threadIdx.y] = 0.0;
        }

        
        s_data[thread_id] = data[true_pos_x * d + pos_y];
        
        
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



