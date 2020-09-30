#include <iostream>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <math.h>
#include "stdio.h"
#include "bitonic.cuh"
using namespace std;

#define MEMORY_ERROR        -1
#define MIN_VALUE          -1000.0
#define MAX_VALUE           1000.0


int cmpfunc (const void * a, const void * b) {
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return (fa > fb) - (fa < fb);
}


void initArray(float *data, int n){
    if(data == NULL){
        cout << "Not enough memory. Aborting ..." << endl;
        free(data);
        exit(MEMORY_ERROR);
    }else{
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator(seed);
        uniform_real_distribution<float> distribution(MIN_VALUE, MAX_VALUE);

        for(int i = 0; i < n; i++){
            data[i] = distribution(generator);
        }
    }
}


int main(int argc, char *argv[]){

    if(argc != 3){
        cout << "Wrong number of arguments. Requires number of points(power of 2) & direction" << endl;
        exit(-1);
    }

    int n = atoi(argv[1]);
    int dir = atoi(argv[2]);

    float *h_data; 
    h_data = (float *)malloc(n * sizeof(float));
    initArray(h_data, n);

    // sort using the cuda bitonic
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
    long gpu_time = bitonic(d_data, n, dir);
    cout << "Parallel Bitonic GPU time: " << gpu_time << endl;
    cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);


    // check according to the given direction
    bool success = 1;
    for(int i = 1; i < n; i++){
        if(dir == 1){
            if(h_data[i - 1] >  h_data[i]){
                success = 0;
                cout << "error in i: " << i << ", " << h_data[i - 1] - h_data[i] << endl;
                break;
            }
        }else{
            if(h_data[i - 1] <  h_data[i]){
                success = 0;
                cout << "error in i: " << i << ", " << h_data[i - 1] - h_data[i] << endl;
                break;
            }
        }
        
    }

    // re suffle array to avoid best case O(n) quicksort
    initArray(h_data, n);

    // compare with std's quicksort
    auto start = std::chrono::high_resolution_clock::now();
    qsort(h_data, n, sizeof(float), cmpfunc);
    auto finish = std::chrono::high_resolution_clock::now();
    long cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
    cout << "Quicksort CPU time: " << cpu_time << endl;

    cout << "The result is: " << success << endl;
    cout << "Speed up: " << (double)cpu_time / gpu_time << endl;
    
    // free memeory
    free(h_data);
    
    return 0;
}


