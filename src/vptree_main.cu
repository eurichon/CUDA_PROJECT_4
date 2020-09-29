#include <iostream>
#include <random>
#include <chrono>
#include "vptree.cuh"
using namespace std;

#define MIN_VALUE               -1000.0
#define MAX_VALUE               1000.0

#define NUM_OF_INPUTS           2
#define INPUT_ERROR             -1 
#define MEMORY_ERROR            -2

void cudaGPUDetails();
void initDataset(float **dataset, unsigned long n, unsigned long d);



int main(int argc, char *argv[]){
    unsigned long n, d;
    float *h_dataset;

    // read inputs
    if(argc != (NUM_OF_INPUTS + 1)){
        cout << "Wrong number of inputs. Exiting..." <<endl;
        exit(-1);
    }else{
        n = atoi(argv[1]);
        d = atoi(argv[2]);
    }

    cudaGPUDetails();

    // initialize dataset
    initDataset(&h_dataset, n, d);

    // create tree
    createVPTree(h_dataset, n, d);


    // free resources
    free(h_dataset);     
    
    return 0;
}


void initDataset(float **dataset, unsigned long n, unsigned long d){
    *dataset = (float *)malloc(n * d * sizeof(float));

    cout << "Initialazing dataset of " << n << " points of " << d << " dimensions" << endl;
    
    if(*dataset == NULL){
        cout << "Not enough memory. Aborting ..." << endl;
        free(*dataset);
        exit(MEMORY_ERROR);
    }else{
        // initialize random generator engine
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator(seed);
        uniform_real_distribution<float> distribution(MIN_VALUE, MAX_VALUE);

        unsigned long length = n * d;

        for(unsigned long i = 0; i < length; i++){
            (*dataset)[i] = distribution(generator);
        }
    }

    cout << "Dataset initialized succesfully" << endl;
}__global__ void cudaInitIndexes(float *d, int n);

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