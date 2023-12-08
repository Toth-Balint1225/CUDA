/**
 * Goal:
 * Use CUDA to probe the GPU on the machine and display its data.
*/

#include <stdio.h>
#include "error.h"

void print_props(int dev)
{
    cudaDeviceProp props;
    CHECK_ERR( cudaGetDeviceProperties(&props, dev) );
    printf("--- General Card Information, device %d ---\n", dev);
    printf("Name: %s\n", props.name);
    printf("Compute capability: %d.%d\n", props.major, props.minor);
    printf("Clock rate: %d\n", props.clockRate);

    printf("--- Memory Information, device %d ---\n", dev);
    printf("Global memory: %ld\n", props.totalGlobalMem);
    printf("Total constant memory: %ld\n", props.totalConstMem);
    printf("Max mem pitch: %ld\n", props.memPitch);
    printf("Texture alignment: %ld\n", props.textureAlignment);
    printf("Shared memory per mp: %ld\n", props.sharedMemPerBlock);

    printf("--- Multiprocess Information, device %d ---\n", dev);
    printf("Multiprocessor count: %d\n", props.multiProcessorCount);
    printf("Registers per block: %d\n", props.regsPerBlock);
    printf("Max threads per block: %d\n", props.maxThreadsPerBlock);
    printf("Max thread dimensions: (%d, %d, %d)\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
    printf("Max grid dimensions: (%d, %d, %d)\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
    printf("\n\n");
}

int main(void)
{
    // Step 1. Get the device count
    int count;
    CHECK_ERR( cudaGetDeviceCount(&count) );
    printf("Number of CUDA devices: %d\n", count);

    // Step 2. Device information
    for (int i=0;i<count;i++)
        print_props(i);


    LAST_ERR();
    return 0;
}