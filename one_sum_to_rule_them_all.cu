#include <iostream>
#include <cstdlib>

#define COLAB
#ifdef COLAB
#include "error.h"
#include "timetest.h"
#else
#include <error.h>
#include <timetest.h>
#endif

#define N (1<<22)
#define T 32

void print(int* data, int sum) {
    std::cout << data[0]; 
    for (unsigned i=1;i<N;i++)
        std::cout << " + " << data[i];
    
    std::cout << " = " << sum << std::endl;
}

__global__ void cheaty_collect(int data[], int* sum) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (unsigned i=0;i<N;i++) {
            *sum += data[i];
        }
    }
}

__global__ void basic_collect(int data[], int* sum) {
    extern __shared__ int shared[]; // manually allocated shared memory

    unsigned thread = threadIdx.x;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    // copy the locality to the shared memory
    shared[thread] = data[tid];
    __syncthreads();

    // do the actual pairwise sums inside the local memory
    for (unsigned i=1; i<blockDim.x; i*=2) {
        if (thread % (2*i) == 0)
            shared[thread] += shared[thread + i];
        __syncthreads();
    }

    // write back to global mem
    if (thread == 0) {
        atomicAdd(sum, shared[0]);
    }
}

__global__ void not_that_shit_collect(int data[], int* sum) {
    // at this point it's the same as the basic one
    extern __shared__ int shared[];
    unsigned thread = threadIdx.x;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    shared[thread] = data[tid];
    __syncthreads();

    // the % operator is a piece of shit. 
    // let's change up the internal logic
    for (unsigned i=1; i<blockDim.x; i*=2) {
        int index = 2 * i * thread;
        if (index < blockDim.x)  {
            shared[index] += shared[index + i];
        }
        __syncthreads();
    }

    if (thread == 0) {
        atomicAdd(sum, shared[0]);
    }

}

__global__ void better_collect(int data[], int* sum) {
    // at this point it's the same as all the other ones
    extern __shared__ int shared[];
    unsigned thread = threadIdx.x;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    shared[thread] = data[tid];
    __syncthreads();

    // problem: shared memory bank conflicts
    // by changing the loop to sequential addressing, we can remedy it
    // at this point, we've made the next index calculation in one step
    for (unsigned i=blockDim.x/2; i>0;i>>=1) {
        if (i < blockDim.x)  {
            shared[thread] += shared[thread + i];
        }
        __syncthreads();
    }

    if (thread == 0) {
        atomicAdd(sum, shared[0]);
    }
}

/*
    Mark Harris' method
*/

// the unrolled last warp -> no need for the last iteration
template <unsigned blockSize>
__device__ void warp_reduce(volatile int* shared, unsigned thread) {
    if (blockSize >= 64) shared[thread] += shared[thread + 32];
    if (blockSize >= 32) shared[thread] += shared[thread + 16];
    if (blockSize >= 16) shared[thread] += shared[thread + 8];
    if (blockSize >= 8) shared[thread] += shared[thread + 4];
    if (blockSize >= 4) shared[thread] += shared[thread + 2];
    if (blockSize >= 2) shared[thread] += shared[thread + 1];
}

template <unsigned blockSize>
__global__ void reduce(int* data, int* sum, unsigned n) {
    extern __shared__ int shared[];
    unsigned thread = threadIdx.x;
    unsigned tid = blockIdx.x * (blockSize*2) + thread;
    unsigned gridSize = blockSize * 2 * gridDim.x;
    shared[thread] = 0;

    while (tid < n) {
        shared[tid] += data[tid] + data[tid + blockSize];
        tid += gridSize; 
    }
    __syncthreads();

    // wtf even is this?
    if (blockSize >= 512) {
        if (thread < 256) {
            shared[thread] += shared[thread + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (thread < 128) {
            shared[thread] += shared[thread + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (thread < 64) {
            shared[thread] += shared[thread + 64];
        }
        __syncthreads();
    }

    // use the unrolled last warp
    if (thread < 32) 
        warp_reduce<blockSize>(shared, thread);
    
    if (thread == 0)
        atomicAdd(sum, shared[0]);

}

auto reference(int data[]) -> int {
    int sum = 0;
    for (unsigned i=0;i<N;i++) {
        sum += data[i];
    }
    return sum;
}

auto main() -> int {
    const unsigned n = N;
    const unsigned t = T;
    auto data = new int[n];
    int sum = 0;

    int *d_data, *d_sum;
    CHECK_ERR(cudaMalloc((void**)&d_data, n*sizeof(int)));
    CHECK_ERR(cudaMalloc((void**)&d_sum, sizeof(int)));

    std::srand(20001225);
    for (unsigned i=0;i<n;i++) {
        data[i] = std::rand() % 10;
    }

    CHECK_ERR(cudaMemcpy(d_data, data, n*sizeof(int), cudaMemcpyHostToDevice));

    CUDA_TIME_START();

    //cheaty_collect<<<(n + t - 1) / t, t>>>(d_data, d_sum);
    //basic_collect<<<(n + t - 1) / t, t, t * sizeof(int)>>>(d_data, d_sum);
    //not_that_shit_collect<<<(n + t - 1) / t, t, t * sizeof(int)>>>(d_data, d_sum);
    //better_collect<<<(n + t - 1) / t, t, t * sizeof(int)>>>(d_data, d_sum);
    reduce<t><<<(n + t - 1) / t, t, t*sizeof(int)>>>(d_data, d_sum, n);


    CUDA_TIME_END();

    CHECK_ERR(cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "diff: " << sum - reference(data) << std::endl;
    // print(data, sum);
    
    CHECK_ERR(cudaFree(d_data));
    CHECK_ERR(cudaFree(d_sum));
    delete[] data;
    return 0;
}