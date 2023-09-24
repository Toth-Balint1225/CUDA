#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "error.h"
#include "timetest.h"

// #define N 16
// #define N 16777216
#define N 33554432
// #define N 1024
#define T 256

// OMG awesome GPU accelerated algorithm 2023 Elon Musk Spacex magic hype XD
__global__ void collect1(int* a, int* sum)
{
    int i = 0;
    while (i<N)
    {
        *sum += a[i];
        i++;
    }
}

// actual SIMT log2 sum
__global__ void collect(int* a, int* sum, int n, int maxloop)
{
    int loop = 0;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while (loop < maxloop) 
    {
        int stride = 1 << loop;
        if (tid % (stride * 2) == 0) // collector cell
        {
            // find its pair and add this to it
            int pair = tid + stride;
            if (pair < n)
                a[tid] += a[pair];
            //printf("[KERNEL] loop %d tid %d pair %d stride %d\n", loop, tid, pair, stride);
        }
        loop++;
        __syncthreads();
    }
    *sum = a[0];
}

// works only if n is a power of 2
__global__ void collect_simplified(int* a, int* sum, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    n /= 2;
    
    while (n != 0)
    {
        if (tid < n)
            a[tid] += a[tid + n];
        __syncthreads();
        n /= 2;
    }
    *sum = a[0];
}

int control(int* a)
{
    int sum = 0;
    for (int i=0;i<N;i++)
        sum += a[i];
    return sum;
}

int add_cpu(int* a, size_t n)
{
    int sum = 0;
    for (size_t i=0; i<n;i++)
        sum += a[i];
    return sum;
}

void randomize(int* a, size_t n) 
{
    srand(20001225);
    for (size_t i=0; i<n; i++)
        a[i] = rand() % 10;
}

void print_res(int* a, int sum, size_t n)
{
    for (size_t i=0; i<n; i++)
        printf("%d\n", a[i]);
    printf("= %d\n", sum);
}

int my_log2(int n)
{
  int k = n, i = 0;
  while(k) {
    k >>= 1;
    i++;
  }
  return i - 1;
}

int main(void)
{
    size_t n = N;
    int* a = (int*)malloc(sizeof(int) * n);
    randomize(a, n);
    int* dev_a;
    CHECK_ERR( cudaMalloc((void**)&dev_a, sizeof(int) * n) );
    CHECK_ERR( cudaMemcpy(dev_a, a, sizeof(int) * n, cudaMemcpyHostToDevice) );

    int sum = 0;
    int* dev_sum = {0};
    CHECK_ERR( cudaMalloc((void**)&dev_sum, sizeof(int)) );

    CUDA_TIME_START();

    //collect1<<<1,1>>>(dev_a, dev_sum);
    //collect_simplified<<<(N + T - 1) / T, T>>>(dev_a, dev_sum, N);
    collect<<<(N + T - 1) / T, T>>>(dev_a, dev_sum, N, my_log2(N));
    //sum = add_cpu(a, n);

    CUDA_TIME_END();
    LAST_ERR();


    CHECK_ERR( cudaMemcpy(&sum, dev_sum, sizeof(int), cudaMemcpyDeviceToHost) );

    //print_res(a, sum, n);
    TIME_START();
    int c = control(a);
    TIME_END();
    //printf("[CONTROL] sum = %d\n", c);
    CHECK_ERR( cudaFree(dev_a) );
    CHECK_ERR( cudaFree(dev_sum) );
    free(a);

    return 0;
}