#include "error.h"
#include <stdlib.h>
#include "time.h"

#define T 2

#define N 16
//#define N 1024
//#define N 65536
//#define N 16777216
//#define N 33554432
//#define N 67108864

/**
 * Saxpy: y = ax + y, where a is scalar, x, y are vectors
*/

void saxpy(int a, int* x, int* y)
{
    for (size_t i=0;i<N;i++)
    {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void saxpy_kernel(int a, int* x, int* y)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
        y[i] = a * x[i] + y[i];
}

void print(int a, int* x, int* y) 
{
    printf("a = %d\n", a);
    for (size_t i=0;i<N;i++)
        printf("[%5lu]: %4d %4d\n", i, x[i], y[i]);
}

void randomize(int* x, int* y, int* a)
{
    srand(20001225);
    for (size_t i=0;i<N;i++)
    {
        x[i] = rand() % 10;
        y[i] = rand() % 10;
    }
    *a = rand() % 100;
}

void test_CPU(void)
{
    int *x, *y, a;
    // allocate CPU and GPU memory
    x = (int*)malloc(N*sizeof(int));
    y = (int*)malloc(N*sizeof(int));

    // generate data
    randomize(x, y, &a);


    clock_t start = clock();
    // compute on the CPU
    saxpy(a, x, y);
    clock_t end = clock();
    double elapsed = ((double)end - start)*1000.0 / (CLOCKS_PER_SEC);
    
    //print(a, x, y);
    printf("Elapsed time: %lf ms\n", elapsed);

    free(x);
    free(y);
}

void test_GPU(void)
{
    int *x, *y, a;
    // allocate CPU and GPU memory
    x = (int*)malloc(N*sizeof(int));
    y = (int*)malloc(N*sizeof(int));

    int *cuda_x, *cuda_y;
    CHECK_ERR( cudaMalloc((void**)&cuda_x, N*sizeof(int)) );
    CHECK_ERR( cudaMalloc((void**)&cuda_y, N*sizeof(int)) );
    

    // generate data
    randomize(x, y, &a);

    // copy data to GPU
    CHECK_ERR( cudaMemcpy(cuda_x, x, N*sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_ERR( cudaMemcpy(cuda_y, y, N*sizeof(int), cudaMemcpyHostToDevice) );

    clock_t start = clock();

    // compute on the GPU
    saxpy_kernel<<<(N + T - 1) / T, T>>>(a, cuda_x, cuda_y);
    LAST_ERR();

    clock_t end = clock();
    double elapsed = ((double)end - start)*1000.0 / (CLOCKS_PER_SEC);
    
    CHECK_ERR( cudaMemcpy(x, cuda_x, N*sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_ERR( cudaMemcpy(y, cuda_y, N*sizeof(int), cudaMemcpyDeviceToHost) );
    print(a, x, y);
    printf("Elapsed time: %lf ms\n", elapsed);

    CHECK_ERR( cudaFree(cuda_x) );
    CHECK_ERR( cudaFree(cuda_y) );
    free(x);
    free(y);
}

int main(void)
{
//    printf("Testing CPU\n");
//    test_CPU();
//    printf("Testing GPU\n");
    test_GPU();
}