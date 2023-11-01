#include <stdlib.h>
#include <stdio.h>
#include "timetest.h"
#include "error.h"

/*
Control Solution
{{ 5,  8,  1,  3},{ 7,  5,  4,  6},{ 1,  0,  6,  6},{ 0,  2,  5,  0}}
{{ 4,  2,  0,  7},{ 7,  0,  8,  9},{ 2,  2,  1,  7},{ 2,  8,  3,  2}}
{{84, 36, 74, 120},{83, 70, 62, 134},{28, 62, 24, 61},{24, 10, 21, 53}}
[TIME] Elapsed time (N 1024) = 13885.393350 ms
*/

#define N 8
#define T 32

// code smells ass at this point, but I'll allow it
#define IND(x, y) x * n + y

void randomize(int* a, int* b, int n)
{
    srand(20001225);
    for (int i=0;i<n*n;i++)
    {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }
}

void print_mat(int* a, int n)
{
    printf("{");
    for (int i=0;i<n;i++)
    {
        printf("{");
        for (int j=0;j<n;j++)
        {
            printf("%2d", a[IND(i,j)]);
            if (j != n-1)
                printf(", ");
            else
                printf("}");
        }
        if (i != n - 1)
            printf(",\n");
    }
	printf("}\n");
}

// reference matrix multiplication implementation
// O(x^3)
void control(int* a, int* b, int* c, int n)
{
    for (int i=0; i<n; i++) // dot product of the i-th row of A and i-th column of B
        for (int j=0; j<n; j++)
            for (int k=0; k<n; k++)
                c[IND(i,j)] += a[IND(i,k)] * b[IND(k,j)];
}

__global__ void multiply_dumb(int* a, int* b, int* c, int n)
{
    int k;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int j = 0;
    while (j < n)
    {
        k = 0;
        while (k < n)
        {
            c[IND(i,j)] += a[IND(i,k)] * b[IND(k,j)];
            __syncthreads();
            k += 1;
        }
        //printf("[KERNEL] i %d j %d c %d\n", i, j, c[IND(i,j)]);
        j++;
    }
}

__global__ void multiply_not_so_good(int* a, int* b, int* c, int n)
{
   /* 
    printf("[KERNEL] grid (%d %d %d) block (%d %d %d), (%d %d %d) thread (%d %d %d)\n",
        gridDim.x, gridDim.y, gridDim.z, 
        blockDim.x, blockDim.y, blockDim.z,
        blockIdx.x, blockIdx.y, blockIdx.z,
        threadIdx.x, threadIdx.y, threadIdx.z);
        c[0] = 0;
    */ 
    int k = 0;
    int i = blockIdx.x;
    int j = blockIdx.y;
    while (k < n)
    {
        c[IND(i,j)] += a[IND(i,k)] * b[IND(k,j)];
        __syncthreads();
        k++;
    }
}

__global__ void multiply(int* a, int* b, int*c, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < n && j < n)
        for (int k = 0; k < n; k++) 
            c[IND(i,j)] += a[IND(i,k)] * b[IND(k,j)];
}

int main(int argc, char** argv)
{
    int n = N;
    int t = T;

    if (argc < 3)
    {
        n = N;
        t = T;
    } 
    else
    {
        n = atoi(argv[1]);
        t = atoi(argv[2]);
    }

    int* a = (int*)malloc(sizeof(int) * n * n);
    int* b = (int*)malloc(sizeof(int) * n * n);
    int* c = (int*)malloc(sizeof(int) * n * n);
    randomize(a, b, n);

    int *dev_a;
	int *dev_b;
	int *dev_c;

    CHECK_ERR(cudaMalloc((void**)& dev_a, sizeof(int) * n * n));
    CHECK_ERR(cudaMalloc((void**)& dev_b, sizeof(int) * n * n));
    CHECK_ERR(cudaMalloc((void**)& dev_c, sizeof(int) * n * n));

    CHECK_ERR(cudaMemcpy(dev_a, a, sizeof(int) * n * n, cudaMemcpyHostToDevice));
    CHECK_ERR(cudaMemcpy(dev_b, b, sizeof(int) * n * n, cudaMemcpyHostToDevice));

    printf("\n---------------------------------\n N = %d, T = %d\n", n, t);
/*
    CUDA_TIME_START();
    multiply_dumb<<<(n + t - 0) / t, t>>>(dev_a, dev_b, dev_c, n);
    
    // ---
    dim3 grid = {(unsigned)n, (unsigned)n};
    multiply_not_so_good<<<grid, t>>>(dev_a, dev_b, dev_c, n);
*/
/* 
    // ---
    dim3 block = {(unsigned)t, (unsigned)t};
    unsigned int grid_size = ((unsigned)n + (unsigned)t - 1) / (unsigned)t;
    //printf("[DBG] grid size %d\n", grid_size);
    dim3 grid = {grid_size, grid_size};
    multiply<<<grid, block>>>(dev_a, dev_b, dev_c, n);
*/
/*
    CUDA_TIME_END();
    LAST_ERR();
    CHECK_ERR(cudaMemcpy(c, dev_c, sizeof(int) * n * n, cudaMemcpyDeviceToHost));
*/
//*
    
    TIME_START();
    control(a, b, c, n);
    TIME_END();
//*/
/*
    print_mat(a);
    print_mat(b);
*/

    //print_mat(c, n);

    free(a);
    free(b);
    free(c);

    CHECK_ERR(cudaFree(dev_a));
    CHECK_ERR(cudaFree(dev_b));
    CHECK_ERR(cudaFree(dev_c));

    return 0;
}
