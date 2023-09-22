#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define N 16
#define T 4

static struct timespec time_measure_start;
static struct timespec time_measure_end;

#define TIME_START() \
do { \
    clock_gettime(CLOCK_MONOTONIC, &time_measure_start); \
    printf("[TIME] Measurement started.\n"); \
} while (0)

#define TIME_END() \
do { \
    clock_gettime(CLOCK_MONOTONIC, &time_measure_end); \
    printf("[TIME] Measurement ended.\n"); \
    printf("[TIME] Elapsed time = %f ms\n", ((time_measure_end.tv_sec * 1000000000 + time_measure_end.tv_nsec) - (time_measure_start.tv_sec * 1000000000 + time_measure_start.tv_nsec)) / 1000000.0); \
} while (0)

#define CHECK_ERR(val) check_err((val), #val, __FILE__, __LINE__)
void check_err(cudaError_t err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[CUDA] Error at %s:%d\n%s %s\n", file, line, cudaGetErrorString(err), func);
    }
}

#define LAST_ERR() last_err(__FILE__, __LINE__)
void last_err(const char* const file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "[CUDA] Error at %s:%d\n%s\n",file, line, cudaGetErrorString(err));
    }
}

// this is where the reduce magic happens
__global__ void add(int* a, int* sum)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int subtotal;


    // increment the subtotals with each thread
    subtotal += a[tid];
    printf("[KERNEL] block %d thread %d tid %d value %d sub %d\n", blockIdx.x, threadIdx.x, tid,a[tid], subtotal);

    __syncthreads();

    *sum = subtotal;
/*
    int i=0;
    while (i < blockDim.x)
    {
        i ++;
    }
*/
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

    TIME_START();

    add<<<(N + T - 1) / T, T>>>(dev_a, dev_sum);
    //sum = add_cpu(a, n);

    TIME_END();
    LAST_ERR();


    CHECK_ERR( cudaMemcpy(&sum, dev_sum, sizeof(int), cudaMemcpyDeviceToHost) );

    print_res(a, sum, n);
    CHECK_ERR( cudaFree(dev_a) );
    CHECK_ERR( cudaFree(dev_sum) );
    free(a);

    return 0;
}