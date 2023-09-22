#include <stdio.h>
#include <cuda_runtime.h>

#define CHECKERR(call) checkerr((call), #call, __FILE__, __LINE__)
void checkerr(cudaError_t err, const char* func, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[CUDA] ERROR at %s:%d\n%s %s\n", file, line, cudaGetErrorString(err), func);
    }
}

#define LASTERR() lasterr(__FILE__, __LINE__)
void lasterr(const char* file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[CUDA] ERROR at %s:%d\n%s\n", file, line, cudaGetErrorString(err));
    }
}
