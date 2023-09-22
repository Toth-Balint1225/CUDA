# CUDA repository for code sharing
Sample and practice codes on my development machine.

## error.h
Standard error handling for CUDA. Defines cuda function call wrapper macro `CHECK_ERR(func)` and kernel check `LAST_ERR()`.

## timetest.h
Time measurement macros `TIME_START()` and `TIME_END()`. Uses the monotonic clock and `clock_gettime()`.

## saxpy.cu
Simple "Hello World" program for CUDA. Executes the y = ax + y vector operation.

## sum.cu
Sum the elements of a vector. (**pain**) *TODO*

## matricx.cu
*TODO*