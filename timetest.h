#ifndef TIMETEST_H
#define TIMETEST_H

#include <time.h>
#include <stdio.h>

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

#endif // TIMETEST_H
