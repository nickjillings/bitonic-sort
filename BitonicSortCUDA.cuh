#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#ifndef _BITONIC_SORT_C
#include "BitonicSort.h"
#endif
namespace BitonicSort {
	cudaError_t BitonicSortCUDA(unsigned int* mem, int N);
	cudaError_t BitonicSortCUDA(float* mem, int N);
	cudaError_t BitonicSortCUDA(double* mem, int N);
}