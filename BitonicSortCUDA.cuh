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

	cudaError_t BitonicSortCUDARank(unsigned int* mem, unsigned int* index, int N);
	cudaError_t BitonicSortCUDARank(float* mem, unsigned int* index, int N);
	cudaError_t BitonicSortCUDARank(double* mem, unsigned int* index, int N);

	cudaError_t BitonicSortCUDAZero(unsigned int* mem, int N);
	cudaError_t BitonicSortCUDAZero(float* mem, int N);
	cudaError_t BitonicSortCUDAZero(double* mem, int N);

	cudaError_t BitonicSortCUDARankZero(unsigned int* mem, unsigned int* index, int N);
	cudaError_t BitonicSortCUDARankZero(float* mem, unsigned int* index, int N);
	cudaError_t BitonicSortCUDARankZero(double* mem, unsigned int* index, int N);
}