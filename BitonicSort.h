#include <stdio.h>
#include <stdint.h>
#include <cmath>

#ifndef _BITONIC_SORT_C
#define _BITONIC_SORT_C

namespace BitonicSort
{
	void compare(unsigned int* a, unsigned int* b);
	void compare(float* a, float* b);
	void compare(double* a, double* b);

	void bitonic(unsigned int* mem, int N);
	void bitonic(float* mem, int N);
	void bitonic(double* mem, int N);

	void BitonicSort(unsigned int* mem, int N);
	void BitonicSort(float* mem, int N);
	void BitonicSort(double* mem, int N);

	void compare(unsigned int *a, unsigned int *b, unsigned int *a_index, unsigned int *b_index);
	void bitonic(unsigned int* mem, unsigned int *index, int N);
	void BitonicSortRank(unsigned int* mem, unsigned int* index, int N);

	void compare(float *a, float *b, unsigned int *a_index, unsigned int *b_index);
	void bitonic(float* mem, unsigned int *index, int N);
	void BitonicSortRank(float* mem, unsigned int* index, int N);

	void compare(double *a, double *b, unsigned int *a_index, unsigned int *b_index);
	void bitonic(double* mem, unsigned int *index, int N);
	void BitonicSortRank(double* mem, unsigned int* index, int N);
}

#endif