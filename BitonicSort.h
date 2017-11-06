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
}

#endif