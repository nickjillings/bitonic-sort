#include "BitonicSort.h"

namespace BitonicSort
{
	void compare(unsigned int* a, unsigned int* b)
	{
		unsigned int A = *a;
		unsigned int B = *b;
		if (A > B)
		{
			*a = B;
			*b = A;
		}
	}
	void compare(float* a, float* b)
	{
		float A = *a;
		float B = *b;
		if (A > B)
		{
			*a = B;
			*b = A;
		}
	}
	void compare(double* a, double* b)
	{
		double A = *a;
		double B = *b;
		if (A > B)
		{
			*a = B;
			*b = A;
		}
	}

	void bitonic(unsigned int* mem, int N)
	{
		int K = log2(N);
		int d = 1 << K;
		for (int n = 0; n < d >> 1; n++)
		{
			compare(&mem[n], &mem[d - n - 1]);
		}
		K--;
		if (K <= 0) {
			return;
		}
		for (int k = K; k > 0; k--)
		{
			d = 1 << k;
			for (int m = 0; m < N; m += d)
			{
				for (int n = 0; n < d >> 1; n++)
				{
					compare(&mem[m + n], &mem[m + (d >> 1) + n]);
				}
			}
		}
	}
	void bitonic(float* mem, int N)
	{
		int K = log2(N);
		int d = 1 << K;
		for (int n = 0; n < d >> 1; n++)
		{
			compare(&mem[n], &mem[d - n - 1]);
		}
		K--;
		if (K <= 0) {
			return;
		}
		for (int k = K; k > 0; k--)
		{
			d = 1 << k;
			for (int m = 0; m < N; m += d)
			{
				for (int n = 0; n < d >> 1; n++)
				{
					compare(&mem[m + n], &mem[m + (d >> 1) + n]);
				}
			}
		}
	}
	void bitonic(double* mem, int N)
	{
		int K = log2(N);
		int d = 1 << K;
		for (int n = 0; n < d >> 1; n++)
		{
			compare(&mem[n], &mem[d - n - 1]);
		}
		K--;
		if (K <= 0) {
			return;
		}
		for (int k = K; k > 0; k--)
		{
			d = 1 << k;
			for (int m = 0; m < N; m += d)
			{
				for (int n = 0; n < d >> 1; n++)
				{
					compare(&mem[m + n], &mem[m + (d >> 1) + n]);
				}
			}
		}
	}

	void BitonicSort(unsigned int* mem, int N)
	{
		unsigned int* map = new unsigned int[N];
		for (int n = 0; n < N; n++)
		{
			map[n] = mem[n];
		}
		int K = log2(N);
		for (int k = 1; k <= K; k++)
		{
			int d = 1 << k;
			for (int n = 0; n < N; n += d) {
				unsigned int* map_ptr = &map[n];
				bitonic(map_ptr, d);
			}
		}
		for (int n = 0; n < N; n++)
		{
			mem[n] = map[n];
		}
		delete[] map;
	}

	void BitonicSort(float* mem, int N)
	{
		float* map = new float[N];
		for (int n = 0; n < N; n++)
		{
			map[n] = mem[n];
		}
		int K = log2(N);
		for (int k = 1; k <= K; k++)
		{
			int d = 1 << k;
			for (int n = 0; n < N; n += d) {
				float* map_ptr = &map[n];
				bitonic(map_ptr, d);
			}
		}
		for (int n = 0; n < N; n++)
		{
			mem[n] = map[n];
		}
		delete[] map;
	}

	void BitonicSort(double* mem, int N)
	{
		double* map = new double[N];
		for (int n = 0; n < N; n++)
		{
			map[n] = mem[n];
		}
		int K = log2(N);
		for (int k = 1; k <= K; k++)
		{
			int d = 1 << k;
			for (int n = 0; n < N; n += d) {
				double* map_ptr = &map[n];
				bitonic(map_ptr, d);
			}
		}
		for (int n = 0; n < N; n++)
		{
			mem[n] = map[n];
		}
		delete[] map;
	}
}