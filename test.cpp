#include "BitonicSort.h"
#include "BitonicSortCUDA.cuh"
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>

#define TYPE 2
#define PROF 1

#ifdef _DEBUG
int main()
{
	int N = 131072;
#if TYPE==1
	float* mem = new float[N];
#elif TYPE ==2
	double* mem = new double[N];
#else
	unsigned int* mem = new unsigned int[N];
#endif
	unsigned int* index = new unsigned int[N];
	for (int n = 0; n < N; n++)
	{
		mem[n] = rand();
		index[n] = n;
	}
	BitonicSort::BitonicSortRank(mem, index, N);
	BitonicSort::BitonicSortCUDARank(mem, index, N);
	return 0;
}
#else
#if PROF == 1
int main()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}
	int N = 131072;
#if TYPE==1
	float* mem = new float[N];
#elif TYPE ==2
	double* mem = new double[N];
#else
	unsigned int* mem = new unsigned int[N];
#endif
	for (int n = 0; n < N; n++)
	{
		mem[n] = rand();
	}
	cudaStatus = BitonicSort::BitonicSortCUDA(mem, N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	delete[] mem;
	return 0;
}
#else
int main()
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}
	std::ofstream myfile;
	myfile.open("example.csv");
	myfile << "N, GPU, CPU\n";
	for (int n = 8; n <= 22; n++)
	{
		int N = 1 << n;
#if TYPE==1
		float* mem = new float[N];
		float* mem2 = new float[N];
#elif TYPE ==2
		double* mem = new double[N];
		double* mem2 = new double[N];
#else
		unsigned int* mem = new unsigned int[N];
		unsigned int* mem2 = new unsigned int[N];
#endif
		std::cout << "For " << N << ": ";
		myfile << N << ",";
		for (int n = 0; n < N; n++)
		{
			mem2[n] = mem[n] = rand();
		}

		// Add vectors in parallel.
		auto t1 = std::chrono::high_resolution_clock::now();
		cudaError_t cudaStatus = BitonicSort::BitonicSortCUDA(mem, N);
		auto t2 = std::chrono::high_resolution_clock::now();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
		std::cout << "GPU :"
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
			<< "ms";
		myfile << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << ",";
		t1 = std::chrono::high_resolution_clock::now();
		BitonicSort::BitonicSort(mem2, N);
		t2 = std::chrono::high_resolution_clock::now();
		std::cout << "CPU : "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
			<< " ms\n";
		myfile << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\n";
		delete[] mem;
	}
	myfile.close();
	return 0;
}
#endif
#endif