
#include "BitonicSortCUDA.cuh"

// Bitonic for uint32_t
__device__ void _bitonicStep1_uint32(unsigned int * smem, int tid, int tpp, int d)
{
	int m = tid / (d >> 1);
	int tib = tid - m*(d >> 1);
	int addr1 = d*m + tib;
	int addr2 = (m + 1)*d - tib - 1;
	
	unsigned int A = smem[addr1];
	unsigned int B = smem[addr2];
	smem[addr1] = max(A, B);
	smem[addr2] = min(A, B);
}

__device__ void _bitonicStep2_uint32(unsigned int * smem, int tid, int tpp, int d)
{
	int m = tid / (d >> 1);
	int tib = tid - m*(d >> 1);
	int addr1 = d*m + tib;
	int addr2 = addr1 + (d >> 1);

	unsigned int A = smem[addr1];
	unsigned int B = smem[addr2];
	smem[addr1] = max(A, B);
	smem[addr2] = min(A, B);
}

__global__ void bitonicSortKernel128_uint32(unsigned int* mem)
{
	// Operating on 64 samples
	int bid = blockIdx.x; // Block UID
	int tpp = threadIdx.x; // Thread position in block
	int tid = blockIdx.x*blockDim.x + threadIdx.x; // Thread global UID
	__shared__ unsigned int smem[256]; // Two blocks worth of shared memory
	smem[tpp] = mem[blockDim.x*(2 * bid) + tpp]; // Coalesced memory load
	smem[tpp + blockDim.x] = mem[blockDim.x*((2 * bid) + 1) + tpp]; // Coalesced memory load
	int blocks = 8;
	for (int blockNum = 1; blockNum <= blocks; blockNum++)
	{
		int d = 1 << blockNum;
		_bitonicStep1_uint32(smem, tpp, tpp, d);
		__syncthreads();
		d = d >> 1;
		while(d >= 2)
		{
			_bitonicStep2_uint32(smem, tpp, tpp, d);
			__syncthreads();
			d = d >> 1;
		}
	}

	mem[blockDim.x*(2 * bid) + tpp] = smem[tpp];
	mem[blockDim.x*((2*bid)+1) + tpp] = smem[tpp + blockDim.x];
}

__global__ void bitonicSortKernelXBlock1_uint32(unsigned int* mem, int blockNum)
{
	int bid = blockIdx.x; // Block UID
	int tpp = threadIdx.x; // Thread position in block
	int tid = blockIdx.x*blockDim.x + threadIdx.x; // Thread global UID
	int d = 1 << blockNum;
	_bitonicStep1_uint32(mem, tid, tpp, d);
}
__global__ void bitonicSortKernelXBlock2_uint32(unsigned int* mem, int blockNum, int d)
{
	int bid = blockIdx.x; // Block UID
	int tpp = threadIdx.x; // Thread position in block
	int tid = blockIdx.x*blockDim.x + threadIdx.x; // Thread global UID
	_bitonicStep2_uint32(mem, tid, tpp, d);
	
}


// For float32

__device__ void _bitonicStep1_fp32(float * smem, int tid, int tpp, int d)
{
	int m = tid / (d >> 1);
	int tib = tid - m*(d >> 1);
	int addr1 = d*m + tib;
	int addr2 = (m + 1)*d - tib - 1;

	float A = smem[addr1];
	float B = smem[addr2];
	smem[addr1] = max(A, B);
	smem[addr2] = min(A, B);
}

__device__ void _bitonicStep2_fp32(float * smem, int tid, int tpp, int d)
{
	int m = tid / (d >> 1);
	int tib = tid - m*(d >> 1);
	int addr1 = d*m + tib;
	int addr2 = addr1 + (d >> 1);

	float A = smem[addr1];
	float B = smem[addr2];
	smem[addr1] = max(A, B);
	smem[addr2] = min(A, B);
}

__global__ void bitonicSortKernel128_fp32(float* mem)
{
	// Operating on 64 samples
	int bid = blockIdx.x; // Block UID
	int tpp = threadIdx.x; // Thread position in block
	int tid = blockIdx.x*blockDim.x + threadIdx.x; // Thread global UID
	__shared__ float smem[256]; // Two blocks worth of shared memory
	smem[tpp] = mem[blockDim.x*(2 * bid) + tpp]; // Coalesced memory load
	smem[tpp + blockDim.x] = mem[blockDim.x*((2 * bid) + 1) + tpp]; // Coalesced memory load
	int blocks = 8;
	for (int blockNum = 1; blockNum <= blocks; blockNum++)
	{
		int d = 1 << blockNum;
		_bitonicStep1_fp32(smem, tpp, tpp, d);
		__syncthreads();
		d = d >> 1;
		while (d >= 2)
		{
			_bitonicStep2_fp32(smem, tpp, tpp, d);
			__syncthreads();
			d = d >> 1;
		}
	}

	mem[blockDim.x*(2 * bid) + tpp] = smem[tpp];
	mem[blockDim.x*((2 * bid) + 1) + tpp] = smem[tpp + blockDim.x];
}

__global__ void bitonicSortKernelXBlock1_fp32(float* mem, int blockNum)
{
	int bid = blockIdx.x; // Block UID
	int tpp = threadIdx.x; // Thread position in block
	int tid = blockIdx.x*blockDim.x + threadIdx.x; // Thread global UID
	int d = 1 << blockNum;
	_bitonicStep1_fp32(mem, tid, tpp, d);
}
__global__ void bitonicSortKernelXBlock2_fp32(float* mem, int blockNum, int d)
{
	int bid = blockIdx.x; // Block UID
	int tpp = threadIdx.x; // Thread position in block
	int tid = blockIdx.x*blockDim.x + threadIdx.x; // Thread global UID
	_bitonicStep2_fp32(mem, tid, tpp, d);

}


// For float32

__device__ void _bitonicStep1_fp64(double * smem, int tid, int tpp, int d)
{
	int m = tid / (d >> 1);
	int tib = tid - m*(d >> 1);
	int addr1 = d*m + tib;
	int addr2 = (m + 1)*d - tib - 1;

	double A = smem[addr1];
	double B = smem[addr2];
	smem[addr1] = max(A, B);
	smem[addr2] = min(A, B);
}

__device__ void _bitonicStep2_fp64(double * smem, int tid, int tpp, int d)
{
	int m = tid / (d >> 1);
	int tib = tid - m*(d >> 1);
	int addr1 = d*m + tib;
	int addr2 = addr1 + (d >> 1);

	double A = smem[addr1];
	double B = smem[addr2];
	smem[addr1] = max(A, B);
	smem[addr2] = min(A, B);
}

__global__ void bitonicSortKernel128_fp64(double* mem)
{
	// Operating on 64 samples
	int bid = blockIdx.x; // Block UID
	int tpp = threadIdx.x; // Thread position in block
	int tid = blockIdx.x*blockDim.x + threadIdx.x; // Thread global UID
	__shared__ double smem[256]; // Two blocks worth of shared memory
	smem[tpp] = mem[blockDim.x*(2 * bid) + tpp]; // Coalesced memory load
	smem[tpp + blockDim.x] = mem[blockDim.x*((2 * bid) + 1) + tpp]; // Coalesced memory load
	int blocks = 8;
	for (int blockNum = 1; blockNum <= blocks; blockNum++)
	{
		int d = 1 << blockNum;
		_bitonicStep1_fp64(smem, tpp, tpp, d);
		__syncthreads();
		d = d >> 1;
		while (d >= 2)
		{
			_bitonicStep2_fp64(smem, tpp, tpp, d);
			__syncthreads();
			d = d >> 1;
		}
	}

	mem[blockDim.x*(2 * bid) + tpp] = smem[tpp];
	mem[blockDim.x*((2 * bid) + 1) + tpp] = smem[tpp + blockDim.x];
}

__global__ void bitonicSortKernelXBlock1_fp64(double* mem, int blockNum)
{
	int bid = blockIdx.x; // Block UID
	int tpp = threadIdx.x; // Thread position in block
	int tid = blockIdx.x*blockDim.x + threadIdx.x; // Thread global UID
	int d = 1 << blockNum;
	_bitonicStep1_fp64(mem, tid, tpp, d);
}
__global__ void bitonicSortKernelXBlock2_fp64(double* mem, int blockNum, int d)
{
	int bid = blockIdx.x; // Block UID
	int tpp = threadIdx.x; // Thread position in block
	int tid = blockIdx.x*blockDim.x + threadIdx.x; // Thread global UID
	_bitonicStep2_fp64(mem, tid, tpp, d);

}


// Helper function for using CUDA to add vectors in parallel.

cudaError_t BitonicSort::BitonicSortCUDA(unsigned int* mem, int N)
{
	cudaError_t cudaStatus;
	unsigned int* dev_mem;

	// Allocate GPU buffers for vector
	cudaStatus = cudaMalloc((void**)&dev_mem, N * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_mem, mem, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	int numBlocks = log2(N);
	
	bitonicSortKernel128_uint32 <<<N/256, 128 >>>(dev_mem);
	for (int b = 9; b <= numBlocks; b++)
	{
		int d = 1 << b;
		bitonicSortKernelXBlock1_uint32 << <N / 512, 256 >> >(dev_mem,b);
		d = d >> 1;
		while (d >= 2)
		{
			bitonicSortKernelXBlock2_uint32 << <N / 512, 256 >> >(dev_mem, b, d);
			d = d >> 1;
		}
	}
	
	//bitonicSortKernelTestDbg <<< N / 256, 128 >>> (dev_mem);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bitonicSortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(mem, dev_mem, N * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
Error:
	cudaFree(dev_mem);

	return cudaStatus;
}

cudaError_t BitonicSort::BitonicSortCUDA(float* mem, int N)
{
	cudaError_t cudaStatus;
	float* dev_mem;

	// Allocate GPU buffers for vector
	cudaStatus = cudaMalloc((void**)&dev_mem, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_mem, mem, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	int numBlocks = log2(N);

	bitonicSortKernel128_fp32 <<<N / 256, 128 >> >(dev_mem);
	for (int b = 9; b <= numBlocks; b++)
	{
		int d = 1 << b;
		bitonicSortKernelXBlock1_fp32 <<<N / 512, 256 >> >(dev_mem, b);
		d = d >> 1;
		while (d >= 2)
		{
			bitonicSortKernelXBlock2_fp32 <<<N / 512, 256 >> >(dev_mem, b, d);
			d = d >> 1;
		}
	}

	//bitonicSortKernelTestDbg <<< N / 256, 128 >>> (dev_mem);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bitonicSortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(mem, dev_mem, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
Error:
	cudaFree(dev_mem);

	return cudaStatus;
}

cudaError_t BitonicSort::BitonicSortCUDA(double* mem, int N)
{
	cudaError_t cudaStatus;
	double* dev_mem;

	// Allocate GPU buffers for vector
	cudaStatus = cudaMalloc((void**)&dev_mem, N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_mem, mem, N * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	int numBlocks = log2(N);

	bitonicSortKernel128_fp64 << <N / 256, 128 >> >(dev_mem);
	for (int b = 9; b <= numBlocks; b++)
	{
		int d = 1 << b;
		bitonicSortKernelXBlock1_fp64 << <N / 512, 256 >> >(dev_mem, b);
		d = d >> 1;
		while (d >= 2)
		{
			bitonicSortKernelXBlock2_fp64 << <N / 512, 256 >> >(dev_mem, b, d);
			d = d >> 1;
		}
	}

	//bitonicSortKernelTestDbg <<< N / 256, 128 >>> (dev_mem);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bitonicSortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(mem, dev_mem, N * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
Error:
	cudaFree(dev_mem);

	return cudaStatus;
}