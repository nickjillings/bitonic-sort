# Parallel Bitonic Sort
[Bitonic Sort](https://en.wikipedia.org/wiki/Bitonic_sorter) for C and CUDA.

Includes both inplace sorting and ranking algorithms.

Include the header files for either C or CUDA libraries. C header is "BitonicSort.h" and CUDA is "BitonicSortCUDA.cuh" (BitonicSortCUDA.cuh includes the BitonicSort.h).

```C
#include "BitonicSort.h"
float* mem_block;
int N = 131072;
BitonicSort::BitonicSort(mem_block, N);
```

```CUDA
#include "BitonicSortCUDA.cuh"
float* mem_block;
int N = 131072;
BitonicSort::BitonicSortCUDA(mem_block, N);
```

## Sorting

The sorting algorithm takes an array of floats, doubles or unsigned integers and sorts them into a descending list. This function is made of three specialist kernels for each data type, allowing the CUDA compiler to make the most efficient targets for your system.

Use `BitonicSortCUDA` to perform the sorting function. This will perform the copy from host to device and back. `BitonicSortCUDAZero` instead assumes the data is already on device memory, so will not do the copies.

## Ranking

Ranking returns a list of indices which would sort the list, without actually touching the list. This is useful if you then need to use one list to sort another!

Use `BitonicSortRankCUDA` to perform the ranking function. This will perform the copy from host to device and back. `BitonicSortRankCUDAZero` instead assumes the data is already on device memory, so will not do the copies.


### Performance

Some quick comparisons of CUDA and CPU:

N       | i5-3317U  | GT 630M
---     |   ---     | ---
4096    | 0.48ms    | 1.55ms
8192    | 1.03ms    | 1.72ms
16384   | 2.27ms    | 2.03ms
32768   | 4.82ms    | 2.74ms
65536   | 10.48ms   | 4.18ms
131072  | 22.97ms   | 7.93ms
262144  | 48.86ms   | 15.65ms
524288  | 106.48ms  | 32.77ms
1048576 | 233.91ms  | 70.48ms
2097152 | 493.46ms  | 152.79ms
4194304 | 1083.80ms | 332.50ms