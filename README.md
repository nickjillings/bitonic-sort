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

N       | i5-3317U  |Xeon x5365	|Xeon E5-2630 v4    | GT 630M	| GTX 1050   | Tesla M40
---     |   ---     | ---       | ---               | ---		| ---        | ---
4096    | 0.48ms    | 0.34ms	| 0.25ms            | 1.55ms	| 0.95ms     | 0.71ms
8192    | 1.03ms    | 0.76ms	| 0.53ms            | 1.72ms	| 1.10ms     | 0.78ms
16384   | 2.27ms    | 1.70ms	| 1.10ms            | 2.03ms	| 1.28ms     | 0.83ms
32768   | 4.82ms    | 3.59ms	| 2.48ms            | 2.74ms	| 1.53ms     | 0.95ms
65536   | 10.48ms   | 7.69ms	| 5.05ms            | 4.18ms	| 1.92ms     | 1.14ms
131072  | 22.97ms   | 17.70ms	| 10.84ms           | 7.93ms	| 2.66ms     | 1.45ms
262144  | 48.86ms   | 39.44ms 	| 24.50ms           | 15.65ms	| 4.88ms     | 2.02ms
524288  | 106.48ms  | 93.80ms	| 51.51ms           | 32.77ms	| 12.72ms    | 3.04ms
1048576 | 233.91ms  | 183.54ms	| 104.58ms          | 70.48ms	| 25.92ms    | 9.22ms
2097152 | 493.46ms  | 430.17ms	| 224.48ms          | 152.79ms	| 54.05ms    | 19.10ms
4194304 | 1083.80ms | 1002.40ms	| 506.80ms          | 332.50ms	| 114.78ms   | 40.47ms
