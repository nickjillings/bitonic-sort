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

## Ranking

Ranking returns a list of indices which would sort the list, without actually touching the list. This is useful if you then need to use one list to sort another!

### Performance

Some quick comparisons of CUDA and CPU:

N       | i5-3317U  | GT 630M
---     |   ---     | ---
4096    | 1ms       | 55ms
8192    | 2ms       | 44ms
16384   | 5ms       | 43ms
32768   | 13ms      | 43ms
65536   | 29ms      | 49ms
131072  | 61ms      | 51ms
262144  | 115ms     | 120ms
524288  | 191ms     | 107ms
1048576 | 388ms     | 105ms
2097152 | 820ms     | 195ms
4194304 | 1703ms    | 385ms