# bitonic-sort
[Bitonic Sort](https://en.wikipedia.org/wiki/Bitonic_sorter) for C and CUDA

Include the header files for either C or CUDA libraries. C header is "BitonicSort.h" and CUDA is "BitonicSortCUDA.cuh" (BitonicSortCUDA.cuh includes the BitonicSort.h).

```C
#include "BitonicSort.h"
float* mem_block;
int N = 131072;
BitonicSort::BitonicSort(mem_block, N);
```

```CUDA
#include "BitonicSort.cuh"
float* mem_block;
int N = 131072;
BitonicSort::BitonicSortCUDA(mem_block, N);
```
