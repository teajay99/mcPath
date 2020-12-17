#include "config.hpp"

#ifndef CUSTOMMATH_HPP
#define CUSTOMMATH_HPP

// Function for x to the power p for integers
__host__ __device__ inline int intPow(int x, int p) {
  if (p == 0)
    return 1;
  if (p == 1)
    return x;

  int tmp = intPow(x, p / 2);
  if (p % 2 == 0)
    return tmp * tmp;
  else
    return x * tmp * tmp;
}

__host__ __device__ inline int getBlockCount(int n) {
  return (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
}

#endif
