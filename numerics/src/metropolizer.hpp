#include "curand_kernel.h"
#include "lagrangian.hpp"

#ifndef METROPOLIZER_HPP
#define METROPOLIZER_HPP

template <typename T> inline __device__ T device_getRand(curandState *s) {
  return 0;
}
template <> inline __device__ double device_getRand<double>(curandState *s) {
  return curand_uniform_double(s);
}
template <> inline __device__ float device_getRand<float>(curandState *s) {
  return curand_uniform(s);
}

inline __global__ void kernel_initIteration(curandState *state, int n) {

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    curand_init(42, idx, 0, &state[idx]);
  }
}

template <typename T, class Lag>
__host__ __device__ T device_evalActionChange(Lag L, T *path, int loc) {
  T out = L.eval(path, loc);
  for (int k = 0; k < L.getRootDim(); k++) {
    out += L.eval(path,
                  (loc + L.getRootDof() - L.getRootBasis(k)) % L.getRootDof());
  }

  return out * pow(L.getLatSpacing(), L.getRootDim());
}

//__global__ void prescan(float *g_odata, float *g_idata, int n) {
// template <typename T, class Lag>
// inline __global__ void kernel_evalAction(T *path, Lag L, T *out) {
//   int thid = threadIdx.x;
//   int n = 2 * CUDA_BLOCK_SIZE;
//
//   __shared__ T temp[2 * CUDA_BLOCK_SIZE];
//
//   int sitesPerThread = getBlockCount(L.getRootDof());
//   temp[2 * thid] = 0;
//   for (int i = thid * sitesPerThread;
//        (i < (thid + 1) * sitesPerThread) && (i < L.getRootDof()); i++) {
//     temp[2 * thid] += L.eval(path, i);
//   }
//   temp[2 * thid + 1] = 0;
//
//   __syncthreads();
//
//   int offset = 1;
//
//   for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
//   {
//     __syncthreads();
//     if (thid < d) {
//       int ai = offset * (2 * thid + 1) - 1;
//       int bi = offset * (2 * thid + 2) - 1;
//       temp[bi] += temp[ai];
//     }
//     offset *= 2;
//   }
//   __syncthreads();
//   if (thid == 0) {
//     *out =
//         pow(L.getLatSpacing(), L.getRootDim()) * temp[2 * CUDA_BLOCK_SIZE -
//         1];
//   }
// }

template <typename T, class Lag>
__global__ void kernel_metroStep(curandState *state, Lag L, T *oldPath,
                                 T *newPath, int offset, T varRange,
                                 int multiProbe) {
  int idx = (threadIdx.x + blockDim.x * blockIdx.x);
  int site = (2 * idx);
  site += ((site / L.getLatSites()) + offset) % 2;

  if (site >= L.getRootDof()) {
    return;
  }

  for (int i = 0; i < L.getTargetDim(); i++) {
    int pathIndex = site + (i * L.getRootDof());
    for (int j = 0; j < multiProbe; j++) {
      newPath[pathIndex] =
          oldPath[pathIndex] +
          (2 * varRange * (device_getRand<T>(&(state[idx])) - 0.5));

      T newActCorr = device_evalActionChange(L, newPath, site);
      T oldActCorr = device_evalActionChange(L, oldPath, site);

      T actChange = newActCorr - oldActCorr;

      if (actChange < 0 || exp(-actChange) > device_getRand<T>(&(state[idx]))) {
        oldPath[pathIndex] = newPath[pathIndex];
      }
    }
  }
}

template <typename T>
__global__ void kernel_randPath(curandState *state, T *path, T bias, T range,
                                int rDof, int dim) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (!((2 * idx + 1) >= rDof)) {
    return;
  }

  for (int i = 0; i < dim; i++) {
    path[(2 * idx) + (i * rDof)] =
        bias + 2 * (device_getRand<T>(&(state[idx])) - 0.5) * range;
    path[(2 * idx) + 1 + (i * rDof)] =
        bias + 2 * (device_getRand<T>(&(state[idx])) - 0.5) * range;
  }
}

double getRand() {
  return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

template <typename T, class Lag> class metropolizer {
public:
  metropolizer(Lag iL, T iVarRange, int iMultiProbe, int iSeed = 42) : L(iL) {
    varRange = iVarRange;
    multiProbe = iMultiProbe;
    threadCount = CUDA_BLOCK_SIZE;
    blockCount = getBlockCount(L.getRootDof() / 2);

    cudaMallocManaged(&rand_state, sizeof(curandState) * (L.getRootDof() / 2));
    kernel_initIteration<<<blockCount, threadCount>>>(rand_state,
                                                      (L.getRootDof() / 2));
    cudaDeviceSynchronize();
  }

  ~metropolizer() { cudaFree(rand_state); }

  void makeMetroStep(T *path) {

    T *newPath;
    cudaMallocManaged(&newPath, L.getRootDof() * L.getTargetDim() * sizeof(T));

    for (int i = 0; i < 2; i++) {
      memcpy(newPath, path, L.getRootDof() * L.getTargetDim() * sizeof(T));

      kernel_metroStep<T, Lag>
          <<<getBlockCount(L.getRootDof() / 2), CUDA_BLOCK_SIZE>>>(
              rand_state, L, path, newPath, i, varRange, multiProbe);
      cudaDeviceSynchronize();
    }

    cudaFree(newPath);
  }
  
  void getRandomPath(T *path, T range = 0, T bias = 0) {
    if (range == 0) {
      range = varRange;
    }
    kernel_randPath<<<getBlockCount(L.getRootDof() / 2), CUDA_BLOCK_SIZE>>>(
        rand_state, path, bias, range, L.getRootDof(), L.getTargetDim());
    cudaDeviceSynchronize();
  }

  __host__ __device__ T getVarRange() { return varRange; }
  __host__ __device__ int getMultiProbe() { return multiProbe; }
  __host__ __device__ Lag getLagrangian() { return L; }

private:
  T varRange;
  int multiProbe;
  Lag L;
  int blockCount;
  int threadCount;
  int seed;
  curandState *rand_state;
};

// template <typename T>
// __global__ void iterate(metropolizer<T> *metro, T *oldPath, T *newPath,
//                         T oldActVal, int offset);

#endif /*METROPOLIZER_HPP*/
