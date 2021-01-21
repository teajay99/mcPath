#include "customMath.hpp"

#ifndef LAGRANGIAN_HPP
#define LAGRANGIAN_HPP

/*
f_0(x_0,x_1,x_2, ...) = path[x_0 + x_1*latSites^1 + x_2*latSites^2 + ...]
f_1(x_0,x_1,x_2, ...) = path[x_0 + x_1*latSites^1 + x_2*latSites^2 + ... +
rootDof] rootSpaceDof]

*/

template <typename T, int rootDim> class lagrangian {
public:
  __host__ __device__ lagrangian(int tDim, int lSites, T lSpacing) {
    latSites = lSites;
    latSpacing = lSpacing;
    rootDof = intPow(latSites, rootDim);
    targetDim = tDim;
    for (int i = 0; i < rootDim; i++) {
      rootBasis[i] = intPow(latSites, i);
    }
  }

  __host__ __device__ ~lagrangian() {}


  // Getter
  __host__ __device__ int getLatSites() { return latSites; }
  __host__ __device__ int getRootDim() { return rootDim; }
  __host__ __device__ int getRootDof() { return rootDof; }
  __host__ __device__ int getTargetDim() { return targetDim; }
  __host__ __device__ int getRootBasis(int i) { return rootBasis[i]; }
  __host__ __device__ T getLatSpacing() { return latSpacing; }

private:
  int latSites;  // Latice sites per root space dimension
  int rootDof;   // latSites^rootSpaceDim
  int targetDim; // Field Values per Lattice Site
  int rootBasis[rootDim];
  T latSpacing;
};

#endif /*LAGRANGIAN_HPP*/
