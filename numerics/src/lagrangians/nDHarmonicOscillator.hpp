#ifndef NDHARMONICOSCILLATOR_H
#define NDHARMONICOSCILLATOR_H

#include "lagrangian.hpp"

template <typename T> class nDHarmonicOscillator : public lagrangian<T, 1> {
public:
  __host__ __device__ nDHarmonicOscillator(T lSpacing, int lSites, int tDim = 1,
                                           T m = 1.0)
      : lagrangian<T, 1>(tDim, lSites, lSpacing) {
    mass = m;
  }
  __host__ __device__ ~nDHarmonicOscillator() {}

  // T evalLagrangian(T *path, int *loc){}
  __host__ __device__ T eval(T *x, T *v) {
    T out = 0;
    for (int i = 0; i < this->getTargetDim(); i++) {
      out += (x[i] * x[i]) + (v[i] * v[i]);
    }
    out *= mass / 2.0;
    return out;
    return 0;
  }

  __host__ __device__ T evalPath(T *path, int loc) {
    T out = 0;
    for (int i = 0; i < this->getTargetDim(); i++) {
      int here = loc + (i * this->getRootDof());
      int next = ((loc + this->getRootBasis(i)) % this->getRootDof()) +
                 (i * this->getRootDof());
      T x = path[here];
      T v = (path[next] - path[here]) / this->getLatSpacing();
      out += x * x + v * v;
    }
    return 0.5 * mass * out;
  }

private:
  T mass;
};

#endif /*NDHARMONICOSCILLATOR_H*/
