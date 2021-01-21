#ifndef NDHARMONICOSCILLATOR_H
#define NDHARMONICOSCILLATOR_H

#include "lagrangian.hpp"

template <typename T> class nDHarmonicOscillator : public lagrangian<T, 1> {
public:
  __host__ __device__ nDHarmonicOscillator(T lSpacing, int lSites, int tDim = 1,
                                           T iOmega = 1.0, T m = 1.0)
      : lagrangian<T, 1>(tDim, lSites, lSpacing) {
    mass = m;
    omega = iOmega;
  }
  __host__ __device__ ~nDHarmonicOscillator() {}

  __host__ __device__ T eval(T *path, int loc) {
    T out = 0;
    for (int i = 0; i < this->getTargetDim(); i++) {
      int here = loc + (i * this->getRootDof());
      int next = ((loc + 1) % this->getRootDof()) + (i * this->getRootDof());
      T x = path[here];
      T v = (path[next] - path[here]) / this->getLatSpacing();
      out += (v * v * mass) + (omega * omega * x * x * mass);
    }
    return 0.5 * out;
  }

  __host__ __device__ T opEvalHamiltonian(T *path, int loc) {
    T out = 0;
    for (int i = 0; i < this->getTargetDim(); i++) {
      int here = loc + (i * this->getRootDof());
      int next = ((loc + 1) % this->getRootDof()) + (i * this->getRootDof());
      int last = ((loc - 1 + this->getRootDof()) % this->getRootDof()) +
                 (i * this->getRootDof());
      out += -((path[next] - path[here]) * (path[here] - path[last]) * mass) /
             (this->getLatSpacing() * this->getLatSpacing());
      out += (omega * omega * path[here] * path[here] * mass);
    }
    return out * 0.5;
  }

  __host__ __device__ T opGroundLevelEnergy(T *path, int loc) {}

  template <int dim = 0, int power = 1>
  __host__ __device__ T opEvalPosition(T *path, int loc) {
    return pow(path[loc + (dim * this->getRootDof())], power);
  }

  template <int dim = 0, int power = 1>
  __host__ __device__ T opEvalMomentum(T *path, int loc) {
    return 0;
  }

  template <int power = 1>
  __host__ __device__ T opEvalPositionNormSquared(T *path, int loc) {
    return 0;
  }

  template <int power = 1>
  __host__ __device__ T opEvalMomentumNormSquared(T *path, int loc) {
    return 0;
  }

private:
  T mass;
  T omega;
};

#endif /*NDHARMONICOSCILLATOR_H*/
