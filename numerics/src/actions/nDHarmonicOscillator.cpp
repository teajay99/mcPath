#include "nDHarmonicOscillator.hpp"

template <typename T>
nDHarmonicOscillator<T>::nDHarmonicOscillator(T iLatSpacing, int iLatSites,
                                              int iDim) {
  latSpacing = iLatSpacing;
  latSites = iLatSites;
  dim = iDim;
  dof = dim * latSites;
}

template <typename T>
nDHarmonicOscillator<T>::nDHarmonicOscillator(T iLatSpacing, int iLatSites) {
  nDHarmonicOscillator(iLatSpacing, iLatSites, 1);
}

template <typename T> nDHarmonicOscillator<T>::~nDHarmonicOscillator() {}

template <typename T> T nDHarmonicOscillator<T>::eval(T *path) {
  T out = 0;
  for (int i = 0; i < dof; i++) {
    T v = (path[(i + 1) % dof] - path[i]) / latSpacing;
    out += (0.5 * v * v) + (0.5 * path[i] * path[i]);
  }
  return latSpacing * out;
}

template <typename T> int nDHarmonicOscillator<T>::getDOF() { return dof; }
