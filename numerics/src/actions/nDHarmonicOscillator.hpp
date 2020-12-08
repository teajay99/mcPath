#ifndef NDHARMONICOSCILLATOR_H
#define NDHARMONICOSCILLATOR_H

#include "action.hpp"

template <typename T>
class nDHarmonicOscillator : public action<T> {
public:
  nDHarmonicOscillator(T latSpacing, int latSites, int dim);
  nDHarmonicOscillator(T latSpacing, int latSites);
  virtual ~nDHarmonicOscillator();


  T eval(T* path);
  int getDOF();

private:
  T latSpacing;
  int latSites;
  int dim;
  int dof;
};

#endif /*NDHARMONICOSCILLATOR_H*/
