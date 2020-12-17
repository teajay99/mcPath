#include "errors.hpp"
#include "lagrangian.hpp"
#include "metropolizer.hpp"
#include "nDHarmonicOscillator.hpp"
#include "numtest.hpp"

#include <cmath>
#include <iostream>

void harmosci_test() {
  double T_E = 3.141 * 4 / 20;
  int N = 2048;
  int dim = 1;

  float a = 20.0 / (N * T_E);

  nDHarmonicOscillator<double> osc(a, N, dim);

  metropolizer<double, nDHarmonicOscillator<double>> metro(osc, 2 * sqrt(a),
                                                           10);

  double *path;
  cudaMallocManaged(&path, dim * N * sizeof(double));
  metro.getRandomPath(path, 10 * sqrt(a), 5);

  for (int i = 0; i < 1; i++) {
    metro.makeMetroStep(path);
    // std::cout << i << std::endl;
  }

  cudaFree(path);
}
