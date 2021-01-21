#include "errors.hpp"
#include "lagrangian.hpp"
#include "metropolizer.hpp"
#include "nDHarmonicOscillator.hpp"
#include "numtest.hpp"
#include "opEvaluator.hpp"

#include <cmath>
#include <iostream>

void harmosci_test() {
  double time = 100.0;
  int N = 2048;
  double a = time / N;
  int dim = 1;
  double omega = 1;
  double mass = 1;

  nDHarmonicOscillator<double> osc(a, N, dim, omega, mass);

  metropolizer<double, nDHarmonicOscillator<double>> metro(osc, 2 * sqrt(a),
                                                           10);

  opEvaluator<double, nDHarmonicOscillator<double>,
              &nDHarmonicOscillator<double>::opEvalPosition<0, 2>>
      x2(osc);

  opEvaluator<double, nDHarmonicOscillator<double>,
              &nDHarmonicOscillator<double>::opEvalHamiltonian>
      ham(osc);

  double *path;
  cudaMallocManaged(&path, dim * N * sizeof(double));
  metro.getRandomPath(path, 1 * sqrt(a), 0);

  // for (int i = 0; i < N; i++) {
  //   path[i] = 0;
  // }

  for (int j = 0; j < 10000; j++) {
    metro.makeMetroStep(path);
  }

  for (int i = 0; i < 10000; i++) {
    for (int j = 0; j < 50; j++) {
      metro.makeMetroStep(path);
    }

    ham.evalPath(path);
    x2.evalPath(path);

    // double E = 0;
    // for (int i = 0; i < N; i++) {
    //    E += path[i] * path[i];
    // }
    // E = E / N;
    // std::cout << E << std::endl;

     std::cout << "Energy is: " << omega * omega * x2.getMean()
               << "   Hamiltonian is: " << ham.getMean() << std::endl;
  }

  // for(int i = 0; i<N; i++){
  //   std::cout << path[i] << std::endl;
  // }

  cudaFree(path);
}
