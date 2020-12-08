#include <random>

#IFNDEF METROPOLIZER_HPP
#DEFINE METROPOLIZER_HPP

template <typename T> class metropolizer {
public:
  metropolizer(action s);
  ~metropolizer();
  void performStep(T *path);

private:
  action<T> *s;
  std::random_device rd;
  std::uniform_real_distribution<> dist(0, 10);
};

#ENDIF /*METROPOLIZER_HPP*/
