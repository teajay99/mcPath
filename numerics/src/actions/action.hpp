#ifndef ACTION_HPP
#define ACTION_HPP

template <typename T>
class action {
public:
  virtual T eval(T* path) = 0;
  virtual int getDOF();
};

#endif /*ACTION_HPP*/
