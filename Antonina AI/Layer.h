#pragma once
#include <fstream>
#include <string>

class Layer {
public:
  int size;
  int nextSize;
  double *neurons;
  double *biases;
  double *weights;

  Layer();
  Layer(int _size, int _nextSize);
  Layer(const Layer &l);
  Layer &operator=(const Layer &l);
  Layer(Layer &&l);
  Layer &operator=(Layer &&l);
  void readFromFile(std::ifstream *fin);
  void writeInFile(std::ofstream *fout);
  void deInit();
  ~Layer();
};
