#include "Layer.h"
#include <cassert>
#include <fstream>
#include <iostream>

Layer::Layer() {
  size = 0;
  nextSize = 0;
  neurons = nullptr;
  biases = nullptr;
  weights = nullptr;
}

Layer::Layer(int _size, int _nextSize) {
  assert(_size > 0 && _size < 10000);
  assert(_nextSize >= 0 && _nextSize < 10000);
  size = _size;
  nextSize = _nextSize;
  neurons = new double[size]();
  if (nextSize > 0) {
    biases = new double[nextSize];
    weights = new double[nextSize * size];
  } else {
    biases = nullptr;
    weights = nullptr;
  }
}

Layer::Layer(const Layer &l) {
  size = l.size;
  nextSize = l.nextSize;
  neurons = new double[size]();
  for (int i = 0; i < size; i++)
    neurons[i] = l.neurons[i];
  if (nextSize > 0) {
    biases = new double[nextSize];
    weights = new double[nextSize * size];
    for (int i = 0; i < nextSize; i++) {
      biases[i] = l.biases[i];
      for (int j = 0; j < size; j++)
        weights[i * size + j] = l.weights[i * size + j];
    }
  } else {
    biases = nullptr;
    weights = nullptr;
  }
}

Layer &Layer::operator=(const Layer &l) {
  if (this == &l)
    return *this;
  int new_size = l.size;
  int new_next = l.nextSize;
  assert(new_size > 0 && new_size < 10000);
  assert(new_next >= 0 && new_next < 10000);
  double *new_neurons = new double[new_size]();
  for (int i = 0; i < new_size; i++)
    new_neurons[i] = l.neurons[i];
  double *new_biases = nullptr;
  double *new_weights = nullptr;
  if (new_next > 0) {
    new_biases = new double[new_next];
    new_weights = new double[new_next * new_size];
    for (int i = 0; i < new_next; i++) {
      new_biases[i] = l.biases[i];
      for (int j = 0; j < new_size; j++)
        new_weights[i * new_size + j] = l.weights[i * new_size + j];
    }
  }
  deInit();
  size = new_size;
  nextSize = new_next;
  neurons = new_neurons;
  biases = new_biases;
  weights = new_weights;
  return *this;
}

Layer::Layer(Layer &&l) {
  size = l.size;
  nextSize = l.nextSize;
  neurons = l.neurons;
  biases = l.biases;
  weights = l.weights;
  l.size = 0;
  l.nextSize = 0;
  l.neurons = nullptr;
  l.biases = nullptr;
  l.weights = nullptr;
}

Layer &Layer::operator=(Layer &&l) {
  if (this != &l) {
    deInit();
    size = l.size;
    nextSize = l.nextSize;
    neurons = l.neurons;
    biases = l.biases;
    weights = l.weights;
    l.size = 0;
    l.nextSize = 0;
    l.neurons = nullptr;
    l.biases = nullptr;
    l.weights = nullptr;
  }
  return *this;
}

void Layer::readFromFile(std::ifstream *fin) {
  this->deInit();
  *fin >> size >> nextSize;
  if (fin->fail() || size <= 0 || size > 10000 || nextSize < 0 ||
      nextSize > 10000) {
    size = 0;
    nextSize = 0;
    return;
  }
  neurons = new double[size]();
  for (int i = 0; i < size; i++)
    *fin >> neurons[i];
  if (nextSize > 0) {
    biases = new double[nextSize];
    weights = new double[nextSize * size];
    for (int i = 0; i < nextSize; i++)
      *fin >> biases[i];
    for (int i = 0; i < nextSize; i++)
      for (int j = 0; j < size; j++)
        *fin >> weights[i * size + j];
  }
}

void Layer::writeInFile(std::ofstream *fout) {
  *fout << size << ' ' << nextSize << '\n';
  for (int i = 0; i < size; i++) {
    if (i > 0)
      *fout << ' ';
    *fout << neurons[i];
  }
  *fout << '\n';
  if (nextSize > 0) {
    for (int i = 0; i < nextSize; i++) {
      if (i > 0)
        *fout << ' ';
      *fout << biases[i];
    }
    *fout << '\n';
    for (int i = 0; i < nextSize; i++) {
      *fout << weights[i * size + 0];
      for (int j = 1; j < size; j++)
        *fout << ' ' << weights[i * size + j];
      *fout << '\n';
    }
  }
}

void Layer::deInit() {
  if (neurons != nullptr) {
    delete[] neurons;
    neurons = nullptr;
  }
  if (biases != nullptr) {
    delete[] biases;
    biases = nullptr;
  }
  if (weights != nullptr) {
    delete[] weights;
    weights = nullptr;
  }
  size = 0;
  nextSize = 0;
}

Layer::~Layer() { deInit(); }
