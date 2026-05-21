#pragma once
#include "Brain.h"
#include "Layer.h"
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <type_traits>

class Perceptron : public Brain {
private:
  double learningRate;
  Layer *layers;
  int length;
  double EPSILON;
  double NOT_MUTAHION;

  template <typename T> T random_in_range(T a, T b) {
    static thread_local std::mt19937 gen(std::random_device{}());
    if constexpr (std::is_integral_v<T>) {
      std::uniform_int_distribution<T> dist(a, b);
      return dist(gen);
    } else {
      std::uniform_real_distribution<T> dist(a, b);
      return dist(gen);
    }
  }

public:
  Perceptron();
  Perceptron(double learningRate_, int length_, int *sizes);
  Perceptron(const Perceptron &p);
  Perceptron(Perceptron &&p);
  Perceptron(Perceptron *p1, Perceptron *p2);
  Perceptron &operator=(const Perceptron &p);
  Perceptron &operator=(Perceptron &&p);
  ~Perceptron();

  void feedForward(double *inputs) override;

  void feedForwardBatch(const double *batch_inputs, double *batch_outputs, int B) const;

  int getOut() override;
  double outputValue(int index) const override;
  std::unique_ptr<Brain> cloneBrain() const override;

  void getOutBatch(const double *batch_outputs, int *out_args, int B) const;

  void mutate(double sigma, double prob);
  bool isInitialized();
  void deInit();

  int getInputSize() const { return layers ? layers[0].size : 0; }
  int getOutputSize() const { return layers ? layers[length - 1].size : 0; }
  int layerCount() const { return layers ? length : 0; }
  int layerSize(int index) const;
  double neuronValue(int layer, int index) const;
  double connectionWeight(int from_layer, int from, int to) const;

  void readFromFile(std::ifstream *fin);
  void writeInFile(std::ofstream *fout);
  void readFromFile(std::string file);
  void writeInFile(std::string file);
};
