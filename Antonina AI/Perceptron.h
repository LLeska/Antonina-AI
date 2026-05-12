#pragma once
#include "Brain.h"
#include "Layer.h"
#include <fstream>
#include <memory>
#include <string>

class Perceptron : public Brain {
private:
  double learningRate;
  Layer *layers;
  int length;
  double EPSILON;
  double NOT_MUTAHION;

  template <typename T> T random_in_range(T a, T b);

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

  void feedForwardBatch(const double *batch_inputs, double *batch_outputs,
                        int B) const;

  int getOut() override;
  std::unique_ptr<Brain> cloneBrain() const override;

  void getOutBatch(const double *batch_outputs, int *out_args, int B) const;

  void mutate(double sigma, double prob);
  bool isInitialized();
  void deInit();

  int getInputSize() const { return layers ? layers[0].size : 0; }
  int getOutputSize() const { return layers ? layers[length - 1].size : 0; }

  void readFromFile(std::ifstream *fin);
  void writeInFile(std::ofstream *fout);
  void readFromFile(std::string file);
  void writeInFile(std::string file);
};
