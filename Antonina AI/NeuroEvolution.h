#pragma once
#include "Perceptron.h"
#include <fstream>
#include <string>

class NeuroEvolution {
private:
  double learningRate;
  int population;
  int length;
  int parents_size;
  int *sizes;
  Perceptron *neuros;
  int *fitness;

  int best_fitness_ever;
  int generations_without_improvement;
  double current_epsilon;
  double current_mutation_prob;

  template <typename T> T random_in_range(T a, T b);

  int partition(int low, int high);
  void quickSort(int low, int high);
  bool allEqual();
  void deinit();

public:
  NeuroEvolution();
  NeuroEvolution(double learningRate_, int length_, int *sizes_,
                 int parents_size_, int population_);
  ~NeuroEvolution();

  void evolution();

  void setFitness(int *fitness_);
  void clearFitness();
  int *getFitness();

  int getPopulation();
  Perceptron *getNeuros(int i);
  Perceptron *demonstrate();

  void resetBestFitness(double new_epsilon = -1.0,
                        double new_mutation_prob = -1.0);

  int getBestEver() const { return best_fitness_ever; }
  int getStagnation() const { return generations_without_improvement; }
  double getEpsilon() const { return current_epsilon; }
  double getMutationProb() const { return current_mutation_prob; }

  void readFromFile(std::string file);
  void writeInFile(std::string file);
  void readFromFile(std::ifstream *fin);
  void writeInFile(std::ofstream *fout);
};
