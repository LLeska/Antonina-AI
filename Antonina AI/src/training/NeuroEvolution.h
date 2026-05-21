#pragma once
#include "AITrainingAlgorithm.h"
#include "TrainingSettings.h"
#include <iosfwd>
#include <memory>
#include <random>
#include <string>
#include <type_traits>

class LiveStats;
class Perceptron;

class NeuroEvolution final : public AITrainingAlgorithm {
private:
  double learningRate;
  int population;
  int length;
  int parents_size;
  int *sizes;
  Perceptron *neuros;
  int *fitness;

  int best_fitness_ever;
  int generation;
  int generations_without_improvement;
  double current_epsilon;
  double current_mutation_prob;
  TrainingSettings training_settings_;
  LiveStats *live_ = nullptr;
  std::string load_file_;

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

  void init(double learningRate_, int length_, const int *sizes_, int parents_size_, int population_);
  int partition(int low, int high);
  void quickSort(int low, int high);
  bool allEqual();
  void deinit();

public:
  NeuroEvolution();
  NeuroEvolution(double learningRate_, int length_, int *sizes_, int parents_size_, int population_);
  NeuroEvolution(const TrainingSettings &settings, LiveStats *live, std::string load_file);
  ~NeuroEvolution() override;

  int train() override;
  std::unique_ptr<Brain> getBestBrain() const override;
  bool save(const std::string &file) const override;
  bool load(const std::string &file) override;
  void evolution();

  void setFitness(int *fitness_);
  void clearFitness();
  int *getFitness();

  int getPopulation();
  Perceptron *getNeuros(int i);
  Perceptron *demonstrate();

  void resetBestFitness(double new_epsilon = -1.0, double new_mutation_prob = -1.0);

  int getGeneration() const override { return generation; }
  int getBestFitness() const override { return best_fitness_ever; }
  int getBestEver() const { return best_fitness_ever; }
  int getStagnation() const { return generations_without_improvement; }
  double getEpsilon() const { return current_epsilon; }
  double getMutationProb() const { return current_mutation_prob; }
  void configureMutation(double epsilon, double mutation_prob);

  void readFromFile(std::string file);
  void writeInFile(std::string file);
  void readFromFile(std::ifstream *fin);
  void writeInFile(std::ofstream *fout);
};
