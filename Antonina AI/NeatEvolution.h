#pragma once

#include "Brain.h"
#include <iosfwd>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

struct NeatNodeGene {
  int id = 0;
  int type = 0;
  double level = 0.0;
  double value = 0.0;
};

struct NeatConnectionGene {
  int in = 0;
  int out = 0;
  int innovation = 0;
  double weight = 0.0;
  bool enabled = true;
  int in_index = -1;
  int out_index = -1;
};

class NeatInnovation {
public:
  explicit NeatInnovation(int next_node_id = 0);
  int connectionInnovation(int in, int out);
  int nodeInnovation(int in, int out);
  void reset(int next_node_id = 0);
  void observeNodeId(int id);
  void observeConnection(int in, int out, int innovation);

private:
  int next_node_id_;
  int next_innovation_;
  std::map<std::pair<int, int>, int> connections_;
  std::map<std::pair<int, int>, int> split_nodes_;
};

class NeatGenome : public Brain {
public:
  NeatGenome();
  static NeatGenome minimal(int input_count, int output_count,
                            NeatInnovation &innovation, std::mt19937 &rng);

  void feedForward(double *inputs) override;
  int getOut() override;
  std::unique_ptr<Brain> cloneBrain() const override;
  void feedForwardBatch(const double *batch_inputs, double *batch_outputs,
                        int B) const;
  void getOutBatch(const double *batch_outputs, int *out_args, int B) const;

  void mutateWeights(double sigma, double prob, std::mt19937 &rng);
  bool mutateAddInputConnection(NeatInnovation &innovation, std::mt19937 &rng);
  bool mutateAddConnection(NeatInnovation &innovation, std::mt19937 &rng);
  bool mutateAddNode(NeatInnovation &innovation, std::mt19937 &rng);

  static NeatGenome crossover(const NeatGenome &a, const NeatGenome &b,
                              std::mt19937 &rng);
  static double compatibility(const NeatGenome &a, const NeatGenome &b);

  int inputCount() const { return input_count_; }
  int outputCount() const { return output_count_; }
  int getInputSize() const { return input_count_; }
  int getOutputSize() const { return output_count_; }
  int nodeCount() const { return (int)nodes_.size(); }
  int connectionCount() const { return (int)connections_.size(); }
  int enabledConnectionCount() const {
    int count = 0;
    for (const auto &conn : connections_)
      if (conn.enabled)
        ++count;
    return count;
  }
  const std::vector<NeatNodeGene> &nodes() const { return nodes_; }
  const std::vector<NeatConnectionGene> &connections() const {
    return connections_;
  }
  double fitness() const { return fitness_; }
  void setFitness(double value) { fitness_ = value; }
  void writeInFile(std::ofstream &fout) const;
  bool readInFile(std::ifstream &fin);
  static int initialSeedInputCount(int input_count);
  static std::vector<int> initialSeedInputIds(int input_count);

private:
  int input_count_;
  int output_count_;
  int bias_id_;
  int bias_index_;
  double fitness_;
  std::vector<NeatNodeGene> nodes_;
  std::vector<NeatConnectionGene> connections_;
  std::vector<int> innovation_order_;
  std::vector<int> output_indices_;
  std::vector<double> outputs_;

  int nodeType(int id) const;
  double nodeLevel(int id) const;
  bool hasConnection(int in, int out) const;
  void sortGenes();
  static double activate(double x);
};

class NeatEvolution {
public:
  NeatEvolution(int input_count, int output_count, int population);

  int getPopulation() const { return (int)population_.size(); }
  NeatGenome *getGenome(int index) { return &population_[index]; }
  const NeatGenome *getGenome(int index) const { return &population_[index]; }
  void setFitness(const int *fitness);
  void evolution();
  int getGeneration() const { return generation_; }
  int getBestFitness() const { return best_fitness_; }
  int getStagnation() const { return stagnation_; }
  int getSpeciesCount() const { return species_count_; }
  double getMutationSigma() const { return mutation_sigma_; }
  double getMutationProb() const { return mutation_prob_; }
  double getCompatibilityThreshold() const { return compatibility_threshold_; }
  double getAvgNodeCount() const;
  double getAvgConnectionCount() const;
  void resetBestFitness();
  bool writeInFile(const std::string &file, int active_tests = -1) const;
  bool readInFile(const std::string &file, int *active_tests = nullptr);
  bool writeBestInFile(const std::string &file) const;

private:
  struct Species {
    int id = 0;
    NeatGenome representative;
    std::vector<int> members;
    double adjusted_sum = 0.0;
    double best_fitness = -1.0e300;
    double mean_fitness = 0.0;
    int stale = 0;
    int spawn = 0;
  };

  int input_count_;
  int output_count_;
  int generation_;
  int best_fitness_;
  int stagnation_;
  int species_count_;
  double compatibility_threshold_;
  double mutation_sigma_;
  double mutation_prob_;
  int next_species_id_;
  NeatInnovation innovation_;
  std::mutex innovation_mutex_;
  std::mt19937 rng_;
  std::vector<NeatGenome> population_;
  std::vector<Species> species_;

  void speciate();
  void updateSpeciesStats();
  void removeStaleSpecies();
  void assignSpawnCounts();
  NeatGenome makeChild(const Species &species,
                       const std::vector<int> &global_parents,
                       std::mt19937 &rng);
  int tournamentPick(const std::vector<int> &members, std::mt19937 &rng) const;
  int bestIndex() const;
};
