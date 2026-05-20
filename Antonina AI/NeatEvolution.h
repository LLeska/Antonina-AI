#pragma once

#include "Brain.h"
#include <iosfwd>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
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

struct NeatExecutionPlan {
  const int *conn_in_indices = nullptr;
  const int *conn_out_indices = nullptr;
  const double *conn_weights = nullptr;
  int connection_count = 0;

  const int *source_indices = nullptr;
  const int *source_offsets = nullptr;
  int source_count = 0;

  const int *non_input_indices = nullptr;
  int non_input_count = 0;

  const int *used_input_indices = nullptr;
  int used_input_count = 0;

  const int *output_indices = nullptr;
  int node_count = 0;
  int input_count = 0;
  int output_count = 0;
};

struct NeatLexicaseSelection {
  const int *case_scores = nullptr;
  int population = 0;
  int case_stride = 0;
  const int *case_indices = nullptr;
  const int *case_epsilons = nullptr;
  int case_count = 0;
  int cases_per_parent = 0;
  double parent_rate = 0.0;

  bool valid() const {
    return case_scores && case_indices && case_epsilons && population > 0 &&
           case_stride > 0 && case_count > 0 && parent_rate > 0.0;
  }
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
  double outputValue(int index) const override;
  std::unique_ptr<Brain> cloneBrain() const override;
  void feedForwardBatch(const double *batch_inputs, double *batch_outputs,
                        int B) const;
  void feedForwardBatchRows(const double *const *batch_inputs,
                            double *batch_outputs, int B) const;
  void getOutBatch(const double *batch_outputs, int *out_args, int B) const;

  void mutateWeights(double sigma, double prob, std::mt19937 &rng);
  bool mutateAddInputConnection(
      NeatInnovation &innovation, std::mt19937 &rng,
      const std::vector<int> *preferred_inputs = nullptr);
  bool mutateAddConnection(NeatInnovation &innovation, std::mt19937 &rng);
  bool mutateAddNode(NeatInnovation &innovation, std::mt19937 &rng);

  static NeatGenome crossover(const NeatGenome &a, const NeatGenome &b,
                              std::mt19937 &rng);
  static NeatGenome crossoverGraft(const NeatGenome &base,
                                   const NeatGenome &donor,
                                   int max_extra_connections,
                                   std::mt19937 &rng);
  static double compatibility(const NeatGenome &a, const NeatGenome &b,
                              double weight_coefficient = 0.4);

  int inputCount() const { return input_count_; }
  int outputCount() const { return output_count_; }
  int getInputSize() const { return input_count_; }
  int getOutputSize() const { return output_count_; }
  int nodeCount() const { return (int)nodes_.size(); }
  int connectionCount() const { return (int)connections_.size(); }
  int activeConnectionCount() const { return (int)active_conn_weights_.size(); }
  int activeSourceCount() const { return (int)active_source_indices_.size(); }
  int activeNonInputNodeCount() const {
    return (int)active_non_input_indices_.size();
  }
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
  NeatExecutionPlan executionPlan() const;
  bool structurallyValid() const;
  bool disableInputConnections(int input_id);
  double fitness() const { return fitness_; }
  void setFitness(double value) { fitness_ = value; }
  void writeInFile(std::ofstream &fout) const;
  bool readInFile(std::ifstream &fin);
  bool upgradeInputCount(int new_input_count);
  static int initialSeedInputCount(int input_count);
  static std::vector<int> initialSeedInputIds(int input_count);

private:
  bool repairLoadedStructure();

  int input_count_;
  int output_count_;
  int bias_id_;
  int bias_index_;
  double fitness_;
  std::vector<NeatNodeGene> nodes_;
  std::vector<NeatConnectionGene> connections_;
  std::vector<int> active_connection_indices_;
  std::vector<int> active_conn_in_indices_;
  std::vector<int> active_conn_out_indices_;
  std::vector<double> active_conn_weights_;
  std::vector<int> active_source_indices_;
  std::vector<int> active_source_offsets_;
  std::vector<int> active_non_input_indices_;
  std::vector<int> used_input_indices_;
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
  void reservePopulation(int capacity);
  void setFitness(const int *fitness);
  void evolution(int target_population = -1,
                 const std::vector<NeatGenome> *protected_elites = nullptr,
                 const NeatLexicaseSelection *lexicase = nullptr,
                 const std::vector<std::pair<int, int>> *bridge_mates =
                     nullptr,
                 int bridge_graft_links = 0);
  int getGeneration() const { return generation_; }
  int getBestFitness() const { return best_fitness_; }
  int getStagnation() const { return stagnation_; }
  int getSpeciesCount() const { return species_count_; }
  double getMutationSigma() const { return mutation_sigma_; }
  double getMutationProb() const { return mutation_prob_; }
  double getCompatibilityThreshold() const { return compatibility_threshold_; }
  void configure(double mutation_sigma, double mutation_prob,
                 double compatibility_threshold, double survival_rate,
                 double add_node_prob, double add_connection_prob,
                 double sparse_connection_prob, double input_probe_prob,
                 double sparse_input_probe_prob,
                 double compatibility_weight_coefficient,
                 int target_species_min, int target_species_max);
  void setInputMutationPrior(const std::vector<int> &input_ids, double bias);
  double getAvgNodeCount() const;
  double getAvgConnectionCount() const;
  double getAvgActiveConnectionCount() const;
  void resetBestFitness();
  bool writeInFile(const std::string &file, int active_tests = -1) const;
  bool readInFile(const std::string &file, int *active_tests = nullptr);
  bool writeBestInFile(const std::string &file) const;
  bool writeGenomeInFile(int index, const std::string &file,
                         int reported_fitness) const;

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
  double compatibility_weight_coefficient_;
  double mutation_sigma_;
  double mutation_prob_;
  double survival_rate_;
  double mutate_only_prob_;
  double interspecies_mate_prob_;
  double add_connection_prob_;
  double add_node_prob_;
  double input_probe_prob_;
  double sparse_input_probe_prob_;
  double sparse_connection_prob_;
  std::vector<int> input_mutation_prior_;
  double input_mutation_prior_bias_;
  int target_species_min_;
  int target_species_max_;
  int next_species_id_;
  NeatInnovation innovation_;
  std::mutex innovation_mutex_;
  std::mt19937 rng_;
  std::vector<NeatGenome> population_;
  std::vector<Species> species_;

  void speciate();
  void updateSpeciesStats();
  void removeStaleSpecies();
  void assignSpawnCounts(int target_population);
  NeatGenome makeChild(const Species &species,
                       const std::vector<int> &global_parents,
                       std::mt19937 &rng,
                       const NeatLexicaseSelection *lexicase);
  void mutateChild(NeatGenome &child, std::mt19937 &rng);
  int pickParent(const std::vector<int> &members, std::mt19937 &rng,
                 const NeatLexicaseSelection *lexicase) const;
  int lexicasePick(const std::vector<int> &members, std::mt19937 &rng,
                   const NeatLexicaseSelection &lexicase) const;
  int tournamentPick(const std::vector<int> &members, std::mt19937 &rng) const;
  int bestIndex() const;
};
