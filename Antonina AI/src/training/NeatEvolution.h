#pragma once

#include "AITrainingAlgorithm.h"
#include "NeatGenome.h"
#include "NeatInnovation.h"
#include "TrainingSettings.h"

#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

class LiveStats;

struct NeatLexicaseSelection {
  const int *case_scores = nullptr;
  int population = 0;
  int case_stride = 0;
  const int *case_indices = nullptr;
  const int *case_epsilons = nullptr;
  int case_count = 0;
  int cases_per_parent = 0;
  double parent_rate = 0.0;

  bool valid() const { return case_scores && case_indices && case_epsilons && population > 0 && case_stride > 0 && case_count > 0 && parent_rate > 0.0; }
};

class NeatEvolution final : public AITrainingAlgorithm {
public:
  NeatEvolution(int input_count, int output_count, int population);
  NeatEvolution(const TrainingSettings &settings, LiveStats *live, std::string load_file);

  int train() override;
  std::unique_ptr<Brain> getBestBrain() const override;
  bool save(const std::string &file) const override;
  bool load(const std::string &file) override;
  int getPopulation() const { return (int)population_.size(); }
  NeatGenome *getGenome(int index) { return &population_[index]; }
  const NeatGenome *getGenome(int index) const { return &population_[index]; }
  void reservePopulation(int capacity);
  void setFitness(const int *fitness);
  void evolution(int target_population = -1, const std::vector<NeatGenome> *protected_elites = nullptr, const NeatLexicaseSelection *lexicase = nullptr, const std::vector<std::pair<int, int>> *bridge_mates = nullptr, int bridge_graft_links = 0);
  int getGeneration() const override { return generation_; }
  int getBestFitness() const override { return best_fitness_; }
  int getStagnation() const { return stagnation_; }
  int getSpeciesCount() const { return species_count_; }
  double getMutationSigma() const { return mutation_sigma_; }
  double getMutationProb() const { return mutation_prob_; }
  double getCompatibilityThreshold() const { return compatibility_threshold_; }
  void configure(double mutation_sigma, double mutation_prob, double compatibility_threshold, double survival_rate, double add_node_prob, double add_connection_prob, double sparse_connection_prob, double input_probe_prob, double sparse_input_probe_prob, double compatibility_weight_coefficient, int target_species_min, int target_species_max);
  void setInputMutationPrior(const std::vector<int> &input_ids, double bias);
  double getAvgNodeCount() const;
  double getAvgConnectionCount() const;
  double getAvgActiveConnectionCount() const;
  void resetBestFitness();
  bool writeInFile(const std::string &file, int active_tests = -1) const;
  bool readInFile(const std::string &file, int *active_tests = nullptr);
  bool writeGenomeInFile(int index, const std::string &file, int reported_fitness) const;

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
  TrainingSettings training_settings_;
  LiveStats *live_ = nullptr;
  std::string load_file_;

  void speciate();
  void updateSpeciesStats();
  void removeStaleSpecies();
  void assignSpawnCounts(int target_population);
  NeatGenome makeChild(const Species &species, const std::vector<int> &global_parents, std::mt19937 &rng, const NeatLexicaseSelection *lexicase);
  void mutateChild(NeatGenome &child, std::mt19937 &rng);
  int pickParent(const std::vector<int> &members, std::mt19937 &rng, const NeatLexicaseSelection *lexicase) const;
  int lexicasePick(const std::vector<int> &members, std::mt19937 &rng, const NeatLexicaseSelection &lexicase) const;
  int tournamentPick(const std::vector<int> &members, std::mt19937 &rng) const;
  int bestIndex() const;
};
