#pragma once

#include <string>
#include <vector>

inline constexpr const char *ALGORITHM_NEAT = "neat";
inline constexpr const char *ALGORITHM_CLASSIC = "classic";

struct TrainingSettings {
  std::string algorithm = ALGORITHM_NEAT;
  int population = 1000;
  int requested_threads = 0;
  int max_generations = 1000001;
  int max_population = 0;
  bool adaptive_population = true;
  int initial_active_tests = 1;
  int failure_log_interval = 25;
  int failure_trace_detail_count = 1;
  int population_stall_generations = 200;
  int population_stable_failure_checks = 4;
  int population_change_cooldown = 200;
  bool test_weighting = true;
  double test_weight_max = 1.5;
  double test_weight_step = 0.1;
  int test_weight_max_tests = 64;
  bool neat_hard_test_archive = true;
  int neat_hard_test_archive_size = 16;
  bool neat_input_prior = true;
  int neat_input_prior_max = 32;
  double neat_input_prior_bias = 0.65;
  double neat_input_prior_reach = 0.05;
  bool neat_epsilon_lexicase = true;
  int neat_lexicase_max_cases = 256;
  int neat_lexicase_cases_per_parent = 64;
  double neat_lexicase_epsilon_fraction = 0.02;
  double neat_lexicase_parent_rate = 0.85;
  int neat_failure_memory = 64;
  int neat_focus_radius = 4;
  int neat_focus_max_cases = 32;
  int neat_bridge_children_per_case = 12;
  int neat_bridge_graft_links = 8;
  double neat_mutation_sigma = 0.25;
  double neat_mutation_prob = 0.8;
  double neat_compatibility_threshold = 3.0;
  double neat_compatibility_weight = 1.0;
  int neat_target_species_min = 12;
  int neat_target_species_max = 48;
  double neat_survival_rate = 0.2;
  double neat_complexity_penalty = 50.0;
  double neat_add_node_prob = 0.03;
  double neat_add_connection_prob = 0.08;
  double neat_sparse_connection_prob = 0.30;
  double neat_input_probe_prob = 0.12;
  double neat_sparse_input_probe_prob = 0.45;
  double classic_learning_rate = 0.01;
  int classic_parents = 100;
  int classic_hidden_layers = 2;
  std::vector<int> classic_hidden_sizes = {128, 32};
  double classic_mutation_sigma = 0.35;
  double classic_mutation_prob = 0.08;
};

bool isNeatAlgorithm(const std::string &algorithm);
bool isClassicAlgorithm(const std::string &algorithm);
const char *trainingAlgorithmName(const std::string &algorithm);
int defaultClassicHiddenSize(int index);
void resizeClassicHiddenSizes(TrainingSettings &settings);
void setClassicHiddenSize(TrainingSettings &settings, int index, int size);
void normalizeSettings(TrainingSettings &settings);
bool parseAlgorithm(const std::string &value, std::string &algorithm);
