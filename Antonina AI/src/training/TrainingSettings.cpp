#include "TrainingSettings.h"

#include "AntoninaAPI.h"

#include <algorithm>
#include <cstddef>

bool isNeatAlgorithm(const std::string &algorithm) { return algorithm == ALGORITHM_NEAT || algorithm == "NEAT"; }

bool isClassicAlgorithm(const std::string &algorithm) { return algorithm == ALGORITHM_CLASSIC || algorithm == "evolution" || algorithm == "evo" || algorithm == "classic-evolution"; }

const char *trainingAlgorithmName(const std::string &algorithm) { return isNeatAlgorithm(algorithm) ? "NEAT" : "Classic evolution"; }

int defaultClassicHiddenSize(int index) { return index == 0 ? 128 : 32; }

void resizeClassicHiddenSizes(TrainingSettings &settings) {
  settings.classic_hidden_layers = std::max(0, settings.classic_hidden_layers);
  const size_t target = (size_t)settings.classic_hidden_layers;
  while (settings.classic_hidden_sizes.size() < target) {
    settings.classic_hidden_sizes.push_back(defaultClassicHiddenSize((int)settings.classic_hidden_sizes.size()));
  }
  if (settings.classic_hidden_sizes.size() > target)
    settings.classic_hidden_sizes.resize(target);
}

void setClassicHiddenSize(TrainingSettings &settings, int index, int size) {
  if (index < 0)
    return;
  if ((int)settings.classic_hidden_sizes.size() <= index)
    settings.classic_hidden_sizes.resize((size_t)index + 1, 32);
  settings.classic_hidden_sizes[(size_t)index] = size;
  settings.classic_hidden_layers = std::max(settings.classic_hidden_layers, index + 1);
}

void normalizeSettings(TrainingSettings &settings) {
  settings.population = std::max(1, settings.population);
  settings.requested_threads = std::max(0, settings.requested_threads);
  settings.max_generations = std::max(1, settings.max_generations);
  settings.max_population = std::max(0, settings.max_population);
  settings.initial_active_tests = std::clamp(settings.initial_active_tests, 1, AntoninaAPI::ALL_TESTS);
  settings.failure_log_interval = std::max(1, settings.failure_log_interval);
  settings.failure_trace_detail_count = std::clamp(settings.failure_trace_detail_count, 0, 8);
  settings.population_stall_generations = std::max(1, settings.population_stall_generations);
  settings.population_stable_failure_checks = std::max(1, settings.population_stable_failure_checks);
  settings.population_change_cooldown = std::max(0, settings.population_change_cooldown);
  settings.test_weight_max = std::clamp(settings.test_weight_max, 1.0, 1.5);
  settings.test_weight_step = std::clamp(settings.test_weight_step, 0.01, 0.5);
  settings.test_weight_max_tests = std::clamp(settings.test_weight_max_tests, 1, AntoninaAPI::ALL_TESTS);
  settings.neat_hard_test_archive_size = std::clamp(settings.neat_hard_test_archive_size, 0, 128);
  settings.neat_input_prior_max = std::clamp(settings.neat_input_prior_max, 0, AntoninaAPI::INPUT_FEATURES);
  settings.neat_input_prior_bias = std::clamp(settings.neat_input_prior_bias, 0.0, 1.0);
  settings.neat_input_prior_reach = std::clamp(settings.neat_input_prior_reach, 0.0, 1.0);
  settings.neat_lexicase_max_cases = std::clamp(settings.neat_lexicase_max_cases, 0, AntoninaAPI::ALL_TESTS);
  settings.neat_lexicase_cases_per_parent = std::clamp(settings.neat_lexicase_cases_per_parent, 0, AntoninaAPI::ALL_TESTS);
  settings.neat_lexicase_epsilon_fraction = std::clamp(settings.neat_lexicase_epsilon_fraction, 0.0, 1.0);
  settings.neat_lexicase_parent_rate = std::clamp(settings.neat_lexicase_parent_rate, 0.0, 1.0);
  settings.neat_failure_memory = std::clamp(settings.neat_failure_memory, 0, AntoninaAPI::ALL_TESTS);
  settings.neat_focus_radius = std::clamp(settings.neat_focus_radius, 0, AntoninaAPI::ALL_TESTS);
  settings.neat_focus_max_cases = std::clamp(settings.neat_focus_max_cases, 0, AntoninaAPI::ALL_TESTS);
  settings.neat_bridge_children_per_case = std::clamp(settings.neat_bridge_children_per_case, 0, 128);
  settings.neat_bridge_graft_links = std::clamp(settings.neat_bridge_graft_links, 0, 64);
  settings.neat_mutation_sigma = std::max(0.001, settings.neat_mutation_sigma);
  settings.neat_mutation_prob = std::clamp(settings.neat_mutation_prob, 0.0, 1.0);
  settings.neat_compatibility_threshold = std::clamp(settings.neat_compatibility_threshold, 0.1, 6.0);
  settings.neat_compatibility_weight = std::clamp(settings.neat_compatibility_weight, 0.1, 5.0);
  settings.neat_target_species_min = std::clamp(settings.neat_target_species_min, 1, 128);
  settings.neat_target_species_max = std::clamp(settings.neat_target_species_max, settings.neat_target_species_min, 256);
  settings.neat_survival_rate = std::clamp(settings.neat_survival_rate, 0.01, 1.0);
  settings.neat_complexity_penalty = std::clamp(settings.neat_complexity_penalty, 0.0, 1000.0);
  settings.neat_add_node_prob = std::clamp(settings.neat_add_node_prob, 0.0, 1.0);
  settings.neat_add_connection_prob = std::clamp(settings.neat_add_connection_prob, 0.0, 1.0);
  settings.neat_sparse_connection_prob = std::clamp(settings.neat_sparse_connection_prob, 0.0, 1.0);
  settings.neat_input_probe_prob = std::clamp(settings.neat_input_probe_prob, 0.0, 1.0);
  settings.neat_sparse_input_probe_prob = std::clamp(settings.neat_sparse_input_probe_prob, 0.0, 1.0);
  settings.classic_learning_rate = std::max(0.0, settings.classic_learning_rate);
  settings.classic_parents = std::max(1, settings.classic_parents);
  resizeClassicHiddenSizes(settings);
  for (int &size : settings.classic_hidden_sizes)
    size = std::max(4, size);
  settings.classic_mutation_sigma = std::max(0.001, settings.classic_mutation_sigma);
  settings.classic_mutation_prob = std::clamp(settings.classic_mutation_prob, 0.0, 1.0);
}

bool parseAlgorithm(const std::string &value, std::string &algorithm) {
  if (value == "neat" || value == "NEAT") {
    algorithm = ALGORITHM_NEAT;
    return true;
  }
  if (value == "classic" || value == "evolution" || value == "evo" || value == "classic-evolution") {
    algorithm = ALGORITHM_CLASSIC;
    return true;
  }
  return false;
}
