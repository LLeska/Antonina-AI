#include "TrainingSettingsJson.h"

#include "TrainingSettings.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static bool jsonReadString(const std::string &json, const std::string &key, std::string &value) {
  std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (!std::regex_search(json, match, re))
    return false;
  value = match[1].str();
  return true;
}

static bool jsonReadNumber(const std::string &json, const std::string &key, double &value) {
  std::regex re("\"" + key + "\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)");
  std::smatch match;
  if (!std::regex_search(json, match, re))
    return false;
  value = std::atof(match[1].str().c_str());
  return true;
}

static bool jsonReadBool(const std::string &json, const std::string &key, bool &value) {
  std::regex re("\"" + key + "\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (!std::regex_search(json, match, re))
    return false;
  value = match[1].str() == "true";
  return true;
}

static void jsonReadIntSetting(const std::string &json, const std::string &key, int &target) {
  double value = 0.0;
  if (jsonReadNumber(json, key, value))
    target = (int)value;
}

static void jsonReadDoubleSetting(const std::string &json, const std::string &key, double &target) {
  double value = 0.0;
  if (jsonReadNumber(json, key, value))
    target = value;
}

static bool jsonReadIntArray(const std::string &json, const std::string &key, std::vector<int> &target) {
  std::regex re("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
  std::smatch match;
  if (!std::regex_search(json, match, re))
    return false;

  std::vector<int> values;
  const std::string body = match[1].str();
  std::regex number_re("-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?");
  for (auto it = std::sregex_iterator(body.begin(), body.end(), number_re); it != std::sregex_iterator(); ++it)
    values.push_back((int)std::atof((*it)[0].str().c_str()));
  target = std::move(values);
  return true;
}

const char *settingsFileName() { return "training_settings.json"; }

bool loadSettingsJson(TrainingSettings &settings) {
  std::ifstream fin(settingsFileName());
  if (!fin.is_open())
    return false;

  std::ostringstream ss;
  ss << fin.rdbuf();
  const std::string json = ss.str();

  std::string algorithm;
  if (jsonReadString(json, "algorithm", algorithm))
    parseAlgorithm(algorithm, settings.algorithm);

  jsonReadIntSetting(json, "population", settings.population);
  jsonReadIntSetting(json, "threads", settings.requested_threads);
  jsonReadIntSetting(json, "max_generations", settings.max_generations);
  jsonReadIntSetting(json, "max_population", settings.max_population);
  jsonReadBool(json, "adaptive_population", settings.adaptive_population);
  jsonReadIntSetting(json, "initial_active_tests", settings.initial_active_tests);
  jsonReadIntSetting(json, "failure_log_interval", settings.failure_log_interval);
  jsonReadIntSetting(json, "failure_trace_detail_count", settings.failure_trace_detail_count);
  jsonReadIntSetting(json, "population_stall_generations", settings.population_stall_generations);
  jsonReadIntSetting(json, "population_stable_failure_checks", settings.population_stable_failure_checks);
  jsonReadIntSetting(json, "population_change_cooldown", settings.population_change_cooldown);
  if (!jsonReadBool(json, "test_weighting", settings.test_weighting))
    jsonReadBool(json, "failure_focus", settings.test_weighting);
  jsonReadDoubleSetting(json, "test_weight_max", settings.test_weight_max);
  jsonReadDoubleSetting(json, "test_weight_step", settings.test_weight_step);
  jsonReadIntSetting(json, "test_weight_max_tests", settings.test_weight_max_tests);
  jsonReadBool(json, "neat_hard_test_archive", settings.neat_hard_test_archive);
  jsonReadIntSetting(json, "neat_hard_test_archive_size", settings.neat_hard_test_archive_size);
  jsonReadBool(json, "neat_input_prior", settings.neat_input_prior);
  jsonReadIntSetting(json, "neat_input_prior_max", settings.neat_input_prior_max);
  jsonReadDoubleSetting(json, "neat_input_prior_bias", settings.neat_input_prior_bias);
  jsonReadDoubleSetting(json, "neat_input_prior_reach", settings.neat_input_prior_reach);
  jsonReadBool(json, "neat_epsilon_lexicase", settings.neat_epsilon_lexicase);
  jsonReadIntSetting(json, "neat_lexicase_max_cases", settings.neat_lexicase_max_cases);
  jsonReadIntSetting(json, "neat_lexicase_cases_per_parent", settings.neat_lexicase_cases_per_parent);
  jsonReadDoubleSetting(json, "neat_lexicase_epsilon_fraction", settings.neat_lexicase_epsilon_fraction);
  jsonReadDoubleSetting(json, "neat_lexicase_parent_rate", settings.neat_lexicase_parent_rate);
  jsonReadIntSetting(json, "neat_failure_memory", settings.neat_failure_memory);
  jsonReadIntSetting(json, "neat_focus_radius", settings.neat_focus_radius);
  jsonReadIntSetting(json, "neat_focus_max_cases", settings.neat_focus_max_cases);
  jsonReadIntSetting(json, "neat_bridge_children_per_case", settings.neat_bridge_children_per_case);
  jsonReadIntSetting(json, "neat_bridge_graft_links", settings.neat_bridge_graft_links);
  jsonReadDoubleSetting(json, "neat_mutation_sigma", settings.neat_mutation_sigma);
  jsonReadDoubleSetting(json, "neat_mutation_prob", settings.neat_mutation_prob);
  jsonReadDoubleSetting(json, "neat_compatibility_threshold", settings.neat_compatibility_threshold);
  jsonReadDoubleSetting(json, "neat_compatibility_weight", settings.neat_compatibility_weight);
  jsonReadIntSetting(json, "neat_target_species_min", settings.neat_target_species_min);
  jsonReadIntSetting(json, "neat_target_species_max", settings.neat_target_species_max);
  jsonReadDoubleSetting(json, "neat_survival_rate", settings.neat_survival_rate);
  jsonReadDoubleSetting(json, "neat_complexity_penalty", settings.neat_complexity_penalty);
  jsonReadDoubleSetting(json, "neat_add_node_prob", settings.neat_add_node_prob);
  jsonReadDoubleSetting(json, "neat_add_connection_prob", settings.neat_add_connection_prob);
  jsonReadDoubleSetting(json, "neat_sparse_connection_prob", settings.neat_sparse_connection_prob);
  jsonReadDoubleSetting(json, "neat_input_probe_prob", settings.neat_input_probe_prob);
  jsonReadDoubleSetting(json, "neat_sparse_input_probe_prob", settings.neat_sparse_input_probe_prob);
  jsonReadDoubleSetting(json, "classic_learning_rate", settings.classic_learning_rate);
  jsonReadIntSetting(json, "classic_parents", settings.classic_parents);

  double hidden_layers_value = 0.0;
  const bool hidden_layers_set = jsonReadNumber(json, "classic_hidden_layers", hidden_layers_value);
  if (hidden_layers_set)
    settings.classic_hidden_layers = (int)hidden_layers_value;
  if (jsonReadIntArray(json, "classic_hidden_sizes", settings.classic_hidden_sizes)) {
    if (!hidden_layers_set)
      settings.classic_hidden_layers = (int)settings.classic_hidden_sizes.size();
  } else {
    auto setLegacyHiddenSize = [&](int index, int size) {
      if ((int)settings.classic_hidden_sizes.size() <= index)
        settings.classic_hidden_sizes.resize((size_t)index + 1, 32);
      settings.classic_hidden_sizes[(size_t)index] = size;
      if (!hidden_layers_set)
        settings.classic_hidden_layers = std::max(settings.classic_hidden_layers, index + 1);
    };
    if (jsonReadNumber(json, "classic_hidden1", hidden_layers_value))
      setLegacyHiddenSize(0, (int)hidden_layers_value);
    if (jsonReadNumber(json, "classic_hidden2", hidden_layers_value))
      setLegacyHiddenSize(1, (int)hidden_layers_value);
    if (jsonReadNumber(json, "classic_hidden3", hidden_layers_value))
      setLegacyHiddenSize(2, (int)hidden_layers_value);
    if (jsonReadNumber(json, "classic_hidden4", hidden_layers_value))
      setLegacyHiddenSize(3, (int)hidden_layers_value);
  }
  jsonReadDoubleSetting(json, "classic_mutation_sigma", settings.classic_mutation_sigma);
  jsonReadDoubleSetting(json, "classic_mutation_prob", settings.classic_mutation_prob);
  normalizeSettings(settings);
  return true;
}

bool saveSettingsJson(const TrainingSettings &settings) {
  std::ofstream fout(settingsFileName(), std::ios::out | std::ios::trunc);
  if (!fout.is_open())
    return false;

  fout << std::boolalpha << std::fixed << std::setprecision(6);
  fout << "{\n";
  fout << "  \"algorithm\": \"" << (isNeatAlgorithm(settings.algorithm) ? ALGORITHM_NEAT : ALGORITHM_CLASSIC) << "\",\n";
  fout << "  \"population\": " << settings.population << ",\n";
  fout << "  \"threads\": " << settings.requested_threads << ",\n";
  fout << "  \"max_generations\": " << settings.max_generations << ",\n";
  fout << "  \"max_population\": " << settings.max_population << ",\n";
  fout << "  \"adaptive_population\": " << settings.adaptive_population << ",\n";
  fout << "  \"initial_active_tests\": " << settings.initial_active_tests << ",\n";
  fout << "  \"failure_log_interval\": " << settings.failure_log_interval << ",\n";
  fout << "  \"failure_trace_detail_count\": " << settings.failure_trace_detail_count << ",\n";
  fout << "  \"population_stall_generations\": " << settings.population_stall_generations << ",\n";
  fout << "  \"population_stable_failure_checks\": " << settings.population_stable_failure_checks << ",\n";
  fout << "  \"population_change_cooldown\": " << settings.population_change_cooldown << ",\n";
  fout << "  \"test_weighting\": " << settings.test_weighting << ",\n";
  fout << "  \"test_weight_max\": " << settings.test_weight_max << ",\n";
  fout << "  \"test_weight_step\": " << settings.test_weight_step << ",\n";
  fout << "  \"test_weight_max_tests\": " << settings.test_weight_max_tests << ",\n";
  fout << "  \"neat_hard_test_archive\": " << settings.neat_hard_test_archive << ",\n";
  fout << "  \"neat_hard_test_archive_size\": " << settings.neat_hard_test_archive_size << ",\n";
  fout << "  \"neat_input_prior\": " << settings.neat_input_prior << ",\n";
  fout << "  \"neat_input_prior_max\": " << settings.neat_input_prior_max << ",\n";
  fout << "  \"neat_input_prior_bias\": " << settings.neat_input_prior_bias << ",\n";
  fout << "  \"neat_input_prior_reach\": " << settings.neat_input_prior_reach << ",\n";
  fout << "  \"neat_epsilon_lexicase\": " << settings.neat_epsilon_lexicase << ",\n";
  fout << "  \"neat_lexicase_max_cases\": " << settings.neat_lexicase_max_cases << ",\n";
  fout << "  \"neat_lexicase_cases_per_parent\": " << settings.neat_lexicase_cases_per_parent << ",\n";
  fout << "  \"neat_lexicase_epsilon_fraction\": " << settings.neat_lexicase_epsilon_fraction << ",\n";
  fout << "  \"neat_lexicase_parent_rate\": " << settings.neat_lexicase_parent_rate << ",\n";
  fout << "  \"neat_failure_memory\": " << settings.neat_failure_memory << ",\n";
  fout << "  \"neat_focus_radius\": " << settings.neat_focus_radius << ",\n";
  fout << "  \"neat_focus_max_cases\": " << settings.neat_focus_max_cases << ",\n";
  fout << "  \"neat_bridge_children_per_case\": " << settings.neat_bridge_children_per_case << ",\n";
  fout << "  \"neat_bridge_graft_links\": " << settings.neat_bridge_graft_links << ",\n";
  fout << "  \"neat_mutation_sigma\": " << settings.neat_mutation_sigma << ",\n";
  fout << "  \"neat_mutation_prob\": " << settings.neat_mutation_prob << ",\n";
  fout << "  \"neat_compatibility_threshold\": " << settings.neat_compatibility_threshold << ",\n";
  fout << "  \"neat_compatibility_weight\": " << settings.neat_compatibility_weight << ",\n";
  fout << "  \"neat_target_species_min\": " << settings.neat_target_species_min << ",\n";
  fout << "  \"neat_target_species_max\": " << settings.neat_target_species_max << ",\n";
  fout << "  \"neat_survival_rate\": " << settings.neat_survival_rate << ",\n";
  fout << "  \"neat_complexity_penalty\": " << settings.neat_complexity_penalty << ",\n";
  fout << "  \"neat_add_node_prob\": " << settings.neat_add_node_prob << ",\n";
  fout << "  \"neat_add_connection_prob\": " << settings.neat_add_connection_prob << ",\n";
  fout << "  \"neat_sparse_connection_prob\": " << settings.neat_sparse_connection_prob << ",\n";
  fout << "  \"neat_input_probe_prob\": " << settings.neat_input_probe_prob << ",\n";
  fout << "  \"neat_sparse_input_probe_prob\": " << settings.neat_sparse_input_probe_prob << ",\n";
  fout << "  \"classic_learning_rate\": " << settings.classic_learning_rate << ",\n";
  fout << "  \"classic_parents\": " << settings.classic_parents << ",\n";
  fout << "  \"classic_hidden_layers\": " << settings.classic_hidden_layers << ",\n";
  fout << "  \"classic_hidden_sizes\": [";
  for (size_t i = 0; i < settings.classic_hidden_sizes.size(); ++i) {
    if (i > 0)
      fout << ", ";
    fout << settings.classic_hidden_sizes[i];
  }
  fout << "],\n";
  fout << "  \"classic_mutation_sigma\": " << settings.classic_mutation_sigma << ",\n";
  fout << "  \"classic_mutation_prob\": " << settings.classic_mutation_prob << "\n";
  fout << "}\n";
  return true;
}
