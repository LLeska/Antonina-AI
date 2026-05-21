#include "AITrainingAlgorithm.h"
#include "AntoninaAPI.h"
#include "LiveStats.h"
#include "ScopedViewerLogRedirect.h"
#include "TrainingAlgorithmFactory.h"
#include "TrainingSettings.h"
#include "TrainingSettingsJson.h"
#include "Viewer.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

int main(int argc, char **argv) {
  std::ios_base::sync_with_stdio(false);
  std::cout.tie(nullptr);
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  TrainingSettings settings;
  const bool loaded_settings = loadSettingsJson(settings);
  bool enable_gui = true;
  std::string load_file;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--load" && i + 1 < argc) {
      load_file = argv[++i];
    } else if (arg == "--algorithm" && i + 1 < argc) {
      std::string parsed;
      if (parseAlgorithm(argv[++i], parsed))
        settings.algorithm = parsed;
    } else if (arg == "--classic") {
      settings.algorithm = ALGORITHM_CLASSIC;
    } else if (arg == "--neat") {
      settings.algorithm = ALGORITHM_NEAT;
    } else if (arg == "--population" && i + 1 < argc) {
      settings.population = std::atoi(argv[++i]);
    } else if (arg == "--threads" && i + 1 < argc) {
      settings.requested_threads = std::atoi(argv[++i]);
    } else if (arg == "--max-generations" && i + 1 < argc) {
      settings.max_generations = std::atoi(argv[++i]);
    } else if (arg == "--max-population" && i + 1 < argc) {
      settings.max_population = std::atoi(argv[++i]);
    } else if (arg == "--active-tests" && i + 1 < argc) {
      settings.initial_active_tests = std::atoi(argv[++i]);
    } else if (arg == "--trace-detail-count" && i + 1 < argc) {
      settings.failure_trace_detail_count = std::atoi(argv[++i]);
    } else if (arg == "--no-trace-detail") {
      settings.failure_trace_detail_count = 0;
    } else if (arg == "--fixed-population") {
      settings.adaptive_population = false;
    } else if (arg == "--no-test-weights" || arg == "--no-failure-focus") {
      settings.test_weighting = false;
    } else if (arg == "--test-weight-max" && i + 1 < argc) {
      settings.test_weight_max = std::atof(argv[++i]);
    } else if (arg == "--test-weight-step" && i + 1 < argc) {
      settings.test_weight_step = std::atof(argv[++i]);
    } else if ((arg == "--test-weight-max-tests" || arg == "--failure-focus-max-tests") && i + 1 < argc) {
      settings.test_weight_max_tests = std::atoi(argv[++i]);
    } else if ((arg == "--failure-focus-weight" || arg == "--failure-focus-stable") && i + 1 < argc) {
      ++i;
    } else if (arg == "--no-hard-test-archive") {
      settings.neat_hard_test_archive = false;
    } else if (arg == "--hard-test-archive-size" && i + 1 < argc) {
      settings.neat_hard_test_archive_size = std::atoi(argv[++i]);
    } else if (arg == "--no-input-prior") {
      settings.neat_input_prior = false;
    } else if (arg == "--input-prior-max" && i + 1 < argc) {
      settings.neat_input_prior_max = std::atoi(argv[++i]);
    } else if (arg == "--input-prior-bias" && i + 1 < argc) {
      settings.neat_input_prior_bias = std::atof(argv[++i]);
    } else if (arg == "--input-prior-reach" && i + 1 < argc) {
      settings.neat_input_prior_reach = std::atof(argv[++i]);
    } else if (arg == "--no-epsilon-lexicase" || arg == "--no-lexicase") {
      settings.neat_epsilon_lexicase = false;
    } else if (arg == "--lexicase-max-cases" && i + 1 < argc) {
      settings.neat_lexicase_max_cases = std::atoi(argv[++i]);
    } else if (arg == "--lexicase-cases-per-parent" && i + 1 < argc) {
      settings.neat_lexicase_cases_per_parent = std::atoi(argv[++i]);
    } else if (arg == "--lexicase-epsilon" && i + 1 < argc) {
      settings.neat_lexicase_epsilon_fraction = std::atof(argv[++i]);
    } else if (arg == "--lexicase-parent-rate" && i + 1 < argc) {
      settings.neat_lexicase_parent_rate = std::atof(argv[++i]);
    } else if (arg == "--failure-memory" && i + 1 < argc) {
      settings.neat_failure_memory = std::atoi(argv[++i]);
    } else if (arg == "--focus-radius" && i + 1 < argc) {
      settings.neat_focus_radius = std::atoi(argv[++i]);
    } else if (arg == "--focus-max-cases" && i + 1 < argc) {
      settings.neat_focus_max_cases = std::atoi(argv[++i]);
    } else if (arg == "--bridge-children-per-case" && i + 1 < argc) {
      settings.neat_bridge_children_per_case = std::atoi(argv[++i]);
    } else if (arg == "--bridge-graft-links" && i + 1 < argc) {
      settings.neat_bridge_graft_links = std::atoi(argv[++i]);
    } else if (arg == "--parents" && i + 1 < argc) {
      settings.classic_parents = std::atoi(argv[++i]);
    } else if (arg == "--learning-rate" && i + 1 < argc) {
      settings.classic_learning_rate = std::atof(argv[++i]);
    } else if (arg == "--hidden-layers" && i + 1 < argc) {
      settings.classic_hidden_layers = std::atoi(argv[++i]);
    } else if (arg == "--hidden1" && i + 1 < argc) {
      setClassicHiddenSize(settings, 0, std::atoi(argv[++i]));
    } else if (arg == "--hidden2" && i + 1 < argc) {
      setClassicHiddenSize(settings, 1, std::atoi(argv[++i]));
    } else if (arg == "--hidden3" && i + 1 < argc) {
      setClassicHiddenSize(settings, 2, std::atoi(argv[++i]));
    } else if (arg == "--hidden4" && i + 1 < argc) {
      setClassicHiddenSize(settings, 3, std::atoi(argv[++i]));
    } else if (arg == "--hidden-size" && i + 2 < argc) {
      int index = std::atoi(argv[++i]) - 1;
      int size = std::atoi(argv[++i]);
      setClassicHiddenSize(settings, index, size);
    } else if (arg == "--neat-sigma" && i + 1 < argc) {
      settings.neat_mutation_sigma = std::atof(argv[++i]);
    } else if (arg == "--neat-mut-prob" && i + 1 < argc) {
      settings.neat_mutation_prob = std::atof(argv[++i]);
    } else if (arg == "--neat-compat" && i + 1 < argc) {
      settings.neat_compatibility_threshold = std::atof(argv[++i]);
    } else if (arg == "--neat-compat-weight" && i + 1 < argc) {
      settings.neat_compatibility_weight = std::atof(argv[++i]);
    } else if (arg == "--neat-target-species-min" && i + 1 < argc) {
      settings.neat_target_species_min = std::atoi(argv[++i]);
    } else if (arg == "--neat-target-species-max" && i + 1 < argc) {
      settings.neat_target_species_max = std::atoi(argv[++i]);
    } else if (arg == "--neat-survival" && i + 1 < argc) {
      settings.neat_survival_rate = std::atof(argv[++i]);
    } else if (arg == "--neat-complexity-penalty" && i + 1 < argc) {
      settings.neat_complexity_penalty = std::atof(argv[++i]);
    } else if (arg == "--neat-add-node" && i + 1 < argc) {
      settings.neat_add_node_prob = std::atof(argv[++i]);
    } else if (arg == "--neat-add-connection" && i + 1 < argc) {
      settings.neat_add_connection_prob = std::atof(argv[++i]);
    } else if (arg == "--neat-sparse-connection" && i + 1 < argc) {
      settings.neat_sparse_connection_prob = std::atof(argv[++i]);
    } else if (arg == "--neat-input-probe" && i + 1 < argc) {
      settings.neat_input_probe_prob = std::atof(argv[++i]);
    } else if (arg == "--neat-sparse-input-probe" && i + 1 < argc) {
      settings.neat_sparse_input_probe_prob = std::atof(argv[++i]);
    } else if (arg == "--classic-sigma" && i + 1 < argc) {
      settings.classic_mutation_sigma = std::atof(argv[++i]);
    } else if (arg == "--classic-mut-prob" && i + 1 < argc) {
      settings.classic_mutation_prob = std::atof(argv[++i]);
    } else if (arg == "--no-gui") {
      enable_gui = false;
    } else if (load_file.empty()) {
      load_file = arg;
    }
  }
  normalizeSettings(settings);

  std::unique_ptr<LiveStats> live;
  std::unique_ptr<AntoninaAPI> anim_api;
  std::thread gui_thread;
  std::unique_ptr<ScopedViewerLogRedirect> cout_redirect;
  std::unique_ptr<ScopedViewerLogRedirect> cerr_redirect;

  if (enable_gui) {
    live = std::make_unique<LiveStats>();
    live->setDefaultSettings(settings);
    cout_redirect = std::make_unique<ScopedViewerLogRedirect>(std::cout, *live);
    cerr_redirect = std::make_unique<ScopedViewerLogRedirect>(std::cerr, *live);

    anim_api = std::make_unique<AntoninaAPI>();
    gui_thread = std::thread(viewerThread, std::ref(*live), std::ref(*anim_api));
    if (loaded_settings)
      std::cout << "loaded settings from " << settingsFileName() << '\n';
    std::cout << "viewer ready: waiting for start" << '\n';

    std::string gui_load_file;
    if (!live->waitForStart(settings, &gui_load_file)) {
      if (gui_thread.joinable())
        gui_thread.join();
      return 0;
    }
    if (!gui_load_file.empty())
      load_file = gui_load_file;
    normalizeSettings(settings);
  }

  if (!enable_gui && loaded_settings)
    std::cout << "loaded settings from " << settingsFileName() << '\n';
  if (enable_gui) {
    if (saveSettingsJson(settings))
      std::cout << "saved settings to " << settingsFileName() << '\n';
    else
      std::cerr << "failed to save settings to " << settingsFileName() << '\n';
  }

  std::unique_ptr<AITrainingAlgorithm> algorithm = createTrainingAlgorithm(settings, live.get(), load_file);
  const int rc = algorithm ? algorithm->train() : 1;

  if (live)
    live->requestStop();
  if (gui_thread.joinable())
    gui_thread.join();

  return rc;
}
