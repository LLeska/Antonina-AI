#include "NeatEvolution.h"
#include "AntoninaAPI.h"
#include "FullMasteryOptimizer.h"
#include "LiveStats.h"
#include "TestWeightTable.h"
#include "ThreadPool.h"
#include "TrainingUtils.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <utility>

constexpr double DEFAULT_SURVIVAL_RATE = 0.2;
constexpr int MAX_STALE = 100;
constexpr double DEFAULT_MUTATE_ONLY_PROB = 0.25;
constexpr double DEFAULT_INTERSPECIES_MATE_PROB = 0.001;
constexpr double DEFAULT_ADD_CONNECTION_PROB = 0.08;
constexpr double DEFAULT_ADD_NODE_PROB = 0.03;
constexpr double DEFAULT_INPUT_PROBE_PROB = 0.12;
constexpr double DEFAULT_SPARSE_INPUT_PROBE_PROB = 0.45;
constexpr double DEFAULT_SPARSE_CONNECTION_PROB = 0.30;
constexpr int DEFAULT_TARGET_SPECIES_MIN = 12;
constexpr int DEFAULT_TARGET_SPECIES_MAX = 48;
constexpr double DEFAULT_COMPATIBILITY_WEIGHT_COEFFICIENT = 1.0;
constexpr double COMPATIBILITY_MIN = 0.1;
constexpr double COMPATIBILITY_MAX = 6.0;

struct HardTestBest {
  int test_index;
  double weight;
  int genome_index;
  int score;
  int result;

  HardTestBest();
};




struct InputGroupRange {
  const char *name;
  int begin;
  int end;

  InputGroupRange(const char *name = "", int begin = 0, int end = 0);
};

static const InputGroupRange *inputGroups(int &count);
static int inputGroupFor(int input_id);



HardTestBest::HardTestBest()
    : test_index(0), weight(1.0), genome_index(-1), score(INT_MIN),
      result(0) {}
InputGroupRange::InputGroupRange(const char *name, int begin, int end)
    : name(name), begin(begin), end(end) {}

static const InputGroupRange *inputGroups(int &count) {
  static const InputGroupRange groups[] = {
      {"base", 0, 20},       {"align", 20, 36},
      {"up", 36, 68},        {"right", 68, 100},
      {"down", 100, 132},    {"left", 132, 164},
      {"control", 164, 165}, {"wall", 165, 177},
      {"edge", 177, 183},    {"bounds", 183, 191},
  };
  count = (int)(sizeof(groups) / sizeof(groups[0]));
  return groups;
}

static int inputGroupFor(int input_id) {
  int group_count = 0;
  const InputGroupRange *groups = inputGroups(group_count);
  for (int i = 0; i < group_count; ++i) {
    if (input_id >= groups[i].begin && input_id < groups[i].end)
      return i;
  }
  return -1;
}



static std::string compactDiagnosticValue(int value) {
  int abs_value = std::abs(value);
  std::ostringstream out;
  if (abs_value >= 1000000) {
    out << std::fixed << std::setprecision(2) << (value / 1000000.0) << "M";
  } else if (abs_value >= 1000) {
    out << std::fixed << std::setprecision(1) << (value / 1000.0) << "k";
  } else {
    out << value;
  }
  return out.str();
}

static std::string signedDiagnosticValue(int value) {
  if (value > 0)
    return "+" + compactDiagnosticValue(value);
  return compactDiagnosticValue(value);
}

static std::string joinTestIds(const std::vector<int> &tests, int max_items) {
  std::ostringstream out;
  int shown = std::min((int)tests.size(), std::max(0, max_items));
  for (int i = 0; i < shown; ++i) {
    if (i > 0)
      out << ' ';
    out << tests[i];
  }
  if ((int)tests.size() > shown)
    out << " ...";
  return out.str();
}

static const char *testWeightsFileName() { return "test_weights.csv"; }

static std::string hardTestBestSummary(const std::vector<HardTestBest> &bests, int max_items) {
  std::ostringstream out;
  int shown = std::min((int)bests.size(), std::max(0, max_items));
  for (int i = 0; i < shown; ++i) {
    if (i > 0)
      out << ' ';
    const auto &best = bests[(size_t)i];
    out << best.test_index << ':';
    if (best.result > 0) {
      out << "win";
    } else {
      out << compactDiagnosticValue(std::max(0, best.score));
    }
    out << '@' << best.genome_index;
  }
  if ((int)bests.size() > shown)
    out << " ...";
  return out.str();
}

static std::vector<int> buildUnderusedInputPrior(const NeatEvolution &ne, int population, int input_count, int max_inputs, double reach_fraction) {
  if (max_inputs <= 0 || population <= 0 || input_count <= 0)
    return {};

  std::vector<int> counts((size_t)input_count, 0);
  for (int i = 0; i < population; ++i) {
    const NeatGenome *genome = ne.getGenome(i);
    if (!genome)
      continue;
    const auto plan = genome->executionPlan();
    for (int j = 0; j < plan.used_input_count; ++j) {
      int input_id = plan.used_input_indices[j];
      if (input_id >= 0 && input_id < input_count)
        ++counts[(size_t)input_id];
    }
  }

  const int threshold =
      std::max(1, (int)std::ceil(population * reach_fraction));
  std::vector<std::pair<int, int>> candidates;
  candidates.reserve((size_t)input_count);
  for (int input_id = 0; input_id < input_count; ++input_id) {
    if (counts[(size_t)input_id] <= threshold)
      candidates.push_back({counts[(size_t)input_id], input_id});
  }
  std::sort(candidates.begin(), candidates.end());
  if ((int)candidates.size() > max_inputs)
    candidates.resize((size_t)max_inputs);

  std::vector<int> prior;
  prior.reserve(candidates.size());
  for (const auto &candidate : candidates)
    prior.push_back(candidate.second);
  return prior;
}

static std::vector<int> buildLexicaseCaseIndices(int active_tests, const std::vector<WeightedTest> &weighted_tests, const std::vector<int> &failures, int max_cases, int gen) {
  active_tests = std::clamp(active_tests, 0, AntoninaAPI::ALL_TESTS);
  if (active_tests <= 0 || max_cases <= 0)
    return {};

  const int cap = std::min(active_tests, max_cases);
  std::vector<unsigned char> seen((size_t)active_tests, 0);
  std::vector<int> cases;
  cases.reserve((size_t)cap);
  auto addCase = [&](int id) {
    if (id < 0 || id >= active_tests || seen[(size_t)id] ||
        (int)cases.size() >= cap)
      return;
    seen[(size_t)id] = 1;
    cases.push_back(id);
  };

  for (const auto &test : weighted_tests)
    addCase(test.index);
  for (int failure : failures)
    addCase(failure);

  const int stride = std::max(1, active_tests / std::max(1, cap));
  const int offset = stride > 1 ? std::abs(gen) % stride : 0;
  for (int id = offset; id < active_tests && (int)cases.size() < cap;
       id += stride) {
    addCase(id);
  }
  for (int id = 0; id < active_tests && (int)cases.size() < cap; ++id)
    addCase(id);
  return cases;
}

static std::vector<WeightedTest> buildFocusedTests(int active_tests, const std::vector<WeightedTest> &weighted, const std::vector<int> &recent_failures, int radius, int max_cases) {
  active_tests = std::clamp(active_tests, 0, AntoninaAPI::ALL_TESTS);
  max_cases = std::clamp(max_cases, 0, active_tests);
  if (active_tests <= 0 || max_cases <= 0)
    return weighted;

  std::vector<WeightedTest> focused;
  focused.reserve((size_t)std::min(active_tests,
                                   (int)weighted.size() + max_cases));
  std::vector<unsigned char> seen((size_t)active_tests, 0);
  auto add = [&](int id, double weight) {
    if (id < 0 || id >= active_tests)
      return;
    if (seen[(size_t)id]) {
      for (auto &test : focused) {
        if (test.index == id) {
          test.weight = std::max(test.weight, weight);
          return;
        }
      }
    }
    seen[(size_t)id] = 1;
    focused.push_back({id, weight});
  };

  for (const auto &test : weighted)
    add(test.index, test.weight);

  int added_focus = 0;
  for (int failure : recent_failures) {
    if (added_focus >= max_cases)
      break;
    for (int delta = 0; delta <= radius && added_focus < max_cases; ++delta) {
      if (delta == 0) {
        int before = (int)focused.size();
        add(failure, 1.0);
        added_focus += (int)focused.size() > before ? 1 : 0;
        continue;
      }
      int before = (int)focused.size();
      add(failure - delta, 1.0);
      added_focus += (int)focused.size() > before ? 1 : 0;
      before = (int)focused.size();
      add(failure + delta, 1.0);
      added_focus += (int)focused.size() > before ? 1 : 0;
    }
  }

  return focused;
}

static std::vector<int> buildLexicaseEpsilons(const std::vector<int> &case_scores, int population, int case_stride, const std::vector<int> &cases, double epsilon_fraction) {
  std::vector<int> epsilons(cases.size(), 0);
  if (population <= 0 || case_stride <= 0 || case_scores.empty())
    return epsilons;

  for (int ci = 0; ci < (int)cases.size(); ++ci) {
    const int case_index = cases[(size_t)ci];
    if (case_index < 0 || case_index >= case_stride)
      continue;
    int min_score = INT_MAX;
    int max_score = INT_MIN;
    for (int gi = 0; gi < population; ++gi) {
      int score = case_scores[(size_t)gi * case_stride + case_index];
      min_score = std::min(min_score, score);
      max_score = std::max(max_score, score);
    }
    const int range = std::max(0, max_score - min_score);
    epsilons[(size_t)ci] =
        (int)std::llround((double)range * epsilon_fraction);
  }
  return epsilons;
}

static void rememberFailures(std::vector<int> &memory, const std::vector<int> &failures, int limit) {
  if (limit <= 0 || failures.empty())
    return;
  for (int failure : failures) {
    if (failure < 0 || failure >= AntoninaAPI::ALL_TESTS)
      continue;
    auto it = std::find(memory.begin(), memory.end(), failure);
    if (it != memory.end())
      memory.erase(it);
    memory.insert(memory.begin(), failure);
  }
  if ((int)memory.size() > limit)
    memory.resize((size_t)limit);
}

static const char *moveName(int move) {
  switch (move) {
  case 0:
    return "U";
  case 1:
    return "R";
  case 2:
    return "D";
  case 3:
    return "L";
  default:
    return "?";
  }
}

static std::string formatTraceSummary(const AntoninaAPI::TestTrace &trace) {
  std::ostringstream out;
  out << "test=" << trace.test_index << " score="
      << compactDiagnosticValue(trace.score) << " result=";
  if (trace.result > 0)
    out << "win@" << trace.result;
  else
    out << "fail";
  out << " steps=" << trace.steps << " r2b=" << trace.initial_r2b << "->"
      << trace.min_r2b << " b2p=" << trace.initial_b2p << "->"
      << trace.min_b2p << " ctrl=" << trace.initial_control << "->"
      << trace.min_control;
  if (trace.initial_solution > 0)
    out << " sol=" << trace.initial_solution << "->" << trace.min_solution
        << " cur=" << trace.current_solution;
  out << " picked="
      << (trace.bucket_picked ? "yes" : "no") << " invalid="
      << trace.invalid_moves << " masked=" << trace.masked_moves
      << " moves U/R/D/L=" << trace.moves[0] << '/' << trace.moves[1] << '/'
      << trace.moves[2] << '/' << trace.moves[3] << " last="
      << moveName(trace.last_move) << " pos a=(" << trace.ax << ','
      << trace.ay << ") b=(" << trace.gx << ',' << trace.gy << ") pad=("
      << trace.Ox << ',' << trace.Oy << ')';
  if (trace.stones > 0)
    out << " stones=" << trace.stones;
  return out.str();
}

static std::string validMoveMask(const std::array<bool, 4> &valid) {
  constexpr char names[4] = {'U', 'R', 'D', 'L'};
  std::string mask;
  mask.reserve(4);
  for (int i = 0; i < 4; ++i)
    mask.push_back(valid[(size_t)i] ? names[i] : '-');
  return mask;
}

static std::string formatTraceStep(const AntoninaAPI::TestTrace &trace, const AntoninaAPI::TestTrace::Step &step) {
  std::ostringstream out;
  out << "test=" << trace.test_index << " s=" << step.step << " pos a=("
      << step.ax << ',' << step.ay << ") b=(" << step.gx << ',' << step.gy
      << ") pad=(" << step.Ox << ',' << step.Oy << ')';
  if (step.solution >= 0)
    out << " sol=" << step.solution;
  out << " raw=" << moveName(step.raw_move)
      << " sel=" << moveName(step.selected_move)
      << " valid=" << validMoveMask(step.valid) << " out=" << std::fixed
      << std::setprecision(3) << step.outputs[0] << '/' << step.outputs[1]
      << '/' << step.outputs[2] << '/' << step.outputs[3]
      << " picked=" << (step.bucket_picked ? "yes" : "no")
      << " invalid=" << step.invalid_moves;
  if (step.result > 0)
    out << " result=win@" << step.result;
  return out.str();
}

static std::vector<std::string> buildNeatInputDiagnostics(AntoninaAPI &env, const NeatGenome &genome, int tests_to_run, int baseline_fitness) {
  constexpr int ABLATION_LIMIT = 48;

  const int input_count = genome.inputCount();
  const auto plan = genome.executionPlan();
  const auto &nodes = genome.nodes();
  const auto &connections = genome.connections();

  std::vector<unsigned char> reachable(input_count, 0);
  std::vector<unsigned char> connected(input_count, 0);
  std::vector<int> active_links(input_count, 0);
  std::vector<double> active_abs(input_count, 0.0);
  std::vector<int> ablation_delta(input_count, 0);
  std::vector<unsigned char> ablated(input_count, 0);

  int reachable_count = 0;
  for (int i = 0; i < plan.used_input_count; ++i) {
    int input_id = plan.used_input_indices[i];
    if (input_id >= 0 && input_id < input_count && !reachable[input_id]) {
      reachable[input_id] = 1;
      ++reachable_count;
    }
  }

  int enabled_count = 0;
  for (const auto &conn : connections) {
    if (!conn.enabled)
      continue;
    ++enabled_count;
    if (conn.in >= 0 && conn.in < input_count)
      connected[conn.in] = 1;
  }

  for (int i = 0; i < plan.connection_count; ++i) {
    int input_id = plan.conn_in_indices[i];
    if (input_id < 0 || input_id >= input_count)
      continue;
    ++active_links[input_id];
    active_abs[input_id] += std::abs(plan.conn_weights[i]);
  }

  int connected_count = 0;
  int connected_dead_count = 0;
  for (int i = 0; i < input_count; ++i) {
    if (!connected[i])
      continue;
    ++connected_count;
    if (!reachable[i])
      ++connected_dead_count;
  }

  std::vector<unsigned char> active_node(nodes.size(), 0);
  for (int i = 0; i < plan.non_input_count; ++i) {
    int idx = plan.non_input_indices[i];
    if (idx >= 0 && idx < (int)active_node.size())
      active_node[idx] = 1;
  }

  int hidden_total = 0;
  int hidden_dead = 0;
  for (int i = 0; i < (int)nodes.size(); ++i) {
    if (nodes[i].type != 2)
      continue;
    ++hidden_total;
    if (!active_node[i])
      ++hidden_dead;
  }

  int group_count = 0;
  const InputGroupRange *groups = inputGroups(group_count);
  std::vector<int> group_total(group_count, 0);
  std::vector<int> group_reachable(group_count, 0);
  std::vector<int> group_connected(group_count, 0);
  std::vector<int> group_delta(group_count, 0);
  std::vector<int> group_ablated(group_count, 0);
  for (int g = 0; g < group_count; ++g) {
    group_total[g] = std::max(0, std::min(input_count, groups[g].end) -
                                     std::min(input_count, groups[g].begin));
  }
  for (int i = 0; i < input_count; ++i) {
    int g = inputGroupFor(i);
    if (g < 0)
      continue;
    if (reachable[i])
      ++group_reachable[g];
    if (connected[i])
      ++group_connected[g];
  }

  std::vector<int> candidates;
  candidates.reserve(input_count);
  for (int i = 0; i < input_count; ++i) {
    if (reachable[i] && active_links[i] > 0)
      candidates.push_back(i);
  }
  std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
    if (active_abs[a] == active_abs[b])
      return a < b;
    return active_abs[a] > active_abs[b];
  });

  const int ablation_count =
      std::min((int)candidates.size(), ABLATION_LIMIT);
  for (int i = 0; i < ablation_count; ++i) {
    int input_id = candidates[i];
    NeatGenome masked = genome;
    if (!masked.disableInputConnections(input_id))
      continue;
    int masked_fitness = env.solveFitnessBatch(&masked, tests_to_run);
    int delta = baseline_fitness - masked_fitness;
    ablation_delta[input_id] = delta;
    ablated[input_id] = 1;
    int g = inputGroupFor(input_id);
    if (g >= 0) {
      group_delta[g] += delta;
      ++group_ablated[g];
    }
  }

  std::vector<std::string> lines;
  {
    std::ostringstream out;
    out << "inputs reachable " << reachable_count << "/" << input_count
        << " | dead " << (input_count - reachable_count)
        << " | connected " << connected_count
        << " | connected-dead " << connected_dead_count;
    lines.push_back(out.str());
  }
  {
    std::ostringstream out;
    out << "links active/enabled " << plan.connection_count << "/"
        << enabled_count << " | inactive enabled "
        << std::max(0, enabled_count - plan.connection_count)
        << " | dead hidden " << hidden_dead << "/" << hidden_total;
    lines.push_back(out.str());
  }

  std::vector<int> useful = candidates;
  std::sort(useful.begin(), useful.end(), [&](int a, int b) {
    return ablation_delta[a] > ablation_delta[b];
  });
  std::ostringstream useful_line;
  useful_line << "top useful ablation " << ablation_count << "/"
              << candidates.size() << ":";
  int useful_written = 0;
  for (int input_id : useful) {
    if (!ablated[input_id] || ablation_delta[input_id] <= 0)
      continue;
    useful_line << " #" << input_id << " "
                << signedDiagnosticValue(ablation_delta[input_id]);
    if (++useful_written >= 5)
      break;
  }
  if (useful_written == 0)
    useful_line << " none";
  lines.push_back(useful_line.str());

  std::vector<int> harmful = candidates;
  std::sort(harmful.begin(), harmful.end(), [&](int a, int b) {
    return ablation_delta[a] < ablation_delta[b];
  });
  std::ostringstream harmful_line;
  harmful_line << "noisy if negative:";
  int harmful_written = 0;
  for (int input_id : harmful) {
    if (!ablated[input_id] || ablation_delta[input_id] >= 0)
      continue;
    harmful_line << " #" << input_id << " "
                 << signedDiagnosticValue(ablation_delta[input_id]);
    if (++harmful_written >= 4)
      break;
  }
  if (harmful_written == 0)
    harmful_line << " none";
  lines.push_back(harmful_line.str());

  std::ostringstream weak_line;
  weak_line << "weak groups:";
  int weak_written = 0;
  for (int g = 0; g < group_count; ++g) {
    if (group_total[g] <= 0)
      continue;
    if (group_reachable[g] * 4 > group_total[g])
      continue;
    weak_line << " " << groups[g].name << " " << group_reachable[g] << "/"
              << group_total[g];
    if (++weak_written >= 6)
      break;
  }
  if (weak_written == 0)
    weak_line << " none under 25% reach";
  lines.push_back(weak_line.str());

  std::vector<int> group_order(group_count);
  std::iota(group_order.begin(), group_order.end(), 0);
  std::sort(group_order.begin(), group_order.end(), [&](int a, int b) {
    return group_delta[a] > group_delta[b];
  });
  std::ostringstream group_line;
  group_line << "group loss:";
  int group_written = 0;
  for (int g : group_order) {
    if (group_ablated[g] <= 0 || group_delta[g] <= 0)
      continue;
    group_line << " " << groups[g].name << " "
               << signedDiagnosticValue(group_delta[g]) << " ("
               << group_reachable[g] << "/" << group_total[g] << ")";
    if (++group_written >= 4)
      break;
  }
  if (group_written == 0)
    group_line << " none";
  lines.push_back(group_line.str());

  return lines;
}

int NeatEvolution::train() {
  const TrainingSettings &settings = training_settings_;
  LiveStats *live = live_;
  const std::string &load_file = load_file_;
  const int input_count = AntoninaAPI::INPUT_FEATURES;
  int population = std::max(1, settings.population);
  int loaded_active_tests =
      std::clamp(settings.initial_active_tests, 1, AntoninaAPI::ALL_TESTS);

  std::cout << "Antonina AI NEAT starting: population=" << population
            << " inputs=" << input_count << " outputs=4";
  if (!load_file.empty())
    std::cout << " checkpoint=" << load_file;
  std::cout << '\n';

  NeatEvolution &ne = *this;
  if (!load_file.empty()) {
    if (!ne.readInFile(load_file, &loaded_active_tests)) {
      std::cerr << "failed to load NEAT checkpoint: " << load_file << '\n';
      return 1;
    }
    std::cout << "Loaded NEAT checkpoint from " << load_file
              << " | generation=" << ne.getGeneration()
              << " population=" << ne.getPopulation()
              << " active_tests=" << loaded_active_tests << '\n';
  }
  ne.configure(settings.neat_mutation_sigma, settings.neat_mutation_prob,
               settings.neat_compatibility_threshold,
               settings.neat_survival_rate, settings.neat_add_node_prob,
               settings.neat_add_connection_prob,
               settings.neat_sparse_connection_prob,
               settings.neat_input_probe_prob,
               settings.neat_sparse_input_probe_prob,
               settings.neat_compatibility_weight,
               settings.neat_target_species_min,
               settings.neat_target_species_max);

  int start = ne.getGeneration();
  population = ne.getPopulation();
  if (population <= 0) {
    std::cerr << "NEAT population is empty" << '\n';
    return 1;
  }

  const int base_population = population;
  const int max_population =
      settings.adaptive_population
          ? std::max(base_population,
                     settings.max_population > 0 ? settings.max_population
                                                 : base_population * 3)
          : base_population;
  int population_target = base_population;
  ne.reservePopulation(max_population);

  const int num_threads = threadCountFromSettings(settings);
  std::cout << "Training threads=" << num_threads << " gui="
            << (live ? "on" : "off") << '\n';
  std::cout << "Adaptive population="
            << (settings.adaptive_population ? "on" : "off")
            << " base=" << base_population << " max=" << max_population
            << '\n';

  ThreadPool pool(num_threads);
  std::vector<AntoninaAPI> envs(num_threads);

  const int MAX_TESTS = AntoninaAPI::ALL_TESTS;
  const int MIN_GEN_BETWEEN_PROMOTIONS = 0;
#ifdef _DEBUG
  const int LOG_EVERY = 1;
  const bool LOG_DEBUG_STAGES = true;
#else
  const int LOG_EVERY = 100;
  const bool LOG_DEBUG_STAGES = false;
#endif

  for (auto &env : envs)
    env.active_tests = std::clamp(loaded_active_tests, 1, MAX_TESTS);
  int last_promotion_gen = start - MIN_GEN_BETWEEN_PROMOTIONS;

  std::vector<int> fitness(population, 0);
  FullMasteryOptimizer final_optimizer("models/best_final.csv", "models/population_final.csv");
  std::vector<int> last_failure_log;
  int last_failure_log_gen = start - 1000;
  const int FAILURE_LOG_LIMIT = 16;
  const int WIN_FITNESS_UNIT = 500000;
  int adaptive_best_fitness = INT_MIN;
  int adaptive_best_wins = 0;
  int adaptive_last_progress_gen = start;
  int adaptive_last_population_change_gen =
      start - settings.population_change_cooldown;
  int adaptive_stable_failure_checks = 0;
  std::vector<int> adaptive_failure_signature;
  int last_input_diagnostics_gen = start - 1000;
  int last_input_diagnostics_fitness = INT_MIN;
  int last_hard_log_gen = start - 1000;
  std::vector<int> selection_fitness(population, 0);
  std::vector<int> lexicase_case_scores;
  std::vector<int> recent_failure_cases;
  TestWeightTable test_weights;
  test_weights.loadOrReset(testWeightsFileName(), AntoninaAPI::ALL_TESTS,
                           settings);
  bool full_demonstration_started = false;
  if (settings.test_weighting) {
    if (test_weights.save(testWeightsFileName()))
      std::cout << "[weights] using " << testWeightsFileName()
                << " max=" << settings.test_weight_max
                << " step=" << settings.test_weight_step
                << " cap=" << settings.test_weight_max_tests << '\n';
    else
      std::cout << "[weights] loaded in memory but failed to write "
                << testWeightsFileName() << '\n';
  }

  for (int gen = start; gen < start + settings.max_generations; gen++) {
    if (live && !live->isRunning())
      break;

    auto gen_begin = std::chrono::steady_clock::now();
    if (LOG_DEBUG_STAGES && gen < 10)
      std::cout << "gen " << gen << ": fitness start" << '\n';

    const int cur_tests = envs[0].active_tests;
    const std::vector<WeightedTest> weighted_tests =
        settings.test_weighting ? test_weights.boostedTests(cur_tests, settings)
                                : std::vector<WeightedTest>{};
    const bool weights_active = !weighted_tests.empty();
    const std::vector<WeightedTest> hard_tests = buildFocusedTests(
        cur_tests, weighted_tests, recent_failure_cases,
        settings.neat_focus_radius, settings.neat_focus_max_cases);
    const bool focus_active = hard_tests.size() > weighted_tests.size();
    const int max_weight_x100 =
        (int)std::lround(test_weights.maxWeightIn(weighted_tests) * 100.0);
    const bool lexicase_active =
        settings.neat_epsilon_lexicase &&
        settings.neat_lexicase_parent_rate > 0.0 &&
        settings.neat_lexicase_max_cases > 0;
    if (lexicase_active) {
      lexicase_case_scores.assign((size_t)population * cur_tests, 0);
    } else {
      lexicase_case_scores.clear();
    }
    int *lexicase_scores_ptr =
        lexicase_case_scores.empty() ? nullptr : lexicase_case_scores.data();

    std::fill(fitness.begin(), fitness.end(), 0);
    selection_fitness.resize(population);
    std::fill(selection_fitness.begin(), selection_fitness.end(), 0);
    int *fitness_ptr = fitness.data();
    int *selection_fitness_ptr = selection_fitness.data();
    std::atomic<int> next_genome{0};
    const int eval_block = 16;
    const int eval_jobs = std::min(population, num_threads);
    const int hard_slots =
        settings.neat_hard_test_archive ? (int)hard_tests.size() : 0;
    std::vector<std::vector<int>> job_hard_scores(
        (size_t)eval_jobs, std::vector<int>((size_t)hard_slots, INT_MIN));
    std::vector<std::vector<int>> job_hard_indices(
        (size_t)eval_jobs, std::vector<int>((size_t)hard_slots, -1));
    std::vector<std::vector<int>> job_hard_results(
        (size_t)eval_jobs, std::vector<int>((size_t)hard_slots, 0));
    for (int job = 0; job < eval_jobs; job++) {
      pool.enqueue([&ne, &envs, &next_genome, fitness_ptr,
                    selection_fitness_ptr, job, num_threads, eval_block,
                    population, live, &hard_tests,
                    hard_slots, &job_hard_scores, &job_hard_indices,
                    &job_hard_results, lexicase_scores_ptr, cur_tests,
                    complexity_penalty = settings.neat_complexity_penalty]() {
        AntoninaAPI &env = envs[job % num_threads];
        for (;;) {
          if (live && !live->isRunning())
            break;
          const int begin = next_genome.fetch_add(eval_block);
          if (begin >= population)
            break;
          const int end = std::min(population, begin + eval_block);
          for (int i = begin; i < end; i++) {
            if (live && !live->isRunning())
              break;
            int *case_row =
                lexicase_scores_ptr
                    ? lexicase_scores_ptr + (size_t)i * cur_tests
                    : nullptr;
            NeatGenome *genome = ne.getGenome(i);
            int value = case_row ? env.solveFitnessBatch(genome, 0, case_row)
                                 : env.solveFitnessBatch(genome, 0);
            long long selection_value = value;
            if (!hard_tests.empty()) {
              for (int slot = 0; slot < (int)hard_tests.size(); ++slot) {
                const auto &test = hard_tests[(size_t)slot];
                int score = 0;
                int result = env.testResult(ne.getGenome(i), test.index, &score);
                if (slot < hard_slots &&
                    score > job_hard_scores[(size_t)job][(size_t)slot]) {
                  job_hard_scores[(size_t)job][(size_t)slot] = score;
                  job_hard_indices[(size_t)job][(size_t)slot] = i;
                  job_hard_results[(size_t)job][(size_t)slot] = result;
                }
                selection_value += (long long)std::llround(
                    std::max(0.0, test.weight - 1.0) * score);
              }
            }
            if (complexity_penalty > 0.0) {
              const NeatGenome *genome = ne.getGenome(i);
              int complexity = genome->activeConnectionCount() +
                               2 * genome->activeNonInputNodeCount();
              selection_value -=
                  (long long)std::llround(complexity_penalty * complexity);
            }
            fitness_ptr[i] = value;
            selection_fitness_ptr[i] = (int)std::clamp(
                selection_value, 0LL,
                (long long)std::numeric_limits<int>::max());
          }
        }
      });
    }
    pool.wait_all();
    if (live && !live->isRunning())
      break;

    std::vector<HardTestBest> hard_bests;
    hard_bests.reserve((size_t)hard_slots);
    for (int slot = 0; slot < hard_slots; ++slot) {
      HardTestBest best;
      best.test_index = hard_tests[(size_t)slot].index;
      best.weight = hard_tests[(size_t)slot].weight;
      for (int job = 0; job < eval_jobs; ++job) {
        int score = job_hard_scores[(size_t)job][(size_t)slot];
        if (score > best.score) {
          best.score = score;
          best.genome_index = job_hard_indices[(size_t)job][(size_t)slot];
          best.result = job_hard_results[(size_t)job][(size_t)slot];
        }
      }
      if (best.genome_index >= 0)
        hard_bests.push_back(best);
    }

    auto fitness_done = std::chrono::steady_clock::now();
    ne.setFitness(selection_fitness_ptr);

    const FitnessSummary fitness_summary = summarizeFitness(fitness);
    int max_fitness = fitness_summary.max_fitness;
    int min_fitness = fitness_summary.min_fitness;
    int avg_fitness = fitness_summary.avg_fitness;
    int best_idx = fitness_summary.best_index;
    NeatGenome *best = ne.getGenome(best_idx);
    Brain *best_brain = best;

    int best_wins = 0;
    std::vector<int> failures;
    int wins = 0;
    bool checked_failures = false;
    bool promote_after_evolution = false;
    int promotion_target = cur_tests;
    if (gen - last_promotion_gen >= MIN_GEN_BETWEEN_PROMOTIONS) {
      bool can_master =
          (long long)max_fitness >= (long long)cur_tests * WIN_FITNESS_UNIT;
      bool should_log_failures =
          gen - last_failure_log_gen >= settings.failure_log_interval;
      best_wins = std::clamp(max_fitness / WIN_FITNESS_UNIT, 0, cur_tests);

      wins = best_wins;
      if (can_master || should_log_failures) {
        wins =
            envs[0].collectFailures(best_brain, cur_tests, failures, cur_tests);
        best_wins = wins;
        checked_failures = true;
      }

      if (checked_failures && wins == cur_tests) {
        if (cur_tests >= MAX_TESTS) {
          final_optimizer.handle(
              gen, wins, MAX_TESTS, cur_tests, best_idx, max_fitness,
              [&ne](int index, const std::string &file, int fitness) {
                return ne.writeGenomeInFile(index, file, fitness);
              },
              [&ne](const std::string &file, int active_tests) {
                return ne.writeInFile(file, active_tests);
              });
          launchFullDemonstrationOnce(live, full_demonstration_started,
                                      *best_brain, gen);
        } else {
          std::cout << "[mastery] gen=" << gen << " wins=" << wins << '/'
                    << cur_tests << '\n';
          promotion_target = std::min(nextTestCount(cur_tests), MAX_TESTS);
          promote_after_evolution = promotion_target > cur_tests;
        }
      } else if (checked_failures) {
        if (failures != last_failure_log ||
            gen - last_failure_log_gen >= settings.failure_log_interval) {
          std::cout << "[failures] gen=" << gen << " wins=" << wins << '/'
                    << cur_tests << " tests:";
          if (failures.empty()) {
            std::cout << " none";
          } else {
            std::cout << ' ' << joinTestIds(failures, FAILURE_LOG_LIMIT);
          }
          std::cout << '\n';
          last_failure_log = failures;
          last_failure_log_gen = gen;
        }
      }
    }

    if (max_fitness > adaptive_best_fitness || best_wins > adaptive_best_wins) {
      adaptive_best_fitness = std::max(adaptive_best_fitness, max_fitness);
      adaptive_best_wins = std::max(adaptive_best_wins, best_wins);
      adaptive_last_progress_gen = gen;
      adaptive_stable_failure_checks = 0;
    }

    if (checked_failures) {
      rememberFailures(recent_failure_cases, failures,
                       settings.neat_failure_memory);
      if (failures == adaptive_failure_signature) {
        ++adaptive_stable_failure_checks;
      } else {
        adaptive_failure_signature = failures;
        adaptive_stable_failure_checks = 1;
      }

      if (settings.test_weighting) {
        bool changed =
            test_weights.updateFromFailures(gen, cur_tests, failures, settings);
        if (changed) {
          if (!test_weights.save(testWeightsFileName())) {
            std::cout << "[weights] gen=" << gen << " failed to save "
                      << testWeightsFileName() << '\n';
          }
          std::vector<WeightedTest> next_weighted =
              test_weights.boostedTests(cur_tests, settings);
          if (!next_weighted.empty()) {
            std::cout << "[weights] gen=" << gen << " boosted "
                      << next_weighted.size() << " tests: "
                      << test_weights.boostedSummary(next_weighted, 12)
                      << '\n';
          }
        }
      }

      if (!failures.empty()) {
        const int trace_count = std::min((int)failures.size(), 4);
        for (int ti = 0; ti < trace_count; ++ti) {
          AntoninaAPI::TestTrace trace;
          envs[0].traceTest(best_brain, failures[(size_t)ti], trace);
          std::cout << "[trace] gen=" << gen << " best "
                    << formatTraceSummary(trace) << '\n';
          if (ti < settings.failure_trace_detail_count) {
            for (const auto &step : trace.steps_detail) {
              std::cout << "[trace-step] gen=" << gen << ' '
                        << formatTraceStep(trace, step) << '\n';
            }
          }
        }
      }
    }

    auto updatePopulationTarget = [&](int new_target, const char *reason) {
      new_target = std::clamp(new_target, base_population, max_population);
      if (new_target == population_target)
        return;
      std::cout << "[population] gen=" << gen << " target "
                << population_target << " -> " << new_target
                << " reason=" << reason << " wins=" << best_wins << '/'
                << cur_tests << " stable_failures="
                << adaptive_stable_failure_checks << '\n';
      population_target = new_target;
      adaptive_last_population_change_gen = gen;
    };

    if (settings.adaptive_population && promote_after_evolution &&
        population_target > base_population) {
      int decrease = std::max(1, population_target / 3);
      updatePopulationTarget(population_target - decrease, "curriculum");
    } else if (settings.adaptive_population && !promote_after_evolution &&
               population_target < max_population && checked_failures &&
               best_wins < cur_tests && !failures.empty() &&
               adaptive_stable_failure_checks >=
                   settings.population_stable_failure_checks &&
               gen - adaptive_last_progress_gen >=
                   settings.population_stall_generations &&
               gen - adaptive_last_population_change_gen >=
                   settings.population_change_cooldown) {
      int increase = std::max(1, population_target / 2);
      updatePopulationTarget(population_target + increase, "stalled");
    }

    auto fitness_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          fitness_done - gen_begin)
                          .count();
    const int species_count_snapshot = ne.getSpeciesCount();
    const double avg_nodes_snapshot = ne.getAvgNodeCount();
    const double avg_connections_snapshot = ne.getAvgConnectionCount();
    const double avg_active_connections_snapshot =
        ne.getAvgActiveConnectionCount();
    const double compatibility_snapshot = ne.getCompatibilityThreshold();
    const double mutation_sigma_snapshot = ne.getMutationSigma();
    const double mutation_prob_snapshot = ne.getMutationProb();
    const int stagnation_snapshot = ne.getStagnation();

    std::unique_ptr<Brain> best_snapshot;
    if (live)
      best_snapshot = best->cloneBrain();

    if (live &&
        (gen - last_input_diagnostics_gen >= 50 ||
         max_fitness > last_input_diagnostics_fitness || checked_failures)) {
      live->publishInputDiagnostics(buildNeatInputDiagnostics(
          envs[0], *best, cur_tests, max_fitness));
      last_input_diagnostics_gen = gen;
      last_input_diagnostics_fitness =
          std::max(last_input_diagnostics_fitness, max_fitness);
    }

    std::vector<NeatGenome> protected_elites;
    if (settings.neat_hard_test_archive &&
        settings.neat_hard_test_archive_size > 0 && !hard_bests.empty()) {
      std::sort(hard_bests.begin(), hard_bests.end(),
                [](const auto &a, const auto &b) {
                  const bool aw = a.result > 0;
                  const bool bw = b.result > 0;
                  if (aw != bw)
                    return aw > bw;
                  if (a.weight != b.weight)
                    return a.weight > b.weight;
                  if (a.score != b.score)
                    return a.score > b.score;
                  return a.test_index < b.test_index;
                });

      std::vector<int> protected_indices;
      protected_indices.reserve(
          (size_t)settings.neat_hard_test_archive_size);
      for (const auto &hard : hard_bests) {
        if ((int)protected_elites.size() >=
            settings.neat_hard_test_archive_size)
          break;
        if (hard.genome_index < 0 || hard.genome_index >= population ||
            hard.score <= 0)
          continue;
        if (std::find(protected_indices.begin(), protected_indices.end(),
                      hard.genome_index) != protected_indices.end())
          continue;
        protected_indices.push_back(hard.genome_index);
        NeatGenome copy = *ne.getGenome(hard.genome_index);
        copy.setFitness(selection_fitness[(size_t)hard.genome_index]);
        protected_elites.push_back(std::move(copy));
      }
    }

    int hard_wins = 0;
    for (const auto &hard : hard_bests)
      if (hard.result > 0)
        ++hard_wins;
    if (!hard_bests.empty() &&
        (checked_failures || gen - last_hard_log_gen >= LOG_EVERY)) {
      std::cout << "[hard] gen=" << gen << " coverage=" << hard_wins << '/'
                << hard_bests.size() << " archive="
                << protected_elites.size() << " tests: "
                << hardTestBestSummary(hard_bests, 12) << '\n';
      last_hard_log_gen = gen;
    }

    std::vector<int> input_prior;
    if (settings.neat_input_prior &&
        (weights_active || focus_active || !recent_failure_cases.empty()) &&
        settings.neat_input_prior_max > 0 && !promote_after_evolution) {
      input_prior = buildUnderusedInputPrior(
          ne, population, input_count, settings.neat_input_prior_max,
          settings.neat_input_prior_reach);
    }
    ne.setInputMutationPrior(input_prior, settings.neat_input_prior_bias);
    const int input_prior_size = (int)input_prior.size();

    std::vector<int> lexicase_cases;
    std::vector<int> lexicase_epsilons;
    NeatLexicaseSelection lexicase_selection;
    const bool use_lexicase =
        lexicase_active && !lexicase_case_scores.empty() &&
        !promote_after_evolution;
    if (use_lexicase) {
      lexicase_cases = buildLexicaseCaseIndices(
          cur_tests, hard_tests, recent_failure_cases,
          settings.neat_lexicase_max_cases,
          gen);
      lexicase_epsilons = buildLexicaseEpsilons(
          lexicase_case_scores, population, cur_tests, lexicase_cases,
          settings.neat_lexicase_epsilon_fraction);
      if (!lexicase_cases.empty()) {
        lexicase_selection.case_scores = lexicase_case_scores.data();
        lexicase_selection.population = population;
        lexicase_selection.case_stride = cur_tests;
        lexicase_selection.case_indices = lexicase_cases.data();
        lexicase_selection.case_epsilons = lexicase_epsilons.data();
        lexicase_selection.case_count = (int)lexicase_cases.size();
        lexicase_selection.cases_per_parent =
            settings.neat_lexicase_cases_per_parent;
        lexicase_selection.parent_rate = settings.neat_lexicase_parent_rate;
      }
    }
    const int lexicase_case_count =
        lexicase_selection.valid() ? lexicase_selection.case_count : 0;

    std::vector<std::pair<int, int>> bridge_mates;
    if (settings.neat_bridge_children_per_case > 0 && checked_failures &&
        !failures.empty() && !promote_after_evolution && !hard_bests.empty()) {
      std::vector<unsigned char> failed_case((size_t)cur_tests, 0);
      for (int test_index : failures) {
        if (test_index >= 0 && test_index < cur_tests)
          failed_case[(size_t)test_index] = 1;
      }
      for (const auto &hard : hard_bests) {
        if (hard.test_index < 0 || hard.test_index >= cur_tests ||
            !failed_case[(size_t)hard.test_index]) {
          continue;
        }
        if (hard.result <= 0 || hard.genome_index < 0 ||
            hard.genome_index >= population || hard.genome_index == best_idx) {
          continue;
        }
        for (int k = 0; k < settings.neat_bridge_children_per_case; ++k)
          bridge_mates.push_back({best_idx, hard.genome_index});
      }
      if (!bridge_mates.empty() && checked_failures) {
        std::cout << "[bridge] gen=" << gen << " children="
                  << bridge_mates.size() << " from failures:";
        int shown = 0;
        for (const auto &hard : hard_bests) {
          if (shown >= 12)
            break;
          if (hard.test_index >= 0 && hard.test_index < cur_tests &&
              failed_case[(size_t)hard.test_index] && hard.result > 0 &&
              hard.genome_index >= 0 && hard.genome_index < population &&
              hard.genome_index != best_idx) {
            std::cout << ' ' << hard.test_index << '@' << hard.genome_index;
            ++shown;
          }
        }
        std::cout << '\n';
      }
    }
    const int bridge_child_count = (int)bridge_mates.size();

    if (LOG_DEBUG_STAGES && gen < 10)
      std::cout << "gen " << gen << ": evolution start" << '\n';
    auto evolution_begin = std::chrono::steady_clock::now();
    ne.evolution(population_target,
                 protected_elites.empty() ? nullptr : &protected_elites,
                 lexicase_selection.valid() ? &lexicase_selection : nullptr,
                 bridge_mates.empty() ? nullptr : &bridge_mates,
                 settings.neat_bridge_graft_links);
    auto evolution_done = std::chrono::steady_clock::now();
    population = ne.getPopulation();
    fitness.resize(population);
    auto evolution_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            evolution_done - evolution_begin)
                            .count();
    if (LOG_DEBUG_STAGES && gen < 10) {
      std::cout << "gen " << gen
                << ": evolution done | evolution_ms: " << evolution_ms << '\n';
    }

    if (gen % LOG_EVERY == 0) {
      std::ostringstream weight_text;
      weight_text << std::fixed << std::setprecision(2)
                  << (max_weight_x100 / 100.0);
      std::cout << "gen " << gen << " | max: " << max_fitness
                << " min: " << min_fitness << " avg: " << avg_fitness
                << " | tests: " << envs[0].active_tests
                << " | pop: " << population << '/' << population_target
                << " | species: " << species_count_snapshot
                << " | topo: " << (int)avg_nodes_snapshot << "n/"
                << (int)avg_connections_snapshot << "c"
                << " active=" << (int)avg_active_connections_snapshot << "c"
                << " | compat: " << compatibility_snapshot
                << " | weights: " << weighted_tests.size() << "x"
                << weight_text.str()
                << " | hard: " << hard_wins << '/' << hard_bests.size()
                << " arch=" << protected_elites.size()
                << " focus=" << (hard_tests.size() - weighted_tests.size())
                << " prior=" << input_prior_size
                << " lex=" << lexicase_case_count
                << " bridge=" << bridge_child_count
                << " graft=" << settings.neat_bridge_graft_links
                << " | fitness_ms: " << fitness_ms
                << " | evolution_ms: " << evolution_ms << '\n';
    }

    if (live) {
      GenSample gs{gen,
                   max_fitness,
                   min_fitness,
                   avg_fitness,
                   envs[0].active_tests,
                   population,
                   population_target,
                   max_population,
                   best_wins,
                   species_count_snapshot,
                   avg_nodes_snapshot,
                   avg_connections_snapshot,
                   avg_active_connections_snapshot,
                   compatibility_snapshot,
                   (int)fitness_ms,
                   (int)evolution_ms,
                   mutation_sigma_snapshot,
                   mutation_prob_snapshot,
                   stagnation_snapshot,
                   (int)weighted_tests.size(),
                   max_weight_x100};
      live->pushSample(gs);

      if (best_snapshot)
        live->publishBest(*best_snapshot);
    }

    if (promote_after_evolution) {
      for (auto &env : envs)
        env.active_tests = promotion_target;
      ne.resetBestFitness();
      adaptive_best_fitness = INT_MIN;
      adaptive_best_wins = 0;
      adaptive_last_progress_gen = gen;
      adaptive_stable_failure_checks = 0;
      adaptive_failure_signature.clear();
      last_promotion_gen = gen;
      std::cout << "[mastery] gen=" << gen << " promoted " << cur_tests
                << " -> " << promotion_target << " tests" << '\n';
    }

    if (live && live->consumeSaveRequest()) {
      std::filesystem::create_directories("models");
      std::string checkpoint_file = "models/population_manual_gen_" +
                                    std::to_string(ne.getGeneration()) +
                                    ".csv";
      std::cout << "saving " << checkpoint_file << "..." << '\n';
      if (ne.writeInFile(checkpoint_file, envs[0].active_tests))
        std::cout << "saved " << checkpoint_file << '\n';
      else
        std::cout << "failed to save " << checkpoint_file << '\n';
    }
  }

  return 0;
}


NeatEvolution::NeatEvolution(int input_count, int output_count, int population)
    : input_count_(input_count), output_count_(output_count), generation_(0),
      best_fitness_(-2147483647), stagnation_(0), species_count_(0),
      compatibility_threshold_(3.0),
      compatibility_weight_coefficient_(DEFAULT_COMPATIBILITY_WEIGHT_COEFFICIENT),
      mutation_sigma_(0.25), mutation_prob_(0.8),
      survival_rate_(DEFAULT_SURVIVAL_RATE),
      mutate_only_prob_(DEFAULT_MUTATE_ONLY_PROB),
      interspecies_mate_prob_(DEFAULT_INTERSPECIES_MATE_PROB),
      add_connection_prob_(DEFAULT_ADD_CONNECTION_PROB),
      add_node_prob_(DEFAULT_ADD_NODE_PROB),
      input_probe_prob_(DEFAULT_INPUT_PROBE_PROB),
      sparse_input_probe_prob_(DEFAULT_SPARSE_INPUT_PROBE_PROB),
      sparse_connection_prob_(DEFAULT_SPARSE_CONNECTION_PROB),
      input_mutation_prior_bias_(0.0),
      target_species_min_(DEFAULT_TARGET_SPECIES_MIN),
      target_species_max_(DEFAULT_TARGET_SPECIES_MAX),
      next_species_id_(0), innovation_(input_count + output_count + 1),
      rng_((unsigned)std::chrono::high_resolution_clock::now()
               .time_since_epoch()
               .count()) {
  population_.reserve(population);
  for (int i = 0; i < population; ++i) {
    auto genome =
        NeatGenome::minimal(input_count_, output_count_, innovation_, rng_);
    genome.mutateWeights(0.5, 1.0, rng_);
    population_.push_back(std::move(genome));
  }
  speciate();
}

NeatEvolution::NeatEvolution(const TrainingSettings &settings, LiveStats *live, std::string load_file)
    : NeatEvolution(AntoninaAPI::INPUT_FEATURES, 4, load_file.empty() ? std::max(1, settings.population) : 0) {
  training_settings_ = settings;
  live_ = live;
  load_file_ = std::move(load_file);
}

std::unique_ptr<Brain> NeatEvolution::getBestBrain() const {
  if (population_.empty())
    return nullptr;
  return population_[(size_t)bestIndex()].cloneBrain();
}

bool NeatEvolution::save(const std::string &file) const { return writeInFile(file); }

bool NeatEvolution::load(const std::string &file) { return readInFile(file); }

void NeatEvolution::setFitness(const int *fitness) {
  for (int i = 0; i < (int)population_.size(); ++i)
    population_[i].setFitness(fitness[i]);
}

void NeatEvolution::reservePopulation(int capacity) {
  if (capacity > (int)population_.capacity())
    population_.reserve(capacity);
}

void NeatEvolution::configure(double mutation_sigma, double mutation_prob, double compatibility_threshold, double survival_rate, double add_node_prob, double add_connection_prob, double sparse_connection_prob, double input_probe_prob, double sparse_input_probe_prob, double compatibility_weight_coefficient, int target_species_min, int target_species_max) {
  mutation_sigma_ = std::max(0.001, mutation_sigma);
  mutation_prob_ = std::clamp(mutation_prob, 0.0, 1.0);
  compatibility_threshold_ =
      std::clamp(compatibility_threshold, COMPATIBILITY_MIN, COMPATIBILITY_MAX);
  compatibility_weight_coefficient_ =
      std::clamp(compatibility_weight_coefficient, 0.1, 5.0);
  survival_rate_ = std::clamp(survival_rate, 0.01, 1.0);
  add_node_prob_ = std::clamp(add_node_prob, 0.0, 1.0);
  add_connection_prob_ = std::clamp(add_connection_prob, 0.0, 1.0);
  sparse_connection_prob_ = std::clamp(sparse_connection_prob, 0.0, 1.0);
  input_probe_prob_ = std::clamp(input_probe_prob, 0.0, 1.0);
  sparse_input_probe_prob_ = std::clamp(sparse_input_probe_prob, 0.0, 1.0);
  target_species_min_ = std::clamp(target_species_min, 1, 128);
  target_species_max_ =
      std::clamp(target_species_max, target_species_min_, 256);
}

void NeatEvolution::setInputMutationPrior(const std::vector<int> &input_ids, double bias) {
  input_mutation_prior_.clear();
  input_mutation_prior_bias_ = std::clamp(bias, 0.0, 1.0);
  std::vector<unsigned char> seen(std::max(0, input_count_), 0);
  for (int input_id : input_ids) {
    if (input_id < 0 || input_id >= input_count_ || seen[(size_t)input_id])
      continue;
    seen[(size_t)input_id] = 1;
    input_mutation_prior_.push_back(input_id);
  }
}

void NeatEvolution::evolution(
    int target_population, const std::vector<NeatGenome> *protected_elites,
    const NeatLexicaseSelection *lexicase,
    const std::vector<std::pair<int, int>> *bridge_mates,
    int bridge_graft_links) {
  if (population_.empty())
    return;
  if (target_population <= 0)
    target_population = (int)population_.size();

  int current_best_index = bestIndex();
  int current_best = (int)population_[current_best_index].fitness();
  if (current_best > best_fitness_) {
    best_fitness_ = current_best;
    stagnation_ = 0;
    std::cout << "NEW NEAT SELECTION RECORD: " << best_fitness_
              << " nodes=" << population_[current_best_index].nodeCount()
              << " conns=" << population_[current_best_index].connectionCount()
              << " active_conns="
              << population_[current_best_index].activeConnectionCount()
              << std::endl;
  } else {
    ++stagnation_;
  }

  speciate();
  updateSpeciesStats();
  removeStaleSpecies();
  assignSpawnCounts(target_population);

  std::vector<NeatGenome> next;
  next.reserve(target_population);

  std::vector<int> global_parents;
  for (auto &species : species_) {
    if (species.members.empty() || species.spawn <= 0)
      continue;
    int survivor_count =
        std::max(1, (int)std::ceil(species.members.size() * survival_rate_));
    survivor_count = std::min(survivor_count, (int)species.members.size());
    for (int i = 0; i < survivor_count; ++i)
      global_parents.push_back(species.members[i]);
    if ((int)next.size() < target_population) {
      next.push_back(population_[species.members[0]]);
      --species.spawn;
    } else {
      species.spawn = 0;
    }
  }

  if (global_parents.empty())
    global_parents.push_back(current_best_index);

  if (protected_elites != nullptr && !protected_elites->empty()) {
    const int protected_limit = std::min(
        (int)protected_elites->size(), std::max(1, target_population / 50));
    for (int i = 0; i < protected_limit && (int)next.size() < target_population;
         ++i) {
      next.push_back((*protected_elites)[(size_t)i]);
    }
  }

  if (bridge_mates != nullptr && !bridge_mates->empty()) {
    const int bridge_limit =
        std::min((int)bridge_mates->size(), std::max(1, target_population / 20));
    for (int i = 0; i < bridge_limit && (int)next.size() < target_population;
         ++i) {
      const int a = (*bridge_mates)[(size_t)i].first;
      const int b = (*bridge_mates)[(size_t)i].second;
      if (a < 0 || b < 0 || a >= (int)population_.size() ||
          b >= (int)population_.size() || a == b) {
        continue;
      }
      NeatGenome child = NeatGenome::crossoverGraft(population_[(size_t)a],
                                                    population_[(size_t)b],
                                                    bridge_graft_links, rng_);
      mutateChild(child, rng_);
      next.push_back(std::move(child));
    }
  }

  int worker_count = std::max(1u, std::thread::hardware_concurrency() == 0
                                      ? 1u
                                      : std::thread::hardware_concurrency());
  worker_count = std::min(worker_count, std::max(1, (int)species_.size()));
  std::vector<unsigned> seeds(worker_count);
  for (int i = 0; i < worker_count; ++i)
    seeds[i] = rng_();
  std::vector<std::vector<NeatGenome>> chunks(worker_count);
  for (auto &chunk : chunks)
    chunk.reserve(target_population / worker_count + 1);
  std::vector<std::thread> workers;
  workers.reserve(worker_count);

  for (int worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&, worker] {
      std::mt19937 local_rng(seeds[worker]);
      for (int si = worker; si < (int)species_.size(); si += worker_count) {
        Species &species = species_[si];
        for (int k = 0; k < species.spawn; ++k)
          chunks[worker].push_back(
              makeChild(species, global_parents, local_rng, lexicase));
      }
    });
  }
  for (auto &worker : workers)
    worker.join();

  for (auto &chunk : chunks)
    for (auto &child : chunk)
      if ((int)next.size() < target_population)
        next.push_back(std::move(child));

  while ((int)next.size() < target_population) {
    auto genome =
        NeatGenome::minimal(input_count_, output_count_, innovation_, rng_);
    genome.mutateWeights(0.5, 1.0, rng_);
    next.push_back(std::move(genome));
  }

  population_ = std::move(next);
  ++generation_;
  speciate();
}

void NeatEvolution::resetBestFitness() {
  best_fitness_ = -2147483647;
  stagnation_ = 0;
}

double NeatEvolution::getAvgNodeCount() const {
  if (population_.empty())
    return 0.0;
  long long total = 0;
  for (const auto &genome : population_)
    total += genome.nodeCount();
  return (double)total / population_.size();
}

double NeatEvolution::getAvgConnectionCount() const {
  if (population_.empty())
    return 0.0;
  long long total = 0;
  for (const auto &genome : population_)
    total += genome.connectionCount();
  return (double)total / population_.size();
}

double NeatEvolution::getAvgActiveConnectionCount() const {
  if (population_.empty())
    return 0.0;
  long long total = 0;
  for (const auto &genome : population_)
    total += genome.activeConnectionCount();
  return (double)total / population_.size();
}

bool NeatEvolution::writeInFile(const std::string &file, int active_tests) const {
  std::ofstream fout(file);
  if (!fout.is_open())
    return false;
  fout << "NEAT " << generation_ << ' ' << input_count_ << ' ' << output_count_
       << ' ' << population_.size() << ' ' << best_fitness_ << ' '
       << mutation_sigma_ << ' ' << mutation_prob_ << ' '
       << compatibility_threshold_ << '\n';
  if (active_tests > 0)
    fout << "STATE " << active_tests << '\n';
  for (const auto &genome : population_)
    genome.writeInFile(fout);
  fout.flush();
  return fout.good();
}

bool NeatEvolution::readInFile(const std::string &file, int *active_tests) {
  std::ifstream fin(file);
  if (!fin.is_open())
    return false;

  std::string tag;
  int generation = 0;
  int input_count = 0;
  int output_count = 0;
  int population = 0;
  int best_fitness = -2147483647;
  double mutation_sigma = 0.0;
  double mutation_prob = 0.0;
  double compatibility_threshold = 0.0;
  if (!(fin >> tag >> generation >> input_count >> output_count >> population >>
        best_fitness >> mutation_sigma >> mutation_prob >>
        compatibility_threshold) ||
      tag != "NEAT" || input_count <= 0 || input_count > input_count_ ||
      output_count != output_count_ || population <= 0) {
    return false;
  }

  int loaded_active_tests = -1;
  std::streampos genome_start = fin.tellg();
  std::string maybe_state;
  if (fin >> maybe_state) {
    if (maybe_state == "STATE") {
      if (!(fin >> loaded_active_tests))
        return false;
    } else {
      fin.clear();
      fin.seekg(genome_start);
    }
  } else {
    return false;
  }

  std::vector<NeatGenome> loaded;
  loaded.reserve(population);
  for (int i = 0; i < population; ++i) {
    NeatGenome genome;
    if (!genome.readInFile(fin))
      return false;
    if (genome.inputCount() != input_count || genome.outputCount() != output_count_)
      return false;
    if (!genome.upgradeInputCount(input_count_))
      return false;
    loaded.push_back(std::move(genome));
  }

  population_ = std::move(loaded);
  generation_ = generation;
  best_fitness_ = best_fitness;
  stagnation_ = 0;
  species_count_ = 0;
  compatibility_threshold_ = compatibility_threshold;
  mutation_sigma_ = mutation_sigma;
  mutation_prob_ = mutation_prob;
  next_species_id_ = 0;

  innovation_.reset(input_count_ + output_count_ + 1);
  for (const auto &genome : population_) {
    for (const auto &node : genome.nodes())
      innovation_.observeNodeId(node.id);
    for (const auto &conn : genome.connections())
      innovation_.observeConnection(conn.in, conn.out, conn.innovation);
  }

  species_.clear();
  speciate();
  if (active_tests && loaded_active_tests > 0)
    *active_tests = loaded_active_tests;
  return true;
}

bool NeatEvolution::writeGenomeInFile(int index, const std::string &file, int reported_fitness) const {
  if (index < 0 || index >= (int)population_.size())
    return false;

  std::ofstream fout(file);
  if (!fout.is_open())
    return false;

  fout << "NEAT_BEST " << generation_ << ' ' << input_count_ << ' '
       << output_count_ << ' ' << reported_fitness << ' ' << mutation_sigma_
       << ' ' << mutation_prob_ << ' ' << compatibility_threshold_ << '\n';
  NeatGenome copy = population_[(size_t)index];
  copy.setFitness(reported_fitness);
  copy.writeInFile(fout);
  return true;
}

void NeatEvolution::speciate() {
  std::vector<Species> next_species = std::move(species_);
  for (auto &species : next_species)
    species.members.clear();

  for (int i = 0; i < (int)population_.size(); ++i) {
    bool placed = false;
    for (auto &species : next_species) {
      if (NeatGenome::compatibility(population_[i], species.representative,
                                    compatibility_weight_coefficient_) <
          compatibility_threshold_) {
        species.members.push_back(i);
        placed = true;
        break;
      }
    }
    if (!placed) {
      Species species;
      species.id = next_species_id_++;
      species.representative = population_[i];
      species.members.push_back(i);
      next_species.push_back(std::move(species));
    }
  }

  species_.clear();
  for (auto &species : next_species) {
    if (species.members.empty())
      continue;
    species.representative = population_[species.members[0]];
    species_.push_back(std::move(species));
  }

  species_count_ = (int)species_.size();
  compatibility_threshold_ =
      std::clamp(compatibility_threshold_, COMPATIBILITY_MIN,
                 COMPATIBILITY_MAX);
  if (species_count_ > target_species_max_) {
    compatibility_threshold_ = std::min(
        COMPATIBILITY_MAX,
        compatibility_threshold_ +
            std::min(0.35,
                     0.01 * (double)(species_count_ - target_species_max_)));
  } else if (species_count_ < target_species_min_) {
    compatibility_threshold_ = std::max(
        COMPATIBILITY_MIN,
        compatibility_threshold_ -
            std::min(0.45,
                     0.05 * (double)(target_species_min_ - species_count_)));
  }
}

void NeatEvolution::updateSpeciesStats() {
  for (auto &species : species_) {
    std::sort(species.members.begin(), species.members.end(),
              [this](int a, int b) {
                return population_[a].fitness() > population_[b].fitness();
              });
    double sum = 0.0;
    species.adjusted_sum = 0.0;
    for (int idx : species.members) {
      double fit = std::max(0.0, population_[idx].fitness());
      sum += fit;
      species.adjusted_sum += fit / species.members.size();
    }
    species.mean_fitness =
        species.members.empty() ? 0.0 : sum / species.members.size();
    double best = species.members.empty()
                      ? -1.0e300
                      : population_[species.members[0]].fitness();
    if (best > species.best_fitness) {
      species.best_fitness = best;
      species.stale = 0;
    } else {
      ++species.stale;
    }
  }
}

void NeatEvolution::removeStaleSpecies() {
  if (species_.size() <= 2)
    return;
  double global_best = -1.0e300;
  for (const auto &species : species_)
    global_best = std::max(global_best, species.best_fitness);

  std::vector<Species> kept;
  kept.reserve(species_.size());
  for (auto &species : species_) {
    if (species.stale < MAX_STALE || species.best_fitness >= global_best)
      kept.push_back(std::move(species));
  }
  if (kept.empty()) {
    auto best = std::max_element(species_.begin(), species_.end(),
                                 [](const auto &a, const auto &b) {
                                   return a.best_fitness < b.best_fitness;
                                 });
    kept.push_back(std::move(*best));
  }
  species_ = std::move(kept);
  species_count_ = (int)species_.size();
}

void NeatEvolution::assignSpawnCounts(int target_population) {
  if (species_.empty())
    return;
  target_population = std::max(1, target_population);
  double total_adjusted = 0.0;
  for (const auto &species : species_)
    total_adjusted += species.adjusted_sum;

  int assigned = 0;
  if (total_adjusted <= 0.0) {
    int base = std::max(1, target_population / (int)species_.size());
    for (auto &species : species_) {
      species.spawn = base;
      assigned += species.spawn;
    }
  } else {
    for (auto &species : species_) {
      species.spawn =
          std::max(1, (int)std::round(species.adjusted_sum / total_adjusted *
                                      target_population));
      assigned += species.spawn;
    }
  }

  std::sort(species_.begin(), species_.end(), [](const auto &a, const auto &b) {
    return a.adjusted_sum > b.adjusted_sum;
  });
  int diff = target_population - assigned;
  int pos = 0;
  while (diff != 0 && !species_.empty()) {
    Species &species = species_[pos % species_.size()];
    if (diff > 0) {
      ++species.spawn;
      --diff;
    } else if (species.spawn > 1) {
      --species.spawn;
      ++diff;
    }
    ++pos;
    if (pos > (int)species_.size() * 4 && diff < 0)
      break;
  }
}

NeatGenome NeatEvolution::makeChild(const Species &species, const std::vector<int> &global_parents, std::mt19937 &rng, const NeatLexicaseSelection *lexicase) {
  int survivor_count =
      std::max(1, (int)std::ceil(species.members.size() * survival_rate_));
  survivor_count = std::min(survivor_count, (int)species.members.size());
  std::vector<int> parents(species.members.begin(),
                           species.members.begin() + survivor_count);

  std::uniform_real_distribution<double> uni(0.0, 1.0);
  int a = pickParent(parents, rng, lexicase);
  NeatGenome child;
  if (parents.size() == 1 || uni(rng) < mutate_only_prob_) {
    child = population_[a];
  } else {
    int b;
    if (uni(rng) < interspecies_mate_prob_) {
      b = pickParent(global_parents, rng, lexicase);
    } else {
      b = pickParent(parents, rng, lexicase);
    }
    child = NeatGenome::crossover(population_[a], population_[b], rng);
  }

  mutateChild(child, rng);
  return child;
}

void NeatEvolution::mutateChild(NeatGenome &child, std::mt19937 &rng) {
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  child.mutateWeights(mutation_sigma_, mutation_prob_, rng);
  int sparse_limit = std::max(output_count_ * 8, input_count_ / 2);
  bool sparse = child.connectionCount() < sparse_limit;
  double input_probe_prob =
      sparse ? sparse_input_probe_prob_ : input_probe_prob_;
  double add_connection_prob =
      sparse ? sparse_connection_prob_ : add_connection_prob_;

  if (uni(rng) < input_probe_prob) {
    std::lock_guard<std::mutex> lock(innovation_mutex_);
    const std::vector<int> *preferred_inputs = nullptr;
    if (!input_mutation_prior_.empty() &&
        uni(rng) < input_mutation_prior_bias_) {
      preferred_inputs = &input_mutation_prior_;
    }
    child.mutateAddInputConnection(innovation_, rng, preferred_inputs);
  }
  if (uni(rng) < add_connection_prob) {
    std::lock_guard<std::mutex> lock(innovation_mutex_);
    child.mutateAddConnection(innovation_, rng);
  }
  if (child.connectionCount() > output_count_ && uni(rng) < add_node_prob_) {
    std::lock_guard<std::mutex> lock(innovation_mutex_);
    child.mutateAddNode(innovation_, rng);
  }
}

int NeatEvolution::pickParent(const std::vector<int> &members, std::mt19937 &rng, const NeatLexicaseSelection *lexicase) const {
  if (members.empty())
    return bestIndex();
  if (!lexicase || !lexicase->valid())
    return tournamentPick(members, rng);

  std::uniform_real_distribution<double> uni(0.0, 1.0);
  if (uni(rng) > std::clamp(lexicase->parent_rate, 0.0, 1.0))
    return tournamentPick(members, rng);
  return lexicasePick(members, rng, *lexicase);
}

int NeatEvolution::lexicasePick(const std::vector<int> &members, std::mt19937 &rng, const NeatLexicaseSelection &lexicase) const {
  if (members.empty())
    return bestIndex();

  std::vector<int> candidates;
  candidates.reserve(members.size());
  for (int idx : members) {
    if (idx >= 0 && idx < lexicase.population && idx < (int)population_.size())
      candidates.push_back(idx);
  }
  if (candidates.empty())
    return tournamentPick(members, rng);
  if (candidates.size() == 1)
    return candidates[0];

  std::vector<int> order((size_t)lexicase.case_count);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), rng);

  const int cases_to_use =
      lexicase.cases_per_parent > 0
          ? std::min(lexicase.case_count, lexicase.cases_per_parent)
          : lexicase.case_count;

  std::vector<int> kept;
  kept.reserve(candidates.size());
  for (int oi = 0; oi < cases_to_use && candidates.size() > 1; ++oi) {
    const int slot = order[(size_t)oi];
    const int case_index = lexicase.case_indices[slot];
    if (case_index < 0 || case_index >= lexicase.case_stride)
      continue;

    int best_score = INT_MIN;
    for (int idx : candidates) {
      const int score =
          lexicase.case_scores[(size_t)idx * lexicase.case_stride + case_index];
      best_score = std::max(best_score, score);
    }

    const int epsilon = std::max(0, lexicase.case_epsilons[slot]);
    const int threshold = best_score - epsilon;
    kept.clear();
    for (int idx : candidates) {
      const int score =
          lexicase.case_scores[(size_t)idx * lexicase.case_stride + case_index];
      if (score >= threshold)
        kept.push_back(idx);
    }
    if (!kept.empty())
      candidates.swap(kept);
  }

  std::uniform_int_distribution<int> dist(0, (int)candidates.size() - 1);
  return candidates[(size_t)dist(rng)];
}

int NeatEvolution::tournamentPick(const std::vector<int> &members, std::mt19937 &rng) const {
  std::uniform_int_distribution<int> dist(0, (int)members.size() - 1);
  int best = members[dist(rng)];
  int rounds = std::min(3, (int)members.size());
  for (int i = 1; i < rounds; ++i) {
    int candidate = members[dist(rng)];
    if (population_[candidate].fitness() > population_[best].fitness())
      best = candidate;
  }
  return best;
}

int NeatEvolution::bestIndex() const {
  return (int)(std::max_element(population_.begin(), population_.end(),
                                [](const auto &a, const auto &b) {
                                  return a.fitness() < b.fitness();
                                }) -
               population_.begin());
}
