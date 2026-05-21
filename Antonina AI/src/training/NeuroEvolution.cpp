#include "NeuroEvolution.h"
#include "AntoninaAPI.h"
#include "FullMasteryOptimizer.h"
#include "LiveStats.h"
#include "Perceptron.h"
#include "ThreadPool.h"
#include "TrainingUtils.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

int NeuroEvolution::partition(int low, int high) {
  int pivot = fitness[high];
  int i = low - 1;

  for (int j = low; j < high; j++) {
    if (fitness[j] >= pivot) {
      i++;
      std::swap(neuros[i], neuros[j]);
      std::swap(fitness[i], fitness[j]);
    }
  }
  std::swap(fitness[i + 1], fitness[high]);
  std::swap(neuros[i + 1], neuros[high]);
  return i + 1;
}

void NeuroEvolution::quickSort(int low, int high) {
  if (low < 0 || high < low || neuros == nullptr || fitness == nullptr)
    return;

  const int count = high - low + 1;
  std::vector<int> order(count);
  std::iota(order.begin(), order.end(), low);
  std::stable_sort(order.begin(), order.end(),
                   [this](int a, int b) { return fitness[a] > fitness[b]; });

  std::unique_ptr<Perceptron[]> sorted_neuros(new Perceptron[count]);
  std::vector<int> sorted_fitness(count);

  for (int i = 0; i < count; ++i) {
    sorted_fitness[i] = fitness[order[i]];
    sorted_neuros[i] = std::move(neuros[order[i]]);
  }
  for (int i = 0; i < count; ++i) {
    fitness[low + i] = sorted_fitness[i];
    neuros[low + i] = std::move(sorted_neuros[i]);
  }
}

NeuroEvolution::NeuroEvolution() {
  learningRate = 0;
  population = 0;
  length = 0;
  parents_size = 0;
  sizes = nullptr;
  neuros = nullptr;
  fitness = nullptr;
  best_fitness_ever = INT_MIN;
  generation = 0;
  generations_without_improvement = 0;
  current_epsilon = 0.35;
  current_mutation_prob = 0.08;
}

NeuroEvolution::NeuroEvolution(double learningRate_, int length_, int *sizes_, int parents_size_, int population_) : NeuroEvolution() {
  init(learningRate_, length_, sizes_, parents_size_, population_);
}

NeuroEvolution::NeuroEvolution(const TrainingSettings &settings, LiveStats *live, std::string load_file) : NeuroEvolution() {
  training_settings_ = settings;
  live_ = live;
  load_file_ = std::move(load_file);

  std::vector<int> topology;
  topology.reserve(settings.classic_hidden_sizes.size() + 2);
  topology.push_back(AntoninaAPI::INPUT_FEATURES);
  for (int hidden_size : settings.classic_hidden_sizes)
    topology.push_back(hidden_size);
  topology.push_back(4);

  init(settings.classic_learning_rate, (int)topology.size(), topology.data(), settings.classic_parents, std::max(1, settings.population));
  configureMutation(settings.classic_mutation_sigma, settings.classic_mutation_prob);
}

void NeuroEvolution::init(double learningRate_, int length_, const int *sizes_, int parents_size_, int population_) {
  deinit();
  learningRate = learningRate_;
  population = population_;
  length = length_;
  parents_size = parents_size_;
  best_fitness_ever = INT_MIN;
  generation = 0;
  generations_without_improvement = 0;
  current_epsilon = 0.35;
  current_mutation_prob = 0.08;
  fitness = nullptr;
  sizes = new int[length];
  for (int i = 0; i < length; i++) {
    sizes[i] = sizes_[i];
  }
  neuros = new Perceptron[population];
  for (int i = 0; i < population; i++) {
    Perceptron p(learningRate, length, sizes);
    neuros[i] = p;
  }
  clearFitness();
}

NeuroEvolution::~NeuroEvolution() { deinit(); }

std::unique_ptr<Brain> NeuroEvolution::getBestBrain() const {
  if (population <= 0 || neuros == nullptr)
    return nullptr;
  return neuros[0].cloneBrain();
}

bool NeuroEvolution::save(const std::string &file) const {
  std::ofstream fout(file);
  if (!fout.is_open())
    return false;
  const_cast<NeuroEvolution *>(this)->writeInFile(&fout);
  return fout.good();
}

bool NeuroEvolution::load(const std::string &file) {
  readFromFile(file);
  return neuros != nullptr && population > 0;
}

int NeuroEvolution::train() {
  const TrainingSettings &settings = training_settings_;
  LiveStats *live = live_;
  const std::string &load_file = load_file_;
  const int input_count = AntoninaAPI::INPUT_FEATURES;
  int population = std::max(1, settings.population);
  std::vector<int> sizes;
  sizes.reserve(settings.classic_hidden_sizes.size() + 2);
  sizes.push_back(input_count);
  for (int hidden_size : settings.classic_hidden_sizes)
    sizes.push_back(hidden_size);
  sizes.push_back(4);
  const int length = (int)sizes.size();

  std::cout << "Antonina AI classic evolution starting: population="
            << population << " topology=";
  for (int i = 0; i < length; ++i) {
    if (i > 0)
      std::cout << '-';
    std::cout << sizes[(size_t)i];
  }
  std::cout << '\n';

  NeuroEvolution &ne = *this;
  if (!load_file.empty()) {
    ne.readFromFile(load_file);
    population = ne.getPopulation();
    std::cout << "Loaded classic population from " << load_file
              << " population=" << population << '\n';
  }

  if (population <= 0) {
    std::cerr << "classic population is empty" << '\n';
    return 1;
  }
  if (settings.adaptive_population) {
    std::cout << "Adaptive population is ignored by classic evolution"
              << '\n';
  }

  const int num_threads = threadCountFromSettings(settings);
  std::cout << "Training threads=" << num_threads << " gui="
            << (live ? "on" : "off") << '\n';

  ThreadPool pool(num_threads);
  std::vector<AntoninaAPI> envs(num_threads);
  const int MAX_TESTS = AntoninaAPI::ALL_TESTS;
  for (auto &env : envs)
    env.active_tests =
        std::clamp(settings.initial_active_tests, 1, AntoninaAPI::ALL_TESTS);

#ifdef _DEBUG
  const int LOG_EVERY = 1;
  const bool LOG_DEBUG_STAGES = true;
#else
  const int LOG_EVERY = 100;
  const bool LOG_DEBUG_STAGES = false;
#endif

  const int WIN_FITNESS_UNIT = 500000;
  const int FAILURE_LOG_LIMIT = 16;
  const int base_population = population;
  const int max_population = population;
  double avg_nodes_snapshot = 0.0;
  double avg_connections_snapshot = 0.0;
  for (int size : sizes)
    avg_nodes_snapshot += size;
  for (int i = 0; i + 1 < length; ++i)
    avg_connections_snapshot +=
        (double)sizes[(size_t)i] * sizes[(size_t)i + 1];

  std::vector<int> fitness(population, 0);
  std::vector<int> last_failure_log;
  int last_failure_log_gen = -1000;
  bool full_demonstration_started = false;
  FullMasteryOptimizer final_optimizer("models/classic_best_final.csv", "models/classic_population_final.csv");

  for (int gen = 0; gen < settings.max_generations; ++gen) {
    if (live && !live->isRunning())
      break;

    auto gen_begin = std::chrono::steady_clock::now();
    if (LOG_DEBUG_STAGES && gen < 10)
      std::cout << "gen " << gen << ": fitness start" << '\n';

    population = ne.getPopulation();
    fitness.resize(population);
    std::fill(fitness.begin(), fitness.end(), 0);
    int *fitness_ptr = fitness.data();
    std::atomic<int> next_genome{0};
    const int eval_block = 16;
    const int eval_jobs = std::min(population, num_threads);
    for (int job = 0; job < eval_jobs; ++job) {
      pool.enqueue([&ne, &envs, &next_genome, fitness_ptr, job,
                    num_threads, eval_block, population, live]() {
        AntoninaAPI &env = envs[job % num_threads];
        for (;;) {
          if (live && !live->isRunning())
            break;
          const int begin = next_genome.fetch_add(eval_block);
          if (begin >= population)
            break;
          const int end = std::min(population, begin + eval_block);
          for (int i = begin; i < end; ++i) {
            if (live && !live->isRunning())
              break;
            fitness_ptr[i] = env.solveFitnessBatch(ne.getNeuros(i), 0);
          }
        }
      });
    }
    pool.wait_all();
    if (live && !live->isRunning())
      break;

    auto fitness_done = std::chrono::steady_clock::now();
    ne.setFitness(fitness_ptr);

    const FitnessSummary fitness_summary = summarizeFitness(fitness);
    int max_fitness = fitness_summary.max_fitness;
    int min_fitness = fitness_summary.min_fitness;
    int avg_fitness = fitness_summary.avg_fitness;
    int best_idx = fitness_summary.best_index;
    Brain *best_brain = ne.getNeuros(best_idx);
    int cur_tests = envs[0].active_tests;
    int best_wins = std::clamp(max_fitness / WIN_FITNESS_UNIT, 0, cur_tests);
    int wins = best_wins;
    std::vector<int> failures;
    bool checked_failures = false;
    bool promote_after_evolution = false;
    int promotion_target = cur_tests;

    bool can_master =
        (long long)max_fitness >= (long long)cur_tests * WIN_FITNESS_UNIT;
    bool should_log_failures =
        gen - last_failure_log_gen >= settings.failure_log_interval;
    if (can_master || should_log_failures) {
      wins =
          envs[0].collectFailures(best_brain, cur_tests, failures,
                                  FAILURE_LOG_LIMIT);
      best_wins = wins;
      checked_failures = true;
    }

    if (checked_failures && wins == cur_tests) {
      if (cur_tests >= MAX_TESTS) {
        final_optimizer.handle(
            gen, wins, MAX_TESTS, cur_tests, best_idx, max_fitness,
            [&ne](int index, const std::string &file, int) {
              ne.getNeuros(index)->writeInFile(file);
              return true;
            },
            [&ne](const std::string &file, int) {
              ne.writeInFile(file);
              return true;
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
        for (int idx : failures)
          std::cout << ' ' << idx;
        std::cout << '\n';
        last_failure_log = failures;
        last_failure_log_gen = gen;
      }
    }

    auto fitness_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          fitness_done - gen_begin)
                          .count();
    std::unique_ptr<Brain> best_snapshot;
    if (live)
      best_snapshot = best_brain->cloneBrain();

    if (LOG_DEBUG_STAGES && gen < 10)
      std::cout << "gen " << gen << ": evolution start" << '\n';
    auto evolution_begin = std::chrono::steady_clock::now();
    ne.evolution();
    auto evolution_done = std::chrono::steady_clock::now();
    auto evolution_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            evolution_done - evolution_begin)
                            .count();

    if (gen % LOG_EVERY == 0) {
      std::cout << "gen " << gen << " | max: " << max_fitness
                << " min: " << min_fitness << " avg: " << avg_fitness
                << " | tests: " << envs[0].active_tests
                << " | pop: " << population
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
                   base_population,
                   max_population,
                   best_wins,
                   0,
                   avg_nodes_snapshot,
                   avg_connections_snapshot,
                   avg_connections_snapshot,
                   0.0,
                   (int)fitness_ms,
                   (int)evolution_ms,
                   ne.getEpsilon(),
                   ne.getMutationProb(),
                   ne.getStagnation()};
      live->pushSample(gs);
      if (best_snapshot)
        live->publishBest(*best_snapshot);
    }

    if (promote_after_evolution) {
      for (auto &env : envs)
        env.active_tests = promotion_target;
      ne.resetBestFitness(settings.classic_mutation_sigma,
                          settings.classic_mutation_prob);
      std::cout << "[mastery] gen=" << gen << " promoted " << cur_tests
                << " -> " << promotion_target << " tests" << '\n';
    }

    if (live && live->consumeSaveRequest()) {
      std::filesystem::create_directories("models");
      std::string checkpoint_file =
          "models/classic_population_manual_gen_" + std::to_string(gen) +
          ".csv";
      std::cout << "saving " << checkpoint_file << "..." << '\n';
      ne.writeInFile(checkpoint_file);
      std::cout << "saved " << checkpoint_file << '\n';
    }
  }

  return 0;
}


int NeuroEvolution::getPopulation() { return population; }

Perceptron *NeuroEvolution::getNeuros(int i) { return &(neuros[i]); }

int *NeuroEvolution::getFitness() { return fitness; }

void NeuroEvolution::clearFitness() {
  if (fitness != nullptr) {
    delete[] fitness;
    fitness = nullptr;
  }
  fitness = new int[population];
  for (int i = 0; i < population; i++) {
    fitness[i] = 0;
  }
}

void NeuroEvolution::setFitness(int *fitness_) {
  for (int i = 0; i < population; i++) {
    fitness[i] = fitness_[i];
  }
}

void NeuroEvolution::evolution() {
  if (population <= 0 || neuros == nullptr)
    return;

  if (!allEqual()) {
    quickSort(0, population - 1);
  }

  int current_best = fitness[0];
  bool improved = false;

  if (current_best > best_fitness_ever) {
    best_fitness_ever = current_best;
    generations_without_improvement = 0;
    improved = true;
    std::cout << "NEW RECORD: " << best_fitness_ever << std::endl;
  } else {
    ++generations_without_improvement;
    improved = false;
  }

  if (improved) {
    current_epsilon = std::max(0.08, current_epsilon * 0.8);
    current_mutation_prob = std::max(0.03, current_mutation_prob * 0.85);
  } else {
    double grow_factor =
        1.0 + 0.015 * std::min(generations_without_improvement, 20);
    current_epsilon = std::min(0.75, current_epsilon * grow_factor);
    current_mutation_prob = std::min(0.16, current_mutation_prob * grow_factor);
  }

  static int log_counter = 0;
  if (++log_counter % 10 == 0) {
    std::cout << "Stagnation: " << generations_without_improvement
              << " | Eps: " << current_epsilon
              << " | MutProb: " << current_mutation_prob
              << " | Best: " << best_fitness_ever << std::endl;
  }

  if (generations_without_improvement > 500) {
    std::cout << "MAJOR RESET after " << generations_without_improvement
              << " gens! Best was: " << best_fitness_ever << std::endl;

    int survivors = std::clamp(population / 10, 1, population);
    std::unique_ptr<Perceptron[]> saved(new Perceptron[survivors]);
    for (int i = 0; i < survivors; ++i)
      saved[i] = neuros[i];

    for (int i = survivors; i < population; ++i) {
      neuros[i] = neuros[i % survivors];
      neuros[i].mutate(0.8, 0.18);
    }

    for (int i = 0; i < survivors; ++i)
      neuros[i] = std::move(saved[i]);

    generations_without_improvement = 0;
    current_epsilon = 0.6;
    current_mutation_prob = 0.14;
    clearFitness();
    return;
  }

  int elite_count = std::clamp(std::max(8, population / 50), 1, population);
  int immigrants =
      std::clamp(std::max(20, population / 25), 0, population - elite_count);

  int use_parents = std::min(parents_size, population);
  std::unique_ptr<Perceptron[]> best(new Perceptron[use_parents]);
  for (int i = 0; i < use_parents; ++i)
    best[i] = neuros[i];

  std::unique_ptr<Perceptron[]> nextGen(new Perceptron[population]);

  for (int i = 0; i < elite_count; ++i) {
    nextGen[i] = neuros[i];
  }

  static thread_local std::mt19937 gen(
      (unsigned)std::chrono::high_resolution_clock::now()
          .time_since_epoch()
          .count());
  std::uniform_int_distribution<int> parentDist(0, use_parents - 1);

  long long avg_fitness = 0;
  for (int i = 0; i < population; ++i) {
    avg_fitness += fitness[i];
  }
  avg_fitness /= population;

  double diversity = (double)avg_fitness / (fitness[0] + 1.0);

  int tournament_size;
  if (generations_without_improvement > 100) {
    tournament_size = 2;
  } else if (generations_without_improvement > 50) {
    tournament_size = diversity > 0.75 ? 3 : 2;
  } else {
    tournament_size = diversity > 0.85 ? 4 : 3;
  }

  auto tournament = [&]() -> int {
    int winner = parentDist(gen);
    for (int t = 1; t < tournament_size; ++t) {
      int cand = parentDist(gen);
      if (fitness[cand] > fitness[winner])
        winner = cand;
    }
    return winner;
  };

  int idx = elite_count;
  while (idx < population - immigrants) {
    int i1 = tournament();

    Perceptron child;

    if (generations_without_improvement > 80 ||
        random_in_range(0.0, 1.0) < 0.5) {
      child = best[i1];
    } else {
      int i2 = tournament();
      child = Perceptron(&best[i1], &best[i2]);
    }

    double eps = current_epsilon;
    double prob = current_mutation_prob;

    if (generations_without_improvement > 30) {
      eps *= random_in_range(0.7, 1.6);
      prob *= random_in_range(0.8, 1.4);
    }

    child.mutate(eps, prob);
    nextGen[idx++] = std::move(child);
  }

  for (int i = population - immigrants; i < population; ++i) {
    Perceptron random_one(learningRate, length, sizes);
    nextGen[i] = random_one;
  }

  delete[] neuros;
  neuros = nextGen.release();
  ++generation;
  clearFitness();
}

void NeuroEvolution::readFromFile(std::string file) {
  std::ifstream fin(file, std::ios::in);
  readFromFile(&fin);
  fin.close();
}

void NeuroEvolution::writeInFile(std::string file) {
  std::ofstream fout(file, std::ios::out);
  writeInFile(&fout);
  fout.close();
}

void NeuroEvolution::readFromFile(std::ifstream *fin) {
  if (!fin->is_open()) {
    std::cerr << "Ошибка открытия файла" << std::endl;
    return;
  }
  this->deinit();
  int population_;
  int parents_size_;
  *fin >> learningRate >> length >> population_ >> parents_size_ >>
      best_fitness_ever >> generations_without_improvement >> current_epsilon >>
      current_mutation_prob;

  std::cout << "Read header: lr=" << learningRate << " len=" << length
            << " pop=" << population_ << " parents=" << parents_size_
            << std::endl;

  int *new_sizes = new int[length];
  for (int i = 0; i < length; i++) {
    *fin >> new_sizes[i];
  }

  int allocated = std::max(population, population_);
  Perceptron *new_neuros = nullptr;
  try {
    new_neuros = new Perceptron[allocated];
  } catch (...) {
    delete[] new_sizes;
    throw;
  }
  sizes = new_sizes;
  neuros = new_neuros;
  population = allocated;
  int actually_read = 0;
  for (int i = 0; i < population_; i++) {
    if (fin->fail() || fin->eof())
      break;
    neuros[i].readFromFile(fin);
    if (neuros[i].isInitialized())
      actually_read++;
  }
  for (int i = actually_read; i < population; i++) {
    Perceptron p(learningRate, length, sizes);
    neuros[i] = p;
  }

  parents_size = parents_size_;
  clearFitness();

  std::cout << "Loaded: best=" << best_fitness_ever
            << " stagnation=" << generations_without_improvement
            << " eps=" << current_epsilon << " prob=" << current_mutation_prob
            << std::endl;

  if (generations_without_improvement > 150) {
    std::cout
        << " WARNING: Long stagnation detected! Resetting mutation params..."
        << std::endl;
    generations_without_improvement = 0;
    current_epsilon = 0.55;
    current_mutation_prob = 0.12;
    std::cout << "Reset to: eps=" << current_epsilon
              << " prob=" << current_mutation_prob << std::endl;
  }
}

void NeuroEvolution::writeInFile(std::ofstream *fout) {
  *fout << learningRate << ' ' << length << ' ' << population << ' '
        << parents_size << ' ' << best_fitness_ever << ' '
        << generations_without_improvement << ' ' << current_epsilon << ' '
        << current_mutation_prob << '\n';
  *fout << sizes[0];
  for (int i = 1; i < length; i++) {
    *fout << ' ' << sizes[i];
  }
  *fout << '\n';
  for (int i = 0; i < population; i++) {
    neuros[i].writeInFile(fout);
  }
}

bool NeuroEvolution::allEqual() {
  for (int i = 1; i < population; i++) {
    if (fitness[0] != fitness[i])
      return false;
  }
  return true;
}

Perceptron *NeuroEvolution::demonstrate() { return &neuros[0]; }

void NeuroEvolution::configureMutation(double epsilon, double mutation_prob) {
  if (epsilon > 0.0)
    current_epsilon = epsilon;
  if (mutation_prob >= 0.0)
    current_mutation_prob = mutation_prob;
}

void NeuroEvolution::resetBestFitness(double new_epsilon, double new_mutation_prob) {
  best_fitness_ever = INT_MIN;
  generations_without_improvement = 0;
  if (new_epsilon > 0.0)
    current_epsilon = new_epsilon;
  if (new_mutation_prob >= 0.0)
    current_mutation_prob = new_mutation_prob;
  std::cout << "[curriculum] best_fitness_ever reset, eps=" << current_epsilon
            << " prob=" << current_mutation_prob << std::endl;
}

void NeuroEvolution::deinit() {
  if (sizes != nullptr) {
    delete[] sizes;
    sizes = nullptr;
  }
  if (neuros != nullptr) {
    delete[] neuros;
    neuros = nullptr;
  }
  if (fitness != nullptr) {
    delete[] fitness;
    fitness = nullptr;
  }
}
