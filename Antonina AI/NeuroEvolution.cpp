#include "NeuroEvolution.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

template <typename T> T NeuroEvolution::random_in_range(T a, T b) {
  static thread_local std::mt19937 gen(std::random_device{}());

  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> dist(a, b);
    return dist(gen);
  } else if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> dist(a, b);
    return dist(gen);
  }
}

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
  clearFitness();
}

NeuroEvolution::NeuroEvolution(double learningRate_, int length_, int *sizes_,
                               int parents_size_ = 10, int population_ = 100) {
  learningRate = learningRate_;
  population = population_;
  length = length_;
  parents_size = parents_size_;
  best_fitness_ever = INT_MIN;
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

void NeuroEvolution::resetBestFitness(double new_epsilon,
                                      double new_mutation_prob) {
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
