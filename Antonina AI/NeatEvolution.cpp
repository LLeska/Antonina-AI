#include "NeatEvolution.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>

namespace {
constexpr double SURVIVAL_RATE = 0.2;
constexpr int MAX_STALE = 30;
constexpr double MUTATE_ONLY_PROB = 0.25;
constexpr double INTERSPECIES_MATE_PROB = 0.001;
constexpr double ADD_CONNECTION_PROB = 0.05;
constexpr double ADD_NODE_PROB = 0.03;
}

NeatInnovation::NeatInnovation(int next_node_id)
    : next_node_id_(next_node_id), next_innovation_(0) {}

int NeatInnovation::connectionInnovation(int in, int out) {
  auto key = std::make_pair(in, out);
  auto it = connections_.find(key);
  if (it != connections_.end())
    return it->second;
  int value = next_innovation_++;
  connections_[key] = value;
  return value;
}

int NeatInnovation::nodeInnovation(int in, int out) {
  auto key = std::make_pair(in, out);
  auto it = split_nodes_.find(key);
  if (it != split_nodes_.end())
    return it->second;
  int value = next_node_id_++;
  split_nodes_[key] = value;
  return value;
}

NeatGenome::NeatGenome()
    : input_count_(0), output_count_(0), bias_id_(-1), bias_index_(-1),
      fitness_(0.0) {}

int NeatGenome::initialSeedInputCount(int input_count) {
  return (int)initialSeedInputIds(input_count).size();
}

std::vector<int> NeatGenome::initialSeedInputIds(int input_count) {
  (void)input_count;
  return {};
}

NeatGenome NeatGenome::minimal(int input_count, int output_count,
                               NeatInnovation &innovation, std::mt19937 &rng) {
  NeatGenome genome;
  genome.input_count_ = input_count;
  genome.output_count_ = output_count;
  genome.bias_id_ = input_count;
  genome.outputs_.assign(output_count, 0.0);

  for (int i = 0; i < input_count; ++i)
    genome.nodes_.push_back({i, 0, 0.0, 0.0});
  genome.nodes_.push_back({genome.bias_id_, 1, 0.0, 1.0});
  for (int i = 0; i < output_count; ++i)
    genome.nodes_.push_back({input_count + 1 + i, 3, 1.0, 0.0});

  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const auto seed_inputs = initialSeedInputIds(input_count);
  for (int out = 0; out < output_count; ++out) {
    int out_id = input_count + 1 + out;
    genome.connections_.push_back(
        {genome.bias_id_, out_id,
         innovation.connectionInnovation(genome.bias_id_, out_id), dist(rng),
         true});
    for (int in_id : seed_inputs) {
      genome.connections_.push_back(
          {in_id, out_id, innovation.connectionInnovation(in_id, out_id),
           dist(rng), true});
    }
  }
  genome.sortGenes();
  return genome;
}

void NeatGenome::feedForward(double *inputs) {
  for (int i = 0; i < (int)nodes_.size(); ++i)
    nodes_[i].value = 0.0;

  for (int i = 0; i < input_count_; ++i)
    nodes_[i].value = inputs[i];
  if (bias_index_ >= 0)
    nodes_[bias_index_].value = 1.0;

  size_t conn_pos = 0;
  for (int idx = 0; idx < (int)nodes_.size(); ++idx) {
    double out_value = nodes_[idx].value;
    if (nodes_[idx].type == 2)
      out_value = activate(out_value);

    while (conn_pos < connections_.size() &&
           connections_[conn_pos].in_index == idx) {
      const auto &conn = connections_[conn_pos++];
      if (!conn.enabled || conn.out_index < 0)
        continue;
      nodes_[conn.out_index].value += out_value * conn.weight;
    }
  }

  if ((int)outputs_.size() != output_count_)
    outputs_.assign(output_count_, 0.0);
  for (int i = 0; i < output_count_; ++i) {
    int out_idx = i < (int)output_indices_.size() ? output_indices_[i] : -1;
    outputs_[i] = out_idx >= 0 ? nodes_[out_idx].value : 0.0;
  }
}

int NeatGenome::getOut() {
  int best = 0;
  for (int i = 1; i < (int)outputs_.size(); ++i)
    if (outputs_[i] > outputs_[best])
      best = i;
  return best;
}

std::unique_ptr<Brain> NeatGenome::cloneBrain() const {
  return std::make_unique<NeatGenome>(*this);
}

void NeatGenome::feedForwardBatch(const double *batch_inputs,
                                  double *batch_outputs, int B) const {
  if (B <= 0 || nodes_.empty())
    return;

  static thread_local std::vector<double> values;
  static thread_local std::vector<double> activated;

  values.assign((size_t)nodes_.size() * B, 0.0);
  activated.resize(B);

  for (int b = 0; b < B; ++b) {
    const double *src = batch_inputs + (size_t)b * input_count_;
    for (int i = 0; i < input_count_; ++i)
      values[(size_t)i * B + b] = src[i];
  }

  if (bias_index_ >= 0) {
    double *bias_row = values.data() + (size_t)bias_index_ * B;
    for (int b = 0; b < B; ++b)
      bias_row[b] = 1.0;
  }

  size_t conn_pos = 0;
  for (int idx = 0; idx < (int)nodes_.size(); ++idx) {
    double *row = values.data() + (size_t)idx * B;
    const double *out_row = row;
    if (nodes_[idx].type == 2) {
      for (int b = 0; b < B; ++b)
        activated[b] = activate(row[b]);
      out_row = activated.data();
    }

    while (conn_pos < connections_.size() &&
           connections_[conn_pos].in_index == idx) {
      const auto &conn = connections_[conn_pos++];
      if (!conn.enabled || conn.out_index < 0)
        continue;
      double *dst = values.data() + (size_t)conn.out_index * B;
      double weight = conn.weight;
      for (int b = 0; b < B; ++b)
        dst[b] += out_row[b] * weight;
    }
  }

  for (int i = 0; i < output_count_; ++i) {
    int out_idx = i < (int)output_indices_.size() ? output_indices_[i] : -1;
    for (int b = 0; b < B; ++b) {
      double value = out_idx >= 0 ? values[(size_t)out_idx * B + b] : 0.0;
      batch_outputs[(size_t)b * output_count_ + i] = value;
    }
  }
}

void NeatGenome::getOutBatch(const double *batch_outputs, int *out_args,
                             int B) const {
  for (int b = 0; b < B; ++b) {
    const double *out = batch_outputs + (size_t)b * output_count_;
    int best = 0;
    for (int i = 1; i < output_count_; ++i)
      if (out[i] > out[best])
        best = i;
    out_args[b] = best;
  }
}

void NeatGenome::mutateWeights(double sigma, double prob, std::mt19937 &rng) {
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::uniform_real_distribution<double> reset(-1.0, 1.0);
  std::normal_distribution<double> perturb(0.0, sigma);
  for (auto &conn : connections_) {
    if (uni(rng) > prob)
      continue;
    if (uni(rng) < 0.1)
      conn.weight = reset(rng);
    else
      conn.weight = std::clamp(conn.weight + perturb(rng), -5.0, 5.0);
  }
}

void NeatGenome::writeInFile(std::ofstream &fout) const {
  fout << "GENOME " << fitness_ << ' ' << input_count_ << ' ' << output_count_
       << ' ' << bias_id_ << ' ' << nodes_.size() << ' ' << connections_.size()
       << '\n';
  for (const auto &node : nodes_)
    fout << "NODE " << node.id << ' ' << node.type << ' ' << node.level << '\n';
  for (const auto &conn : connections_)
    fout << "CONN " << conn.in << ' ' << conn.out << ' ' << conn.innovation
         << ' ' << conn.weight << ' ' << (conn.enabled ? 1 : 0) << '\n';
}

bool NeatGenome::readInFile(std::ifstream &fin) {
  std::string tag;
  int node_count = 0;
  int conn_count = 0;
  if (!(fin >> tag >> fitness_ >> input_count_ >> output_count_ >> bias_id_ >>
        node_count >> conn_count) ||
      tag != "GENOME" || input_count_ <= 0 || output_count_ <= 0 ||
      node_count <= 0 || conn_count < 0)
    return false;

  nodes_.clear();
  connections_.clear();
  nodes_.reserve(node_count);
  connections_.reserve(conn_count);
  outputs_.assign(output_count_, 0.0);

  for (int i = 0; i < node_count; ++i) {
    NeatNodeGene node;
    if (!(fin >> tag >> node.id >> node.type >> node.level) || tag != "NODE")
      return false;
    node.value = 0.0;
    nodes_.push_back(node);
  }

  for (int i = 0; i < conn_count; ++i) {
    NeatConnectionGene conn;
    int enabled = 0;
    if (!(fin >> tag >> conn.in >> conn.out >> conn.innovation >> conn.weight >>
          enabled) ||
        tag != "CONN")
      return false;
    conn.enabled = enabled != 0;
    connections_.push_back(conn);
  }

  sortGenes();
  return true;
}

bool NeatGenome::mutateAddConnection(NeatInnovation &innovation,
                                     std::mt19937 &rng) {
  if (nodes_.size() < 2)
    return false;

  std::uniform_int_distribution<int> dist(0, (int)nodes_.size() - 1);
  std::uniform_real_distribution<double> weight_dist(-1.0, 1.0);

  for (int attempt = 0; attempt < 64; ++attempt) {
    const auto &a = nodes_[dist(rng)];
    const auto &b = nodes_[dist(rng)];
    if (a.id == b.id)
      continue;
    if (a.type == 3 || b.type == 0 || b.type == 1)
      continue;
    if (a.level >= b.level)
      continue;
    if (hasConnection(a.id, b.id))
      continue;

    connections_.push_back({a.id, b.id,
                            innovation.connectionInnovation(a.id, b.id),
                            weight_dist(rng), true});
    sortGenes();
    return true;
  }
  return false;
}

bool NeatGenome::mutateAddNode(NeatInnovation &innovation, std::mt19937 &rng) {
  std::vector<int> enabled;
  for (int i = 0; i < (int)connections_.size(); ++i)
    if (connections_[i].enabled)
      enabled.push_back(i);
  if (enabled.empty())
    return false;

  std::uniform_int_distribution<int> dist(0, (int)enabled.size() - 1);
  int conn_index = enabled[dist(rng)];
  auto old = connections_[conn_index];
  connections_[conn_index].enabled = false;

  int node_id = innovation.nodeInnovation(old.in, old.out);
  double level = (nodeLevel(old.in) + nodeLevel(old.out)) * 0.5;
  nodes_.push_back({node_id, 2, level, 0.0});
  connections_.push_back({old.in, node_id,
                          innovation.connectionInnovation(old.in, node_id), 1.0,
                          true});
  connections_.push_back({node_id, old.out,
                          innovation.connectionInnovation(node_id, old.out),
                          old.weight, true});
  sortGenes();
  return true;
}

NeatGenome NeatGenome::crossover(const NeatGenome &a, const NeatGenome &b,
                                 std::mt19937 &rng) {
  const NeatGenome *fit = &a;
  const NeatGenome *other = &b;
  if (b.fitness_ > a.fitness_) {
    fit = &b;
    other = &a;
  }

  NeatGenome child;
  child.input_count_ = fit->input_count_;
  child.output_count_ = fit->output_count_;
  child.bias_id_ = fit->bias_id_;
  child.nodes_ = fit->nodes_;
  child.outputs_.assign(child.output_count_, 0.0);

  std::unordered_map<int, NeatConnectionGene> other_genes;
  for (const auto &conn : other->connections_)
    other_genes[conn.innovation] = conn;

  std::uniform_int_distribution<int> coin(0, 1);
  for (const auto &conn : fit->connections_) {
    auto it = other_genes.find(conn.innovation);
    NeatConnectionGene selected = conn;
    if (it != other_genes.end() && coin(rng))
      selected = it->second;
    if (it != other_genes.end() && (!conn.enabled || !it->second.enabled)) {
      std::uniform_real_distribution<double> uni(0.0, 1.0);
      selected.enabled = uni(rng) >= 0.75;
    }
    child.connections_.push_back(selected);
  }

  child.sortGenes();
  return child;
}

double NeatGenome::compatibility(const NeatGenome &a, const NeatGenome &b) {
  int i = 0, j = 0;
  int disjoint = 0;
  int excess = 0;
  int matching = 0;
  double weight_diff = 0.0;
  int max_a = a.innovation_order_.empty()
                  ? -1
                  : a.connections_[a.innovation_order_.back()].innovation;
  int max_b = b.innovation_order_.empty()
                  ? -1
                  : b.connections_[b.innovation_order_.back()].innovation;

  while (i < (int)a.innovation_order_.size() &&
         j < (int)b.innovation_order_.size()) {
    const auto &ca = a.connections_[a.innovation_order_[i]];
    const auto &cb = b.connections_[b.innovation_order_[j]];
    if (ca.innovation == cb.innovation) {
      ++matching;
      weight_diff += std::abs(ca.weight - cb.weight);
      ++i;
      ++j;
    } else if (ca.innovation < cb.innovation) {
      if (ca.innovation > max_b)
        ++excess;
      else
        ++disjoint;
      ++i;
    } else {
      if (cb.innovation > max_a)
        ++excess;
      else
        ++disjoint;
      ++j;
    }
  }

  excess +=
      (int)a.innovation_order_.size() - i + (int)b.innovation_order_.size() - j;
  double n = (double)std::max(a.connections_.size(), b.connections_.size());
  if (n < 20.0)
    n = 1.0;
  double w = matching > 0 ? weight_diff / matching : 0.0;
  return 1.0 * excess / n + 1.0 * disjoint / n + 0.4 * w;
}

int NeatGenome::nodeType(int id) const {
  for (const auto &node : nodes_)
    if (node.id == id)
      return node.type;
  return -1;
}

double NeatGenome::nodeLevel(int id) const {
  for (const auto &node : nodes_)
    if (node.id == id)
      return node.level;
  return 0.0;
}

bool NeatGenome::hasConnection(int in, int out) const {
  for (const auto &conn : connections_)
    if (conn.in == in && conn.out == out)
      return true;
  return false;
}

void NeatGenome::sortGenes() {
  std::sort(nodes_.begin(), nodes_.end(), [](const auto &a, const auto &b) {
    if (a.level == b.level)
      return a.id < b.id;
    return a.level < b.level;
  });

  std::unordered_map<int, int> node_index;
  node_index.reserve(nodes_.size() * 2);
  for (int i = 0; i < (int)nodes_.size(); ++i)
    node_index[nodes_[i].id] = i;

  auto indexOf = [&](int id) {
    auto it = node_index.find(id);
    return it == node_index.end() ? -1 : it->second;
  };

  for (auto &conn : connections_) {
    conn.in_index = indexOf(conn.in);
    conn.out_index = indexOf(conn.out);
  }

  bias_index_ = indexOf(bias_id_);
  output_indices_.assign(output_count_, -1);
  for (int i = 0; i < output_count_; ++i)
    output_indices_[i] = indexOf(input_count_ + 1 + i);

  std::sort(connections_.begin(), connections_.end(),
            [](const auto &a, const auto &b) {
              if (a.in_index != b.in_index)
                return a.in_index < b.in_index;
              if (a.out_index != b.out_index)
                return a.out_index < b.out_index;
              return a.innovation < b.innovation;
            });
  innovation_order_.resize(connections_.size());
  std::iota(innovation_order_.begin(), innovation_order_.end(), 0);
  std::sort(innovation_order_.begin(), innovation_order_.end(),
            [this](int a, int b) {
              return connections_[a].innovation < connections_[b].innovation;
            });
}

double NeatGenome::activate(double x) {
  if (x > 60.0)
    return 1.0;
  if (x < -60.0)
    return 0.0;
  return 1.0 / (1.0 + std::exp(-4.9 * x));
}

NeatEvolution::NeatEvolution(int input_count, int output_count, int population)
    : input_count_(input_count), output_count_(output_count), generation_(0),
      best_fitness_(-2147483647), stagnation_(0), species_count_(0),
      compatibility_threshold_(3.0), mutation_sigma_(0.25), mutation_prob_(0.8),
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

void NeatEvolution::setFitness(const int *fitness) {
  for (int i = 0; i < (int)population_.size(); ++i)
    population_[i].setFitness(fitness[i]);
}

void NeatEvolution::evolution() {
  if (population_.empty())
    return;

  int current_best_index = bestIndex();
  int current_best = (int)population_[current_best_index].fitness();
  if (current_best > best_fitness_) {
    best_fitness_ = current_best;
    stagnation_ = 0;
    std::cout << "NEW NEAT RECORD: " << best_fitness_
              << " nodes=" << population_[current_best_index].nodeCount()
              << " conns=" << population_[current_best_index].connectionCount()
              << std::endl;
  } else {
    ++stagnation_;
  }

  speciate();
  updateSpeciesStats();
  removeStaleSpecies();
  assignSpawnCounts();

  std::vector<NeatGenome> next;
  next.reserve(population_.size());

  std::vector<int> global_parents;
  for (auto &species : species_) {
    if (species.members.empty() || species.spawn <= 0)
      continue;
    int survivor_count =
        std::max(1, (int)std::ceil(species.members.size() * SURVIVAL_RATE));
    survivor_count = std::min(survivor_count, (int)species.members.size());
    for (int i = 0; i < survivor_count; ++i)
      global_parents.push_back(species.members[i]);
    next.push_back(population_[species.members[0]]);
    --species.spawn;
  }

  if (global_parents.empty())
    global_parents.push_back(current_best_index);

  int worker_count = std::max(1u, std::thread::hardware_concurrency() == 0
                                      ? 1u
                                      : std::thread::hardware_concurrency());
  worker_count = std::min(worker_count, std::max(1, (int)species_.size()));
  std::vector<unsigned> seeds(worker_count);
  for (int i = 0; i < worker_count; ++i)
    seeds[i] = rng_();
  std::vector<std::vector<NeatGenome>> chunks(worker_count);
  std::vector<std::thread> workers;
  workers.reserve(worker_count);

  for (int worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&, worker] {
      std::mt19937 local_rng(seeds[worker]);
      for (int si = worker; si < (int)species_.size(); si += worker_count) {
        Species &species = species_[si];
        for (int k = 0; k < species.spawn; ++k)
          chunks[worker].push_back(
              makeChild(species, global_parents, local_rng));
      }
    });
  }
  for (auto &worker : workers)
    worker.join();

  for (auto &chunk : chunks)
    for (auto &child : chunk)
      if ((int)next.size() < (int)population_.size())
        next.push_back(std::move(child));

  while ((int)next.size() < (int)population_.size()) {
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

void NeatEvolution::writeInFile(const std::string &file) const {
  std::ofstream fout(file);
  if (!fout.is_open())
    return;
  fout << "NEAT " << generation_ << ' ' << input_count_ << ' ' << output_count_
       << ' ' << population_.size() << ' ' << best_fitness_ << ' '
       << mutation_sigma_ << ' ' << mutation_prob_ << ' '
       << compatibility_threshold_ << '\n';
  for (const auto &genome : population_)
    genome.writeInFile(fout);
}

bool NeatEvolution::writeBestInFile(const std::string &file) const {
  if (population_.empty())
    return false;

  std::ofstream fout(file);
  if (!fout.is_open())
    return false;

  int idx = bestIndex();
  double current_best = population_[idx].fitness();
  fout << "NEAT_BEST " << generation_ << ' ' << input_count_ << ' '
       << output_count_ << ' ' << current_best << ' ' << mutation_sigma_ << ' '
       << mutation_prob_ << ' ' << compatibility_threshold_ << '\n';
  population_[idx].writeInFile(fout);
  return true;
}

void NeatEvolution::speciate() {
  std::vector<Species> next_species = std::move(species_);
  for (auto &species : next_species)
    species.members.clear();

  for (int i = 0; i < (int)population_.size(); ++i) {
    bool placed = false;
    for (auto &species : next_species) {
      if (NeatGenome::compatibility(population_[i], species.representative) <
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
  if (species_count_ > 32)
    compatibility_threshold_ +=
        std::min(1.0, 0.02 * (double)(species_count_ - 32));
  else if (species_count_ < 10)
    compatibility_threshold_ = std::max(0.5, compatibility_threshold_ - 0.05);
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

void NeatEvolution::assignSpawnCounts() {
  if (species_.empty())
    return;
  double total_adjusted = 0.0;
  for (const auto &species : species_)
    total_adjusted += species.adjusted_sum;

  int assigned = 0;
  if (total_adjusted <= 0.0) {
    int base = std::max(1, (int)population_.size() / (int)species_.size());
    for (auto &species : species_) {
      species.spawn = base;
      assigned += species.spawn;
    }
  } else {
    for (auto &species : species_) {
      species.spawn =
          std::max(1, (int)std::round(species.adjusted_sum / total_adjusted *
                                      population_.size()));
      assigned += species.spawn;
    }
  }

  std::sort(species_.begin(), species_.end(), [](const auto &a, const auto &b) {
    return a.adjusted_sum > b.adjusted_sum;
  });
  int diff = (int)population_.size() - assigned;
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

NeatGenome NeatEvolution::makeChild(const Species &species,
                                    const std::vector<int> &global_parents,
                                    std::mt19937 &rng) {
  int survivor_count =
      std::max(1, (int)std::ceil(species.members.size() * SURVIVAL_RATE));
  survivor_count = std::min(survivor_count, (int)species.members.size());
  std::vector<int> parents(species.members.begin(),
                           species.members.begin() + survivor_count);

  std::uniform_real_distribution<double> uni(0.0, 1.0);
  int a = tournamentPick(parents, rng);
  NeatGenome child;
  if (parents.size() == 1 || uni(rng) < MUTATE_ONLY_PROB) {
    child = population_[a];
  } else {
    int b;
    if (uni(rng) < INTERSPECIES_MATE_PROB) {
      b = tournamentPick(global_parents, rng);
    } else {
      b = tournamentPick(parents, rng);
    }
    child = NeatGenome::crossover(population_[a], population_[b], rng);
  }

  child.mutateWeights(mutation_sigma_, mutation_prob_, rng);
  if (uni(rng) < ADD_CONNECTION_PROB) {
    std::lock_guard<std::mutex> lock(innovation_mutex_);
    child.mutateAddConnection(innovation_, rng);
  }
  if (uni(rng) < ADD_NODE_PROB) {
    std::lock_guard<std::mutex> lock(innovation_mutex_);
    child.mutateAddNode(innovation_, rng);
  }
  return child;
}

int NeatEvolution::tournamentPick(const std::vector<int> &members,
                                  std::mt19937 &rng) const {
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
