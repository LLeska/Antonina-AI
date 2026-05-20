#include "NeatEvolution.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>

namespace {
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
}

NeatInnovation::NeatInnovation(int next_node_id)
    : next_node_id_(next_node_id), next_innovation_(0) {}

void NeatInnovation::reset(int next_node_id) {
  next_node_id_ = next_node_id;
  next_innovation_ = 0;
  connections_.clear();
  split_nodes_.clear();
}

void NeatInnovation::observeNodeId(int id) {
  if (id >= next_node_id_)
    next_node_id_ = id + 1;
}

void NeatInnovation::observeConnection(int in, int out, int innovation) {
  connections_[std::make_pair(in, out)] = innovation;
  if (innovation >= next_innovation_)
    next_innovation_ = innovation + 1;
}

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
  for (int idx : active_non_input_indices_)
    nodes_[idx].value = 0.0;

  for (int i : used_input_indices_)
    nodes_[i].value = inputs[i];
  if (bias_index_ >= 0)
    nodes_[bias_index_].value = 1.0;

  for (int group = 0; group < (int)active_source_indices_.size(); ++group) {
    int idx = active_source_indices_[group];
    double out_value = nodes_[idx].value;
    if (nodes_[idx].type == 2)
      out_value = activate(out_value);

    int begin = active_source_offsets_[group];
    int end = active_source_offsets_[group + 1];
    for (int pos = begin; pos < end; ++pos) {
      nodes_[active_conn_out_indices_[pos]].value +=
          out_value * active_conn_weights_[pos];
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

double NeatGenome::outputValue(int index) const {
  if (index < 0 || index >= (int)outputs_.size())
    return 0.0;
  return outputs_[(size_t)index];
}

std::unique_ptr<Brain> NeatGenome::cloneBrain() const {
  return std::make_unique<NeatGenome>(*this);
}

NeatExecutionPlan NeatGenome::executionPlan() const {
  NeatExecutionPlan plan;
  plan.conn_in_indices = active_conn_in_indices_.empty()
                             ? nullptr
                             : active_conn_in_indices_.data();
  plan.conn_out_indices = active_conn_out_indices_.empty()
                              ? nullptr
                              : active_conn_out_indices_.data();
  plan.conn_weights =
      active_conn_weights_.empty() ? nullptr : active_conn_weights_.data();
  plan.connection_count = (int)active_conn_weights_.size();

  plan.source_indices =
      active_source_indices_.empty() ? nullptr : active_source_indices_.data();
  plan.source_offsets =
      active_source_offsets_.empty() ? nullptr : active_source_offsets_.data();
  plan.source_count = (int)active_source_indices_.size();

  plan.non_input_indices = active_non_input_indices_.empty()
                               ? nullptr
                               : active_non_input_indices_.data();
  plan.non_input_count = (int)active_non_input_indices_.size();

  plan.used_input_indices =
      used_input_indices_.empty() ? nullptr : used_input_indices_.data();
  plan.used_input_count = (int)used_input_indices_.size();

  plan.output_indices =
      output_indices_.empty() ? nullptr : output_indices_.data();
  plan.node_count = (int)nodes_.size();
  plan.input_count = input_count_;
  plan.output_count = output_count_;
  return plan;
}

void NeatGenome::feedForwardBatch(const double *batch_inputs,
                                  double *batch_outputs, int B) const {
  if (B <= 0)
    return;

  static thread_local std::vector<const double *> input_rows;
  input_rows.resize(B);
  for (int b = 0; b < B; ++b)
    input_rows[b] = batch_inputs + (size_t)b * input_count_;
  feedForwardBatchRows(input_rows.data(), batch_outputs, B);
}

void NeatGenome::feedForwardBatchRows(const double *const *batch_inputs,
                                      double *batch_outputs, int B) const {
  if (B <= 0 || nodes_.empty())
    return;

  static thread_local std::vector<double> values;
  static thread_local std::vector<double> activated;

  const size_t total_values = (size_t)nodes_.size() * B;
  if (values.size() < total_values)
    values.resize(total_values);
  activated.resize(B);

  for (int idx : active_non_input_indices_) {
    double *row = values.data() + (size_t)idx * B;
    std::fill(row, row + B, 0.0);
  }

  for (int idx : used_input_indices_) {
    double *row = values.data() + (size_t)idx * B;
    for (int b = 0; b < B; ++b)
      row[b] = batch_inputs[b][idx];
  }

  if (bias_index_ >= 0) {
    double *bias_row = values.data() + (size_t)bias_index_ * B;
    for (int b = 0; b < B; ++b)
      bias_row[b] = 1.0;
  }

  for (int group = 0; group < (int)active_source_indices_.size(); ++group) {
    int idx = active_source_indices_[group];
    double *row = values.data() + (size_t)idx * B;
    const double *out_row = row;
    if (nodes_[idx].type == 2) {
      for (int b = 0; b < B; ++b)
        activated[b] = activate(row[b]);
      out_row = activated.data();
    }

    int begin = active_source_offsets_[group];
    int end = active_source_offsets_[group + 1];
    for (int pos = begin; pos < end; ++pos) {
      double *dst = values.data() + (size_t)active_conn_out_indices_[pos] * B;
      double weight = active_conn_weights_[pos];
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

  repairLoadedStructure();
  sortGenes();
  return structurallyValid();
}

bool NeatGenome::upgradeInputCount(int new_input_count) {
  if (new_input_count == input_count_)
    return structurallyValid();
  if (new_input_count < input_count_)
    return false;

  const int old_input_count = input_count_;
  const int delta = new_input_count - old_input_count;

  for (auto &node : nodes_) {
    if (node.id >= old_input_count)
      node.id += delta;
  }
  for (auto &conn : connections_) {
    if (conn.in >= old_input_count)
      conn.in += delta;
    if (conn.out >= old_input_count)
      conn.out += delta;
  }

  for (int id = old_input_count; id < new_input_count; ++id)
    nodes_.push_back({id, 0, 0.0, 0.0});

  input_count_ = new_input_count;
  bias_id_ = new_input_count;
  repairLoadedStructure();
  sortGenes();
  return structurallyValid();
}

bool NeatGenome::repairLoadedStructure() {
  if (input_count_ <= 0 || output_count_ <= 0)
    return false;

  bias_id_ = input_count_;
  const int first_output = input_count_ + 1;
  const int last_output = first_output + output_count_;

  std::unordered_map<int, NeatNodeGene> by_id;
  by_id.reserve(nodes_.size() * 2 + (size_t)input_count_ + output_count_ + 1);
  auto requiredType = [&](int id) {
    if (id >= 0 && id < input_count_)
      return 0;
    if (id == bias_id_)
      return 1;
    if (id >= first_output && id < last_output)
      return 3;
    return -1;
  };

  for (auto node : nodes_) {
    if (node.type < 0 || node.type > 3)
      continue;
    node.value = 0.0;
    int required = requiredType(node.id);
    if (required >= 0)
      node.type = required;
    auto inserted = by_id.emplace(node.id, node);
    if (!inserted.second && required >= 0)
      inserted.first->second = node;
  }

  for (int id = 0; id < input_count_; ++id)
    by_id[id] = {id, 0, 0.0, 0.0};
  by_id[bias_id_] = {bias_id_, 1, 0.0, 1.0};
  for (int i = 0; i < output_count_; ++i)
    by_id[first_output + i] = {first_output + i, 3, 1.0, 0.0};

  nodes_.clear();
  nodes_.reserve(by_id.size());
  for (const auto &item : by_id)
    nodes_.push_back(item.second);

  std::vector<NeatConnectionGene> repaired;
  repaired.reserve(connections_.size());
  for (auto conn : connections_) {
    auto in = by_id.find(conn.in);
    auto out = by_id.find(conn.out);
    if (in == by_id.end() || out == by_id.end())
      continue;
    if (conn.in == conn.out || in->second.type == 3)
      continue;
    if (out->second.type == 0 || out->second.type == 1)
      continue;
    if (in->second.level >= out->second.level)
      continue;
    conn.in_index = -1;
    conn.out_index = -1;
    repaired.push_back(conn);
  }
  connections_.swap(repaired);
  return true;
}

bool NeatGenome::structurallyValid() const {
  if (input_count_ <= 0 || output_count_ <= 0 || bias_id_ != input_count_)
    return false;

  std::unordered_map<int, const NeatNodeGene *> by_id;
  by_id.reserve(nodes_.size() * 2);
  for (const auto &node : nodes_) {
    if (node.type < 0 || node.type > 3)
      return false;
    if (!by_id.emplace(node.id, &node).second)
      return false;
  }

  for (int id = 0; id < input_count_; ++id) {
    auto it = by_id.find(id);
    if (it == by_id.end() || it->second->type != 0)
      return false;
  }

  auto bias = by_id.find(bias_id_);
  if (bias == by_id.end() || bias->second->type != 1)
    return false;

  for (int i = 0; i < output_count_; ++i) {
    auto out = by_id.find(input_count_ + 1 + i);
    if (out == by_id.end() || out->second->type != 3)
      return false;
  }

  int enabled_count = 0;
  for (const auto &conn : connections_) {
    auto in = by_id.find(conn.in);
    auto out = by_id.find(conn.out);
    if (in == by_id.end() || out == by_id.end())
      return false;
    if (conn.in == conn.out || in->second->type == 3)
      return false;
    if (out->second->type == 0 || out->second->type == 1)
      return false;
    if (in->second->level >= out->second->level)
      return false;
    if (conn.enabled)
      ++enabled_count;
  }

  if (active_conn_in_indices_.size() != active_conn_out_indices_.size() ||
      active_conn_in_indices_.size() != active_conn_weights_.size())
    return false;
  if (active_source_offsets_.size() != active_source_indices_.size() + 1)
    return false;
  if (!active_source_offsets_.empty() &&
      active_source_offsets_.back() != (int)active_conn_weights_.size())
    return false;
  return (int)active_conn_weights_.size() <= enabled_count;
}

bool NeatGenome::disableInputConnections(int input_id) {
  if (input_id < 0 || input_id >= input_count_)
    return false;

  bool changed = false;
  for (auto &conn : connections_) {
    if (conn.enabled && conn.in == input_id) {
      conn.enabled = false;
      changed = true;
    }
  }
  if (changed)
    sortGenes();
  return changed;
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

bool NeatGenome::mutateAddInputConnection(
    NeatInnovation &innovation, std::mt19937 &rng,
    const std::vector<int> *preferred_inputs) {
  if (input_count_ <= 0 || nodes_.empty())
    return false;

  std::vector<int> targets;
  targets.reserve(nodes_.size());
  for (int i = 0; i < (int)nodes_.size(); ++i) {
    if (nodes_[i].type == 2 || nodes_[i].type == 3)
      targets.push_back(i);
  }
  if (targets.empty())
    return false;

  std::uniform_int_distribution<int> input_dist(0, input_count_ - 1);
  std::uniform_int_distribution<int> target_dist(0, (int)targets.size() - 1);
  std::uniform_real_distribution<double> weight_dist(-1.0, 1.0);
  const bool has_preferred =
      preferred_inputs != nullptr && !preferred_inputs->empty();
  std::uniform_int_distribution<int> preferred_dist(
      0, has_preferred ? (int)preferred_inputs->size() - 1 : 0);

  for (int attempt = 0; attempt < 96; ++attempt) {
    int in_id = input_dist(rng);
    if (has_preferred && attempt < 64) {
      int preferred = (*preferred_inputs)[(size_t)preferred_dist(rng)];
      if (preferred < 0 || preferred >= input_count_)
        continue;
      in_id = preferred;
    }
    const auto &target = nodes_[targets[target_dist(rng)]];
    if (nodeLevel(in_id) >= target.level)
      continue;
    if (hasConnection(in_id, target.id))
      continue;

    connections_.push_back(
        {in_id, target.id, innovation.connectionInnovation(in_id, target.id),
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

NeatGenome NeatGenome::crossoverGraft(const NeatGenome &base,
                                      const NeatGenome &donor,
                                      int max_extra_connections,
                                      std::mt19937 &rng) {
  NeatGenome child = base;
  max_extra_connections = std::max(0, max_extra_connections);
  if (max_extra_connections == 0 || donor.active_connection_indices_.empty()) {
    child.sortGenes();
    return child;
  }

  std::unordered_map<int, int> child_nodes;
  child_nodes.reserve(child.nodes_.size() * 2 + 16);
  for (const auto &node : child.nodes_)
    child_nodes[node.id] = 1;

  std::unordered_map<int, int> child_innovations;
  child_innovations.reserve(child.connections_.size() * 2 + 16);
  std::unordered_map<long long, int> child_edges;
  child_edges.reserve(child.connections_.size() * 2 + 16);
  auto edgeKey = [](int in, int out) {
    return (long long)(unsigned int)in << 32 | (unsigned int)out;
  };
  for (const auto &conn : child.connections_) {
    child_innovations[conn.innovation] = 1;
    child_edges[edgeKey(conn.in, conn.out)] = 1;
  }

  std::unordered_map<int, const NeatNodeGene *> donor_nodes;
  donor_nodes.reserve(donor.nodes_.size() * 2);
  for (const auto &node : donor.nodes_)
    donor_nodes[node.id] = &node;

  std::unordered_map<int, std::vector<int>> donor_active_inputs;
  donor_active_inputs.reserve(donor.nodes_.size());
  std::vector<int> output_edges;
  std::vector<int> other_edges;
  output_edges.reserve(donor.active_connection_indices_.size());
  other_edges.reserve(donor.active_connection_indices_.size());
  for (int conn_index : donor.active_connection_indices_) {
    if (conn_index < 0 || conn_index >= (int)donor.connections_.size())
      continue;
    const auto &conn = donor.connections_[(size_t)conn_index];
    if (!conn.enabled)
      continue;
    donor_active_inputs[conn.out].push_back(conn_index);
    if (donor.nodeType(conn.out) == 3)
      output_edges.push_back(conn_index);
    else
      other_edges.push_back(conn_index);
  }
  std::shuffle(output_edges.begin(), output_edges.end(), rng);
  std::shuffle(other_edges.begin(), other_edges.end(), rng);

  int budget = max_extra_connections;
  std::unordered_map<int, int> visiting;
  std::function<bool(int)> addPath = [&](int conn_index) -> bool {
    if (budget <= 0 || conn_index < 0 ||
        conn_index >= (int)donor.connections_.size()) {
      return false;
    }
    const auto &conn = donor.connections_[(size_t)conn_index];
    if (!conn.enabled)
      return false;
    if (child_innovations.find(conn.innovation) != child_innovations.end() ||
        child_edges.find(edgeKey(conn.in, conn.out)) != child_edges.end()) {
      return false;
    }

    const int source_type = donor.nodeType(conn.in);
    if (source_type == 2 && budget > 1 &&
        visiting.emplace(conn.in, 1).second) {
      auto incoming = donor_active_inputs.find(conn.in);
      if (incoming != donor_active_inputs.end()) {
        std::vector<int> upstream = incoming->second;
        std::shuffle(upstream.begin(), upstream.end(), rng);
        for (int upstream_conn : upstream) {
          if (budget <= 1)
            break;
          if (addPath(upstream_conn))
            break;
        }
      }
      visiting.erase(conn.in);
    }

    if (budget <= 0)
      return false;
    auto addNode = [&](int id) {
      if (child_nodes.find(id) != child_nodes.end())
        return;
      auto node = donor_nodes.find(id);
      if (node == donor_nodes.end())
        return;
      child.nodes_.push_back(*node->second);
      child_nodes[id] = 1;
    };
    addNode(conn.in);
    addNode(conn.out);
    child.connections_.push_back(conn);
    child_innovations[conn.innovation] = 1;
    child_edges[edgeKey(conn.in, conn.out)] = 1;
    --budget;
    return true;
  };

  for (int conn_index : output_edges) {
    if (budget <= 0)
      break;
    addPath(conn_index);
  }
  for (int conn_index : other_edges) {
    if (budget <= 0)
      break;
    addPath(conn_index);
  }

  child.sortGenes();
  return child;
}

double NeatGenome::compatibility(const NeatGenome &a, const NeatGenome &b,
                                 double weight_coefficient) {
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
  return 1.0 * excess / n + 1.0 * disjoint / n +
         weight_coefficient * w;
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

  active_connection_indices_.clear();
  active_conn_in_indices_.clear();
  active_conn_out_indices_.clear();
  active_conn_weights_.clear();
  active_source_indices_.clear();
  active_source_offsets_.clear();
  active_non_input_indices_.clear();
  used_input_indices_.clear();

  std::vector<std::vector<int>> reverse_edges(nodes_.size());
  for (const auto &conn : connections_) {
    if (!conn.enabled || conn.in_index < 0 || conn.out_index < 0)
      continue;
    reverse_edges[conn.out_index].push_back(conn.in_index);
  }

  std::vector<unsigned char> reaches_output(nodes_.size(), 0);
  std::vector<int> stack;
  stack.reserve(output_indices_.size());
  for (int out_idx : output_indices_) {
    if (out_idx < 0 || out_idx >= (int)nodes_.size() || reaches_output[out_idx])
      continue;
    reaches_output[out_idx] = 1;
    stack.push_back(out_idx);
  }
  while (!stack.empty()) {
    int out = stack.back();
    stack.pop_back();
    for (int in : reverse_edges[out]) {
      if (reaches_output[in])
        continue;
      reaches_output[in] = 1;
      stack.push_back(in);
    }
  }

  for (int i = 0; i < (int)nodes_.size(); ++i) {
    if (reaches_output[i] && nodes_[i].type != 0)
      active_non_input_indices_.push_back(i);
  }

  std::vector<unsigned char> input_used(std::max(0, input_count_), 0);
  int current_source = -1;
  for (int i = 0; i < (int)connections_.size(); ++i) {
    const auto &conn = connections_[i];
    if (!conn.enabled || conn.in_index < 0 || conn.out_index < 0)
      continue;
    if (!reaches_output[conn.in_index] || !reaches_output[conn.out_index])
      continue;
    active_connection_indices_.push_back(i);
    if (conn.in_index != current_source) {
      current_source = conn.in_index;
      active_source_indices_.push_back(current_source);
      active_source_offsets_.push_back((int)active_conn_weights_.size());
    }
    active_conn_in_indices_.push_back(conn.in_index);
    active_conn_out_indices_.push_back(conn.out_index);
    active_conn_weights_.push_back(conn.weight);
    if (conn.in_index < input_count_ && !input_used[conn.in_index]) {
      input_used[conn.in_index] = 1;
      used_input_indices_.push_back(conn.in_index);
    }
  }
  active_source_offsets_.push_back((int)active_conn_weights_.size());

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

void NeatEvolution::setFitness(const int *fitness) {
  for (int i = 0; i < (int)population_.size(); ++i)
    population_[i].setFitness(fitness[i]);
}

void NeatEvolution::reservePopulation(int capacity) {
  if (capacity > (int)population_.capacity())
    population_.reserve(capacity);
}

void NeatEvolution::configure(double mutation_sigma, double mutation_prob,
                              double compatibility_threshold,
                              double survival_rate, double add_node_prob,
                              double add_connection_prob,
                              double sparse_connection_prob,
                              double input_probe_prob,
                              double sparse_input_probe_prob,
                              double compatibility_weight_coefficient,
                              int target_species_min,
                              int target_species_max) {
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

void NeatEvolution::setInputMutationPrior(const std::vector<int> &input_ids,
                                          double bias) {
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

bool NeatEvolution::writeInFile(const std::string &file,
                                int active_tests) const {
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

bool NeatEvolution::writeGenomeInFile(int index, const std::string &file,
                                      int reported_fitness) const {
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

NeatGenome NeatEvolution::makeChild(const Species &species,
                                    const std::vector<int> &global_parents,
                                    std::mt19937 &rng,
                                    const NeatLexicaseSelection *lexicase) {
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

int NeatEvolution::pickParent(const std::vector<int> &members,
                              std::mt19937 &rng,
                              const NeatLexicaseSelection *lexicase) const {
  if (members.empty())
    return bestIndex();
  if (!lexicase || !lexicase->valid())
    return tournamentPick(members, rng);

  std::uniform_real_distribution<double> uni(0.0, 1.0);
  if (uni(rng) > std::clamp(lexicase->parent_rate, 0.0, 1.0))
    return tournamentPick(members, rng);
  return lexicasePick(members, rng, *lexicase);
}

int NeatEvolution::lexicasePick(const std::vector<int> &members,
                                std::mt19937 &rng,
                                const NeatLexicaseSelection &lexicase) const {
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
