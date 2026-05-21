#include "NeatGenome.h"
#include "NeatInnovation.h"

#include <sstream>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include <functional>
#include <numeric>
#include <algorithm>  

NeatGenome::NeatGenome()
    : input_count_(0), output_count_(0), bias_id_(-1), bias_index_(-1),
    fitness_(0.0) {
}

int NeatGenome::initialSeedInputCount(int input_count) {
    return (int)initialSeedInputIds(input_count).size();
}

std::vector<int> NeatGenome::initialSeedInputIds(int input_count) {
    (void)input_count;
    return {};
}

NeatGenome NeatGenome::minimal(int input_count, int output_count, NeatInnovation& innovation, std::mt19937& rng) {
    NeatGenome genome;
    genome.input_count_ = input_count;
    genome.output_count_ = output_count;
    genome.bias_id_ = input_count;
    genome.outputs_.assign(output_count, 0.0);

    for (int i = 0; i < input_count; ++i)
        genome.nodes_.push_back({ i, 0, 0.0, 0.0 });
    genome.nodes_.push_back({ genome.bias_id_, 1, 0.0, 1.0 });
    for (int i = 0; i < output_count; ++i)
        genome.nodes_.push_back({ input_count + 1 + i, 3, 1.0, 0.0 });

    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    const auto seed_inputs = initialSeedInputIds(input_count);
    for (int out = 0; out < output_count; ++out) {
        int out_id = input_count + 1 + out;
        genome.connections_.push_back(
            { genome.bias_id_, out_id,
             innovation.connectionInnovation(genome.bias_id_, out_id), dist(rng),
             true });
        for (int in_id : seed_inputs) {
            genome.connections_.push_back(
                { in_id, out_id, innovation.connectionInnovation(in_id, out_id),
                 dist(rng), true });
        }
    }
    genome.sortGenes();
    return genome;
}

void NeatGenome::feedForward(double* inputs) {
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

void NeatGenome::feedForwardBatch(const double* batch_inputs, double* batch_outputs, int B) const {
    if (B <= 0)
        return;

    static thread_local std::vector<const double*> input_rows;
    input_rows.resize(B);
    for (int b = 0; b < B; ++b)
        input_rows[b] = batch_inputs + (size_t)b * input_count_;
    feedForwardBatchRows(input_rows.data(), batch_outputs, B);
}

void NeatGenome::feedForwardBatchRows(const double* const* batch_inputs, double* batch_outputs, int B) const {
    if (B <= 0 || nodes_.empty())
        return;

    static thread_local std::vector<double> values;
    static thread_local std::vector<double> activated;

    const size_t total_values = (size_t)nodes_.size() * B;
    if (values.size() < total_values)
        values.resize(total_values);
    activated.resize(B);

    for (int idx : active_non_input_indices_) {
        double* row = values.data() + (size_t)idx * B;
        std::fill(row, row + B, 0.0);
    }

    for (int idx : used_input_indices_) {
        double* row = values.data() + (size_t)idx * B;
        for (int b = 0; b < B; ++b)
            row[b] = batch_inputs[b][idx];
    }

    if (bias_index_ >= 0) {
        double* bias_row = values.data() + (size_t)bias_index_ * B;
        for (int b = 0; b < B; ++b)
            bias_row[b] = 1.0;
    }

    for (int group = 0; group < (int)active_source_indices_.size(); ++group) {
        int idx = active_source_indices_[group];
        double* row = values.data() + (size_t)idx * B;
        const double* out_row = row;
        if (nodes_[idx].type == 2) {
            for (int b = 0; b < B; ++b)
                activated[b] = activate(row[b]);
            out_row = activated.data();
        }

        int begin = active_source_offsets_[group];
        int end = active_source_offsets_[group + 1];
        for (int pos = begin; pos < end; ++pos) {
            double* dst = values.data() + (size_t)active_conn_out_indices_[pos] * B;
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

void NeatGenome::getOutBatch(const double* batch_outputs, int* out_args, int B) const {
    for (int b = 0; b < B; ++b) {
        const double* out = batch_outputs + (size_t)b * output_count_;
        int best = 0;
        for (int i = 1; i < output_count_; ++i)
            if (out[i] > out[best])
                best = i;
        out_args[b] = best;
    }
}

void NeatGenome::mutateWeights(double sigma, double prob, std::mt19937& rng) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_real_distribution<double> reset(-1.0, 1.0);
    std::normal_distribution<double> perturb(0.0, sigma);
    for (auto& conn : connections_) {
        if (uni(rng) > prob)
            continue;
        if (uni(rng) < 0.1)
            conn.weight = reset(rng);
        else
            conn.weight = std::clamp(conn.weight + perturb(rng), -5.0, 5.0);
    }
}

void NeatGenome::writeInFile(std::ofstream& fout) const {
    fout << "GENOME " << fitness_ << ' ' << input_count_ << ' ' << output_count_
        << ' ' << bias_id_ << ' ' << nodes_.size() << ' ' << connections_.size()
        << '\n';
    for (const auto& node : nodes_)
        fout << "NODE " << node.id << ' ' << node.type << ' ' << node.level << '\n';
    for (const auto& conn : connections_)
        fout << "CONN " << conn.in << ' ' << conn.out << ' ' << conn.innovation
        << ' ' << conn.weight << ' ' << (conn.enabled ? 1 : 0) << '\n';
}

bool NeatGenome::readInFile(std::ifstream& fin) {
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

    for (auto& node : nodes_) {
        if (node.id >= old_input_count)
            node.id += delta;
    }
    for (auto& conn : connections_) {
        if (conn.in >= old_input_count)
            conn.in += delta;
        if (conn.out >= old_input_count)
            conn.out += delta;
    }

    for (int id = old_input_count; id < new_input_count; ++id)
        nodes_.push_back({ id, 0, 0.0, 0.0 });

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
        by_id[id] = { id, 0, 0.0, 0.0 };
    by_id[bias_id_] = { bias_id_, 1, 0.0, 1.0 };
    for (int i = 0; i < output_count_; ++i)
        by_id[first_output + i] = { first_output + i, 3, 1.0, 0.0 };

    nodes_.clear();
    nodes_.reserve(by_id.size());
    for (const auto& item : by_id)
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

    std::unordered_map<int, const NeatNodeGene*> by_id;
    by_id.reserve(nodes_.size() * 2);
    for (const auto& node : nodes_) {
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
    for (const auto& conn : connections_) {
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
    for (auto& conn : connections_) {
        if (conn.enabled && conn.in == input_id) {
            conn.enabled = false;
            changed = true;
        }
    }
    if (changed)
        sortGenes();
    return changed;
}

bool NeatGenome::mutateAddConnection(NeatInnovation& innovation, std::mt19937& rng) {
    if (nodes_.size() < 2)
        return false;

    std::uniform_int_distribution<int> dist(0, (int)nodes_.size() - 1);
    std::uniform_real_distribution<double> weight_dist(-1.0, 1.0);

    for (int attempt = 0; attempt < 64; ++attempt) {
        const auto& a = nodes_[dist(rng)];
        const auto& b = nodes_[dist(rng)];
        if (a.id == b.id)
            continue;
        if (a.type == 3 || b.type == 0 || b.type == 1)
            continue;
        if (a.level >= b.level)
            continue;
        if (hasConnection(a.id, b.id))
            continue;

        connections_.push_back({ a.id, b.id,
                                innovation.connectionInnovation(a.id, b.id),
                                weight_dist(rng), true });
        sortGenes();
        return true;
    }
    return false;
}

bool NeatGenome::mutateAddInputConnection(
    NeatInnovation& innovation, std::mt19937& rng,
    const std::vector<int>* preferred_inputs) {
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
        const auto& target = nodes_[targets[target_dist(rng)]];
        if (nodeLevel(in_id) >= target.level)
            continue;
        if (hasConnection(in_id, target.id))
            continue;

        connections_.push_back(
            { in_id, target.id, innovation.connectionInnovation(in_id, target.id),
             weight_dist(rng), true });
        sortGenes();
        return true;
    }
    return false;
}

bool NeatGenome::mutateAddNode(NeatInnovation& innovation, std::mt19937& rng) {
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
    nodes_.push_back({ node_id, 2, level, 0.0 });
    connections_.push_back({ old.in, node_id,
                            innovation.connectionInnovation(old.in, node_id), 1.0,
                            true });
    connections_.push_back({ node_id, old.out,
                            innovation.connectionInnovation(node_id, old.out),
                            old.weight, true });
    sortGenes();
    return true;
}

NeatGenome NeatGenome::crossover(const NeatGenome& a, const NeatGenome& b, std::mt19937& rng) {
    const NeatGenome* fit = &a;
    const NeatGenome* other = &b;
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
    for (const auto& conn : other->connections_)
        other_genes[conn.innovation] = conn;

    std::uniform_int_distribution<int> coin(0, 1);
    for (const auto& conn : fit->connections_) {
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

NeatGenome NeatGenome::crossoverGraft(const NeatGenome& base, const NeatGenome& donor, int max_extra_connections, std::mt19937& rng) {
    NeatGenome child = base;
    max_extra_connections = std::max(0, max_extra_connections);
    if (max_extra_connections == 0 || donor.active_connection_indices_.empty()) {
        child.sortGenes();
        return child;
    }

    std::unordered_map<int, int> child_nodes;
    child_nodes.reserve(child.nodes_.size() * 2 + 16);
    for (const auto& node : child.nodes_)
        child_nodes[node.id] = 1;

    std::unordered_map<int, int> child_innovations;
    child_innovations.reserve(child.connections_.size() * 2 + 16);
    std::unordered_map<long long, int> child_edges;
    child_edges.reserve(child.connections_.size() * 2 + 16);
    auto edgeKey = [](int in, int out) {
        return (long long)(unsigned int)in << 32 | (unsigned int)out;
        };
    for (const auto& conn : child.connections_) {
        child_innovations[conn.innovation] = 1;
        child_edges[edgeKey(conn.in, conn.out)] = 1;
    }

    std::unordered_map<int, const NeatNodeGene*> donor_nodes;
    donor_nodes.reserve(donor.nodes_.size() * 2);
    for (const auto& node : donor.nodes_)
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
        const auto& conn = donor.connections_[(size_t)conn_index];
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
        const auto& conn = donor.connections_[(size_t)conn_index];
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

double NeatGenome::compatibility(const NeatGenome& a, const NeatGenome& b, double weight_coefficient) {
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
        const auto& ca = a.connections_[a.innovation_order_[i]];
        const auto& cb = b.connections_[b.innovation_order_[j]];
        if (ca.innovation == cb.innovation) {
            ++matching;
            weight_diff += std::abs(ca.weight - cb.weight);
            ++i;
            ++j;
        }
        else if (ca.innovation < cb.innovation) {
            if (ca.innovation > max_b)
                ++excess;
            else
                ++disjoint;
            ++i;
        }
        else {
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
    for (const auto& node : nodes_)
        if (node.id == id)
            return node.type;
    return -1;
}

double NeatGenome::nodeLevel(int id) const {
    for (const auto& node : nodes_)
        if (node.id == id)
            return node.level;
    return 0.0;
}

bool NeatGenome::hasConnection(int in, int out) const {
    for (const auto& conn : connections_)
        if (conn.in == in && conn.out == out)
            return true;
    return false;
}

void NeatGenome::sortGenes() {
    std::sort(nodes_.begin(), nodes_.end(), [](const auto& a, const auto& b) {
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

    for (auto& conn : connections_) {
        conn.in_index = indexOf(conn.in);
        conn.out_index = indexOf(conn.out);
    }

    bias_index_ = indexOf(bias_id_);
    output_indices_.assign(output_count_, -1);
    for (int i = 0; i < output_count_; ++i)
        output_indices_[i] = indexOf(input_count_ + 1 + i);

    std::sort(connections_.begin(), connections_.end(),
        [](const auto& a, const auto& b) {
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
    for (const auto& conn : connections_) {
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
        const auto& conn = connections_[i];
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
