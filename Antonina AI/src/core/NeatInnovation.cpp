#include "NeatInnovation.h"

NeatInnovation::NeatInnovation(int next_node_id)
    : next_node_id_(next_node_id), next_innovation_(0) {
}

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