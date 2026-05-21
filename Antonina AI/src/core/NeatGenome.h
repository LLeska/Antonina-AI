#pragma once

#include "Brain.h"

#include <random>
#include <vector>

class NeatInnovation;

struct NeatNodeGene {
    int id = 0;
    int type = 0;
    double level = 0.0;
    double value = 0.0;
};

struct NeatConnectionGene {
    int in = 0;
    int out = 0;
    int innovation = 0;
    double weight = 0.0;
    bool enabled = true;
    int in_index = -1;
    int out_index = -1;
};

struct NeatExecutionPlan {
    const int* conn_in_indices = nullptr;
    const int* conn_out_indices = nullptr;
    const double* conn_weights = nullptr;
    int connection_count = 0;

    const int* source_indices = nullptr;
    const int* source_offsets = nullptr;
    int source_count = 0;

    const int* non_input_indices = nullptr;
    int non_input_count = 0;

    const int* used_input_indices = nullptr;
    int used_input_count = 0;

    const int* output_indices = nullptr;
    int node_count = 0;
    int input_count = 0;
    int output_count = 0;
};

class NeatGenome : public Brain {
public:
    NeatGenome();
    static NeatGenome minimal(int input_count, int output_count, NeatInnovation& innovation, std::mt19937& rng);

    void feedForward(double* inputs) override;
    int getOut() override;
    double outputValue(int index) const override;
    std::unique_ptr<Brain> cloneBrain() const override;
    void feedForwardBatch(const double* batch_inputs, double* batch_outputs, int B) const;
    void feedForwardBatchRows(const double* const* batch_inputs, double* batch_outputs, int B) const;
    void getOutBatch(const double* batch_outputs, int* out_args, int B) const;

    void mutateWeights(double sigma, double prob, std::mt19937& rng);
    bool mutateAddInputConnection(NeatInnovation& innovation, std::mt19937& rng, const std::vector<int>* preferred_inputs = nullptr);
    bool mutateAddConnection(NeatInnovation& innovation, std::mt19937& rng);
    bool mutateAddNode(NeatInnovation& innovation, std::mt19937& rng);

    static NeatGenome crossover(const NeatGenome& a, const NeatGenome& b, std::mt19937& rng);
    static NeatGenome crossoverGraft(const NeatGenome& base, const NeatGenome& donor, int max_extra_connections, std::mt19937& rng);
    static double compatibility(const NeatGenome& a, const NeatGenome& b, double weight_coefficient = 0.4);

    int inputCount() const { return input_count_; }
    int outputCount() const { return output_count_; }
    int getInputSize() const { return input_count_; }
    int getOutputSize() const { return output_count_; }
    int nodeCount() const { return (int)nodes_.size(); }
    int connectionCount() const { return (int)connections_.size(); }
    int activeConnectionCount() const { return (int)active_conn_weights_.size(); }
    int activeSourceCount() const { return (int)active_source_indices_.size(); }
    int activeNonInputNodeCount() const { return (int)active_non_input_indices_.size(); }
    int enabledConnectionCount() const {
        int count = 0;
        for (const auto& conn : connections_)
            if (conn.enabled)
                ++count;
        return count;
    }
    const std::vector<NeatNodeGene>& nodes() const { return nodes_; }
    const std::vector<NeatConnectionGene>& connections() const { return connections_; }
    NeatExecutionPlan executionPlan() const;
    bool structurallyValid() const;
    bool disableInputConnections(int input_id);
    double fitness() const { return fitness_; }
    void setFitness(double value) { fitness_ = value; }
    void writeInFile(std::ofstream& fout) const;
    bool readInFile(std::ifstream& fin);
    bool upgradeInputCount(int new_input_count);
    static int initialSeedInputCount(int input_count);
    static std::vector<int> initialSeedInputIds(int input_count);

private:
    bool repairLoadedStructure();

    int input_count_;
    int output_count_;
    int bias_id_;
    int bias_index_;
    double fitness_;
    std::vector<NeatNodeGene> nodes_;
    std::vector<NeatConnectionGene> connections_;
    std::vector<int> active_connection_indices_;
    std::vector<int> active_conn_in_indices_;
    std::vector<int> active_conn_out_indices_;
    std::vector<double> active_conn_weights_;
    std::vector<int> active_source_indices_;
    std::vector<int> active_source_offsets_;
    std::vector<int> active_non_input_indices_;
    std::vector<int> used_input_indices_;
    std::vector<int> innovation_order_;
    std::vector<int> output_indices_;
    std::vector<double> outputs_;
    int nodeType(int id) const;
    double nodeLevel(int id) const;
    bool hasConnection(int in, int out) const;
    void sortGenes();
    static double activate(double x);
};

