#pragma once

#include <map>

class NeatInnovation {
public:
	NeatInnovation(int next_node_id = 0);
	int connectionInnovation(int in, int out);
	int nodeInnovation(int in, int out);
	void reset(int next_node_id = 0);
	void observeNodeId(int id);
	void observeConnection(int in, int out, int innovation);

private:
	int next_node_id_;
	int next_innovation_;
	std::map<std::pair<int, int>, int> connections_;
	std::map<std::pair<int, int>, int> split_nodes_;
};
