#pragma once

#include <string>
#include <vector>

class Brain;
class LiveStats;
struct TrainingSettings;

struct FitnessSummary {
  int max_fitness = 0;
  int min_fitness = 0;
  int avg_fitness = 0;
  int best_index = 0;
};

FitnessSummary summarizeFitness(const std::vector<int> &fitness);
int nextTestCount(int current);
int threadCountFromSettings(const TrainingSettings &settings);
void launchFullDemonstrationOnce(LiveStats *live, bool &started, const Brain &brain, int gen);
