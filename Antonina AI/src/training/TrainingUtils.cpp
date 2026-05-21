#include "TrainingUtils.h"

#include "AntoninaAPI.h"
#include "Brain.h"
#include "LiveStats.h"
#include "TrainingSettings.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <thread>

FitnessSummary summarizeFitness(const std::vector<int> &fitness) {
  FitnessSummary summary;
  if (fitness.empty())
    return summary;

  auto minmax = std::minmax_element(fitness.begin(), fitness.end());
  summary.min_fitness = *minmax.first;
  summary.max_fitness = *minmax.second;
  summary.best_index = (int)(std::max_element(fitness.begin(), fitness.end()) - fitness.begin());
  long long total = 0;
  for (int value : fitness)
    total += value;
  summary.avg_fitness = (int)(total / (long long)fitness.size());
  return summary;
}

int nextTestCount(int current) {
  if (current < 5)
    return 5;
  if (current < 20)
    return current + 10;
  return current + 50;
}

int threadCountFromSettings(const TrainingSettings &settings) {
  const unsigned hw_threads = std::thread::hardware_concurrency();
  return settings.requested_threads > 0 ? settings.requested_threads : std::max(1u, hw_threads);
}

void launchFullDemonstrationOnce(LiveStats *live, bool &started, const Brain &brain, int gen) {
  if (started)
    return;
  started = true;
  if (live)
    live->appendLog("viewer: full mastery reached, starting original Antonina demonstration");
  std::cout << "[demonstration] gen=" << gen << " starting AntoninaAPI::demonstrate once" << '\n';

  std::unique_ptr<Brain> demo_brain = brain.cloneBrain();
  std::thread([demo_brain = std::move(demo_brain)]() mutable {
    auto demo_api = std::make_unique<AntoninaAPI>();
    demo_api->demonstrate(demo_brain.get());
  }).detach();
}
