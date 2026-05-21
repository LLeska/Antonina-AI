#pragma once

#include "AITrainingAlgorithm.h"
#include "NeatEvolution.h"
#include "NeuroEvolution.h"
#include "TrainingSettings.h"

#include <memory>
#include <string>

class LiveStats;

inline std::unique_ptr<AITrainingAlgorithm> createTrainingAlgorithm(const TrainingSettings &settings, LiveStats *live, const std::string &load_file) {
  if (isNeatAlgorithm(settings.algorithm))
    return std::make_unique<NeatEvolution>(settings, live, load_file);
  return std::make_unique<NeuroEvolution>(settings, live, load_file);
}
