#pragma once

#include <memory>
#include <string>

class Brain;

class AITrainingAlgorithm {
public:
  virtual ~AITrainingAlgorithm() = default;

  virtual int train() = 0;
  virtual std::unique_ptr<Brain> getBestBrain() const = 0;
  virtual bool save(const std::string &file) const = 0;
  virtual bool load(const std::string &file) = 0;
  virtual int getGeneration() const = 0;
  virtual int getBestFitness() const = 0;
};

