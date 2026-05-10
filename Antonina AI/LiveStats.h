#pragma once

#include "Brain.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

struct GenSample {
  int gen;
  int max_fitness;
  int min_fitness;
  int avg_fitness;
  int active_tests;
  int best_wins;
  int species_count;
  double avg_nodes;
  double avg_connections;
  double compatibility_threshold;
  int fitness_ms;
  double epsilon;
  double mut_prob;
  int stagnation;
};

class LiveStats {
public:
  LiveStats() : has_best(false), running(true), save_pending(false) {}

  void requestSave() { save_pending.store(true); }
  bool consumeSaveRequest() {
    bool expected = true;
    return save_pending.compare_exchange_strong(expected, false);
  }

  void pushSample(const GenSample &s) {
    std::lock_guard<std::mutex> lk(mu);
    if ((int)samples.size() >= MAX_SAMPLES) {

      std::vector<GenSample> compact;
      compact.reserve(MAX_SAMPLES / 2 + 1);
      for (size_t i = 0; i < samples.size(); i += 2)
        compact.push_back(samples[i]);
      samples.swap(compact);
    }
    samples.push_back(s);
  }

  void publishBest(const Brain &src) {
    std::lock_guard<std::mutex> lk(mu);
    best = src.cloneBrain();
    has_best = true;
    ++best_version;
  }

  bool snapshotBest(std::unique_ptr<Brain> &dst, int &version_out) {
    std::lock_guard<std::mutex> lk(mu);
    if (!has_best || !best)
      return false;
    dst = best->cloneBrain();
    version_out = best_version;
    return true;
  }

  bool snapshotBestIfNew(std::unique_ptr<Brain> &dst, int &known_version) {
    std::lock_guard<std::mutex> lk(mu);
    if (!has_best || !best || known_version == best_version)
      return false;
    dst = best->cloneBrain();
    known_version = best_version;
    return true;
  }

  void snapshotSamples(std::vector<GenSample> &dst) {
    std::lock_guard<std::mutex> lk(mu);
    dst = samples;
  }

  void requestStop() { running.store(false); }
  bool isRunning() const { return running.load(); }

private:
  static constexpr int MAX_SAMPLES = 4096;

  std::mutex mu;
  std::vector<GenSample> samples;
  std::unique_ptr<Brain> best;
  bool has_best;
  int best_version = 0;
  std::atomic<bool> running;
  std::atomic<bool> save_pending;
};
