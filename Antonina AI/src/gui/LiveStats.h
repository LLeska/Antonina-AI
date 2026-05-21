#pragma once

#include "Brain.h"
#include "TrainingSettings.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct GenSample {
  int gen;
  int max_fitness;
  int min_fitness;
  int avg_fitness;
  int active_tests;
  int population;
  int population_target;
  int max_population;
  int best_wins;
  int species_count;
  double avg_nodes;
  double avg_connections;
  double avg_active_connections;
  double compatibility_threshold;
  int fitness_ms;
  int evolution_ms;
  double epsilon;
  double mut_prob;
  int stagnation;
  int weighted_tests = 0;
  int max_test_weight_x100 = 100;
};

class LiveStats {
public:
  LiveStats() : has_best(false), running(true), save_pending(false), training_started(false), start_requested(false) {}

  void setDefaultSettings(const TrainingSettings &settings) {
    std::lock_guard<std::mutex> lk(control_mu);
    default_settings = settings;
  }

  TrainingSettings getDefaultSettings() const {
    std::lock_guard<std::mutex> lk(control_mu);
    return default_settings;
  }

  void requestStart(const TrainingSettings &settings, const std::string &load_file = "") {
    {
      std::lock_guard<std::mutex> lk(control_mu);
      start_settings = settings;
      start_load_file = load_file;
      start_requested = true;
      training_started.store(true);
    }
    start_cv.notify_all();
  }

  bool waitForStart(TrainingSettings &settings, std::string *load_file = nullptr) {
    std::unique_lock<std::mutex> lk(control_mu);
    start_cv.wait(lk, [this] { return start_requested || !running.load(); });
    if (!running.load() || !start_requested)
      return false;
    settings = start_settings;
    if (load_file)
      *load_file = start_load_file;
    return true;
  }

  bool isTrainingStarted() const { return training_started.load(); }

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

  void publishInputDiagnostics(const std::vector<std::string> &lines) {
    std::lock_guard<std::mutex> lk(mu);
    input_diagnostics = lines;
  }

  void snapshotInputDiagnostics(std::vector<std::string> &dst) {
    std::lock_guard<std::mutex> lk(mu);
    dst = input_diagnostics;
  }

  void appendLog(const std::string &line) {
    if (line.empty())
      return;
    std::lock_guard<std::mutex> lk(log_mu);
    if ((int)logs.size() >= MAX_LOG_LINES) {
      logs.erase(logs.begin(), logs.begin() + MAX_LOG_LINES / 4);
    }
    logs.push_back(line);
  }

  void snapshotLogs(std::vector<std::string> &dst) {
    std::lock_guard<std::mutex> lk(log_mu);
    dst = logs;
  }

  void requestStop() {
    running.store(false);
    start_cv.notify_all();
  }
  bool isRunning() const { return running.load(); }

private:
  static constexpr int MAX_SAMPLES = 4096;
  static constexpr int MAX_LOG_LINES = 2048;

  std::mutex mu;
  std::vector<GenSample> samples;
  std::unique_ptr<Brain> best;
  std::vector<std::string> input_diagnostics;
  bool has_best;
  int best_version = 0;
  std::atomic<bool> running;
  std::atomic<bool> save_pending;

  mutable std::mutex control_mu;
  std::condition_variable start_cv;
  TrainingSettings default_settings;
  TrainingSettings start_settings;
  std::string start_load_file;
  std::atomic<bool> training_started;
  bool start_requested;

  std::mutex log_mu;
  std::vector<std::string> logs;
};
