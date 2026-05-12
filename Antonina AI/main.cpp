#include "NeatEvolution.h"
#include "AntoninaAPI.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#ifndef NO_GUI
#include "LiveStats.h"
#include "Viewer.h"
#endif

class ThreadPool {
public:
  explicit ThreadPool(size_t num_threads) : stop_(false), active_tasks_(0) {
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] {
        for (;;) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty())
              return;
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
          {
            std::unique_lock<std::mutex> lock(done_mutex_);
            --active_tasks_;
          }
          done_cv_.notify_one();
        }
      });
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &w : workers_)
      w.join();
  }

  void enqueue(std::function<void()> task) {
    {
      std::unique_lock<std::mutex> lock(done_mutex_);
      ++active_tasks_;
    }
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      tasks_.push(std::move(task));
    }
    cv_.notify_one();
  }

  void wait_all() {
    std::unique_lock<std::mutex> lock(done_mutex_);
    done_cv_.wait(lock, [this] { return active_tasks_ == 0; });
  }

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_, done_mutex_;
  std::condition_variable cv_, done_cv_;
  std::atomic<int> active_tasks_;
  bool stop_;
};

int main(int argc, char **argv) {
  std::ios_base::sync_with_stdio(false);
  std::cout.tie(nullptr);
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  int population = 5000;
  int requested_threads = 0;
  bool enable_gui = true;
  const int input_count = AntoninaAPI::INPUT_FEATURES;
  std::string load_file;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--load" && i + 1 < argc) {
      load_file = argv[++i];
    } else if (arg == "--population" && i + 1 < argc) {
      population = std::max(1, std::atoi(argv[++i]));
    } else if (arg == "--threads" && i + 1 < argc) {
      requested_threads = std::max(1, std::atoi(argv[++i]));
    } else if (arg == "--no-gui") {
      enable_gui = false;
    } else if (load_file.empty()) {
      load_file = arg;
    }
  }

  std::cout << "Antonina AI NEAT starting: population=" << population
            << " inputs=" << input_count << " outputs=4";
  if (!load_file.empty())
    std::cout << " checkpoint=" << load_file;
  std::cout << '\n';

  NeatEvolution ne(input_count, 4, load_file.empty() ? population : 0);
  int loaded_active_tests = 1;
  if (!load_file.empty()) {
    if (!ne.readInFile(load_file, &loaded_active_tests)) {
      std::cerr << "failed to load NEAT checkpoint: " << load_file << '\n';
      return 1;
    }
    std::cout << "Loaded NEAT checkpoint from " << load_file
              << " | generation=" << ne.getGeneration()
              << " population=" << ne.getPopulation()
              << " active_tests=" << loaded_active_tests << '\n';
  }

#ifndef NO_GUI
  std::unique_ptr<LiveStats> live;
  std::unique_ptr<AntoninaAPI> anim_api;
  std::thread gui_thread;
  if (enable_gui) {
    live = std::make_unique<LiveStats>();
    anim_api = std::make_unique<AntoninaAPI>();
    gui_thread = std::thread(viewerThread, std::ref(*live),
                             std::ref(*anim_api));
  }
#endif

  int start = ne.getGeneration();

  population = ne.getPopulation();

  const unsigned hw_threads = std::thread::hardware_concurrency();
  const int num_threads =
      requested_threads > 0 ? requested_threads : std::max(1u, hw_threads);
  std::cout << "Training threads=" << num_threads;
#ifndef NO_GUI
  std::cout << " gui=" << (enable_gui ? "on" : "off");
#endif
  std::cout << '\n';
  ThreadPool pool(num_threads);

  std::vector<AntoninaAPI> envs(num_threads);

  const int MAX_TESTS = AntoninaAPI::ALL_TESTS;
  const int MIN_GEN_BETWEEN_PROMOTIONS = 0;
#ifdef _DEBUG
  const int LOG_EVERY = 1;
  const bool LOG_DEBUG_STAGES = true;
#else
  const int LOG_EVERY = 100;
  const bool LOG_DEBUG_STAGES = false;
#endif

  for (auto &env : envs)
    env.active_tests = std::clamp(loaded_active_tests, 1, MAX_TESTS);
  int last_promotion_gen = start - MIN_GEN_BETWEEN_PROMOTIONS;

  auto nextTestCount = [](int current) {
    if (current < 5)
      return 5;
    if (current < 20)
      return current + 5;
    return current + 10;
  };

  std::vector<int> fitness(population, 0);
  bool training_complete = false;
  int saved_best_fitness = -2147483647;
  std::vector<int> last_failure_log;
  int last_failure_log_gen = start - 1000;
  const int FAILURE_LOG_LIMIT = 16;
  const int FAILURE_LOG_INTERVAL = 25;

  for (int gen = start; gen < 1000001 + start; gen++) {
    auto gen_begin = std::chrono::steady_clock::now();
    if (LOG_DEBUG_STAGES && gen < 10)
      std::cout << "gen " << gen << ": fitness start" << '\n';

    std::fill(fitness.begin(), fitness.end(), 0);
    int *fitness_ptr = fitness.data();
    const int job_count = std::min(population, num_threads * 16);
    const int chunk_size = (population + job_count - 1) / job_count;
    for (int job = 0; job < job_count; job++) {
      const int begin = job * chunk_size;
      const int end = std::min(population, begin + chunk_size);
      if (begin >= end)
        continue;
      pool.enqueue([&ne, &envs, fitness_ptr, begin, end, job, num_threads]() {
        AntoninaAPI &env = envs[job % num_threads];
        for (int i = begin; i < end; i++)
          fitness_ptr[i] = env.solveFitnessBatch(ne.getGenome(i), 0);
      });
    }
    pool.wait_all();
    auto fitness_done = std::chrono::steady_clock::now();
    ne.setFitness(fitness_ptr);

    int max_fitness = *std::max_element(fitness.begin(), fitness.end());
    int min_fitness = *std::min_element(fitness.begin(), fitness.end());
    long long avg_ll = 0;
    for (int f : fitness)
      avg_ll += f;
    int avg_fitness = (int)(avg_ll / population);
    int best_idx = (int)(std::max_element(fitness.begin(), fitness.end()) -
                         fitness.begin());
    NeatGenome *best = ne.getGenome(best_idx);
    Brain *best_brain = static_cast<Brain *>(best);

    int cur_tests = envs[0].active_tests;
    int best_wins = 0;
    bool promote_after_evolution = false;
    int promotion_target = cur_tests;
    if (gen - last_promotion_gen >= MIN_GEN_BETWEEN_PROMOTIONS) {
      std::vector<int> failures;
      int wins = envs[0].collectFailures(best_brain, cur_tests, failures,
                                         FAILURE_LOG_LIMIT);
      best_wins = wins;
      if (wins == cur_tests) {
        std::cout << "[mastery] gen=" << gen << " wins=" << wins << '/'
                  << cur_tests << '\n';
        if (cur_tests >= MAX_TESTS) {
          std::filesystem::create_directories("models");
          std::string final_file = "models/best_final.csv";
          std::string gen_file =
              "models/best_final_gen_" + std::to_string(gen) + ".csv";
          std::string checkpoint_file =
              "models/population_final_gen_" + std::to_string(gen) + ".csv";
          bool saved_final = ne.writeBestInFile(final_file);
          bool saved_gen = ne.writeBestInFile(gen_file);
          bool saved_checkpoint = ne.writeInFile(checkpoint_file, cur_tests);
          std::cout << "[done] gen=" << gen << " wins=" << wins << '/'
                    << MAX_TESTS << " fitness=" << max_fitness << '\n';
          if (saved_final && saved_gen && saved_checkpoint) {
            std::cout << "[done] saved best model to " << final_file << " and "
                      << gen_file << "; checkpoint to " << checkpoint_file
                      << '\n';
          } else {
            std::cout << "[done] failed to save final model files" << '\n';
          }
          training_complete = true;
        } else {
          promotion_target = std::min(nextTestCount(cur_tests), MAX_TESTS);
          promote_after_evolution = promotion_target > cur_tests;
        }
      } else {
        if (failures != last_failure_log ||
            gen - last_failure_log_gen >= FAILURE_LOG_INTERVAL) {
          std::cout << "[failures] gen=" << gen << " wins=" << wins << '/'
                    << cur_tests << " tests:";
          for (int idx : failures)
            std::cout << ' ' << idx;
          std::cout << '\n';
          last_failure_log = failures;
          last_failure_log_gen = gen;
        }
      }
    }

    if (training_complete)
      break;

    if (max_fitness > saved_best_fitness) {
      std::filesystem::create_directories("models");
      if (ne.writeBestInFile("models/best_latest.csv"))
        saved_best_fitness = max_fitness;
    }

    auto fitness_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          fitness_done - gen_begin)
                          .count();

    if (gen % LOG_EVERY == 0) {
      std::cout << "gen " << gen << " | max: " << max_fitness
                << " min: " << min_fitness << " avg: " << avg_fitness
                << " | tests: " << envs[0].active_tests
                << " | species: " << ne.getSpeciesCount()
                << " | topo: " << (int)ne.getAvgNodeCount() << "n/"
                << (int)ne.getAvgConnectionCount() << "c"
                << " | compat: " << ne.getCompatibilityThreshold()
                << " | fitness_ms: " << fitness_ms << '\n';
    }

#ifndef NO_GUI

    if (live) {
      GenSample gs{gen,
                   max_fitness,
                   min_fitness,
                   avg_fitness,
                   envs[0].active_tests,
                   best_wins,
                   ne.getSpeciesCount(),
                   ne.getAvgNodeCount(),
                   ne.getAvgConnectionCount(),
                   ne.getCompatibilityThreshold(),
                   (int)fitness_ms,
                   ne.getMutationSigma(),
                   ne.getMutationProb(),
                   ne.getStagnation()};
      live->pushSample(gs);

      live->publishBest(*ne.getGenome(best_idx));
    }
#endif

    if (LOG_DEBUG_STAGES && gen < 10)
      std::cout << "gen " << gen << ": evolution start" << '\n';
    auto evolution_begin = std::chrono::steady_clock::now();
    ne.evolution();
    auto evolution_done = std::chrono::steady_clock::now();
    if (LOG_DEBUG_STAGES && gen < 10) {
      auto evolution_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              evolution_done - evolution_begin)
                              .count();
      std::cout << "gen " << gen
                << ": evolution done | evolution_ms: " << evolution_ms << '\n';
    }

    if (promote_after_evolution) {
      for (auto &env : envs)
        env.active_tests = promotion_target;
      ne.resetBestFitness();
      last_promotion_gen = gen;
      std::cout << "[mastery] gen=" << gen << " promoted " << cur_tests
                << " -> " << promotion_target << " tests" << '\n';
    }

    if (gen > start && gen % 1000 == 0) {
      std::string best_file = "models/best_gen_" + std::to_string(gen) + ".csv";
      std::cout << "saving " << best_file << "..." << '\n';
      ne.writeBestInFile(best_file);
      std::cout << "saved " << best_file << '\n';
    }

#ifndef NO_GUI

    if (live && live->consumeSaveRequest()) {
      std::filesystem::create_directories("models");
      std::string checkpoint_file = "models/population_manual_gen_" +
                                    std::to_string(ne.getGeneration()) +
                                    ".csv";
      std::cout << "saving " << checkpoint_file << "..." << '\n';
      if (ne.writeInFile(checkpoint_file, envs[0].active_tests))
        std::cout << "saved " << checkpoint_file << '\n';
      else
        std::cout << "failed to save " << checkpoint_file << '\n';
    }
#endif
#ifndef NO_GUI

    if (live && !live->isRunning())
      break;
#endif
  }

#ifndef NO_GUI
  if (live)
    live->requestStop();
  if (gui_thread.joinable())
    gui_thread.join();
#endif
  return 0;
}
