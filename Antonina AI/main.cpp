#include <iostream>
#include <string>
#include <thread>
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <algorithm>
#include "AntoninaAPI.h"
#include "NeuroEvolution.h"


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
                        if (stop_ && tasks_.empty()) return;
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
        for (auto& w : workers_) w.join();
    }

    void enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.push(std::move(task));
        }
        {
            std::unique_lock<std::mutex> lock(done_mutex_);
            ++active_tasks_;
        }
        cv_.notify_one();
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(done_mutex_);
        done_cv_.wait(lock, [this] { return active_tasks_ == 0; });
    }

private:
    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex                        queue_mutex_, done_mutex_;
    std::condition_variable           cv_, done_cv_;
    std::atomic<int>                  active_tasks_;
    bool                              stop_;
};


int main() {
    std::ios_base::sync_with_stdio(false);
    std::cout.tie(nullptr);

    double learning_rate = 0.01;
    int length = 5;
    std::unique_ptr<int[]> sizes(new int[length] { 64, 32, 16, 8, 4 });
    int parent_size = 100;
    int population = 1000;

    NeuroEvolution ne(learning_rate, length, sizes.get(), parent_size, population);

    int start = 8000;
    ne.readFromFile("models/gen_" + std::to_string(start) + ".csv");
    population = ne.getPopulation();

    const int num_threads = std::thread::hardware_concurrency();
    ThreadPool pool(num_threads);

    std::vector<AntoninaAPI> envs(num_threads);

    const int CURRICULUM_GENS = 50;
    const int CURRICULUM_STEP = 30;
    const int MAX_TESTS = 360;

    for (auto& env : envs) env.active_tests = CURRICULUM_STEP;

    for (int gen = start; gen < 1000001 + start; gen++) {

        if (gen > start && (gen - start) % CURRICULUM_GENS == 0) {
            for (auto& env : envs)
                env.active_tests = std::min(env.active_tests + CURRICULUM_STEP, MAX_TESTS);
            std::cout << "active_tests=" << envs[0].active_tests << '\n';
        }

        std::vector<int> fitness(population, 0);
        int* fitness_ptr = fitness.data();

        for (int i = 0; i < population; i++) {
            Perceptron* neuron = ne.getNeuros(i);
            pool.enqueue([neuron, &envs, fitness_ptr, i, num_threads]() {
                fitness_ptr[i] = envs[i % num_threads].solveFitness(neuron, 0);
                });
        }
        pool.wait_all();

        ne.setFitness(fitness_ptr);

        if (gen % 100 == 0) {
            int max_fitness = *std::max_element(fitness.begin(), fitness.end());
            int min_fitness = *std::min_element(fitness.begin(), fitness.end());
            long long avg = 0;
            for (int f : fitness) avg += f;
            avg /= population;

            std::cout << "gen " << gen
                << " | max: " << max_fitness
                << " min: " << min_fitness
                << " avg: " << avg
                << " | tests: " << envs[0].active_tests << '\n';
        }

        ne.evolution();

        if (gen % 1000 == 0) {
            ne.writeInFile("models/gen_" + std::to_string(gen) + ".csv");
        }

        if (gen % 1000 == 0) {
            envs[0].demonstrate(ne.demonstrate());
        }
    }

    return 0;
}