#pragma once
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>

class FullMasteryOptimizer {
public:
    FullMasteryOptimizer(std::string best_file, std::string checkpoint_file) : best_file_(std::move(best_file)), checkpoint_file_(std::move(checkpoint_file)), saved_(false) {
    }

    template <typename SaveBestFn, typename SavePopulationFn>
    void handle(int gen, int wins, int max_tests, int active_tests, int best_index, int max_fitness, SaveBestFn save_best, SavePopulationFn save_population) {
        if (active_tests < max_tests || wins != active_tests)
            return;
        if (saved_)
            return;

        std::filesystem::create_directories("models");
        bool saved_best = save_best(best_index, best_file_, max_fitness);
        bool saved_checkpoint = save_population(checkpoint_file_, active_tests);

        if (saved_best && saved_checkpoint) {
            saved_ = true;
            std::cout << "[mastery] gen=" << gen << " wins=" << wins << '/' << max_tests << " fitness=" << max_fitness << " saved " << best_file_ << " and " << checkpoint_file_ << '\n';
        } else {
            std::cout << "[mastery] gen=" << gen << " failed to save final model files" << '\n';
        }
    }

private:
    std::string best_file_;
    std::string checkpoint_file_;
    bool saved_;
};

