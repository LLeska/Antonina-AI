#include <iostream>
#include <string>
#include <thread>
#include <memory>
#include <vector>
#include "AntoninaAPI.h"
#include "NeuroEvolution.h"


int main() {
    double learning_rate = 0.01;
    int length = 5;
    std::unique_ptr<int[]> sizes(new int[length] { 64, 32, 16, 8, 4 });
    int parent_size = 100;
    int population = 1000;

    NeuroEvolution ne(learning_rate, length, sizes.get(), parent_size, population);
    AntoninaAPI environment;

    int start = 8000;


    ne.readFromFile("models/gen_" + std::to_string(start) + ".csv");

    for (int gen = start; gen < 1000001 + start; gen++) {
        if (gen % 10 == 0) {

            ne.writeInFile("models/gen_" + std::to_string(gen) + ".csv");
        }
        std::cout << "gen " << gen << std::endl;
        std::vector<int> fitness(population, 0);
        std::vector<std::thread> threads;
        threads.reserve(population);

        for (int i = 0; i < population; i++) {
            Perceptron* neuron = ne.getNeuros(i);
            int* fitness_ptr = fitness.data();
            threads.emplace_back(
                [neuron, &environment, fitness_ptr, i]() {
                    fitness_ptr[i] = environment.solveFitness(neuron, 0);
                }
            );
        }


        for (auto& t : threads) {
            t.join();
        }

        ne.setFitness(fitness.data());

        int max_fitness = fitness[0];
        int min_fitness = fitness[0];
        long long int avg = 0;

        for (int i = 0; i < population; i++) {
            if (fitness[i] > max_fitness) max_fitness = fitness[i];
            if (fitness[i] < min_fitness) min_fitness = fitness[i];
            avg += fitness[i];
        }
        avg /= population;

        std::cout << "max: " << max_fitness
            << " min: " << min_fitness
            << " avg: " << avg << std::endl;

        ne.evolution();

        if (gen % 50 == 0) {
            environment.demonstrate(ne.demonstrate());
        }
    }

    return 0;
}
