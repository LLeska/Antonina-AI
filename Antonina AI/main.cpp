#include <iostream>
#include <string>
#include <thread>
#include "AntoninaAPI.h"
#include "NeuroEvolution.h"


int main() {
    double learning_rate = 0.01;
    int length = 5;
    int* sizes = new int[length] { 64, 32, 16, 8, 4 };
    int parent_size = 100;
    int population = 1000;
    
    NeuroEvolution ne(learning_rate, length, sizes, parent_size, population);
    AntoninaAPI environment;
    
    int start = 8000;


    ne.readFromFile("models/gen_" + std::to_string(start) + ".csv");

    for (int gen = start; gen < 1000001 + start; gen++) {
        if (gen % 10 == 0) {

            ne.writeInFile("models/gen_" + std::to_string(gen) + ".csv");
        }
        std::cout << "gen " << gen << std::endl;
        int* fitness = new int[population];
        std::thread* threads = new std::thread[population];

        for (int i = 0; i < population; i++) {
            Perceptron* neuron = ne.getNeuros(i);
            threads[i] = std::thread(
                [neuron, &environment, fitness, i]() {
                    fitness[i] = environment.solveFitness(neuron, 0);
                }
            );
        }


        for (int i = 0; i < population; i++) {
            threads[i].join();
        }
        delete[] threads;

        ne.setFitness(fitness);

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

        delete[] fitness;

        ne.evolution();

        if (gen % 50 == 0) {
            environment.demonstrate(ne.demonstrate());
        }
    }

    delete[] sizes;
    return 0;
}
