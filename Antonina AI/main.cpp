#include <iostream>
#include <string>
#include "AntoninaAPI.h"
#include "NeuroEvolution.h"


int main() {
    int* sizes = new int[5] { 64, 32, 16, 8, 4 };  
    int population = 100;  
    NeuroEvolution ne(0.01, 5, sizes, 10, population);

    AntoninaAPI environment;
    int start = 9;
    ne.readFromFile("models/gen_" + std::to_string(start) + ".csv");
    for (int gen = start; gen < 1000+start; gen++) {
        std::cout << "gen " << gen << std::endl;

        int* fitness = environment.solveFitness(ne.getNeuros(), population, 0);
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
        if (gen % 10 == 0) {
            environment.demonstrate(ne.demonstrate());
        }
		ne.writeInFile("models/gen_" + std::to_string(gen) + ".csv");
    }

    delete[] sizes;
    return 0;
}
