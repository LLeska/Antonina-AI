#include <iostream>
#include <string>
#include "AntoninaAPI.h"
#include "NeuroEvolution.h"


int main() {
    int* sizes = new int[6] { 64, 6, 50, 50, 50, 4 };
    int population = 50;
    NeuroEvolution ne(0.1, 6, sizes, 10, population);
    int generations = 1000;
    int epochs = 10;

    AntoninaAPI environment;
    for (int gen = 0; gen < generations; gen++) {
        std::cout << gen << '\n';
        for (int epoch = 0; epoch < epochs; epoch++) {
            ne.setFitness(environment.solveFitness(ne.getNeuros(), population));
        }
        ne.evolution();
        ne.writeInFile("models/gen_" + std::to_string(gen) + ".csv");
        //environment.demonstrate(ne.demonstrate());
    }
}
