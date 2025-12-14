#include <iostream>
#include <string>
#include "AntoninaAPI.h"
#include "NeuroEvolution.h"


int main() {
    int* sizes = new int[6] { 64, 64, 48, 32, 16, 4 };
    int population = 1000;
    NeuroEvolution ne(1, 6, sizes, 10, population);
    int generations = 1000;
    int epochs = 1000;// 896 4032 116928

    AntoninaAPI environment;
    //environment.writeInFile();
    int gen = 0;
    int tests = 1;
    while (true){
        gen++;
        std::cout << gen << '\n';
        for (int epoch = 1; epoch < epochs+1; epoch++) {
            ne.setFitness(environment.solveFitness(ne.getNeuros(), population, tests));
            
            ne.evolution();
            ne.writeInFile("models/gen_" + std::to_string(gen) + "epoch_" + std::to_string(epoch) + ".csv");
        }
        tests++;
        environment.demonstrate(ne.demonstrate());
    }
    return 0;
}
