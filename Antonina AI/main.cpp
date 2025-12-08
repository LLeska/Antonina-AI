#include <iostream>
#include <string>
#include "AntoninaAPI.h"
#include "NeuroEvolution.h"


int main() {
    int* sizes = new int[6] { 64, 20, 20, 20, 20, 4 };
    int population = 100;
    NeuroEvolution ne(0.2, 6, sizes, 4, population);
    int generations = 1000;
    int epochs = 1;// 896 4032 116928

    AntoninaAPI environment;
    //environment.writeInFile();
    int gen = 0;
    while (true){
        gen++;
        std::cout << gen << '\n';
        for (int epoch = 0; epoch < epochs; epoch++) {
            ne.setFitness(environment.solveFitness(ne.getNeuros(), population));
        }
        ne.evolution();
        ne.writeInFile("models/gen_" + std::to_string(gen) + ".csv");
        //environment.demonstrate(ne.demonstrate());
    }
    return 0;
}
