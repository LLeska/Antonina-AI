#include <iostream>
#include <cmath>
#include <ctime>
#include <random>
#include <chrono>
#include "Perceptron.h"


int main() {
    int inputs_size = 2;/*
    double learning_rate = 0.01;
    int size = 4;
    int sizes[4] = { inputs_size, 8, 4, 2 };
    Perceptron p(learning_rate, size, sizes);*/
    Perceptron p;
    p.readFromFile("AND.csv");
    int inputs_length = 4;
    double inputs[4][2] = { {0, 0},{0,1},{1,0},{1,1} };
    int output_size = 2;
    double targets[4][2] = { {1, 0}, {1, 0}, {1, 0}, {0, 1} };

    unsigned int epochs = 1000000;
    double* outputs;
    for (unsigned int num = 0; num < epochs; num++) {
        if (num == 0 || num == epochs - 1) {
            std::cout << num << '\n';
        }
        //cout << num << '\n';
        for (int i = 0; i < inputs_length; i++) {
            outputs = p.feedForward(inputs[i], inputs_size);
            if (num == 0 || num == epochs - 1) {
                for (int k = 0; k < output_size; k++) {
                    double tar = targets[i][k];
                    double out = outputs[k];
                    std::cout << out << " : " << tar << ';';
                }
                std::cout << '\n';
            }
            p.backpropagation(targets[i]);
        }

    }
    p.writeInFile("AND1.csv");
    p.deinit();

}
