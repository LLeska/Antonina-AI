#pragma once
#include "Layer.h"
#include <string>

class Perceptron {
private:
    double learningRate;
    Layer* layers;
    int length;
    double EPSILON;
    double NOT_MUTAHION;

    template<typename T>
    T random_in_range(T a, T b);

public:
    Perceptron();
    Perceptron(double learningRate_, int length_, int* sizes);
    Perceptron(const Perceptron& p);
    Perceptron(Perceptron&& p);
    Perceptron(Perceptron* p1, Perceptron* p2);
    Perceptron& operator=(const Perceptron& p);
    Perceptron& operator=(Perceptron&& p);
    ~Perceptron();

    void feedForward(double* inputs);
    int getOut();
    void mutate(double sigma, double prob);
    bool isInitialized();
    void deInit();

    void readFromFile(std::ifstream* fin);
    void writeInFile(std::ofstream* fout);
    void readFromFile(std::string file);
    void writeInFile(std::string file);
};