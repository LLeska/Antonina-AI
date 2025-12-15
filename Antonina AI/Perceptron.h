#pragma once
#include "Layer.h"
#include <string>


class Perceptron
{
private:
    double learningRate;
    Layer* layers;
    int length;
    double EPSILON;
    double NOT_MUTAHION;

    template<typename T>
    T random_in_range(T a, T b);

    double activation(double x);

    double dactivation(double y);

    double solveZ(int L, int j);

    template <typename T>
    void copyArray(int n, T* array_source, T* array_destination);
public:    
    bool isInitialized();

    Perceptron(double learningRate_, int layers_length_, int* sizes);

    Perceptron();

    Perceptron(Perceptron& p);

    Perceptron& operator=(const Perceptron& other);

    ~Perceptron();

    Perceptron(Perceptron* p1, Perceptron* p2);

    void feedForward(double* inputs);

    int getOut();

    void backpropagation(double* targets);

    void readFromFile(std::string file);

    void writeInFile(std::string file);

    void readFromFile(std::ifstream* fin);

    void writeInFile(std::ofstream* fout);

    void deInit();
};

